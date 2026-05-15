"""
Minimal stdio LSP client/manager for coder.

This is intentionally lightweight: it targets the highest-value editor-style
operations without pulling in additional heavy dependencies.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote_from_bytes, unquote, urlparse
from oats.log import cl

log = cl("lsp.client")


def path_to_uri(path: Path) -> str:
    """Convert a filesystem path to a file:// URI."""
    return "file://" + quote_from_bytes(str(path.resolve()).encode("utf-8"))


def uri_to_path(uri: str) -> str:
    """Convert a file:// URI back to a filesystem path."""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return uri
    return unquote(parsed.path)


def language_id_for_path(path: Path) -> str:
    """Map a file extension to its LSP language ID."""
    ext = path.suffix.lower()
    return {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescriptreact",
        ".js": "javascript",
        ".jsx": "javascriptreact",
        ".rs": "rust",
        ".go": "go",
        ".c": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".json": "json",
        ".md": "markdown",
    }.get(ext, ext.lstrip(".") or "plaintext")


def detect_server_command(path: Path) -> list[str] | None:
    """Detect the LSP server command for a file based on its extension."""
    ext = path.suffix.lower()
    env_map = {
        ".py": "CODER_LSP_SERVER_PYTHON",
        ".ts": "CODER_LSP_SERVER_TYPESCRIPT",
        ".tsx": "CODER_LSP_SERVER_TYPESCRIPT",
        ".js": "CODER_LSP_SERVER_TYPESCRIPT",
        ".jsx": "CODER_LSP_SERVER_TYPESCRIPT",
        ".rs": "CODER_LSP_SERVER_RUST",
        ".go": "CODER_LSP_SERVER_GO",
        ".c": "CODER_LSP_SERVER_CPP",
        ".cc": "CODER_LSP_SERVER_CPP",
        ".cpp": "CODER_LSP_SERVER_CPP",
        ".h": "CODER_LSP_SERVER_CPP",
        ".hpp": "CODER_LSP_SERVER_CPP",
    }
    env_key = env_map.get(ext)
    if env_key and os.getenv(env_key):
        return os.getenv(env_key, "").split()

    candidates: dict[str, list[list[str]]] = {
        ".py": [
            ["basedpyright-langserver", "--stdio"],
            ["pyright-langserver", "--stdio"],
            ["pylsp"],
        ],
        ".ts": [["typescript-language-server", "--stdio"]],
        ".tsx": [["typescript-language-server", "--stdio"]],
        ".js": [["typescript-language-server", "--stdio"]],
        ".jsx": [["typescript-language-server", "--stdio"]],
        ".rs": [["rust-analyzer"]],
        ".go": [["gopls"]],
        ".c": [["clangd"]],
        ".cc": [["clangd"]],
        ".cpp": [["clangd"]],
        ".h": [["clangd"]],
        ".hpp": [["clangd"]],
    }
    for cmd in candidates.get(ext, []):
        if shutil.which(cmd[0]):
            return cmd
    return None


def detect_workspace_server_command(root_dir: Path) -> list[str] | None:
    """Detect the best LSP server command for a workspace root."""
    for env_key in [
        "CODER_LSP_SERVER_WORKSPACE",
        "CODER_LSP_SERVER_TYPESCRIPT",
        "CODER_LSP_SERVER_PYTHON",
        "CODER_LSP_SERVER_CPP",
        "CODER_LSP_SERVER_GO",
        "CODER_LSP_SERVER_RUST",
    ]:
        value = os.getenv(env_key)
        if value:
            return value.split()
    for binary, args in [
        ("typescript-language-server", ["typescript-language-server", "--stdio"]),
        ("basedpyright-langserver", ["basedpyright-langserver", "--stdio"]),
        ("pyright-langserver", ["pyright-langserver", "--stdio"]),
        ("clangd", ["clangd"]),
        ("gopls", ["gopls"]),
        ("rust-analyzer", ["rust-analyzer"]),
    ]:
        if shutil.which(binary):
            return args
    return None


async def _read_lsp_message(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Read a single LSP/Content-Length message from the server stdout."""
    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        if not line:
            raise EOFError("LSP server closed stdout")
        if line == b"\r\n":
            break
        key, value = line.decode("utf-8").split(":", 1)
        headers[key.strip().lower()] = value.strip()

    content_length = int(headers.get("content-length", "0"))
    if content_length <= 0:
        raise ValueError("Missing Content-Length in LSP response")
    body = await reader.readexactly(content_length)
    return json.loads(body.decode("utf-8"))


async def _write_lsp_message(
    writer: asyncio.StreamWriter,
    payload: dict[str, Any],
) -> None:
    """Write a JSON-RPC message to the LSP server stdin."""
    body = json.dumps(payload).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    writer.write(header + body)
    await writer.drain()


@dataclass
class LSPServerInstance:
    """Represents a running LSP server process and its state."""
    root_dir: Path
    command: list[str]
    process: asyncio.subprocess.Process | None = None
    stdout: asyncio.StreamReader | None = None
    stdin: asyncio.StreamWriter | None = None
    next_id: int = 1
    initialized: bool = False
    file_versions: dict[str, int] = field(default_factory=dict)
    diagnostics_by_uri: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    async def start(self) -> None:
        """Start the LSP server subprocess and initialize it."""
        if self.process is not None:
            return
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.root_dir),
        )
        assert self.process.stdout is not None
        assert self.process.stdin is not None
        self.stdout = self.process.stdout
        self.stdin = self.process.stdin
        await self._initialize()

    async def _initialize(self) -> None:
        """Send the initialize/initialized handshake to the server."""
        if self.initialized:
            return
        response = await self.request(
            "initialize",
            {
                "processId": None,
                "clientInfo": {"name": "coder", "version": "1.0"},
                "rootUri": path_to_uri(self.root_dir),
                "capabilities": {},
                "workspaceFolders": [
                    {"uri": path_to_uri(self.root_dir), "name": self.root_dir.name}
                ],
            },
        )
        _ = response
        await self.notify("initialized", {})
        self.initialized = True

    async def request(self, method: str, params: dict[str, Any]) -> Any:
        """Send a JSON-RPC request and wait for the matching response."""
        await self.start()
        assert self.stdin is not None
        assert self.stdout is not None
        msg_id = self.next_id
        self.next_id += 1
        await _write_lsp_message(
            self.stdin,
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "method": method,
                "params": params,
            },
        )
        while True:
            msg = await _read_lsp_message(self.stdout)
            if self._handle_notification(msg):
                continue
            if "id" in msg and msg["id"] == msg_id:
                if "error" in msg:
                    raise RuntimeError(f"LSP {method} failed: {msg['error']}")
                return msg.get("result")

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        await self.start()
        assert self.stdin is not None
        await _write_lsp_message(
            self.stdin,
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            },
        )

    def _handle_notification(self, msg: dict[str, Any]) -> bool:
        """Process incoming server notifications; return True if consumed."""
        method = msg.get("method")
        if not method:
            return False
        if method == "textDocument/publishDiagnostics":
            params = msg.get("params", {})
            uri = params.get("uri")
            if uri:
                diagnostics = params.get("diagnostics", [])
                if isinstance(diagnostics, list):
                    self.diagnostics_by_uri[uri] = diagnostics
            return True
        return True

    async def sync_file(self, path: Path, content: str) -> None:
        """Sync file contents to the LSP server (didOpen or didChange)."""
        uri = path_to_uri(path)
        version = self.file_versions.get(uri, 0)
        if version == 0:
            await self.notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": language_id_for_path(path),
                        "version": 1,
                        "text": content,
                    }
                },
            )
            self.file_versions[uri] = 1
            return

        version += 1
        await self.notify(
            "textDocument/didChange",
            {
                "textDocument": {"uri": uri, "version": version},
                "contentChanges": [{"text": content}],
            },
        )
        self.file_versions[uri] = version

    async def collect_diagnostics(
        self,
        path: Path,
        wait_timeout_s: float = 0.75,
    ) -> list[dict[str, Any]]:
        """
        Best-effort collection of diagnostics after a sync/open/change.

        Many servers publish diagnostics asynchronously, so this polls cached
        notifications briefly instead of assuming they arrive immediately.
        """
        uri = path_to_uri(path)
        deadline = asyncio.get_running_loop().time() + max(0.0, wait_timeout_s)
        seen = self.diagnostics_by_uri.get(uri)
        while asyncio.get_running_loop().time() < deadline:
            current = self.diagnostics_by_uri.get(uri)
            if current is not None and current is not seen:
                return current
            await asyncio.sleep(0.05)
        return self.diagnostics_by_uri.get(uri, [])


class LSPManager:
    """Small manager that reuses server processes per root + command."""

    def __init__(self) -> None:
        self._instances: dict[tuple[str, tuple[str, ...]], LSPServerInstance] = {}

    def get_instance(self, root_dir: Path, command: list[str]) -> LSPServerInstance:
        """Get or create an LSP server instance for the given root and command."""
        key = (str(root_dir.resolve()), tuple(command))
        if key not in self._instances:
            self._instances[key] = LSPServerInstance(root_dir=root_dir, command=command)
        return self._instances[key]


_manager: LSPManager | None = None


def get_lsp_manager() -> LSPManager:
    """Return the process-wide LSP manager singleton."""
    global _manager
    if _manager is None:
        _manager = LSPManager()
    return _manager


async def sync_file_if_supported(root_dir: Path, path: Path, content: str) -> bool:
    """
    Best-effort LSP sync for a file after edits/writes.

    Returns True when a compatible language server exists and was updated.
    """
    command = detect_server_command(path)
    if not command:
        return False
    instance = get_lsp_manager().get_instance(root_dir, command)
    await instance.sync_file(path, content)
    return True


def format_diagnostics_summary(
    path: Path,
    diagnostics: list[dict[str, Any]],
    limit: int = 5,
) -> str | None:
    """Format cached LSP diagnostics into a compact tool-friendly summary."""
    if not diagnostics:
        return None

    severity_names = {
        1: "error",
        2: "warning",
        3: "info",
        4: "hint",
    }
    shown = diagnostics[: max(1, limit)]
    lines = [f"LSP diagnostics for {path}:"]
    for item in shown:
        severity = severity_names.get(item.get("severity"), "diagnostic")
        message = str(item.get("message", "")).strip() or "No message"
        rng = item.get("range", {})
        start = rng.get("start", {})
        line = int(start.get("line", 0)) + 1
        column = int(start.get("character", 0)) + 1
        lines.append(f"- {severity} at {line}:{column}: {message}")
    remaining = len(diagnostics) - len(shown)
    if remaining > 0:
        lines.append(f"- ... and {remaining} more")
    return "\n".join(lines)


async def sync_file_and_collect_diagnostics(
    root_dir: Path,
    path: Path,
    content: str,
) -> tuple[bool, str | None]:
    """
    Best-effort helper used by write/edit paths.

    Returns `(lsp_synced, diagnostics_summary)`.
    """
    command = detect_server_command(path)
    if not command:
        return False, None
    instance = get_lsp_manager().get_instance(root_dir, command)
    await instance.sync_file(path, content)
    diagnostics = await instance.collect_diagnostics(path)
    return True, format_diagnostics_summary(path, diagnostics)
