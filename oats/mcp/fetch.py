#!/usr/bin/env python3
"""
mcp_fetch.py — Fetch tools from a running Playwright MCP HTTP server and
emit them as an OpenAPI 3.0 JSON specification.

The Playwright MCP server exposes a Streamable HTTP transport at /mcp
(MCP spec 2024-11-05) and a legacy SSE transport at /sse.  This module
uses the Streamable HTTP transport exclusively.

Protocol flow
-------------
1. POST /mcp  — initialize (no session header)  → get mcp-session-id from response
2. POST /mcp  — notifications/initialized (with session header)
3. POST /mcp  — tools/list (with session header)  → emit OpenAPI JSON

Usage:
    python mcp_fetch.py -u http://0.0.0.0:8931
    python mcp_fetch.py -u http://0.0.0.0:8931 -o openapi.json -p
"""

import argparse
import json
import sys
from typing import Any
import requests
from oats.log import cl

log = cl('mcp.fetch')

# MCP protocol version this client advertises
_MCP_PROTOCOL_VERSION = "2024-11-05"
_CLIENT_INFO = {"name": "mcp-fetch", "version": "1.0.0"}

# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------


def _parse_sse_body(text: str) -> Any:
    """
    Extract the first JSON payload from an SSE-formatted response body.

    SSE lines look like::

        event: message
        data: {"jsonrpc":"2.0","id":1,"result":{...}}

    Returns the parsed dict from the first ``data:`` line that contains JSON.
    Raises ValueError if no data line is found.
    """
    for line in text.splitlines():
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                return json.loads(payload)
    raise ValueError(f"No data line found in SSE body: {text!r}")


def _decode_response(resp: requests.Response) -> Any:
    """
    Decode an MCP HTTP response, handling both JSON and SSE content-types.

    The MCP Streamable HTTP transport may respond with either
    ``application/json`` or ``text/event-stream`` depending on whether the
    server has a single synchronous reply or needs to stream multiple events.
    """
    ct = resp.headers.get("content-type", "")
    if "text/event-stream" in ct:
        return _parse_sse_body(resp.text)
    return resp.json()


def _post_mcp(
    session: requests.Session,
    base_url: str,
    body: dict,
    session_id: str | None = None,
    timeout: int = 30,
) -> requests.Response:
    """POST a JSON-RPC message to ``{base_url}/mcp``."""
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id:
        headers["mcp-session-id"] = session_id

    resp = session.post(
        f"{base_url}/mcp",
        json=body,
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# MCP session helpers
# ---------------------------------------------------------------------------


def _initialize(session: requests.Session, base_url: str) -> str:
    """
    Send the MCP ``initialize`` request and return the ``mcp-session-id``.

    Raises RuntimeError if the server returns a JSON-RPC error.
    """
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": _MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": _CLIENT_INFO,
        },
    }
    resp = _post_mcp(session, base_url, body)
    data = _decode_response(resp)

    if "error" in data:
        raise RuntimeError(f"initialize failed: {data['error']}")

    session_id = resp.headers.get("mcp-session-id")
    if not session_id:
        raise RuntimeError(
            "Server did not return an mcp-session-id header after initialize. "
            "Ensure the MCP server is started with --port (HTTP transport mode)."
        )
    return session_id


def _send_initialized(
    session: requests.Session, base_url: str, session_id: str
) -> None:
    """
    Send the ``notifications/initialized`` notification.

    This is a required protocol step after ``initialize`` succeeds.
    The server returns 202 Accepted with no body for notifications.
    """
    body = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    }
    try:
        _post_mcp(session, base_url, body, session_id=session_id, timeout=10)
    except requests.exceptions.HTTPError:
        # Some servers return 202 or 204 for notifications; ignore HTTP errors here
        pass


def _list_tools(
    session: requests.Session, base_url: str, session_id: str
) -> list[dict]:
    """Send ``tools/list`` and return the tool array."""
    body = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }
    resp = _post_mcp(session, base_url, body, session_id=session_id)
    data = _decode_response(resp)

    if "error" in data:
        raise RuntimeError(f"tools/list failed: {data['error']}")

    return data.get("result", {}).get("tools", [])


# ---------------------------------------------------------------------------
# Public fetch interface
# ---------------------------------------------------------------------------


def fetch_tools(url: str) -> list[dict]:
    """
    Connect to the Playwright MCP server at *url* and return the raw tool list.

    Args:
        url: Base URL of the MCP HTTP server, e.g. ``http://0.0.0.0:8931``.

    Returns:
        List of MCP tool objects as returned by ``tools/list``.
    """
    base_url = url.rstrip("/")
    http = requests.Session()

    session_id = _initialize(http, base_url)
    _send_initialized(http, base_url, session_id)
    return _list_tools(http, base_url, session_id)


# ---------------------------------------------------------------------------
# OpenAPI 3.0 conversion
# ---------------------------------------------------------------------------


def _tag_for_tool(name: str) -> str:
    """Derive a logical grouping tag from the tool name (e.g. ``browser_click`` → ``browser``)."""
    return name.split("_")[0] if "_" in name else name


def _mcp_tool_to_path_item(tool: dict) -> dict:
    """Convert a single MCP tool definition to an OpenAPI path item."""
    name: str = tool.get("name", "unknown")
    description: str = tool.get("description", "")
    input_schema: dict = tool.get("inputSchema") or {}

    # First line of description → operation summary
    summary = description.split("\n")[0].rstrip(".") if description else name

    return {
        "post": {
            "operationId": name,
            "summary": summary,
            "description": description or None,
            "tags": [_tag_for_tool(name)],
            "requestBody": {
                "required": bool(input_schema.get("properties")),
                "content": {
                    "application/json": {
                        "schema": input_schema or {"type": "object"},
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Tool executed successfully",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "content": {
                                        "type": "array",
                                        "description": "Ordered list of result parts",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "type": {
                                                    "type": "string",
                                                    "enum": ["text", "image", "resource"],
                                                },
                                                "text": {
                                                    "type": "string",
                                                    "description": "Text content (when type=text)",
                                                },
                                            },
                                            "required": ["type"],
                                        },
                                    },
                                    "isError": {
                                        "type": "boolean",
                                        "description": "True when the tool call produced an error",
                                    },
                                },
                                "required": ["content"],
                            }
                        }
                    },
                }
            },
        }
    }


def tools_to_openapi(tools: list[dict], server_url: str) -> dict:
    """
    Convert a list of MCP tool definitions to an OpenAPI 3.0 specification.

    Each tool becomes a ``POST /{tool_name}`` path entry.  The request body
    schema is taken directly from the MCP ``inputSchema`` (which is already
    JSON Schema Draft-07 compatible).

    Args:
        tools:      List of MCP tool objects from ``tools/list``.
        server_url: Base URL used to populate the ``servers`` block.

    Returns:
        OpenAPI 3.0 dict ready to be serialised as JSON.
    """
    paths = {f"/{t['name']}": _mcp_tool_to_path_item(t) for t in tools}

    # Collect unique tags for the tags section
    tags_seen: set[str] = set()
    tags_list: list[dict] = []
    for tool in tools:
        tag = _tag_for_tool(tool.get("name", ""))
        if tag not in tags_seen:
            tags_seen.add(tag)
            tags_list.append({"name": tag})

    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Playwright MCP Server API",
            "description": (
                "OpenAPI 3.0 specification auto-generated from the Playwright MCP "
                "server's tool definitions.  Each operation maps directly to one MCP "
                "tool invocable via the /mcp endpoint."
            ),
            "version": "1.0.0",
        },
        "servers": [
            {
                "url": server_url.rstrip("/"),
                "description": "Playwright MCP HTTP server",
            }
        ],
        "tags": tags_list,
        "paths": paths,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp_fetch",
        description=(
            "Fetch Playwright MCP tools from a running HTTP server "
            "and emit an OpenAPI 3.0 JSON specification."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # print to stdout
  python mcp_fetch.py -u http://0.0.0.0:8931

  # write pretty-printed JSON to a file
  python mcp_fetch.py -u http://0.0.0.0:8931 -o openapi.json -p

  # headed MCP server on non-default port
  python mcp_fetch.py -u http://0.0.0.0:8932 -o openapi-headed.json -p
""",
    )
    parser.add_argument(
        "-u", "--url",
        required=True,
        metavar="URL",
        help="Base URL of the MCP server, e.g. http://0.0.0.0:8931",
    )
    parser.add_argument(
        "-o", "--output",
        default="-",
        metavar="FILE",
        help="Output file path (default: stdout, use '-' for stdout)",
    )
    parser.add_argument(
        "-p", "--pretty",
        action="store_true",
        help="Pretty-print JSON output (2-space indent)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress messages to stderr",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point — parse args, fetch tools, emit OpenAPI JSON."""
    args = _build_parser().parse_args(argv)

    log.info(f"Connecting to MCP server at {args.url} ...")
    try:
        tools = fetch_tools(args.url)
    except requests.exceptions.ConnectionError as exc:
        print(f"ERROR: Cannot connect to {args.url}: {exc}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        print(f"ERROR: HTTP {exc.response.status_code} from {args.url}: {exc}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"ERROR: Request to {args.url} timed out", file=sys.stderr)
        sys.exit(1)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    log.info(f"Retrieved {len(tools)} tool(s), converting to OpenAPI 3.0 ...")
    spec = tools_to_openapi(tools, args.url)

    indent = 2 if args.pretty else None
    payload = json.dumps(spec, indent=indent)

    if args.output == "-":
        print(payload)
    else:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.write("\n")
        log.info(f"Wrote openapi.json to {args.output} ({len(tools)} tools)")


if __name__ == "__main__":
    main()
