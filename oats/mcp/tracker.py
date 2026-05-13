"""
MD file tracking system for tool call results.

Uses a single consolidated markdown log per session (append-mode) rather than
individual files per call. This avoids filesystem bloat at 200-1000+ calls per
session. Similar to how LiteLLM logs to a single structured destination per
session rather than per-event files.

Structure:
    .coder/mcp_tracking/
    ├── {session_id}.md          # Consolidated session log (appended per call)
    ├── {session_id}_ranking.md  # Latest ranking snapshot (overwritten)
    └── _global_stats.md         # Cross-session tool stats (overwritten)

Rotation: When a session log exceeds MAX_LOG_SIZE_BYTES, it is rotated to
{session_id}.1.md and a fresh file is started. This keeps any single file
from growing unbounded during very long sessions.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from oats.log import cl
from oats.mcp.models import (
    OrchestrationSession,
    RankingIndex,
    ToolCallRecord,
    ToolCallStatus,
)

log = cl("mcp.tracker")

# Rotate session logs after 5MB to keep files manageable
MAX_LOG_SIZE_BYTES = 5 * 1024 * 1024
# Truncate individual result/error output to keep log lean
MAX_RESULT_CHARS = 1000
MAX_ERROR_CHARS = 500


class ToolCallTracker:
    """
    Tracks tool calls in a single consolidated markdown file per session.

    Each call is appended as a section. The file is human-readable and
    grep-friendly. Ranking snapshots and global stats are separate files
    that get overwritten (not appended) since only the latest matters.
    """

    def __init__(self, tracking_dir: Path | None = None) -> None:
        self._base_dir = tracking_dir or Path(
            os.getenv("MCP_TRACKING_DIR", ".coder/mcp_tracking")
        )

    def init_session(self, session: OrchestrationSession) -> Path:
        """Create tracking dir and write session header."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        filepath = self._session_log_path(session.session_id)

        header = (
            f"# MCP Session: {session.session_id}\n"
            f"Started: {datetime.fromtimestamp(session.started_at).isoformat()}\n\n"
            f"---\n\n"
        )
        filepath.write_text(header)
        return filepath

    def record_call(
        self,
        session: OrchestrationSession,
        record: ToolCallRecord,
    ) -> Path:
        """Append a tool call record to the session log."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        filepath = self._session_log_path(session.session_id)

        # Rotate if needed
        self._maybe_rotate(filepath, session.session_id)

        entry = self._format_call_entry(record, session.total_calls)

        with open(filepath, "a") as f:
            f.write(entry)

        return filepath

    def update_ranking(
        self,
        session: OrchestrationSession,
        ranking: RankingIndex,
    ) -> Path:
        """Write the current ranking index (overwrite, not append)."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        filepath = self._base_dir / f"{session.session_id}_ranking.md"
        filepath.write_text(self._format_ranking(ranking))
        return filepath

    def write_session_summary(self, session: OrchestrationSession) -> Path:
        """Append a summary block to the end of the session log."""
        filepath = self._session_log_path(session.session_id)

        summary = self._format_summary(session)
        with open(filepath, "a") as f:
            f.write(summary)

        return filepath

    def write_global_stats(self, stats: dict[str, Any]) -> Path:
        """Write global cross-session statistics (overwrite)."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        filepath = self._base_dir / "_global_stats.md"

        lines = [
            "# MCP Tool Calling - Global Statistics",
            f"\nLast updated: {datetime.now().isoformat()}",
            "",
            "## Tool Performance",
            "",
            "| Tool | Calls | Success Rate | Avg Latency |",
            "|------|-------|-------------|-------------|",
        ]

        for tool_name, tool_stats in sorted(stats.items()):
            calls = tool_stats.get("total_calls", 0)
            success_rate = tool_stats.get("success_rate", 0) * 100
            avg_latency = tool_stats.get("avg_latency_ms", 0)
            lines.append(
                f"| {tool_name} | {calls} | {success_rate:.1f}% | {avg_latency:.0f}ms |"
            )

        filepath.write_text("\n".join(lines) + "\n")
        return filepath

    # --- Private ---

    def _session_log_path(self, session_id: str) -> Path:
        return self._base_dir / f"{session_id}.md"

    def _maybe_rotate(self, filepath: Path, session_id: str) -> None:
        """Rotate log file if it exceeds MAX_LOG_SIZE_BYTES."""
        if not filepath.exists():
            return
        if filepath.stat().st_size < MAX_LOG_SIZE_BYTES:
            return

        # Find next rotation number
        i = 1
        while True:
            rotated = self._base_dir / f"{session_id}.{i}.md"
            if not rotated.exists():
                break
            i += 1

        filepath.rename(rotated)
        log.info(f"log_rotated: {filepath.name} -> {rotated.name}")

        # Start fresh with a continuation header
        filepath.write_text(
            f"# MCP Session: {session_id} (continued, part {i + 1})\n\n---\n\n"
        )

    def _format_call_entry(self, record: ToolCallRecord, seq: int) -> str:
        """Format a single call as a compact markdown section."""
        ts = datetime.fromtimestamp(record.started_at).strftime("%H:%M:%S")
        latency = f"{record.latency_ms:.0f}ms" if record.latency_ms else "pending"
        status = record.status.value

        lines = [
            f"### #{seq} `{record.tool_name}` [{status}] — {ts}",
            f"Server: {record.server_name} | Depth: {record.depth} | Latency: {latency}",
        ]

        if record.parent_call_id:
            lines.append(f"Parent: {record.parent_call_id}")

        # Compact arguments (single line if short, block if long)
        args_str = json.dumps(record.arguments, separators=(",", ":"))
        if len(args_str) <= 120:
            lines.append(f"Args: `{args_str}`")
        else:
            lines.extend(["", "```json", json.dumps(record.arguments, indent=2)[:500], "```"])

        # Result (truncated)
        if record.result:
            preview = record.result[:MAX_RESULT_CHARS]
            if len(record.result) > MAX_RESULT_CHARS:
                preview += f"\n... ({len(record.result)} chars total)"
            lines.extend(["", f"**Result:** {preview}"])

        # Error
        if record.error:
            lines.append(f"\n**Error:** {record.error[:MAX_ERROR_CHARS]}")

        # Resolution chain (only if non-empty)
        if record.resolution_chain:
            chain = " → ".join(record.resolution_chain)
            lines.append(f"\n**Resolution chain:** {chain}")

        lines.append("\n---\n")
        return "\n".join(lines)

    def _format_summary(self, session: OrchestrationSession) -> str:
        """Format session summary block."""
        completed = [
            r for r in session.call_records
            if r.status in (ToolCallStatus.SUCCESS, ToolCallStatus.ERROR, ToolCallStatus.RESOLVED)
        ]
        successes = sum(
            1 for r in completed
            if r.status in (ToolCallStatus.SUCCESS, ToolCallStatus.RESOLVED)
        )
        avg_latency = (
            sum(r.latency_ms or 0 for r in completed) / len(completed)
            if completed else 0
        )
        stuck = sum(1 for r in session.call_records if r.status == ToolCallStatus.STUCK)
        duration = time.time() - session.started_at

        return (
            f"\n## Session Summary\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Total calls | {session.total_calls} |\n"
            f"| Successes | {successes} |\n"
            f"| Errors | {len(completed) - successes} |\n"
            f"| Stuck | {stuck} |\n"
            f"| Success rate | {successes / max(len(completed), 1) * 100:.1f}% |\n"
            f"| Avg latency | {avg_latency:.0f}ms |\n"
            f"| Max depth | {session.max_depth} |\n"
            f"| Duration | {duration:.1f}s |\n"
        )

    def _format_ranking(self, ranking: RankingIndex) -> str:
        """Format the ranking index as markdown."""
        lines = [
            "# Tool Ranking Index",
            f"\nLast updated: {datetime.fromtimestamp(ranking.last_updated).isoformat()}",
            "",
            "| Rank | Tool | Server | Score | Relevance | Reliability | Latency |",
            "|------|------|--------|-------|-----------|-------------|---------|",
        ]

        for i, entry in enumerate(ranking.top_k(50), 1):
            lines.append(
                f"| {i} | {entry.tool_name} | {entry.server_name} "
                f"| {entry.score:.3f} | {entry.relevance_score:.3f} "
                f"| {entry.reliability_score:.3f} | {entry.latency_score:.3f} |"
            )

        return "\n".join(lines) + "\n"
