"""
Retention policy for tool results stored in session history.

The goal is to keep enough signal for continuation without letting large tool
outputs dominate long sessions.
"""
from __future__ import annotations
from dataclasses import dataclass
from oats.tool.registry import ToolResult


@dataclass
class RetainedToolResult:
    output: str
    metadata: dict


def retain_tool_result(tool_name: str, result: ToolResult) -> RetainedToolResult:
    """
    Compress a tool result for session retention.

    The original `ToolResult` still drives the immediate turn. This helper only
    determines what should be kept in conversation history for future turns.
    """
    output = result.output or ""
    original_length = len(output)
    retained = output

    if tool_name == "read":
        retained = _compress_read_output(output)
    elif tool_name == "grep":
        retained = _compress_grep_output(output)
    elif tool_name == "bash":
        retained = _compress_bash_output(output, result.error)
    elif tool_name == "lsp":
        retained = _compress_lsp_output(output)
    elif len(output) > 4000:
        retained = _compress_generic(output, head_lines=40, tail_lines=20)

    metadata = dict(result.metadata or {})
    metadata["retained_output"] = retained
    metadata["retention_applied"] = retained != output
    metadata["original_output_chars"] = original_length
    metadata["retained_output_chars"] = len(retained)
    return RetainedToolResult(output=retained, metadata=metadata)


def _compress_read_output(output: str) -> str:
    return _compress_generic(output, head_lines=120, tail_lines=40)


def _compress_grep_output(output: str) -> str:
    return _compress_generic(output, head_lines=80, tail_lines=20)


def _compress_bash_output(output: str, error: str | None) -> str:
    if error:
        return _compress_generic(output, head_lines=120, tail_lines=80)
    return _compress_generic(output, head_lines=80, tail_lines=40)


def _compress_lsp_output(output: str) -> str:
    return _compress_generic(output, head_lines=120, tail_lines=20)


def _compress_generic(output: str, head_lines: int, tail_lines: int) -> str:
    if not output:
        return output

    lines = output.splitlines()
    total = len(lines)
    keep = head_lines + tail_lines
    if total <= keep:
        return output

    head = lines[:head_lines]
    tail = lines[-tail_lines:] if tail_lines > 0 else []
    removed = total - len(head) - len(tail)
    marker = [f"... [{removed} lines omitted for session retention] ..."]
    return "\n".join(head + marker + tail)
