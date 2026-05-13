"""Build a high-signal system prompt for autonomous coding sessions."""
from __future__ import annotations

import asyncio
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from oats.log import cl
from oats.core.features import active_tool_guidance_enabled

log = cl("session.prompt")

BASE_PROMPT = """# Role

You are a software engineering agent in the user's environment. Complete requests end-to-end with strong judgment and minimal back-and-forth.

# Behavior

- Be proactive: inspect, plan internally, carry work to completion.
- Prefer doing work over describing it. Read files before editing.
- On ambiguous requests with clear intent, assume and continue.
- On meaningful tradeoffs, explain briefly and choose the safest high-value path.

# Tool Use

- Use tools to reduce uncertainty. Emit tool calls, not pseudo-code.
- ALWAYS use `read`/`glob`/`grep` tools — never `bash` with cat/find/grep.
- Use `bash` only for builds, tests, git, and commands with no dedicated tool.
- Prefer precise edits over broad rewrites. Anchor edits to inspected content.
- Prefer few high-signal tool calls over long speculative prose.
- Keep tool arguments clean, explicit, and valid JSON.
- After tool results, continue the task — don't stop early.

# Editing

- Match existing code style. Avoid unnecessary churn.
- Verify important changes with tests or builds when possible.

# Communication

- Be concise, precise, action-oriented. Surface risks plainly.
- Optimize for usefulness, speed, and quality.
"""


async def build_system_prompt(
    working_dir: str,
    project_dir: str,
    session_id: str | None = None,
    active_tool_names: list[str] | None = None,
    budget_guidance: str | None = None,
    task_guidance: str | None = None,
    mode_guidance: str | None = None,
    user_prompt: str | None = None,
) -> str:
    """
    Build the complete system prompt with all dynamic sections.

    Args:
        working_dir: Current working directory
        project_dir: Project root directory
        session_id: Optional session ID for hook context
    """
    sections = [BASE_PROMPT]

    # Current date/time
    sections.append(
        "\n# Environment\n\n"
        f"Current date: {datetime.now().strftime('%Y-%m-%d')}\n"
        f"Current working directory: {working_dir}\n"
        f"Project directory: {project_dir}"
    )

    # Git status
    git_section = await _get_git_status(working_dir)
    if git_section:
        sections.append(git_section)

    # Persistent memories
    memory_section = await _get_memory_section(project_dir)
    if memory_section:
        sections.append(memory_section)

    if active_tool_guidance_enabled():
        tool_section = await _get_tool_guidance(active_tool_names or [])
        if tool_section:
            sections.append(tool_section)

    if budget_guidance:
        sections.append(budget_guidance)

    if task_guidance:
        sections.append(task_guidance)

    return "\n\n".join(sections)


async def _get_git_status(working_dir: str) -> str | None:
    """Get git status information for the system prompt."""
    try:
        # Get current branch
        branch_proc = await asyncio.create_subprocess_shell(
            f"git -C {working_dir} rev-parse --abbrev-ref HEAD 2>/dev/null",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        branch_out, _ = await asyncio.wait_for(branch_proc.communicate(), timeout=5)
        branch = branch_out.decode().strip()

        if not branch:
            return None

        lines = [f"# Git Status\n\nCurrent branch: {branch}"]

        # Get dirty files
        status_proc = await asyncio.create_subprocess_shell(
            f"git -C {working_dir} status --porcelain 2>/dev/null | head -20",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        status_out, _ = await asyncio.wait_for(status_proc.communicate(), timeout=5)
        status = status_out.decode().strip()

        if status:
            lines.append(f"\nUncommitted changes:\n```\n{status}\n```")
        else:
            lines.append("\nWorking tree clean.")

        # Get recent commits
        log_proc = await asyncio.create_subprocess_shell(
            f"git -C {working_dir} log --oneline -5 2>/dev/null",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        log_out, _ = await asyncio.wait_for(log_proc.communicate(), timeout=5)
        git_log = log_out.decode().strip()

        if git_log:
            lines.append(f"\nRecent commits:\n```\n{git_log}\n```")

        return "\n".join(lines)

    except Exception as e:
        log.info(f"### Sorry!! _get_git_status - failed:\n```\n{traceback.format_exc()}\n```\n")
        return None


async def _get_memory_section(project_dir: str) -> str | None:
    """Load persistent memories for the system prompt."""
    try:
        from oats.memory.manager import MemoryManager
        mm = MemoryManager(project_dir=Path(project_dir))
        section = await mm.build_system_prompt_section()
        if section:
            return f"# Persistent Memories\n\n{section}"
        return None
    except Exception as e:
        log.info(f"### Sorry!! _get_memory_section - memory_load_failed:\n```\n{traceback.format_exc()}\n```\n")
        return None


async def _get_tool_guidance(active_tool_names: list[str]) -> str | None:
    """Minimal tool loading hint — tool schemas are already sent as definitions."""
    try:
        from oats.tool.registry import list_tools

        active = set(active_tool_names)
        deferred_count = sum(
            1 for t in list_tools()
            if t.name not in active and not t.always_load and t.name != "tool_search"
        )

        if deferred_count == 0:
            return None

        return (
            "# Tool Loading\n\n"
            f"{deferred_count} additional tools available via `tool_search`."
        )
    except Exception as e:
        log.info(f"### Sorry!! _get_tool_guidance - failed:\n```\n{active_tool_names}\n```\n\n#### Error\n```\n{traceback.format_exc()}\n```\n")
        return None
