"""
Hook system for lifecycle event customization.

Hooks allow users to run custom commands at key lifecycle events:
- pre_tool_use: Before a tool executes (can block or modify input)
- post_tool_use: After a tool executes
- user_prompt_submit: When a user message is submitted (can block)
- session_start: When a session begins
- file_changed: When a file is modified by a tool
"""
from oats.hook.engine import (
    HookEngine,
    HookEvent,
    HookContext,
    HookResult,
)

__all__ = [
    "HookEngine",
    "HookEvent",
    "HookContext",
    "HookResult",
]
