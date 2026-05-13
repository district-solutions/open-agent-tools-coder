"""
Interactive approval system for coder2.

Provides Yes / Yes-to-all / No / No+instructions approval prompts
for tool operations that need user confirmation.

Modes:
  AUTO       — approve everything (default with -y flag)
  SUPERVISED — ask before write/bash/delete operations
  PLAN       — review-only, no execution
"""
from __future__ import annotations

import sys
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from rich.console import Console

console = Console()

# Read-only tools that never need approval
_READ_ONLY_TOOLS = frozenset({
    "read", "glob", "grep", "tool_search", "todoread",
    "memory_read", "plan_status", "agent_status",
    "question", "askuser", "webfetch", "websearch",
    "check_certificate_expiration", "convert_pq_to_json",
    "get_app_manifest", "get_app_logs", "get_app_credentials",
    "lsp", "generate_readme", "plan_enter", "plan_exit",
})


class ApprovalMode(str, Enum):
    """Operating modes for the approval system."""
    AUTO = "auto"              # Auto-approve everything
    SUPERVISED = "supervised"  # Ask for each operation
    PLAN = "plan"              # Review-only, no execution


class ApprovalAction(str, Enum):
    """Result of an approval prompt."""
    YES = "yes"
    YES_ALL = "yes_all"
    NO = "no"
    NO_WITH_INSTRUCTIONS = "no_with_instructions"


@dataclass
class ApprovalResult:
    """Result from an approval prompt."""
    action: ApprovalAction
    instructions: Optional[str] = None


class ApprovalManager:
    """Manages the approval flow for tool operations."""

    def __init__(self, mode: ApprovalMode = ApprovalMode.AUTO):
        self._mode = mode
        self._auto_approved_tools: set[str] = set()

    @property
    def mode(self) -> ApprovalMode:
        return self._mode

    @mode.setter
    def mode(self, value: ApprovalMode):
        self._mode = value
        if value == ApprovalMode.AUTO:
            self._auto_approved_tools.clear()

    def needs_approval(self, tool_name: str) -> bool:
        """Check if a tool operation needs user approval."""
        if self._mode == ApprovalMode.AUTO:
            return False
        if tool_name in self._auto_approved_tools:
            return False
        if tool_name in _READ_ONLY_TOOLS:
            return False
        return True

    def prompt_approval(self, tool_name: str, description: str) -> ApprovalResult:
        """
        Prompt the user for approval of a tool operation.

        Uses plain input() to avoid conflicting with the main REPL's
        prompt_toolkit application (which is already running in the
        asyncio event loop when this is called via run_in_executor).

        y/Enter = yes, a = yes to all, n = no, i = no + instructions
        """
        console.print()
        console.print(
            f"  [bold yellow]? Approve:[/bold yellow] "
            f"[cyan]{tool_name}[/cyan] {description}"
        )
        console.print(
            f"  [dim]  (y)es / Enter  |  (a)ll  |  (n)o  |  (i)nstructions[/dim]"
        )

        try:
            raw = input("  > ").strip().lower()

            if raw in ("", "y", "yes"):
                return ApprovalResult(action=ApprovalAction.YES)

            if raw in ("a", "all"):
                self._mode = ApprovalMode.AUTO
                console.print(
                    f"  [green]auto-approve enabled for this session[/green]"
                )
                return ApprovalResult(action=ApprovalAction.YES_ALL)

            if raw in ("i", "instructions", "inst"):
                console.print(
                    f"  [dim]Enter additional instructions (Enter to submit):[/dim]"
                )
                instructions = input("  instructions> ").strip()
                return ApprovalResult(
                    action=ApprovalAction.NO_WITH_INSTRUCTIONS,
                    instructions=instructions if instructions else None,
                )

            # Anything else (n, no, etc.) = decline
            console.print(f"  [yellow]skipped[/yellow]")
            return ApprovalResult(action=ApprovalAction.NO)

        except (EOFError, KeyboardInterrupt):
            return ApprovalResult(action=ApprovalAction.NO)

    def auto_approve_tool(self, tool_name: str):
        """Mark a specific tool type as auto-approved for this session."""
        self._auto_approved_tools.add(tool_name)

    def reset(self):
        """Reset to initial state."""
        self._auto_approved_tools.clear()


_manager: Optional["ApprovalManager"] = None


def get_approval_manager() -> "ApprovalManager":
    """Return the process-wide approval manager, creating one in AUTO on first call."""
    global _manager
    if _manager is None:
        _manager = ApprovalManager(mode=ApprovalMode.AUTO)
    return _manager


def set_approval_mode(mode: ApprovalMode) -> None:
    """Set the approval mode on the shared manager."""
    get_approval_manager().mode = mode
