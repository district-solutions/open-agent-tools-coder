"""Interaction modes for coder.

Modes change two things:
  - The approval posture (auto/supervised/plan)
  - A system-prompt overlay that steers the model's behavior and output style

Modes are process-wide (last write wins) and read by the session processor
when building each turn's system prompt.
"""
from __future__ import annotations

from enum import Enum

from oats.cli.approval import ApprovalMode


class InteractionMode(str, Enum):
    """Interaction modes that control approval posture and system-prompt overlay."""
    EDIT = "edit"
    AUTO = "auto"
    PLAN = "plan"
    CAVEMAN = "caveman"


_EDIT_GUIDANCE = (
    "# Mode: edit (supervised)\n\n"
    "You may read, search, and propose edits. The user approves each write/bash action. "
    "Before a destructive or wide-ranging edit, explain briefly why it's needed."
)

_AUTO_GUIDANCE = (
    "# Mode: auto-edit\n\n"
    "All actions auto-approved. Carry tasks to completion without asking for permission on each step. "
    "Still confirm before hard-to-reverse actions (force push, rm -rf, dropping tables)."
)

_PLAN_GUIDANCE = (
    "# Mode: plan (review-only)\n\n"
    "You are in PLAN mode. Do NOT perform any writes, edits, or state-changing bash commands. "
    "Reads, greps, globs, and read-only inspections are fine. "
    "Produce a concrete, numbered implementation plan with: files to touch (paths + line hints), "
    "order of operations, risks, and validation steps. End with a one-line summary the user can approve."
)

_CAVEMAN_GUIDANCE = (
    "# Mode: caveman (terse + auto)\n\n"
    "Output style: TERSE. Drop articles, filler, hedging, pleasantries. Fragments OK.\n"
    "Keep: identifiers, paths, URLs, versions, error messages, code verbatim, logical connectors (because/so/then).\n"
    "Drop: 'just', 'really', 'basically', 'I'll now', 'let me', preamble, recap summaries.\n"
    "Format: `[thing] [action] [reason]. [next step].` Pattern.\n"
    "Code blocks and tool arguments unchanged — full fidelity there. Style applies to prose only.\n"
    "All actions auto-approved. Carry work to completion."
)


_MODE_GUIDANCE: dict[InteractionMode, str] = {
    InteractionMode.EDIT: _EDIT_GUIDANCE,
    InteractionMode.AUTO: _AUTO_GUIDANCE,
    InteractionMode.PLAN: _PLAN_GUIDANCE,
    InteractionMode.CAVEMAN: _CAVEMAN_GUIDANCE,
}


_MODE_APPROVAL: dict[InteractionMode, ApprovalMode] = {
    InteractionMode.EDIT: ApprovalMode.SUPERVISED,
    InteractionMode.AUTO: ApprovalMode.AUTO,
    InteractionMode.PLAN: ApprovalMode.PLAN,
    InteractionMode.CAVEMAN: ApprovalMode.AUTO,
}


_current: InteractionMode = InteractionMode.EDIT


def get_mode() -> InteractionMode:
    """Return the current interaction mode."""
    return _current


def set_mode(mode: InteractionMode) -> None:
    """Set the current interaction mode and update the approval posture."""
    global _current
    _current = mode
    try:
        from oats.cli.approval import set_approval_mode
        set_approval_mode(_MODE_APPROVAL[mode])
    except Exception:
        pass
    try:
        from oats.session.metrics import incr
        incr(f"mode_activations.{mode.value}")
    except Exception:
        pass


def mode_guidance(mode: InteractionMode | None = None) -> str:
    """Return the system-prompt guidance text for the given (or current) mode."""
    return _MODE_GUIDANCE[mode or _current]


def approval_for(mode: InteractionMode) -> ApprovalMode:
    """Return the approval mode corresponding to an interaction mode."""
    return _MODE_APPROVAL[mode]


def describe(mode: InteractionMode) -> str:
    """Return a human-readable description of the interaction mode."""
    return {
        InteractionMode.EDIT: "edit — supervised, ask before writes",
        InteractionMode.AUTO: "auto — all operations auto-approved",
        InteractionMode.PLAN: "plan — review-only, no writes",
        InteractionMode.CAVEMAN: "caveman — terse output + auto-approve",
    }[mode]
