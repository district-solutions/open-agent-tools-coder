"""
Hook-driven trajectory logger.

Registers global Python handlers on the :class:`HookEngine` that persist each
turn into the SQLite/FTS5 trajectory store. Registration is idempotent and
gated by ``CODER_FEATURE_TRAJECTORY_STORE``; calling :func:`install` when the
flag is off is a no-op.

Turn indexing uses a per-session in-memory counter keyed on ``session_id``
and backfilled from :meth:`TrajectoryStore.session_turns` on first touch, so
a restarted process continues numbering where it left off.
"""
from __future__ import annotations

import threading
from typing import Optional

from oats.core.features import trajectory_store_enabled
from oats.hook.engine import HookContext, HookEngine, HookEvent
from oats.log import cl
from oats.trajectory.store import (
    KIND_PROMPT,
    KIND_TOOL_RESULT,
    TrajectoryStore,
    get_store,
)

log = cl("oats.trajectory.logger")

_installed = False
_install_lock = threading.Lock()
_counter_lock = threading.Lock()
_turn_counters: dict[str, int] = {}


def _next_turn(session_id: str, store: TrajectoryStore) -> int:
    """Return the next ``turn_idx`` for ``session_id``, backfilling from disk."""
    with _counter_lock:
        if session_id not in _turn_counters:
            existing = store.session_turns(session_id)
            _turn_counters[session_id] = max((r.turn_idx for r in existing), default=-1) + 1
        idx = _turn_counters[session_id]
        _turn_counters[session_id] = idx + 1
        return idx


async def _on_user_prompt(ctx: HookContext) -> None:
    """Hook handler: persist the user's prompt to the trajectory store.

    Skips empty prompts. Assigns the next sequential ``turn_idx`` for the
    session and records the prompt content as a ``KIND_PROMPT`` row.
    """
    if not ctx.user_prompt:
        return
    store = get_store()
    turn = _next_turn(ctx.session_id, store)
    await store.arecord(
        session_id=ctx.session_id,
        turn_idx=turn,
        role="user",
        kind=KIND_PROMPT,
        content=ctx.user_prompt,
    )


async def _on_tool_result(ctx: HookContext) -> None:
    """Hook handler: persist a tool result to the trajectory store.

    Records the tool name, a compact representation of its arguments, the
    output (truncated to 2000 chars), and any error (truncated to 500 chars)
    as a ``KIND_TOOL_RESULT`` row. Assigns the next sequential ``turn_idx``.
    """
    # We log tool *results* (post_tool_use) because that record carries the
    # full picture — args, outcome, and any error. Pre-tool events would be
    # redundant once this is written.
    if not ctx.tool_name:
        return
    store = get_store()
    turn = _next_turn(ctx.session_id, store)
    # Compact, grep-friendly serialization. Keep args inline so BM25 can
    # rank against "bash: ls -la", not just tool names.
    args_repr = ""
    if ctx.tool_args:
        args_repr = " ".join(f"{k}={str(v)[:80]}" for k, v in ctx.tool_args.items())
    content_parts = [f"{ctx.tool_name}: {args_repr}".strip()]
    if ctx.tool_result_output:
        content_parts.append(ctx.tool_result_output[:2000])
    if ctx.tool_result_error:
        content_parts.append(f"error: {ctx.tool_result_error[:500]}")
    await store.arecord(
        session_id=ctx.session_id,
        turn_idx=turn,
        role="tool",
        kind=KIND_TOOL_RESULT,
        tool_name=ctx.tool_name,
        content="\n".join(content_parts),
    )


def install() -> bool:
    """Register the logger's global handlers. Idempotent. Returns True if active.

    Called once at process startup when the feature flag is on. Safe to call
    more than once — subsequent calls short-circuit so handlers aren't
    registered multiple times.
    """
    global _installed
    if not trajectory_store_enabled():
        return False
    with _install_lock:
        if _installed:
            return True
        HookEngine.register_global(
            HookEvent.USER_PROMPT_SUBMIT, _on_user_prompt, name="trajectory_logger.prompt"
        )
        HookEngine.register_global(
            HookEvent.POST_TOOL_USE, _on_tool_result, name="trajectory_logger.tool"
        )
        _installed = True
        log.info("trajectory_logger_installed")
        return True


def reset_for_tests() -> None:
    """Testing hook — forget the installed flag and clear per-session counters."""
    global _installed
    with _install_lock:
        _installed = False
    with _counter_lock:
        _turn_counters.clear()
