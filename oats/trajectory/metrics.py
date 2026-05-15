"""
Per-turn self-improvement metrics.

Writes to the ``turn_metrics`` table in the same SQLite db as the trajectory
store so retrieval records and turn outcomes can be joined for A/B analysis.

Two entry points:

- :func:`log_retrieval_used` — at prompt-injection time, records what was
  retrieved (if anything) for a given ``(session_id, turn_idx)``.
- :func:`log_turn_outcome` — at turn end, records whether the model
  completed cleanly, how many agent-loop iterations it took, and tool-error
  count.

:func:`report` aggregates the table into a diagnostics dict callers can
render to stdout, Markdown, or JSON. The CLI at ``python -m
coder.trajectory.report`` prints a weekly summary.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from oats.log import cl
from oats.trajectory.store import TrajectoryStore, get_store

log = cl("oats.trajectory.metrics")


@dataclass
class TurnMetricRow:
    """One row from the ``turn_metrics`` table.

    Attributes:
        session_id: The session this turn belongs to.
        turn_idx: Sequential turn index within the session.
        user_prompt: The user's prompt text (truncated to 2000 chars).
        retrieved_ids: JSON-encoded list of trajectory IDs that were retrieved.
        retrieved_scores: JSON-encoded list of BM25 scores for retrieved items.
        retrieval_used: Whether retrieval was used for this turn.
        iterations: Number of agent-loop iterations the turn took.
        tool_error_count: Number of tool errors encountered.
        completed: Whether the turn completed successfully.
        duration_ms: Turn duration in milliseconds.
        model_id: The model ID used for this turn.
        created_at: Timestamp when the row was first inserted.
        updated_at: Timestamp when the row was last updated.
    """
    session_id: str
    turn_idx: int
    user_prompt: str | None
    retrieved_ids: list[int]
    retrieved_scores: list[float]
    retrieval_used: bool
    iterations: int | None
    tool_error_count: int | None
    completed: bool | None
    duration_ms: int | None
    model_id: str | None
    created_at: float
    updated_at: float


def _upsert(
    store: TrajectoryStore,
    *,
    session_id: str,
    turn_idx: int,
    fields: dict,
) -> None:
    """Insert-or-update a row keyed on (session_id, turn_idx)."""
    now = time.time()
    cols = ["session_id", "turn_idx"] + list(fields.keys()) + ["created_at", "updated_at"]
    vals: list = [session_id, turn_idx] + list(fields.values()) + [now, now]
    placeholders = ",".join("?" for _ in cols)
    updates = ",".join(f"{k}=excluded.{k}" for k in list(fields.keys()) + ["updated_at"])

    sql = (
        f"INSERT INTO turn_metrics ({','.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(session_id, turn_idx) DO UPDATE SET {updates}"
    )
    with store._lock:
        store._conn.execute(sql, vals)


def log_retrieval_used(
    *,
    session_id: str,
    turn_idx: int,
    user_prompt: str,
    retrieved: Iterable[tuple[float, int]],
    model_id: str | None = None,
    store: TrajectoryStore | None = None,
) -> None:
    """Record what was retrieved for this turn. ``retrieved`` is pairs of (score, trajectory_id)."""
    store = store or get_store()
    scores: list[float] = []
    ids: list[int] = []
    for score, tid in retrieved:
        scores.append(float(score))
        ids.append(int(tid))
    _upsert(
        store,
        session_id=session_id,
        turn_idx=turn_idx,
        fields={
            "user_prompt": user_prompt[:2000],
            "retrieved_ids": json.dumps(ids),
            "retrieved_scores": json.dumps(scores),
            "retrieval_used": 1 if ids else 0,
            "model_id": model_id,
        },
    )


def log_turn_outcome(
    *,
    session_id: str,
    turn_idx: int,
    iterations: int,
    tool_error_count: int,
    completed: bool,
    duration_ms: int | None = None,
    model_id: str | None = None,
    store: TrajectoryStore | None = None,
) -> None:
    """Record the outcome of a finished turn."""
    store = store or get_store()
    _upsert(
        store,
        session_id=session_id,
        turn_idx=turn_idx,
        fields={
            "iterations": int(iterations),
            "tool_error_count": int(tool_error_count),
            "completed": 1 if completed else 0,
            "duration_ms": int(duration_ms) if duration_ms is not None else None,
            "model_id": model_id,
        },
    )


# ── Reporting ──────────────────────────────────────────────────────

@dataclass
class CohortStats:
    """Aggregate stats for one slice of turns."""
    label: str
    turns: int = 0
    completed: int = 0
    avg_iterations: float = 0.0
    avg_tool_errors: float = 0.0
    avg_duration_ms: float = 0.0

    @property
    def completion_rate(self) -> float:
        """Fraction of turns in this cohort that completed successfully."""
        return (self.completed / self.turns) if self.turns else 0.0


def report(
    since_days: float = 7.0,
    *,
    store: TrajectoryStore | None = None,
) -> dict:
    """Aggregate turn_metrics across the last ``since_days``.

    Splits into two cohorts — retrieval-used vs. not — so callers can tell
    whether in-context retrieval is pulling its weight. Returns a dict ready
    for JSON or Markdown rendering.
    """
    store = store or get_store()
    cutoff = time.time() - since_days * 86400.0

    with store._lock:
        rows = store._conn.execute(
            """
            SELECT retrieval_used, iterations, tool_error_count, completed, duration_ms
            FROM turn_metrics
            WHERE updated_at >= ? AND iterations IS NOT NULL
            """,
            (cutoff,),
        ).fetchall()

    cohorts = {0: CohortStats("no_retrieval"), 1: CohortStats("retrieval_used")}
    for used, iters, errs, done, dur in rows:
        c = cohorts[int(used)]
        c.turns += 1
        if done:
            c.completed += 1
        c.avg_iterations += float(iters or 0)
        c.avg_tool_errors += float(errs or 0)
        c.avg_duration_ms += float(dur or 0)

    for c in cohorts.values():
        if c.turns:
            c.avg_iterations /= c.turns
            c.avg_tool_errors /= c.turns
            c.avg_duration_ms /= c.turns

    return {
        "since_days": since_days,
        "total_turns": sum(c.turns for c in cohorts.values()),
        "cohorts": {
            c.label: {
                "turns": c.turns,
                "completion_rate": round(c.completion_rate, 4),
                "avg_iterations": round(c.avg_iterations, 2),
                "avg_tool_errors": round(c.avg_tool_errors, 2),
                "avg_duration_ms": round(c.avg_duration_ms, 0),
            }
            for c in cohorts.values()
        },
    }


def format_report_markdown(data: dict) -> str:
    """Render the report dict as a Markdown table.

    Produces a header with the time window and total turns, followed by a
    table of cohort statistics and an interpretation note.

    Args:
        data: The dict returned by :func:`report`.

    Returns:
        A Markdown-formatted string.
    """
    lines = [
        "# Coder2 Self-Improvement Report",
        "",
        f"Window: last **{data['since_days']:g} days** — {data['total_turns']} turns recorded.",
        "",
        "| cohort | turns | completion rate | avg iter | avg tool errors | avg dur (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, c in data["cohorts"].items():
        lines.append(
            f"| {label} | {c['turns']} | {c['completion_rate']:.1%} "
            f"| {c['avg_iterations']:.2f} | {c['avg_tool_errors']:.2f} "
            f"| {c['avg_duration_ms']:.0f} |"
        )
    lines.extend([
        "",
        "**Interpretation:** when `retrieval_used` has meaningfully higher completion "
        "rate and/or lower avg iterations than `no_retrieval`, retrieval is net-positive. "
        "If it regresses, re-tune retrieval (min_score, top_k) or corpus coverage.",
    ])
    return "\n".join(lines)
