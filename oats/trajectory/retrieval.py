"""
Retrieval-augmented examples from the trajectory store.

Given a fresh user prompt, find the top past prompts that resemble it, then
pull the tool outcomes that immediately followed them in the same session.
That ``(matched prompt → successful continuation)`` pair becomes an
in-context example the current model can imitate.

Design notes:

- Retrieval uses :meth:`TrajectoryStore.search` over ``KIND_PROMPT`` rows so
  we're ranking user intents, not tool noise.
- Continuation lookup is scoped to the original session via
  :meth:`TrajectoryStore.session_turns` so we never mix threads across
  sessions. We take at most ``continuation_limit`` turns following the
  matched prompt's ``turn_idx``, cap each turn's content length, and stop at
  the next prompt (which would belong to a different user intent).
- Examples are formatted plainly — no special delimiters — so the injection
  costs a predictable number of tokens regardless of retrieval depth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from oats.log import cl
from oats.trajectory.store import (
    KIND_PROMPT,
    KIND_TOOL_CALL,
    KIND_TOOL_RESULT,
    TrajectoryRecord,
    TrajectoryStore,
    get_store,
)

log = cl("oats.trajectory.retrieval")

DEFAULT_TOP_K = 2
DEFAULT_MIN_SCORE = 0.0  # bm25 scores vary by corpus; keep permissive by default
DEFAULT_CONTINUATION_LIMIT = 4
DEFAULT_CONTENT_CAP = 400


@dataclass(frozen=True)
class Example:
    """One retrieval-augmented example."""
    score: float
    prompt_record: TrajectoryRecord
    continuation: list[TrajectoryRecord]

    def format(self, *, content_cap: int = DEFAULT_CONTENT_CAP) -> str:
        lines = [f"- past prompt: {self.prompt_record.content.strip()[:content_cap]}"]
        for rec in self.continuation:
            tag = rec.tool_name or rec.kind
            body = rec.content.strip()
            if len(body) > content_cap:
                body = body[:content_cap] + "…"
            lines.append(f"  {tag}: {body}")
        return "\n".join(lines)


def retrieve_examples(
    user_prompt: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    continuation_limit: int = DEFAULT_CONTINUATION_LIMIT,
    store: TrajectoryStore | None = None,
    exclude_session_id: str | None = None,
) -> list[Example]:
    """Return up to ``top_k`` examples for ``user_prompt``.

    ``exclude_session_id`` lets the caller skip the current session so the
    model isn't handed back a stale version of what it just did.
    """
    if not user_prompt.strip():
        return []
    store = store or get_store()

    ranked = store.search(user_prompt, limit=top_k * 3, kinds=[KIND_PROMPT])
    if not ranked:
        return []

    examples: list[Example] = []
    seen_sessions: set[str] = set()
    for score, prompt_rec in ranked:
        if score < min_score:
            continue
        if exclude_session_id and prompt_rec.session_id == exclude_session_id:
            continue
        # One example per session — avoids leaking an entire session verbatim.
        if prompt_rec.session_id in seen_sessions:
            continue
        continuation = _continuation(store, prompt_rec, continuation_limit)
        if not continuation:
            continue
        examples.append(Example(score=score, prompt_record=prompt_rec, continuation=continuation))
        seen_sessions.add(prompt_rec.session_id)
        if len(examples) >= top_k:
            break
    return examples


def _continuation(
    store: TrajectoryStore,
    prompt_rec: TrajectoryRecord,
    limit: int,
) -> list[TrajectoryRecord]:
    """Turns strictly after ``prompt_rec``, stopping at the next prompt."""
    turns = store.session_turns(prompt_rec.session_id, limit=prompt_rec.turn_idx + limit + 2)
    out: list[TrajectoryRecord] = []
    for t in turns:
        if t.turn_idx <= prompt_rec.turn_idx:
            continue
        if t.kind == KIND_PROMPT:
            break
        if t.kind not in (KIND_TOOL_CALL, KIND_TOOL_RESULT):
            continue
        out.append(t)
        if len(out) >= limit:
            break
    return out


def format_examples_section(examples: list[Example]) -> str | None:
    """Render an examples block for the system prompt; None if no examples."""
    if not examples:
        return None
    lines = [
        "# Past Trajectories (from this project)",
        "",
        "These past sessions handled similar requests. Use them as hints, "
        "not ground truth — verify any assumption before acting.",
        "",
    ]
    for i, ex in enumerate(examples, 1):
        lines.append(f"## Example {i}  (score {ex.score:.2f})")
        lines.append(ex.format())
        lines.append("")
    return "\n".join(lines).rstrip()
