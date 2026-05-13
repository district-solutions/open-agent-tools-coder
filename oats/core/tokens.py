"""
Shared token estimation.

Used by the context ledger, session token budget, and conversation compactor
so all three agree on what "tokens" means. Previously each site used
``len(text) // 4``, which under-counts code (~2.5 chars/token) and is close
to right for English prose (~3.5–4). The divergence matters when the budget
decision (trigger compaction, elide sections) is based on these estimates:
the real input payload routinely exceeds the approximation by 10–30%.

When ``tiktoken`` is importable we use ``cl100k_base`` as a stand-in — it's
not Claude's exact BPE, but it's a far better estimator than chars/4 for
both prose and code and it's stable across calls (so relative comparisons
between sections are faithful).

When ``tiktoken`` is not available we fall back to ``len(text) / 3.5``,
which biases toward *over*-estimating. That's the safer direction for
budget decisions — we'd rather compact a turn early than blow the window.

The tokenizer (if any) is loaded lazily and memoized on the module so we
don't pay encoding-loader cost on every ``count_tokens()`` call.
"""
from __future__ import annotations

import os
from typing import Any

_encoder: Any = None
_encoder_loaded = False


def _get_encoder() -> Any:
    global _encoder, _encoder_loaded
    if _encoder_loaded:
        return _encoder
    _encoder_loaded = True
    if os.getenv("CODER_DISABLE_TIKTOKEN") == "1":
        return None
    try:
        import tiktoken
        _encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _encoder = None
    return _encoder


def count_tokens(text: str | None) -> int:
    """Return an estimated token count for ``text``.

    Uses tiktoken's cl100k_base when available, otherwise a conservative
    chars/3.5 approximation. Empty/None returns 0.
    """
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    # Bias toward over-estimating — safer for budget decisions.
    return max(1, int(len(text) / 3.5))


def count_message_tokens(messages) -> int:
    """Sum token estimate across a list of session Message objects.

    Accepts anything with ``get_text_content()``, ``get_tool_calls()``, and
    ``get_tool_results()`` so the existing SessionTokenBudget / compactor
    can call through without importing the Message type.
    """
    total = 0
    for msg in messages:
        total += count_tokens(msg.get_text_content() or "")
        for tc in msg.get_tool_calls():
            total += count_tokens(str(tc.arguments))
        for tr in msg.get_tool_results():
            total += count_tokens(tr.output or "")
            total += count_tokens(tr.error or "")
    return total
