"""
Caveman-style compression helper.

Takes a long text (tool output, doc, memory capsule) and asks the model to
rewrite it in terse "caveman" style, then validates that structural tokens
the downstream agent relies on (code fences, file paths, URLs, identifiers,
error strings) survived the rewrite.

If validation fails we return the original — a broken compression is worse
than no compression. The caller decides whether to retry at a shorter input
length or give up.

This is a standalone helper so callers can invoke it outside of interactive
mode: from a tool, from compaction, or from a CLI script.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from oats.log import cl
from oats.provider.provider import (
    CompletionRequest,
    Message as ProviderMessage,
    get_provider,
)

log = cl("session.caveman")


_CAVEMAN_SYSTEM = (
    "Rewrite the user's text in caveman style. Reduce prose tokens without losing technical content.\n"
    "\n"
    "RULES:\n"
    "- Drop: articles (a/an/the), filler (just/really/basically), hedging, preamble, recap summaries.\n"
    "- Keep VERBATIM: code inside ``` fences, file paths, URLs, command flags, version numbers,\n"
    "  error messages, identifier names, numbers/units, quoted strings.\n"
    "- Fragments OK. Use `[thing] [action] [reason].` pattern.\n"
    "- Preserve markdown structure: headings stay headings, lists stay lists, code fences stay intact.\n"
    "- Preserve the SAME information. You are compressing style, not dropping facts.\n"
    "- If the input is mostly code or a table, return it unchanged.\n"
    "\n"
    "Return only the rewritten text. No preface, no explanation of what you changed."
)


_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
_PATH_RE = re.compile(r"[/~][\w./\-_]+\.\w{1,8}\b|/[\w./\-_]+/\w+")
_URL_RE = re.compile(r"https?://[^\s)>\]]+?(?=[.,;:!?)]?(?:\s|$))")


@dataclass
class CompressionResult:
    text: str                   # the text to use (compressed if ok, else original)
    original_chars: int
    compressed_chars: int
    compressed: bool            # True if compression was applied + validated
    reason: str = ""            # validation failure reason, if compressed=False

    @property
    def reduction(self) -> float:
        if not self.compressed or self.original_chars == 0:
            return 0.0
        return 1.0 - (self.compressed_chars / self.original_chars)


def _extract_invariants(text: str) -> tuple[set[str], set[str], list[str]]:
    """Return (paths, urls, code_fences) that must survive compression."""
    paths = set(_PATH_RE.findall(text))
    urls = set(_URL_RE.findall(text))
    fences = _CODE_FENCE_RE.findall(text)
    return paths, urls, fences


def _validate(original: str, compressed: str) -> tuple[bool, str]:
    """Check that structural invariants survived the rewrite."""
    orig_paths, orig_urls, orig_fences = _extract_invariants(original)
    _, new_urls, new_fences = _extract_invariants(compressed)

    missing_urls = orig_urls - new_urls
    if missing_urls:
        return False, f"lost {len(missing_urls)} url(s)"

    if len(orig_fences) != len(new_fences):
        return False, f"code fence count changed: {len(orig_fences)} → {len(new_fences)}"
    for a, b in zip(orig_fences, new_fences):
        if a != b:
            return False, "code fence content altered"

    new_text_lower = compressed.lower()
    missing_paths = [p for p in orig_paths if p.lower() not in new_text_lower]
    if missing_paths:
        return False, f"lost {len(missing_paths)} path(s): {missing_paths[:3]}"

    if len(compressed) >= len(original):
        return False, "no reduction"

    return True, ""


async def compress(
    text: str,
    *,
    provider_id: str,
    model_id: str,
    min_input_chars: int = 1500,
    max_tokens: int = 4000,
) -> CompressionResult:
    """Rewrite *text* in caveman style. Returns original if input too short or validation fails."""
    original_chars = len(text)

    if original_chars < min_input_chars:
        try:
            from oats.session.metrics import incr
            incr("caveman_compress_skipped")
        except Exception:
            pass
        return CompressionResult(
            text=text,
            original_chars=original_chars,
            compressed_chars=original_chars,
            compressed=False,
            reason="below_min_input",
        )

    try:
        provider = get_provider(provider_id)
        request = CompletionRequest(
            messages=[
                ProviderMessage(role="system", content=_CAVEMAN_SYSTEM),
                ProviderMessage(role="user", content=text),
            ],
            model=model_id,
            provider_id=provider_id,
            max_tokens=max_tokens,
        )
        response = await provider.complete(request)
        compressed_text = (response.content or "").strip()
    except Exception as e:
        log.warn(f"caveman_compress_failed: {e}")
        return CompressionResult(
            text=text,
            original_chars=original_chars,
            compressed_chars=original_chars,
            compressed=False,
            reason=f"provider_error: {e}",
        )

    ok, why = _validate(text, compressed_text)
    if not ok:
        log.info(f"caveman_validation_failed: {why}")
        try:
            from oats.session.metrics import incr
            incr("caveman_compress_skipped")
        except Exception:
            pass
        return CompressionResult(
            text=text,
            original_chars=original_chars,
            compressed_chars=len(compressed_text),
            compressed=False,
            reason=why,
        )

    try:
        from oats.session.metrics import incr
        incr("caveman_compressions")
        incr("caveman_bytes_saved", max(0, original_chars - len(compressed_text)))
    except Exception:
        pass

    return CompressionResult(
        text=compressed_text,
        original_chars=original_chars,
        compressed_chars=len(compressed_text),
        compressed=True,
    )
