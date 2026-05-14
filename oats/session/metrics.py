"""In-memory per-session counters for feature telemetry.

Tracks usage of the features added in the 2026-04-16 batch so we can
tell whether they're paying off on real workloads:

  - caveman_compressions       — count of successful compress() calls
  - caveman_bytes_saved        — cumulative (original - compressed) chars
  - caveman_compress_skipped   — count of skipped / failed compressions
  - qs_summary_hits            — sidecar found and returned
  - qs_summary_misses          — sidecar absent; fell back
  - aws_classified.<risk>      — distribution of classified aws commands
  - aws_secret_redactions      — AKIA/secret/session-token scrubs in output

All counters are in-process and reset per Python session (cheap, zero deps).
Can be persisted later if we want lifetime stats.
"""
from __future__ import annotations

from collections import defaultdict
from threading import Lock
from time import time
from typing import Any


_counters: dict[str, int] = defaultdict(int)
_lock = Lock()
_started_at = time()


def incr(key: str, n: int = 1) -> None:
    """Increment a counter by *n*.

    Args:
        key: The counter name.
        n: Amount to increment by (default 1).
    """
    with _lock:
        _counters[key] += n


def get(key: str) -> int:
    """Get the current value of a counter.

    Args:
        key: The counter name.

    Returns:
        The counter value, or 0 if not set.
    """
    return _counters.get(key, 0)


def snapshot() -> dict[str, Any]:
    """Return a copy of current counters plus session uptime."""
    with _lock:
        counters = dict(_counters)
    return {
        "counters": counters,
        "session_uptime_s": int(time() - _started_at),
    }


def reset() -> None:
    """Reset all counters. Mostly for tests; production code should never call this."""
    with _lock:
        _counters.clear()
