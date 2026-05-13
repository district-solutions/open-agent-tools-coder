"""
Lightweight per-session debug tracing.

When enabled, this writes JSONL events so live validation on another machine can
inspect tool selection, provider assembly, compaction, and retention behavior.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def trace_enabled() -> bool:
    value = os.getenv("CODER_FEATURE_DEBUG_TRACE", "0").strip().lower()
    return value in {"1", "true", "yes", "on", "enabled"}


def _trace_dir() -> Path:
    return Path(os.getenv("CODER_DEBUG_TRACE_DIR", "/tmp/oats_coder_debug_traces")).resolve()


def trace_event(session_id: str | None, event_type: str, data: dict[str, Any]) -> None:
    """Append a JSONL trace event for a session when tracing is enabled."""
    if not trace_enabled() or not session_id:
        return

    trace_dir = _trace_dir()
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{session_id}.jsonl"
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "event": event_type,
        "data": data,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True, default=str))
        f.write("\n")
