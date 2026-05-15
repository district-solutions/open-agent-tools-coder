"""
SQLite + FTS5 trajectory store.

One row per "turn record" — a user prompt, an assistant reply, a tool call,
or a tool result. FTS5 mirrors the ``content`` column so callers can rank
past turns by BM25 using SQLite's builtin ``bm25()`` function.

Concurrency strategy (borrowed from the Hermes state-store pattern and
adapted to coder2's layering):

- WAL journal mode — concurrent readers tolerate a single writer.
- Short ``busy_timeout`` (1.0s) + application-level retry with random jitter
  (20–150 ms per retry, 15 retries). SQLite's built-in busy handler is
  deterministic and causes convoys under parallel writers; jitter disperses
  them.
- ``PRAGMA synchronous = NORMAL`` — durable after fsync'd checkpoints while
  keeping per-insert cost low. Trajectory data is observability, not a
  money-transfer log; this is the right durability/throughput trade.
- Writes run through ``asyncio.to_thread`` so they never block the event
  loop that drives ``HookEngine.fire``.
"""
from __future__ import annotations

import asyncio
import random
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from oats.core.config import get_data_dir
from oats.log import cl

log = cl("oats.trajectory")


# Kind tags for turn records. Kept as a frozenset so accidental typos fail at
# write time and callers can grep the full vocabulary.
KIND_PROMPT = "prompt"
KIND_RESPONSE = "response"
KIND_TOOL_CALL = "tool_call"
KIND_TOOL_RESULT = "tool_result"

_VALID_KINDS = frozenset({KIND_PROMPT, KIND_RESPONSE, KIND_TOOL_CALL, KIND_TOOL_RESULT})


@dataclass(frozen=True)
class TrajectoryRecord:
    """One row from the trajectory store."""
    id: int
    session_id: str
    parent_session_id: str | None
    turn_idx: int
    role: str
    kind: str
    tool_name: str | None
    content: str
    created_at: float


@dataclass
class _WriteRetry:
    """Tunables for write-contention handling."""
    attempts: int = 15
    min_sleep: float = 0.020
    max_sleep: float = 0.150


class TrajectoryStore:
    """Persistent, full-text searchable log of agent turns.

    Thread-safe: a single process-wide connection guarded by an RLock, which
    is the simpler correct choice on SQLite WAL (multiple connections also
    work but buy nothing here since we serialize writers anyway).

    Call :meth:`record` (sync) from any thread, or :meth:`arecord` (async)
    from coroutine code.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize the trajectory store, creating the database if needed.

        Opens a SQLite connection in WAL mode with autocommit, creates the
        ``trajectories`` table, the FTS5 virtual table, and the ``turn_metrics``
        table (with triggers to keep FTS5 in sync).

        Args:
            db_path: Path to the SQLite database file. Defaults to
                ``<data_dir>/trajectories.db``.
        """
        self._path = Path(db_path) if db_path is not None else get_data_dir() / "trajectories.db"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._retry = _WriteRetry()
        self._conn = self._connect()
        self._init_schema()

    # ── Connection / schema ────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        """Open a new SQLite connection with WAL mode and relaxed sync.

        Returns:
            A configured :class:`sqlite3.Connection`.
        """
        conn = sqlite3.connect(
            str(self._path),
            timeout=1.0,
            isolation_level=None,  # autocommit; we manage transactions explicitly
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 1000")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        """Create tables, indexes, and FTS5 triggers if they don't exist.

        Creates the ``trajectories`` table, the ``trajectories_fts`` virtual
        table, insert/delete triggers to keep FTS5 in sync, and the
        ``turn_metrics`` table for per-turn self-improvement metrics.
        """
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    parent_session_id TEXT,
                    turn_idx INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    tool_name TEXT,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traj_session ON trajectories(session_id, turn_idx)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traj_parent ON trajectories(parent_session_id)"
            )
            # FTS5 mirror over content + tool_name, synced to trajectories via triggers.
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS trajectories_fts USING fts5(
                    content,
                    tool_name,
                    content='trajectories',
                    content_rowid='id'
                )
                """
            )
            self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_traj_ai AFTER INSERT ON trajectories BEGIN
                  INSERT INTO trajectories_fts(rowid, content, tool_name)
                  VALUES (new.id, new.content, COALESCE(new.tool_name, ''));
                END
                """
            )
            self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_traj_ad AFTER DELETE ON trajectories BEGIN
                  INSERT INTO trajectories_fts(trajectories_fts, rowid, content, tool_name)
                  VALUES ('delete', old.id, old.content, COALESCE(old.tool_name, ''));
                END
                """
            )
            # Per-turn self-improvement metrics. Populated by the retrieval
            # layer (at injection time) and by SessionProcessor (at turn end).
            # Rows are keyed on (session_id, turn_idx) and updated by the
            # outcome pass, so we use UPSERT semantics downstream.
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turn_metrics (
                    session_id TEXT NOT NULL,
                    turn_idx INTEGER NOT NULL,
                    user_prompt TEXT,
                    retrieved_ids TEXT,            -- JSON array of trajectory ids
                    retrieved_scores TEXT,         -- JSON array of floats
                    retrieval_used INTEGER NOT NULL DEFAULT 0,
                    iterations INTEGER,
                    tool_error_count INTEGER,
                    completed INTEGER,
                    duration_ms INTEGER,
                    model_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (session_id, turn_idx)
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_created ON turn_metrics(created_at)"
            )

    # ── Writes ─────────────────────────────────────────────────────
    def record(
        self,
        *,
        session_id: str,
        turn_idx: int,
        role: str,
        kind: str,
        content: str,
        tool_name: str | None = None,
        parent_session_id: str | None = None,
        created_at: float | None = None,
    ) -> int:
        """Insert one turn record. Returns the new row id.

        Retries under SQLITE_BUSY with random jitter so concurrent writers
        don't convoy on the builtin busy handler.
        """
        if kind not in _VALID_KINDS:
            raise ValueError(f"invalid kind: {kind!r} (valid: {sorted(_VALID_KINDS)})")
        ts = created_at if created_at is not None else time.time()

        last_err: Exception | None = None
        for attempt in range(self._retry.attempts):
            try:
                with self._lock:
                    cur = self._conn.execute(
                        """
                        INSERT INTO trajectories
                            (session_id, parent_session_id, turn_idx, role, kind,
                             tool_name, content, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (session_id, parent_session_id, turn_idx, role, kind,
                         tool_name, content, ts),
                    )
                    return int(cur.lastrowid)
            except sqlite3.OperationalError as e:
                if "locked" not in str(e).lower() and "busy" not in str(e).lower():
                    raise
                last_err = e
                time.sleep(random.uniform(self._retry.min_sleep, self._retry.max_sleep))
        assert last_err is not None
        raise last_err

    async def arecord(self, **kwargs) -> int:
        """Async wrapper around :meth:`record` — offloads to a thread."""
        return await asyncio.to_thread(self.record, **kwargs)

    # ── Reads ──────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        session_id: str | None = None,
        kinds: Iterable[str] | None = None,
    ) -> list[tuple[float, TrajectoryRecord]]:
        """BM25-rank past turns by query. Returns ``(score, record)`` pairs.

        Lower rank is better in SQLite FTS5 (``bm25()`` returns a *negative*
        score). We flip sign before returning so higher = more relevant, to
        match the convention used elsewhere in coder2.
        """
        if not query.strip():
            return []
        fts_query = _quote_fts(query)
        if not fts_query:
            return []
        params: list = [fts_query]
        where = ["t.id = f.rowid", "trajectories_fts MATCH ?"]
        if session_id:
            where.append("t.session_id = ?")
            params.append(session_id)
        if kinds:
            kinds_list = [k for k in kinds if k in _VALID_KINDS]
            if kinds_list:
                placeholders = ",".join("?" for _ in kinds_list)
                where.append(f"t.kind IN ({placeholders})")
                params.extend(kinds_list)
        params.append(limit)

        sql = f"""
            SELECT bm25(trajectories_fts) AS rank, t.id, t.session_id,
                   t.parent_session_id, t.turn_idx, t.role, t.kind,
                   t.tool_name, t.content, t.created_at
            FROM trajectories t, trajectories_fts f
            WHERE {' AND '.join(where)}
            ORDER BY rank
            LIMIT ?
        """
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        out: list[tuple[float, TrajectoryRecord]] = []
        for rank, rid, sid, psid, tidx, role, kind, tname, content, ts in rows:
            rec = TrajectoryRecord(
                id=rid, session_id=sid, parent_session_id=psid,
                turn_idx=tidx, role=role, kind=kind, tool_name=tname,
                content=content, created_at=ts,
            )
            out.append((-float(rank), rec))  # flip sign: higher = better
        return out

    async def asearch(self, query: str, **kwargs) -> list[tuple[float, TrajectoryRecord]]:
        """Async wrapper around :meth:`search` — offloads to a thread."""
        return await asyncio.to_thread(self.search, query, **kwargs)

    def session_turns(self, session_id: str, limit: int = 1000) -> list[TrajectoryRecord]:
        """All turns for a session, ordered by ``turn_idx``."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, session_id, parent_session_id, turn_idx, role, kind,
                       tool_name, content, created_at
                FROM trajectories
                WHERE session_id = ?
                ORDER BY turn_idx, id
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            TrajectoryRecord(
                id=r[0], session_id=r[1], parent_session_id=r[2], turn_idx=r[3],
                role=r[4], kind=r[5], tool_name=r[6], content=r[7], created_at=r[8],
            )
            for r in rows
        ]

    def count(self) -> int:
        """Total number of records. Handy for tests."""
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0])

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()


# ── Module-level singleton ─────────────────────────────────────────
_store: TrajectoryStore | None = None
_store_lock = threading.Lock()


def get_store(db_path: Path | str | None = None) -> TrajectoryStore:
    """Return the process-wide trajectory store, creating it on first use."""
    global _store
    with _store_lock:
        if _store is None:
            _store = TrajectoryStore(db_path=db_path)
        return _store


def reset_store() -> None:
    """Testing hook — drop the module-level singleton."""
    global _store
    with _store_lock:
        if _store is not None:
            _store.close()
        _store = None


# ── Helpers ────────────────────────────────────────────────────────
import re as _re

# Keep only word chars inside tokens. Drops FTS5 operator characters
# (",()[]{}:^*+-~) and punctuation that would otherwise make ``parse_args():``
# a syntax error.
_TOKEN_RE = _re.compile(r"[A-Za-z0-9_]+")


def _quote_fts(q: str) -> str:
    """Sanitize an arbitrary user query for FTS5 MATCH.

    FTS5 parses unquoted input as a query expression with operators like
    ``()``, ``:``, ``*``, ``-``. Rather than quoting the whole query into
    a strict phrase (misses partial matches and hurts BM25 ranking), we
    tokenize into word-characters-only tokens and join with ``OR``. That
    gives forgiving retrieval with BM25 scoring across any overlapping
    token — the right default for "find similar past turns".
    """
    tokens = _TOKEN_RE.findall(q)
    if not tokens:
        return ""
    return " OR ".join(tokens)
