"""
File state cache — tracks files read/written during a session.

Avoids redundant file reads by caching file metadata (mtime + size).
When a tool requests a file read, the cache can indicate if the file
is unchanged since the last read.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Optional

from oats.log import cl

log = cl("session.cache")


@dataclass
class FileCacheEntry:
    """A cached file state entry."""
    path: str
    mtime: float
    size: int
    content_hash: str


class FileStateCache:
    """
    Tracks files read and written during a session.

    Used to:
    - Detect if a file has changed since last read (avoid redundant reads)
    - Track which files were modified during the session
    - Provide session-level file activity summaries
    """

    def __init__(self) -> None:
        self._read_files: dict[str, FileCacheEntry] = {}
        self._written_files: dict[str, float] = {}  # path -> mtime at write

    def mark_read(self, path: str, content: str | None = None) -> None:
        """Mark a file as read, caching its state."""
        try:
            abs_path = os.path.abspath(path)
            stat = os.stat(abs_path)
            content_hash = ""
            if content:
                content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            self._read_files[abs_path] = FileCacheEntry(
                path=abs_path,
                mtime=stat.st_mtime,
                size=stat.st_size,
                content_hash=content_hash,
            )
        except OSError:
            pass

    def mark_written(self, path: str) -> None:
        """Mark a file as written during this session."""
        try:
            abs_path = os.path.abspath(path)
            stat = os.stat(abs_path)
            self._written_files[abs_path] = stat.st_mtime
            # Invalidate read cache since file changed
            self._read_files.pop(abs_path, None)
        except OSError:
            self._written_files[os.path.abspath(path)] = 0

    def is_fresh(self, path: str) -> bool:
        """
        Check if a file is already cached and unchanged.

        Returns True if the file has been read and its mtime hasn't changed.
        """
        abs_path = os.path.abspath(path)
        if abs_path not in self._read_files:
            return False
        cached = self._read_files[abs_path]
        try:
            stat = os.stat(abs_path)
            return stat.st_mtime == cached.mtime and stat.st_size == cached.size
        except OSError:
            return False

    def get_read_count(self) -> int:
        """Number of unique files read this session."""
        return len(self._read_files)

    def get_written_count(self) -> int:
        """Number of unique files written this session."""
        return len(self._written_files)

    def get_read_files(self) -> list[str]:
        """List of file paths read this session."""
        return list(self._read_files.keys())

    def get_written_files(self) -> list[str]:
        """List of file paths written this session."""
        return list(self._written_files.keys())

    def get_summary(self) -> dict[str, int]:
        """Session file activity summary."""
        return {
            "files_read": len(self._read_files),
            "files_written": len(self._written_files),
        }
