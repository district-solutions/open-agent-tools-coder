"""
Trajectory store — persistent, searchable log of past agent turns.

SQLite (+ FTS5) lives at ``<data_dir>/trajectories.db`` and is opt-in via
``CODER_FEATURE_TRAJECTORY_STORE`` (default off). When enabled, Python hook
handlers in :mod:`coder.trajectory.logger` record every user prompt and tool
outcome so downstream features (in-context example retrieval, RL trace
export) can query past sessions by relevance.

Extends the existing ``coder.core.storage``/``coder.hook.engine`` primitives
rather than introducing a parallel persistence system.
"""
from __future__ import annotations

from oats.trajectory.store import TrajectoryRecord, TrajectoryStore, get_store

__all__ = ["TrajectoryRecord", "TrajectoryStore", "get_store"]
