"""
Lightweight task-budget tracking for agent loops.

This helps local-model sessions avoid runaway tool churn by monitoring turn
count, tool-call volume, and repeated tool patterns.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


def _env_int(name: str, default: int) -> int:
    """Parse an integer from an environment variable, falling back to *default*."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class TaskBudgetSnapshot:
    """Point-in-time snapshot of task budget state."""
    iteration: int
    max_iterations: int
    tool_calls: int
    max_tool_calls: int
    repeated_tool_streak: int
    pressure: str
    should_stop: bool
    guidance: str | None = None


@dataclass
class SessionTaskBudget:
    """Track iteration/tool-call budget and detect repeated-tool churn."""
    max_iterations: int = field(default_factory=lambda: _env_int("CODER_MAX_ITERATIONS", 150))
    max_tool_calls: int = field(default_factory=lambda: _env_int("CODER_MAX_TOOL_CALLS", 300))
    repeated_tool_limit: int = 3
    commit_extension_iterations: int = 8
    _tool_calls: int = 0
    _history: list[tuple[str, str]] = field(default_factory=list)
    _committed: bool = False
    _commit_iteration: int = 0
    _commit_tool_calls: int = 0

    def record_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Record a tool call for budget tracking and repeat detection."""
        normalized = json.dumps(arguments, sort_keys=True, ensure_ascii=True)
        self._tool_calls += 1
        self._history.append((tool_name, normalized))
        if len(self._history) > 24:
            self._history.pop(0)

    def commit(self, iteration: int) -> None:
        """Switch to commit mode: stop the discovery loop but allow the model
        to continue calling tools needed to finalize work (edits, writes, etc.).
        Budget limits are extended by commit_extension_iterations as a safety
        net so a stuck model still terminates eventually."""
        if self._committed:
            return
        self._committed = True
        self._commit_iteration = iteration
        self._commit_tool_calls = self._tool_calls
        self._history.clear()

    def snapshot(self, iteration: int) -> TaskBudgetSnapshot:
        """Compute the current budget snapshot and pressure level."""
        repeated_streak = self._repeated_streak()

        if self._committed:
            iters_since = iteration - self._commit_iteration
            calls_since = self._tool_calls - self._commit_tool_calls
            hard_stop = (
                iters_since >= self.commit_extension_iterations
                or calls_since >= self.commit_extension_iterations * 2
                or repeated_streak >= self.repeated_tool_limit + 2
            )
            return TaskBudgetSnapshot(
                iteration=iteration,
                max_iterations=self.max_iterations,
                tool_calls=self._tool_calls,
                max_tool_calls=self.max_tool_calls,
                repeated_tool_streak=repeated_streak,
                pressure="critical",
                should_stop=hard_stop,
                guidance=self._build_commit_guidance(iters_since, repeated_streak),
            )

        iter_ratio = iteration / max(1, self.max_iterations)
        tool_ratio = self._tool_calls / max(1, self.max_tool_calls)

        if (
            iteration >= self.max_iterations
            or self._tool_calls >= self.max_tool_calls
            or repeated_streak >= self.repeated_tool_limit + 2
        ):
            pressure = "critical"
            should_stop = True
        elif (
            iter_ratio >= 0.85
            or tool_ratio >= 0.85
            or repeated_streak >= self.repeated_tool_limit
        ):
            pressure = "high"
            should_stop = False
        elif (
            iter_ratio >= 0.65
            or tool_ratio >= 0.65
            or repeated_streak >= max(2, self.repeated_tool_limit - 2)
        ):
            pressure = "medium"
            should_stop = False
        else:
            pressure = "low"
            should_stop = False

        guidance = None
        if pressure in {"medium", "high", "critical"}:
            guidance = self._build_guidance(pressure, repeated_streak)

        return TaskBudgetSnapshot(
            iteration=iteration,
            max_iterations=self.max_iterations,
            tool_calls=self._tool_calls,
            max_tool_calls=self.max_tool_calls,
            repeated_tool_streak=repeated_streak,
            pressure=pressure,
            should_stop=should_stop,
            guidance=guidance,
        )

    def _repeated_streak(self) -> int:
        if not self._history:
            return 0
        streak = 1
        last = self._history[-1]
        for item in reversed(self._history[:-1]):
            if item == last:
                streak += 1
            else:
                break
        return streak

    def _build_guidance(self, pressure: str, repeated_streak: int) -> str:
        line = f"# Task: pressure={pressure}, calls={self._tool_calls}, repeat_streak={repeated_streak}"
        if pressure in ("high", "critical"):
            line += "\nFinish current subtask. Avoid repeating identical tool calls."
        return line

    def _build_commit_guidance(self, iters_since: int, repeated_streak: int) -> str:
        return (
            f"# COMMIT MODE (discovery budget exhausted; {iters_since} iters since commit, "
            f"{self._tool_calls} total tool calls, repeat_streak={repeated_streak})\n"
            "STOP exploring. Do NOT run more searches, greps, finds, or file reads for "
            "discovery. Use ONLY the context you already have in this conversation to "
            "complete the user's original task now.\n"
            "Tool calls are still allowed, but ONLY to commit work the user asked for "
            "(edits, writes, running the final command, etc.). If information is missing, "
            "state what you know, state what's missing, and deliver the best answer you "
            "can with what you have. Do not start new investigation threads."
        )
