"""
Lightweight token-budget tracking for long-running sessions.

This is intentionally heuristic rather than tokenizer-accurate. The goal is to
help the runtime adapt before local-model context pressure becomes a failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from oats.session.message import Message


@dataclass
class BudgetSnapshot:
    """Point-in-time snapshot of token budget state."""
    estimated_input_tokens: int
    context_window: int
    remaining_tokens: int
    recommended_max_output_tokens: int
    pressure: str


class SessionTokenBudget:
    """Estimate context pressure and recommend output budgets."""

    def __init__(
        self,
        context_window: int,
        reserve_output_tokens: int = 4096,
        minimum_output_tokens: int = 768,
    ) -> None:
        self.context_window = max(2048, context_window)
        self.reserve_output_tokens = max(256, reserve_output_tokens)
        self.minimum_output_tokens = max(256, minimum_output_tokens)

    def snapshot(
        self,
        messages: list[Message],
        requested_max_tokens: int | None = None,
    ) -> BudgetSnapshot:
        """Compute the current token budget snapshot and pressure level."""
        estimated_input = self._estimate_tokens(messages)
        remaining = max(0, self.context_window - estimated_input)

        available_for_output = max(
            self.minimum_output_tokens,
            remaining - min(self.reserve_output_tokens, max(0, remaining // 3)),
        )
        recommended = available_for_output
        if requested_max_tokens is not None:
            recommended = min(recommended, requested_max_tokens)
        recommended = max(self.minimum_output_tokens, recommended)

        ratio = estimated_input / max(1, self.context_window)
        if ratio >= 0.92:
            pressure = "critical"
        elif ratio >= 0.82:
            pressure = "high"
        elif ratio >= 0.65:
            pressure = "medium"
        else:
            pressure = "low"

        return BudgetSnapshot(
            estimated_input_tokens=estimated_input,
            context_window=self.context_window,
            remaining_tokens=remaining,
            recommended_max_output_tokens=recommended,
            pressure=pressure,
        )

    def _estimate_tokens(self, messages: list[Message]) -> int:
        total_chars = 0
        for msg in messages:
            total_chars += len(msg.get_text_content() or "")
            for tc in msg.get_tool_calls():
                total_chars += len(str(tc.arguments))
            for tr in msg.get_tool_results():
                total_chars += len(tr.output or "")
                total_chars += len(tr.error or "")
        return total_chars // 4


def format_budget_guidance(snapshot: BudgetSnapshot) -> str:
    """Create a compact prompt section describing current context pressure."""
    line = (
        f"# Budget: {snapshot.estimated_input_tokens}/{snapshot.context_window} tokens used, "
        f"pressure={snapshot.pressure}, max_output={snapshot.recommended_max_output_tokens}"
    )
    if snapshot.pressure in ("high", "critical"):
        line += "\nFinish current work concisely. Avoid long prose."
    return line
