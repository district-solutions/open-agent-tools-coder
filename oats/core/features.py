"""
Lightweight runtime feature flags for oats coder.

These are intentionally env-driven so new agent/runtime behaviors can be
tested safely without requiring a larger rollout framework.
"""
from __future__ import annotations

import os


def _is_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def deferred_tools_enabled() -> bool:
    """Enable ToolSearch-style deferred tool loading."""
    return _is_enabled("CODER_FEATURE_DEFERRED_TOOLS", True)


def rich_compaction_enabled() -> bool:
    """Enable denser engineering-oriented compaction prompts."""
    return _is_enabled("CODER_FEATURE_RICH_COMPACTION", True)


def streaming_tool_assembly_enabled() -> bool:
    """Enable robust assembly of fragmented streamed tool calls."""
    return _is_enabled("CODER_FEATURE_STREAMING_TOOL_ASSEMBLY", True)


def strict_tool_schemas_enabled() -> bool:
    """Pass strict tool schema hints through to compatible providers."""
    return _is_enabled("CODER_FEATURE_STRICT_TOOL_SCHEMAS", True)


def active_tool_guidance_enabled() -> bool:
    """Explain active/deferred tools in the system prompt."""
    return _is_enabled("CODER_FEATURE_ACTIVE_TOOL_GUIDANCE", True)


def reactive_compaction_candidate_enabled() -> bool:
    """
    Enable prompt-too-long recovery by compacting the session and retrying the
    request instead of failing immediately.
    """
    return _is_enabled("CODER_FEATURE_REACTIVE_COMPACTION", False)


def context_collapse_candidate_enabled() -> bool:
    """
    Enable state-capsule style compaction that preserves summarized history as a
    system message for stronger continuation.
    """
    return _is_enabled("CODER_FEATURE_CONTEXT_COLLAPSE", False)


def lsp_tools_candidate_enabled() -> bool:
    """Enable LSP-backed code-intelligence tools."""
    return _is_enabled("CODER_FEATURE_LSP_TOOLS", False)


def token_budget_candidate_enabled() -> bool:
    """Enable lightweight per-turn token budgeting and prompt guidance."""
    return _is_enabled("CODER_FEATURE_TOKEN_BUDGET", False)


def task_budget_candidate_enabled() -> bool:
    """Enable lightweight task-loop budgeting and anti-churn guidance."""
    return _is_enabled("CODER_FEATURE_TASK_BUDGET", False)


def result_retention_enabled() -> bool:
    """Enable compressed retention of large tool outputs in session history."""
    return _is_enabled("CODER_FEATURE_RESULT_RETENTION", True)


def trajectory_store_enabled() -> bool:
    """Log every user prompt and tool outcome to the on-disk trajectory store."""
    return _is_enabled("CODER_FEATURE_TRAJECTORY_STORE", False)


def trajectory_retrieval_enabled() -> bool:
    """Inject top-K past trajectories into the system prompt. Requires trajectory_store."""
    return _is_enabled("CODER_FEATURE_TRAJECTORY_RETRIEVAL", False)


def cron_enabled() -> bool:
    """Enable the coder cron/task scheduler tick in the interactive session."""
    return _is_enabled("CODER_FEATURE_CRON", False)


def plugins_enabled() -> bool:
    """Enable the declarative plugin loader (oats.plugins.loader.install)."""
    return _is_enabled("CODER_FEATURE_PLUGINS", False)


def should_disable_streaming(model_id: str | None = None) -> bool:
    """Check if streaming should be disabled for the current model.

    Returns True if:
    - CODER_FEATURE_NON_STREAMING is explicitly enabled, OR
    - The model ID looks like Gemma 4 (auto-detection)

    Auto-detection can be overridden with CODER_FEATURE_NON_STREAMING=0.
    """
    explicit = os.getenv("CODER_FEATURE_NON_STREAMING")
    if explicit is not None:
        return explicit.strip().lower() in {"1", "true", "yes", "on", "enabled"}

    if model_id:
        model_lower = model_id.lower()
        if "gemma-4" in model_lower or "gemma4" in model_lower:
            return True

    return False
