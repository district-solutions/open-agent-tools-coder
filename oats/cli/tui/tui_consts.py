"""Terminal UI constants, environment defaults, and display helpers.

Defines symbols used throughout the TUI (tool icons, status glyphs),
default environment variables for the coder profile, and utility
functions for formatting token counts, shortening model IDs, and
detecting the best image rendering protocol for the current terminal.
"""

import os
from rich.console import Console

# ── Symbols ───────────────────────────────────────────────────────────
SYM_TOOL = "▸"
SYM_OK = "✓"
SYM_ERR = "✗"
SYM_ITER = "↻"
SYM_COMPACT = "◇"
SYM_WARN = "▵"
SYM_SEP = "─"

# ── Tool display icons ───────────────────────────────────────────────
TOOL_ICONS = {
    "read": "📄",
    "write": " ",
    "edit": " ",
    "multiedit": " ",
    "bash": "⚡",
    "glob": "🔍",
    "grep": "🔍",
    "webfetch": "🌐",
    "websearch": "🌐",
    "tool_search": "🔎",
    "lsp": "◈",
    "patch": " ",
    "agent": "⬡",
    "plan_enter": "📋",
    "plan_exit": "📋",
    "memory_read": "💾",
    "memory_write": "💾",
    "todowrite": "☑",
    "todoread": "☑",
    "question": "❓",
    "askuser": "❓",
}

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg", ".tiff", ".ico"}

# ── Environment defaults for local-only mode ──────────────────────────
# CODER_PROFILE controls which feature groups are loaded (minimal|standard|full|custom).
# Default to "full" in interactive mode so all REPL commands are available.
# Override with CODER_PROFILE=minimal for lightweight/embedded usage.
_ENV_DEFAULTS = {
    "CODER_PROFILE": "full",
    "CODER_LOG_QUIET": "1",
    "CODER_DB_DISABLED": "1",
    "CODER_RC_DISABLED": "1",
    "CODER_S3_DISABLED": "1",
    "CODER_FEATURE_DEFERRED_TOOLS": "1",
    "CODER_FEATURE_RICH_COMPACTION": "1",
    "CODER_FEATURE_STREAMING_TOOL_ASSEMBLY": "1",
    "CODER_FEATURE_STRICT_TOOL_SCHEMAS": "1",
    "CODER_FEATURE_ACTIVE_TOOL_GUIDANCE": "1",
    "CODER_FEATURE_REACTIVE_COMPACTION": "1",
    "CODER_FEATURE_TOKEN_BUDGET": "1",
    "CODER_FEATURE_TASK_BUDGET": "1",
    "CODER_FEATURE_RESULT_RETENTION": "1",
    "CODER_FEATURE_LSP_TOOLS": "1",
    "CODER_FEATURE_DEBUG_TRACE": "1",
    "CODER_DEBUG_TRACE_DIR": "/tmp/oats_coder_debug_traces",
}

for key, val in _ENV_DEFAULTS.items():
    os.environ.setdefault(key, val)

if not os.path.exists('CODER_DEBUG_TRACE_DIR'):
    os.makedirs(_ENV_DEFAULTS['CODER_DEBUG_TRACE_DIR'], exist_ok=True)
if os.path.exists('CODER_DEBUG_TRACE_DIR'):
    os.chmod(_ENV_DEFAULTS['CODER_DEBUG_TRACE_DIR'], 0o777)

def _format_tokens(n: int) -> str:
    """Format token count: 1234 -> 1.2k, 12345 -> 12.3k"""
    if n < 1000:
        return str(n)
    elif n < 10000:
        return f"{n / 1000:.1f}k"
    else:
        return f"{n / 1000:.0f}k"

def _short_model(model_id: str) -> str:
    """Shorten model ID for display: hosted_vllm/Qwen3.5-27B-AWQ-4bit -> Qwen3.5-27B-AWQ-4bit"""
    if "/" in model_id:
        return model_id.rsplit("/", 1)[-1]
    return model_id

def _detect_image_protocol() -> str:
    """Return the best chafa output format for the current terminal.

    Returns one of: "kitty", "iterm", "sixels", "symbols".
    Uses chafa's format names directly for consistency.
    """
    # Kitty graphics protocol — pixel-perfect, true color
    if os.environ.get("KITTY_PID") or os.environ.get("TERM_PROGRAM") == "kitty":
        return "kitty"

    term_prog = os.environ.get("TERM_PROGRAM", "")

    # iTerm2 inline images — true color, widely supported
    if term_prog in ("iTerm.app", "iTerm2"):
        return "iterm"

    # WezTerm supports kitty, iterm, AND sixel — prefer kitty for best quality
    if term_prog in ("WezTerm", "wezterm"):
        return "kitty"

    # Sixel — reliable pixel graphics for supporting terminals
    sixel_progs = {"foot", "mlterm", "mintty", "contour"}
    if term_prog in sixel_progs:
        return "sixels"

    # Explicit env hints
    if os.environ.get("SIXEL") == "1":
        return "sixels"

    # Ghostty supports kitty graphics protocol
    if term_prog == "ghostty":
        return "kitty"

    return "symbols"
