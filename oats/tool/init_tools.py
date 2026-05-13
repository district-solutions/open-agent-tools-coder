"""
Default tool system for an ai agent
"""
from __future__ import annotations

from oats.tool.registry import (
    Tool,
    ToolContext,
    ToolResult,
    ToolRegistry,
    get_tool,
    list_tools,
    register_tool,
)
from oats.tool.bash import BashTool
from oats.tool.read import ReadTool
from oats.tool.write import WriteTool
from oats.tool.edit import EditTool
from oats.tool.glob_tool import GlobTool
from oats.tool.grep import GrepTool
from oats.tool.webfetch import WebFetchTool
from oats.tool.websearch import WebSearchTool
from oats.tool.todowrite import TodoWriteTool, TodoReadTool
from oats.tool.question import QuestionTool, AskUserTool
from oats.tool.multiedit import MultiEditTool
from oats.tool.patch import ApplyPatchTool
from oats.tool.plan import PlanEnterTool, PlanExitTool, PlanStatusTool
from oats.tool.generate_readme import GenerateREADMETool
from oats.tool.agent_tool import AgentTool, AgentStatusTool
from oats.tool.memory_tool import MemoryReadTool, MemoryWriteTool, MemoryDeleteTool
from oats.tool.tool_search import ToolSearchTool
from oats.tool.playwright_search import PlaywrightSearchTool
from oats.tool.lsp_tool import LSPTool
from oats.core.features import deferred_tools_enabled
from oats.core.features import lsp_tools_candidate_enabled
from oats.core.profiles import is_feature_enabled

__all__ = [
    "Tool",
    "ToolContext",
    "ToolResult",
    "ToolRegistry",
    "get_tool",
    "list_tools",
    "register_tool",
    "BashTool",
    "ReadTool",
    "WriteTool",
    "EditTool",
    "GlobTool",
    "GrepTool",
    "WebFetchTool",
    "WebSearchTool",
    "TodoWriteTool",
    "TodoReadTool",
    "QuestionTool",
    "AskUserTool",
    "MultiEditTool",
    "ApplyPatchTool",
    "PlanEnterTool",
    "PlanExitTool",
    "PlanStatusTool",
    "GenerateREADMETool",
    "AgentTool",
    "AgentStatusTool",
    "MemoryReadTool",
    "MemoryWriteTool",
    "MemoryDeleteTool",
    "ToolSearchTool",
    "PlaywrightSearchTool",
    "LSPTool",
]


def init_tools() -> None:
    """Initialize and register tools based on the active feature profile.

    The active profile is set via CODER_PROFILE env var (minimal|standard|full|custom).
    Individual groups can be overridden with CODER_FEATURE_<GROUP>=0|1.

    Core tools (read, write, edit, glob, grep, bash) are always registered.
    """
    # ── Core tools (always loaded) ──────────────────────────────
    register_tool(ReadTool())
    register_tool(WriteTool())
    register_tool(EditTool())
    register_tool(MultiEditTool())
    register_tool(GlobTool())
    register_tool(GrepTool())
    register_tool(BashTool())
    register_tool(ApplyPatchTool())

    # Task management (always — lightweight)
    register_tool(TodoWriteTool())
    register_tool(TodoReadTool())

    # User interaction (always)
    register_tool(QuestionTool())
    register_tool(AskUserTool())

    # README generation (always — lightweight)
    register_tool(GenerateREADMETool())

    # Deferred tool discovery (feature-flag gated, not profile)
    if deferred_tools_enabled():
        register_tool(ToolSearchTool())

    # ── Web tools ───────────────────────────────────────────────
    if is_feature_enabled("web_tools"):
        register_tool(WebFetchTool())
        register_tool(WebSearchTool())
        register_tool(PlaywrightSearchTool())

    # ── Planning mode ───────────────────────────────────────────
    if is_feature_enabled("planning"):
        register_tool(PlanEnterTool())
        register_tool(PlanExitTool())
        register_tool(PlanStatusTool())

    # ── Memory tools ────────────────────────────────────────────
    if is_feature_enabled("memory"):
        register_tool(MemoryReadTool())
        register_tool(MemoryWriteTool())
        register_tool(MemoryDeleteTool())

    # ── Sub-agent tools ─────────────────────────────────────────
    if is_feature_enabled("agents"):
        register_tool(AgentTool())
        register_tool(AgentStatusTool())

    # ── LSP code intelligence ───────────────────────────────────
    if is_feature_enabled("lsp") or lsp_tools_candidate_enabled():
        register_tool(LSPTool())

    # ── MCP protocol tools ──────────────────────────────────────
    if is_feature_enabled("mcp"):
        from oats.mcp.tools import (
            MCPDiscoverTool,
            MCPCallTool,
            MCPCallChainTool,
            MCPFanOutTool,
            MCPRankTool,
            MCPSessionSummaryTool,
            MCPServerManageTool,
        )
        register_tool(MCPDiscoverTool())
        register_tool(MCPCallTool())
        register_tool(MCPCallChainTool())
        register_tool(MCPFanOutTool())
        register_tool(MCPRankTool())
        register_tool(MCPSessionSummaryTool())
        register_tool(MCPServerManageTool())
