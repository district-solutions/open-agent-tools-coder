"""
MCP tool implementations that plug into the existing coder tool registry.

These tools expose the MCP orchestration capabilities to the AI agent,
allowing it to discover, call, and manage MCP tools during sessions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oats.tool.registry import Tool, ToolContext, ToolResult
from oats.mcp.orchestrator import MCPOrchestrator
from oats.mcp.registry import get_mcp_registry, init_mcp_registry
from oats.mcp.ranking import ToolRanker
from oats.mcp.tracker import ToolCallTracker

# Lazy-initialized orchestrator
_orchestrator: MCPOrchestrator | None = None


async def _get_orchestrator(ctx: ToolContext) -> MCPOrchestrator:
    """Get or create the MCP orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        registry = await init_mcp_registry(ctx.project_dir)
        tracker = ToolCallTracker(ctx.project_dir / ".coder" / "mcp_tracking")
        ranker = ToolRanker()
        _orchestrator = MCPOrchestrator(registry, tracker, ranker)
        await _orchestrator.initialize()
    return _orchestrator


class MCPDiscoverTool(Tool):
    """Discover available tools across all connected MCP servers."""

    @property
    def name(self) -> str:
        return "mcp_discover"

    @property
    def description(self) -> str:
        return (
            "Discover and list available tools across all connected MCP servers. "
            "Use this to find what tools are available before calling them. "
            "Can filter by server name or search by keyword."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "Filter tools by MCP server name (optional)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant tools (optional)",
                },
                "refresh": {
                    "type": "boolean",
                    "description": "Force re-discovery of tools from servers (default: false)",
                },
            },
            "required": [],
        }

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            orchestrator = await _get_orchestrator(ctx)
            registry = orchestrator._registry

            if args.get("refresh", False):
                await registry.discover_all()

            server = args.get("server")
            query = args.get("query")

            if query:
                tools = registry.search_tools(query)
            else:
                tools = registry.list_tools(server)

            # Format output
            lines = [f"Found {len(tools)} tools:"]
            for t in tools:
                lines.append(f"\n**{t.name}** (server: {t.server_name})")
                if t.description:
                    lines.append(f"  {t.description[:200]}")
                if t.tags:
                    lines.append(f"  Tags: {', '.join(t.tags)}")

            # Also include server health
            health = registry.get_server_health()
            if health:
                lines.append("\n**Server Health:**")
                for name, is_healthy in health.items():
                    status = "healthy" if is_healthy else "unhealthy"
                    lines.append(f"  {name}: {status}")

            return ToolResult(
                title="MCP Discover",
                output="\n".join(lines),
                metadata={
                    "tool_count": len(tools),
                    "server_count": len(registry.list_servers()),
                    "tools": [
                        {"name": t.name, "server": t.server_name, "description": t.description[:100]}
                        for t in tools
                    ],
                },
            )
        except Exception as e:
            return ToolResult(title="MCP Discover", output="", error=str(e))


class MCPCallTool(Tool):
    """Call a tool on a connected MCP server."""

    @property
    def name(self) -> str:
        return "mcp_call"

    @property
    def description(self) -> str:
        return (
            "Call a specific tool on a connected MCP server. The tool name should be "
            "in qualified format: 'server_name.tool_name'. Use mcp_discover first to "
            "find available tools. Handles stuck detection and automatic resolution."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Qualified tool name (server.tool) to call",
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments to pass to the tool",
                },
                "task_description": {
                    "type": "string",
                    "description": "Description of the task (helps with stuck resolution)",
                },
            },
            "required": ["tool_name"],
        }

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            orchestrator = await _get_orchestrator(ctx)

            record = await orchestrator.call_tool(
                tool_name=args["tool_name"],
                arguments=args.get("arguments", {}),
                session_id=ctx.session_id,
                task_description=args.get("task_description", ""),
            )

            if record.error:
                return ToolResult(
                    title=f"MCP Call: {args['tool_name']}",
                    output=record.result or "",
                    error=record.error,
                    metadata={
                        "call_id": record.call_id,
                        "status": record.status.value,
                        "latency_ms": record.latency_ms,
                        "resolution_chain": record.resolution_chain,
                    },
                )

            return ToolResult(
                title=f"MCP Call: {args['tool_name']}",
                output=record.result or "",
                metadata={
                    "call_id": record.call_id,
                    "status": record.status.value,
                    "latency_ms": record.latency_ms,
                    "server": record.server_name,
                    "depth": record.depth,
                },
            )
        except Exception as e:
            return ToolResult(
                title=f"MCP Call: {args.get('tool_name', 'unknown')}",
                output="",
                error=str(e),
            )


class MCPCallChainTool(Tool):
    """Execute a chain of MCP tool calls sequentially."""

    @property
    def name(self) -> str:
        return "mcp_chain"

    @property
    def description(self) -> str:
        return (
            "Execute a chain of MCP tool calls sequentially. Each call can build "
            "on previous results. Use this for multi-step workflows that need to "
            "cross-reference data across different MCP servers."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "calls": {
                    "type": "array",
                    "description": "Ordered list of tool calls to execute",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string", "description": "Qualified tool name"},
                            "arguments": {"type": "object", "description": "Tool arguments"},
                        },
                        "required": ["tool"],
                    },
                },
                "task_description": {
                    "type": "string",
                    "description": "Overall task description for the chain",
                },
            },
            "required": ["calls"],
        }

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            orchestrator = await _get_orchestrator(ctx)

            records = await orchestrator.call_tool_chain(
                calls=args["calls"],
                session_id=ctx.session_id,
                task_description=args.get("task_description", ""),
            )

            lines = [f"Chain completed: {len(records)} calls"]
            for i, r in enumerate(records, 1):
                status = r.status.value
                latency = f"{r.latency_ms:.0f}ms" if r.latency_ms else "n/a"
                lines.append(f"\n{i}. **{r.tool_name}** [{status}] ({latency})")
                if r.result:
                    preview = r.result[:300]
                    lines.append(f"   Result: {preview}")
                if r.error:
                    lines.append(f"   Error: {r.error}")

            return ToolResult(
                title="MCP Chain",
                output="\n".join(lines),
                metadata={
                    "total_calls": len(records),
                    "successes": sum(1 for r in records if r.status.value in ("success", "resolved")),
                    "errors": sum(1 for r in records if r.status.value == "error"),
                },
            )
        except Exception as e:
            return ToolResult(title="MCP Chain", output="", error=str(e))


class MCPFanOutTool(Tool):
    """Execute multiple MCP tool calls concurrently (fan-out)."""

    @property
    def name(self) -> str:
        return "mcp_fan_out"

    @property
    def description(self) -> str:
        return (
            "Execute multiple MCP tool calls concurrently for parallel data gathering. "
            "This is the spoke pattern - dispatch to multiple servers at once and "
            "collect all results. Use when you need data from multiple independent sources."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "calls": {
                    "type": "array",
                    "description": "List of tool calls to execute in parallel",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string"},
                            "arguments": {"type": "object"},
                        },
                        "required": ["tool"],
                    },
                },
                "max_concurrent": {
                    "type": "integer",
                    "description": "Max parallel calls (default: 10)",
                },
            },
            "required": ["calls"],
        }

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            orchestrator = await _get_orchestrator(ctx)

            records = await orchestrator.fan_out(
                tool_calls=args["calls"],
                session_id=ctx.session_id,
                max_concurrent=args.get("max_concurrent", 10),
            )

            lines = [f"Fan-out completed: {len(records)} parallel calls"]
            for r in records:
                status = r.status.value
                lines.append(f"\n**{r.tool_name}** [{status}]")
                if r.result:
                    lines.append(f"  {r.result[:200]}")
                if r.error:
                    lines.append(f"  Error: {r.error}")

            return ToolResult(
                title="MCP Fan-Out",
                output="\n".join(lines),
                metadata={
                    "total": len(records),
                    "successes": sum(1 for r in records if r.status.value in ("success", "resolved")),
                },
            )
        except Exception as e:
            return ToolResult(title="MCP Fan-Out", output="", error=str(e))


class MCPRankTool(Tool):
    """Rank available MCP tools for a specific task."""

    @property
    def name(self) -> str:
        return "mcp_rank"

    @property
    def description(self) -> str:
        return (
            "Rank available MCP tools by relevance to a specific task description. "
            "Uses BM25 text matching combined with reliability and latency scores. "
            "Use this before mcp_call to pick the best tool for your task."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Description of the task to rank tools for",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top tools to return (default: 10)",
                },
            },
            "required": ["task"],
        }

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            orchestrator = await _get_orchestrator(ctx)
            ranked = orchestrator.rank_tools_for_task(
                args["task"],
                top_k=args.get("top_k", 10),
            )

            lines = [f"Top tools for: {args['task'][:100]}"]
            lines.append("")
            lines.append("| Rank | Tool | Server | Score | Relevance | Reliability |")
            lines.append("|------|------|--------|-------|-----------|-------------|")

            for i, entry in enumerate(ranked, 1):
                lines.append(
                    f"| {i} | {entry['tool_name']} | {entry['server_name']} "
                    f"| {entry['score']:.3f} | {entry['relevance']:.3f} "
                    f"| {entry['reliability']:.3f} |"
                )

            return ToolResult(
                title="MCP Rank",
                output="\n".join(lines),
                metadata={"ranked_tools": ranked},
            )
        except Exception as e:
            return ToolResult(title="MCP Rank", output="", error=str(e))


class MCPSessionSummaryTool(Tool):
    """Get a summary of the current MCP orchestration session."""

    @property
    def name(self) -> str:
        return "mcp_session"

    @property
    def description(self) -> str:
        return (
            "Get a summary of the current MCP tool calling session including "
            "total calls, success rates, stuck resolutions, and performance stats."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID (defaults to current session)",
                },
            },
            "required": [],
        }

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            orchestrator = await _get_orchestrator(ctx)
            sid = args.get("session_id", ctx.session_id)
            summary = orchestrator.get_session_summary(sid)

            if "error" in summary:
                return ToolResult(title="MCP Session", output="", error=summary["error"])

            lines = [
                f"# MCP Session: {summary['session_id']}",
                f"- Total calls: {summary['total_calls']}",
                f"- Successes: {summary['successes']}",
                f"- Errors: {summary['errors']}",
                f"- Stuck: {summary['stuck']}",
                f"- Success rate: {summary['success_rate']:.1%}",
                f"- Max depth: {summary['max_depth']}",
                f"- Duration: {summary['duration_seconds']:.1f}s",
            ]

            return ToolResult(
                title="MCP Session",
                output="\n".join(lines),
                metadata=summary,
            )
        except Exception as e:
            return ToolResult(title="MCP Session", output="", error=str(e))


class MCPServerManageTool(Tool):
    """Add or remove MCP servers at runtime."""

    @property
    def name(self) -> str:
        return "mcp_server_manage"

    @property
    def description(self) -> str:
        return (
            "Add, remove, or health-check MCP servers at runtime. "
            "Use 'add' to connect a new server, 'remove' to disconnect, "
            "or 'health' to check server status."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "remove", "health", "list"],
                    "description": "Action to perform",
                },
                "name": {
                    "type": "string",
                    "description": "Server name (required for add/remove/health)",
                },
                "url": {
                    "type": "string",
                    "description": "Server URL (required for add with http transport)",
                },
                "command": {
                    "type": "string",
                    "description": "Server command (required for add with stdio transport)",
                },
                "transport": {
                    "type": "string",
                    "enum": ["http", "sse", "stdio"],
                    "description": "Transport type (default: http)",
                },
                "description": {
                    "type": "string",
                    "description": "Server description",
                },
            },
            "required": ["action"],
        }

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            orchestrator = await _get_orchestrator(ctx)
            registry = orchestrator._registry
            action = args["action"]

            if action == "list":
                servers = registry.list_servers()
                health = registry.get_server_health()
                lines = [f"Registered MCP servers ({len(servers)}):"]
                for name, config in servers.items():
                    status = "healthy" if health.get(name, False) else "unknown"
                    lines.append(
                        f"\n**{name}** [{status}]"
                        f"\n  URL: {config.url or 'stdio'}"
                        f"\n  Transport: {config.transport.value}"
                        f"\n  Description: {config.description}"
                    )
                return ToolResult(title="MCP Servers", output="\n".join(lines))

            elif action == "add":
                from oats.mcp.models import MCPServerConfig, MCPTransport
                name = args.get("name", "")
                if not name:
                    return ToolResult(title="MCP Server", output="", error="Name required")

                config = MCPServerConfig(
                    name=name,
                    description=args.get("description", ""),
                    transport=MCPTransport(args.get("transport", "http")),
                    url=args.get("url"),
                    command=args.get("command"),
                )
                registry.add_server(name, config)
                await registry._discover_server(name, config)
                tools = registry.list_tools(name)
                return ToolResult(
                    title="MCP Server Added",
                    output=f"Added server '{name}' with {len(tools)} tools discovered",
                )

            elif action == "remove":
                name = args.get("name", "")
                if not name:
                    return ToolResult(title="MCP Server", output="", error="Name required")
                registry.remove_server(name)
                return ToolResult(
                    title="MCP Server Removed",
                    output=f"Removed server '{name}'",
                )

            elif action == "health":
                name = args.get("name", "")
                if not name:
                    return ToolResult(title="MCP Server", output="", error="Name required")
                is_healthy = await registry.health_check(name)
                return ToolResult(
                    title="MCP Health",
                    output=f"Server '{name}': {'healthy' if is_healthy else 'unhealthy'}",
                )

            return ToolResult(title="MCP Server", output="", error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(title="MCP Server", output="", error=str(e))
