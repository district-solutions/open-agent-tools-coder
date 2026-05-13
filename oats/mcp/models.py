"""
Pydantic models for the MCP tool calling protocol.

Defines the data structures for MCP server configuration, tool definitions,
call tracking, ranking, and orchestration state.
"""
from __future__ import annotations

import hashlib
import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# MCP Server Configuration
# ---------------------------------------------------------------------------

class MCPTransport(str, Enum):
    """Supported MCP transport types.

    Per MCP spec 2025-06-18: SSE is deprecated, replaced by Streamable HTTP.
    STREAMABLE_HTTP is the recommended transport for remote servers.
    STDIO remains for local per-user integrations.
    """
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable-http"
    HTTP = "http"


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""
    name: str
    description: str = ""
    transport: MCPTransport = MCPTransport.HTTP
    url: Optional[str] = None
    command: Optional[str] = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)
    # Rate limiting
    max_concurrent: int = 10
    timeout_seconds: int = 30


class MCPServersFile(BaseModel):
    """Root schema for mcp_servers.json config file."""
    version: str = "1.0"
    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool Definitions (reduced from LiteLLM spec)
# ---------------------------------------------------------------------------

class ToolParameter(BaseModel):
    """A single parameter for a tool."""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    enum: Optional[list[str]] = None
    default: Optional[Any] = None


class MCPToolDefinition(BaseModel):
    """Definition of a tool exposed by an MCP server."""
    name: str
    description: str = ""
    server_name: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    # Routing: the actual endpoint path for this tool's MCP server
    # For LiteLLM: /{mcp_function_name}/tools/call
    # For generic: /mcp-rest/tools/call
    mcp_function_name: str = ""
    call_endpoint: str = ""
    list_endpoint: str = ""
    # Ranking metadata
    call_count: int = 0
    success_count: int = 0
    avg_latency_ms: float = 0.0
    last_used: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count

    def to_litellm_format(self) -> dict[str, Any]:
        """Convert to LiteLLM/OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Tool Call Tracking
# ---------------------------------------------------------------------------

class ToolCallStatus(str, Enum):
    """Status of a tool call."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    STUCK = "stuck"
    RESOLVED = "resolved"
    CIRCUIT_OPEN = "circuit_open"
    DEGRADED = "degraded"


class ErrorCategory(str, Enum):
    """Classification of errors for retry strategy selection."""
    TRANSIENT = "transient"       # 429, 502, 503, timeout — backoff + retry
    SERVER = "server"             # 500, unknown server error — circuit breaker
    CLIENT = "client"             # 400, 404, validation — no retry, fix args
    UNKNOWN = "unknown"           # Unclassified


class ToolCallRecord(BaseModel):
    """Record of a single tool call for tracking."""
    call_id: str
    tool_name: str
    server_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: ToolCallStatus = ToolCallStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    error_category: ErrorCategory = ErrorCategory.UNKNOWN
    started_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    latency_ms: Optional[float] = None
    # Resolution chain - which tools were asked when stuck
    resolution_chain: list[str] = Field(default_factory=list)
    # Depth in the hub-and-spoke graph
    depth: int = 0
    parent_call_id: Optional[str] = None
    # Idempotency key for safe retries
    idempotency_key: Optional[str] = None
    # Retry metadata
    attempt: int = 1
    max_attempts: int = 1

    def mark_complete(self, result: str) -> None:
        self.status = ToolCallStatus.SUCCESS
        self.result = result
        self.completed_at = time.time()
        self.latency_ms = (self.completed_at - self.started_at) * 1000

    def mark_error(self, error: str, category: ErrorCategory = ErrorCategory.UNKNOWN) -> None:
        self.status = ToolCallStatus.ERROR
        self.error = error
        self.error_category = category
        self.completed_at = time.time()
        self.latency_ms = (self.completed_at - self.started_at) * 1000

    def mark_stuck(self) -> None:
        self.status = ToolCallStatus.STUCK

    def mark_circuit_open(self, server_name: str) -> None:
        self.status = ToolCallStatus.CIRCUIT_OPEN
        self.error = f"Circuit breaker open for server: {server_name}"
        self.completed_at = time.time()
        self.latency_ms = 0.0

    def mark_degraded(self, result: str) -> None:
        """Mark as degraded — partial/cached result returned."""
        self.status = ToolCallStatus.DEGRADED
        self.result = result
        self.completed_at = time.time()
        self.latency_ms = (self.completed_at - self.started_at) * 1000

    def compute_idempotency_key(self) -> str:
        """Compute idempotency key from tool name + arguments."""
        payload = json.dumps(
            {"tool": self.tool_name, "args": self.arguments},
            sort_keys=True,
        )
        self.idempotency_key = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return self.idempotency_key


# ---------------------------------------------------------------------------
# Circuit Breaker State
# ---------------------------------------------------------------------------

class CircuitState(str, Enum):
    """Circuit breaker states per distributed systems pattern."""
    CLOSED = "closed"        # Normal operation, counting failures
    OPEN = "open"            # Blocking requests, cooldown active
    HALF_OPEN = "half_open"  # Probing with single request


# ---------------------------------------------------------------------------
# Ranking Index
# ---------------------------------------------------------------------------

class ToolRankEntry(BaseModel):
    """Entry in the tool ranking index."""
    tool_name: str
    server_name: str
    score: float = 0.0
    relevance_score: float = 0.0
    reliability_score: float = 0.0
    latency_score: float = 0.0
    inertia_score: float = 0.0
    tags: list[str] = Field(default_factory=list)


class RankingIndex(BaseModel):
    """The full ranking index for tool selection."""
    entries: list[ToolRankEntry] = Field(default_factory=list)
    last_updated: float = Field(default_factory=time.time)

    def top_k(self, k: int = 10) -> list[ToolRankEntry]:
        """Get top-k tools by score."""
        return sorted(self.entries, key=lambda e: e.score, reverse=True)[:k]

    def for_query(self, query: str, k: int = 10) -> list[ToolRankEntry]:
        """Get top-k tools relevant to a query (simple keyword match)."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored = []
        for entry in self.entries:
            text = f"{entry.tool_name} {entry.server_name} {' '.join(entry.tags)}".lower()
            text_words = set(text.split())
            overlap = len(query_words & text_words)
            if overlap > 0 or any(w in text for w in query_words):
                match_score = overlap / max(len(query_words), 1)
                combined = entry.score * 0.5 + match_score * 0.5
                scored.append((combined, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k]]


# ---------------------------------------------------------------------------
# Orchestrator State
# ---------------------------------------------------------------------------

class OrchestrationSession(BaseModel):
    """State for a hub-and-spoke orchestration session."""
    session_id: str
    call_records: list[ToolCallRecord] = Field(default_factory=list)
    total_calls: int = 0
    max_depth: int = 0
    started_at: float = Field(default_factory=time.time)
    ranking_index: RankingIndex = Field(default_factory=RankingIndex)
    # Wall-clock timeout for watchdog
    timeout_seconds: float = 1800.0  # 30 minutes default

    def add_record(self, record: ToolCallRecord) -> None:
        self.call_records.append(record)
        self.total_calls += 1
        if record.depth > self.max_depth:
            self.max_depth = record.depth

    @property
    def is_timed_out(self) -> bool:
        """Check if session has exceeded wall-clock timeout."""
        return (time.time() - self.started_at) > self.timeout_seconds


# ---------------------------------------------------------------------------
# LiteLLM API Spec Filter
# ---------------------------------------------------------------------------

class LiteLLMEndpoint(BaseModel):
    """Reduced representation of a LiteLLM API endpoint."""
    path: str
    method: str
    summary: str = ""
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    request_body: dict[str, Any] = Field(default_factory=dict)
    response_schema: dict[str, Any] = Field(default_factory=dict)


class LiteLLMFilteredSpec(BaseModel):
    """Filtered/reduced LiteLLM API spec - only what we need for tool calling."""
    version: str = ""
    base_url: str = ""
    endpoints: list[LiteLLMEndpoint] = Field(default_factory=list)
    schemas: dict[str, Any] = Field(default_factory=dict)
    total_original_size_bytes: int = 0
    filtered_size_bytes: int = 0

    @property
    def reduction_ratio(self) -> float:
        if self.total_original_size_bytes == 0:
            return 0.0
        return 1.0 - (self.filtered_size_bytes / self.total_original_size_bytes)
