"""
Tool registry and base definitions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
from typing import Union
from pydantic import BaseModel
from oats.log import cl

log = cl('tool.registry')


class ToolContext(BaseModel):
    """Context passed to tool execution."""

    session_id: str
    project_dir: Path
    working_dir: Path
    user_confirmed: bool = False

    # Sub-agent support
    parent_session_id: Optional[str] = None
    agent_depth: int = 0
    max_agent_depth: int = 3

    # File state cache (optional, set by SessionProcessor)
    file_cache: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True


@dataclass
class ToolResult:
    """Result from tool execution."""

    title: str
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    attachments: list[dict[str, Any]] = field(default_factory=list)


class Tool(ABC):
    """Base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for the tool parameters."""
        pass

    @abstractmethod
    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Execute the tool with the given arguments."""
        pass

    def is_concurrency_safe(self, args: dict[str, Any] | None = None) -> bool:
        """
        Whether this tool can safely run concurrently with other tools.

        Read-only tools (read, glob, grep) are safe. Write tools (write, edit,
        bash) are not, because they can have side effects that conflict.

        Override in subclasses to return True for read-only tools.
        """
        return False

    @property
    def aliases(self) -> list[str]:
        """Optional alternate names for backwards compatibility."""
        return []

    @property
    def keywords(self) -> list[str]:
        """Short search terms describing when this tool should be used."""
        return []

    @property
    def always_load(self) -> bool:
        """
        Whether this tool should almost always be available to the model.

        Mirrors Claude Code's "alwaysLoad" concept in a lightweight way so
        essential tools stay visible even when we rank the broader tool set.
        """
        return False

    @property
    def strict(self) -> bool:
        """
        Whether provider-side tool schema adherence should be as strict as
        the serving stack supports.
        """
        return False

    def requires_permission(self, args: dict[str, Any], ctx: ToolContext) -> str | None:
        """
        Check if this execution requires user permission.

        Returns None if no permission needed, or a description of what permission is needed.
        """
        return None

    def to_definition(self) -> dict[str, Any]:
        """Convert to LLM tool definition format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        tool = self._tools.get(name)
        if tool is not None:
            return tool
        for candidate in self._tools.values():
            if name in candidate.aliases:
                return candidate
        return None

    def list(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def to_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions for LLM."""
        return [tool.to_definition() for tool in self._tools.values()]


# Global tool registry
_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def register_tool(tool: Tool) -> None:
    """Register a tool in the global registry."""
    get_tool_registry().register(tool)


def get_tool(name: str) -> Tool | None:
    """Get a tool by name."""
    return get_tool_registry().get(name)


def list_tools() -> list[Tool]:
    """List all registered tools."""
    return get_tool_registry().list()
