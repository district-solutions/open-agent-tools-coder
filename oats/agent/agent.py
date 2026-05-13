"""
Agent definitions and management.
"""
from __future__ import annotations


from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentType(str, Enum):
    """Types of sub-agents with different tool access levels."""

    GENERAL = "general"
    EXPLORE = "explore"
    PLAN = "plan"
    VERIFY = "verify"


# Per-type tool restrictions. None means all tools allowed.
AGENT_TYPE_TOOLS: dict[AgentType, set[str] | None] = {
    AgentType.GENERAL: None,  # all tools
    AgentType.EXPLORE: {
        "read", "glob", "grep", "bash", "webfetch", "websearch",
        "question", "todowrite", "todoread",
    },
    AgentType.PLAN: {
        "read", "glob", "grep", "question",
        "plan_enter", "plan_exit", "plan_status",
        "todowrite", "todoread", "webfetch", "websearch",
    },
    AgentType.VERIFY: {
        "read", "glob", "grep", "bash", "question",
        "todowrite", "todoread",
    },
}

# Max iterations per agent type
AGENT_TYPE_MAX_ITERATIONS: dict[AgentType, int] = {
    AgentType.GENERAL: 200,
    AgentType.EXPLORE: 100,
    AgentType.PLAN: 100,
    AgentType.VERIFY: 100,
}


@dataclass
class Agent:
    """Definition of an AI agent."""

    name: str
    description: str = ""
    prompt: str = ""
    agent_type: AgentType = AgentType.GENERAL
    model_id: str | None = None
    provider_id: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[str] = field(default_factory=list)  # Tool names to enable
    allowed_tools: set[str] | None = None  # Restrict tool access
    max_iterations: int = 200
    options: dict[str, Any] = field(default_factory=dict)


# Built-in agents
BUILTIN_AGENTS: list[Agent] = [
    Agent(
        name="default",
        description="General-purpose coding assistant",
        prompt="You are a helpful AI coding assistant.",
        agent_type=AgentType.GENERAL,
    ),
    Agent(
        name="coder",
        description="Focused on writing and modifying code",
        prompt="""You are an expert programmer. Focus on:
- Writing clean, efficient code
- Following best practices
- Using appropriate design patterns
- Adding helpful comments when needed""",
        agent_type=AgentType.GENERAL,
    ),
    Agent(
        name="explorer",
        description="Read-only codebase exploration specialist",
        prompt="""You are a codebase exploration specialist. Focus on:
- Reading and understanding code structure
- Finding files, classes, and functions
- Answering questions about the codebase
- Do NOT modify any files""",
        agent_type=AgentType.EXPLORE,
        allowed_tools=AGENT_TYPE_TOOLS[AgentType.EXPLORE],
        max_iterations=100,
    ),
    Agent(
        name="planner",
        description="Plans implementation approaches",
        prompt="""You are a software architect. Focus on:
- Designing implementation approaches
- Identifying files that need changes
- Considering trade-offs between approaches
- Creating step-by-step plans""",
        agent_type=AgentType.PLAN,
        allowed_tools=AGENT_TYPE_TOOLS[AgentType.PLAN],
        max_iterations=100,
    ),
    Agent(
        name="reviewer",
        description="Code review specialist",
        prompt="""You are a code review expert. Focus on:
- Finding bugs and issues
- Suggesting improvements
- Checking for security vulnerabilities
- Ensuring code quality""",
        agent_type=AgentType.VERIFY,
        allowed_tools=AGENT_TYPE_TOOLS[AgentType.VERIFY],
        max_iterations=100,
    ),
    Agent(
        name="explainer",
        description="Explains code and concepts",
        prompt="""You are a technical educator. Focus on:
- Clear, simple explanations
- Breaking down complex concepts
- Providing examples
- Answering follow-up questions""",
        agent_type=AgentType.EXPLORE,
        allowed_tools=AGENT_TYPE_TOOLS[AgentType.EXPLORE],
        max_iterations=100,
    ),
]


class AgentRegistry:
    """Registry of available agents."""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        # Load built-in agents
        for agent in BUILTIN_AGENTS:
            self.register(agent)

    def register(self, agent: Agent) -> None:
        """Register an agent."""
        self._agents[agent.name] = agent

    def get(self, name: str) -> Agent | None:
        """Get an agent by name."""
        return self._agents.get(name)

    def list(self) -> list[Agent]:
        """List all agents."""
        return list(self._agents.values())


# Global agent registry
_registry: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry


def get_agent(name: str) -> Agent | None:
    """Get an agent by name."""
    return get_agent_registry().get(name)


def list_agents() -> list[Agent]:
    """List all available agents."""
    return get_agent_registry().list()
