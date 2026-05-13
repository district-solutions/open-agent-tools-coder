"""
Persistent memory system for cross-session context.

Memories are stored as markdown files with YAML frontmatter in:
- ~/.coder/memory/ (user-global)
- <project>/.coder/memory/ (project-local)

Each directory has a MEMORY.md index file (max 200 lines).
Memories are loaded into the system prompt at session start.
"""
from oats.memory.models import Memory, MemoryType
from oats.memory.manager import MemoryManager

__all__ = ["Memory", "MemoryType", "MemoryManager"]
