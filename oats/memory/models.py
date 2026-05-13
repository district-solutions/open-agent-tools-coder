"""
Memory data models.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from oats.core.id import generate_id
from oats.date import utc


class MemoryType(str, Enum):
    """Types of persistent memories."""

    USER = "user"          # User role, preferences, knowledge
    FEEDBACK = "feedback"  # How to approach work (corrections + confirmations)
    PROJECT = "project"    # Ongoing work, goals, decisions
    REFERENCE = "reference"  # Pointers to external resources


class Memory(BaseModel):
    """A persistent memory entry."""

    id: str = Field(default_factory=generate_id)
    type: MemoryType = MemoryType.PROJECT
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc)
    updated_at: datetime = Field(default_factory=utc)
    source: str | None = None  # "user", "agent", "feedback"

    def to_frontmatter(self) -> str:
        """Serialize to markdown with YAML frontmatter."""
        tags_str = ", ".join(self.tags) if self.tags else ""
        lines = [
            "---",
            f"id: {self.id}",
            f"type: {self.type.value}",
            f"title: {self.title}",
        ]
        if tags_str:
            lines.append(f"tags: [{tags_str}]")
        lines.extend([
            f"created_at: {self.created_at.isoformat()}",
            f"updated_at: {self.updated_at.isoformat()}",
        ])
        if self.source:
            lines.append(f"source: {self.source}")
        lines.extend([
            "---",
            "",
            self.content,
        ])
        return "\n".join(lines)

    @classmethod
    def from_frontmatter(cls, text: str) -> Optional["Memory"]:
        """Parse from markdown with YAML frontmatter."""
        if not text.startswith("---"):
            return None

        parts = text.split("---", 2)
        if len(parts) < 3:
            return None

        frontmatter = parts[1].strip()
        content = parts[2].strip()

        # Simple YAML parsing (no external dependency)
        meta = {}
        for line in frontmatter.split("\n"):
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                value = value.strip()
                # Handle bracketed lists: [tag1, tag2]
                if value.startswith("[") and value.endswith("]"):
                    value = [v.strip() for v in value[1:-1].split(",") if v.strip()]
                meta[key.strip()] = value

        try:
            mem_type = MemoryType(meta.get("type", "project"))
        except ValueError:
            mem_type = MemoryType.PROJECT

        return cls(
            id=meta.get("id", generate_id()),
            type=mem_type,
            title=meta.get("title", "Untitled"),
            content=content,
            tags=meta.get("tags", []) if isinstance(meta.get("tags"), list) else [],
            source=meta.get("source"),
        )
