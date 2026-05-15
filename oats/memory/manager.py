"""
Memory manager — load, save, search, and build system prompt sections.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from oats.memory.models import Memory, MemoryType
from oats.date import utc
from oats.log import cl

log = cl("memory.mgr")

MAX_INDEX_LINES = 200
MAX_PROMPT_CHARS = int(os.getenv('CODER_MEMORY_MAX_CHARS', '25000'))  # ~6K tokens


class MemoryManager:
    """
    Manages persistent memories across sessions.

    Memories stored as markdown files with YAML frontmatter in:
    - user_dir: ~/.coder/memory/ (user-global)
    - project_dir: <project>/.coder/memory/ (project-local)
    """

    def __init__(
        self,
        user_dir: Path | None = None,
        project_dir: Path | None = None,
    ) -> None:
        """Initialize the memory manager with user and project directories.

        Args:
            user_dir: Directory for user-global memories (default: ``~/.coder/memory``).
            project_dir: Directory for project-local memories (default: ``<project>/.coder/memory``).
        """
        self._user_dir = user_dir or Path.home() / ".coder" / "memory"
        self._project_dir = (
            Path(str(project_dir)) / ".coder" / "memory"
            if project_dir
            else None
        )

    async def load_all(self) -> list[Memory]:
        """Load all memories from both directories."""
        memories = []
        for d in [self._user_dir, self._project_dir]:
            if d and d.exists():
                for f in sorted(d.glob("*.md")):
                    if f.name == "MEMORY.md":
                        continue
                    try:
                        text = f.read_text(encoding="utf-8")
                        mem = Memory.from_frontmatter(text)
                        if mem:
                            memories.append(mem)
                    except Exception as e:
                        log.warn(f"failed to load memory {f}: {e}")
        return memories

    async def save(self, memory: Memory, scope: str = "project") -> Memory:
        """Save a memory to the appropriate directory."""
        memory.updated_at = utc()

        if scope == "user":
            target_dir = self._user_dir
        else:
            target_dir = self._project_dir or self._user_dir

        target_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename from title (sanitized)
        safe_name = "".join(
            c if c.isalnum() or c in "-_ " else "" for c in memory.title
        ).strip().replace(" ", "_").lower()
        if not safe_name:
            safe_name = memory.id[:8]
        filename = f"{safe_name}.md"

        filepath = target_dir / filename
        filepath.write_text(memory.to_frontmatter(), encoding="utf-8")

        # Update MEMORY.md index
        await self._update_index(target_dir)

        log.info(f"saved memory: {filepath}")
        return memory

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        for d in [self._user_dir, self._project_dir]:
            if not d or not d.exists():
                continue
            for f in d.glob("*.md"):
                if f.name == "MEMORY.md":
                    continue
                try:
                    text = f.read_text(encoding="utf-8")
                    if f"id: {memory_id}" in text:
                        f.unlink()
                        await self._update_index(d)
                        log.info(f"deleted memory: {f}")
                        return True
                except Exception:
                    continue
        return False

    async def search(self, query: str) -> list[Memory]:
        """Simple keyword search across memories."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        results = []

        all_memories = await self.load_all()
        for mem in all_memories:
            text = f"{mem.title} {mem.content} {' '.join(mem.tags)}".lower()
            score = sum(1 for w in query_words if w in text)
            if score > 0:
                results.append((score, mem))

        results.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in results]

    async def build_system_prompt_section(self) -> str:
        """
        Build the memory section for the system prompt.

        Loads MEMORY.md index and referenced files, truncated to MAX_PROMPT_CHARS.
        """
        sections = []
        total_chars = 0

        # Load MEMORY.md index from both directories
        for d in [self._project_dir, self._user_dir]:
            if not d or not d.exists():
                continue
            index_path = d / "MEMORY.md"
            if index_path.exists():
                try:
                    content = index_path.read_text(encoding="utf-8")
                    # Truncate to MAX_INDEX_LINES
                    lines = content.split("\n")[:MAX_INDEX_LINES]
                    index_text = "\n".join(lines)
                    sections.append(index_text)
                    total_chars += len(index_text)
                except Exception:
                    pass

        # Load individual memory files
        memories = await self.load_all()
        for mem in memories:
            if total_chars >= MAX_PROMPT_CHARS:
                break
            entry = f"## [{mem.type.value}] {mem.title}\n{mem.content}"
            if total_chars + len(entry) > MAX_PROMPT_CHARS:
                remaining = MAX_PROMPT_CHARS - total_chars
                entry = entry[:remaining] + "\n...(truncated)"
            sections.append(entry)
            total_chars += len(entry)

        if not sections:
            return ""

        return "\n\n".join(sections)

    async def _update_index(self, directory: Path) -> None:
        """Rebuild the MEMORY.md index file."""
        entries = []
        for f in sorted(directory.glob("*.md")):
            if f.name == "MEMORY.md":
                continue
            try:
                text = f.read_text(encoding="utf-8")
                mem = Memory.from_frontmatter(text)
                if mem:
                    tags = f" ({', '.join(mem.tags)})" if mem.tags else ""
                    entries.append(f"- [{mem.title}]({f.name}) — {mem.type.value}{tags}")
            except Exception:
                continue

        index_content = "# Memory Index\n\n" + "\n".join(entries[:MAX_INDEX_LINES])
        index_path = directory / "MEMORY.md"
        index_path.write_text(index_content, encoding="utf-8")
