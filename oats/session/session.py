"""
Session management for AI conversations.
"""
from __future__ import annotations


from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from oats.core.id import generate_id
from oats.core.storage import Storage
from oats.core.bus import bus, Event, EventType
from oats.session.message import Message
from oats.date import utc


class SessionTime(BaseModel):
    """Timestamps for session lifecycle."""

    created: datetime = Field(default_factory=utc)
    updated: datetime = Field(default_factory=utc)
    archived: datetime | None = None


class SessionInfo(BaseModel):
    """Metadata about a session."""

    model_config = {"protected_namespaces": ()}

    id: str = Field(default_factory=generate_id)
    title: str = "New Session"
    project_dir: str
    working_dir: str
    time: SessionTime = Field(default_factory=SessionTime)
    model_id: str | None = None
    provider_id: str | None = None
    root_session_id: str | None = None
    message_count: int = 0
    total_tokens: int = 0
    parent_session_id: str | None = None


class Session(BaseModel):
    """A conversation session with messages."""

    info: SessionInfo
    messages: list[Message] = Field(default_factory=list)

    @property
    def id(self) -> str:
        return self.info.id

    @property
    def title(self) -> str:
        return self.info.title

    def add_message(self, message: Message) -> Message:
        """Add a message to the session."""
        self.messages.append(message)
        self.info.message_count = len(self.messages)
        self.info.time.updated = utc()
        return message

    def create_user_message(
        self,
        content: str,
        images: list[dict[str, str]] | None = None,
    ) -> Message:
        """Create and add a user message.

        *images* is an optional list of dicts with keys:
            media_type  – e.g. "image/png"
            data        – base64-encoded bytes  (mutually exclusive with url)
            url         – image URL             (mutually exclusive with data)
        """
        message = Message(
            session_id=self.id,
            role="user",
        )
        message.add_text(content)
        for img in images or []:
            message.add_image(
                media_type=img["media_type"],
                data=img.get("data"),
                url=img.get("url"),
            )
        return self.add_message(message)

    def create_assistant_message(self) -> Message:
        """Create and add an assistant message."""
        message = Message(
            session_id=self.id,
            role="assistant",
            model=self.info.model_id,
            provider=self.info.provider_id,
        )
        return self.add_message(message)

    def create_system_message(self, content: str) -> Message:
        """Create and add a system message."""
        message = Message(
            session_id=self.id,
            role="system",
        )
        message.add_text(content)
        return self.add_message(message)

    def get_messages_for_llm(self) -> list[dict[str, Any]]:
        """Get messages formatted for LLM API."""
        return [m.to_llm_format() for m in self.messages]

    def update_title(self, title: str) -> None:
        """Update the session title."""
        self.info.title = title
        self.info.time.updated = utc()

    def add_usage(self, usage: dict[str, int]) -> None:
        """Add token usage from a completion."""
        if "total_tokens" in usage:
            self.info.total_tokens += usage["total_tokens"]


# Storage for sessions
class SessionStorage:
    """Storage manager for sessions."""

    def __init__(self) -> None:
        self._storage = Storage(namespace="sessions", model_class=Session)

    async def create(self, session: Session) -> Session:
        """Create a new session."""
        await self._storage.create(session.id, session)
        await bus.publish(
            Event(
                type=EventType.SESSION_CREATED,
                data={"session_id": session.id, "title": session.title},
            )
        )
        return session

    async def get(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        return await self._storage.read(session_id)

    async def update(self, session: Session) -> Session:
        """Update a session."""
        await self._storage.upsert(session.id, session)
        await bus.publish(
            Event(
                type=EventType.SESSION_UPDATED,
                data={"session_id": session.id},
            )
        )
        return session

    async def delete(self, session_id: str) -> bool:
        """Delete a session."""
        result = await self._storage.delete(session_id)
        if result:
            await bus.publish(
                Event(
                    type=EventType.SESSION_DELETED,
                    data={"session_id": session_id},
                )
            )
        return result

    async def list(self) -> list[Session]:
        """List all sessions."""
        return await self._storage.list()

    async def list_infos(self) -> list[SessionInfo]:
        """List session infos only (lighter weight)."""
        sessions = await self.list()
        return [s.info for s in sessions]


# Global session storage
_storage: SessionStorage | None = None


def get_session_storage() -> SessionStorage:
    """Get the global session storage."""
    global _storage
    if _storage is None:
        _storage = SessionStorage()
    return _storage


async def create_session(
    project_dir: Path,
    working_dir: Path | None = None,
    title: str = "New Session",
    model_id: str | None = None,
    provider_id: str | None = None,
    root_session_id: str | None = None,
) -> Session:
    """Create a new session."""
    if working_dir is None:
        working_dir = project_dir

    info = SessionInfo(
        title=title,
        project_dir=str(project_dir),
        working_dir=str(working_dir),
        model_id=model_id,
        provider_id=provider_id,
        root_session_id=root_session_id,
    )
    session = Session(info=info)
    return await get_session_storage().create(session)


async def get_session(session_id: str) -> Session | None:
    """Get a session by ID."""
    return await get_session_storage().get(session_id)


async def list_sessions() -> list[SessionInfo]:
    """List all session infos."""
    return await get_session_storage().list_infos()
