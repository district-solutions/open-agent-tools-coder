"""
Event bus for pub/sub pattern communication.
"""
from __future__ import annotations


import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from enum import Enum


class EventType(str, Enum):
    """Event types for the event bus."""

    # Session events
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    SESSION_DELETED = "session.deleted"
    SESSION_BUDGET = "session.budget"
    SESSION_COMPACTED = "session.compacted"
    SESSION_TASK_BUDGET = "session.task_budget"

    # Message events
    MESSAGE_CREATED = "message.created"
    MESSAGE_UPDATED = "message.updated"
    MESSAGE_PART_CREATED = "message.part.created"
    MESSAGE_PART_UPDATED = "message.part.updated"

    # Tool events
    TOOL_START = "tool.start"
    TOOL_COMPLETE = "tool.complete"
    TOOL_ERROR = "tool.error"

    # Provider events
    PROVIDER_REQUEST = "provider.request"
    PROVIDER_RESPONSE = "provider.response"
    PROVIDER_ERROR = "provider.error"

    # Permission events
    PERMISSION_REQUEST = "permission.request"
    PERMISSION_GRANTED = "permission.granted"
    PERMISSION_DENIED = "permission.denied"

    # Hook events
    HOOK_FIRED = "hook.fired"
    HOOK_BLOCKED = "hook.blocked"

    # File events
    FILE_CHANGED = "file.changed"

    # Agent events
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"


@dataclass
class Event:
    """An event published to the bus."""

    type: EventType | str
    data: dict[str, Any] = field(default_factory=dict)
    source: str | None = None


EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """
    Simple pub/sub event bus for internal communication.

    Supports both sync and async handlers.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._all_handlers: list[EventHandler] = []
        self._once_handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: EventType | str, handler: EventHandler) -> Callable[[], None]:
        """
        Subscribe to events of a specific type.

        Returns an unsubscribe function.
        """
        key = str(event_type)
        self._handlers[key].append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers[key]:
                self._handlers[key].remove(handler)

        return unsubscribe

    def subscribe_all(self, handler: EventHandler) -> Callable[[], None]:
        """
        Subscribe to all events.

        Returns an unsubscribe function.
        """
        self._all_handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._all_handlers:
                self._all_handlers.remove(handler)

        return unsubscribe

    def once(self, event_type: EventType | str, handler: EventHandler) -> Callable[[], None]:
        """
        Subscribe to a single event of a specific type.

        The handler will be automatically unsubscribed after being called once.
        Returns an unsubscribe function.
        """
        key = str(event_type)
        self._once_handlers[key].append(handler)

        def unsubscribe() -> None:
            if handler in self._once_handlers[key]:
                self._once_handlers[key].remove(handler)

        return unsubscribe

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        key = str(event.type)

        # Call type-specific handlers
        for handler in self._handlers.get(key, []):
            try:
                await handler(event)
            except Exception as e:
                # Log but don't fail on handler errors
                print(f"Error in event handler for {key}: {e}")

        # Call once handlers
        once_handlers = self._once_handlers.pop(key, [])
        for handler in once_handlers:
            try:
                await handler(event)
            except Exception as e:
                print(f"Error in once event handler for {key}: {e}")

        # Call all-event handlers
        for handler in self._all_handlers:
            try:
                await handler(event)
            except Exception as e:
                print(f"Error in all-event handler for {key}: {e}")

    def publish_sync(self, event: Event) -> None:
        """Publish an event synchronously (creates a new event loop if needed)."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.publish(event))
        except RuntimeError:
            asyncio.run(self.publish(event))

    def clear(self) -> None:
        """Clear all handlers."""
        self._handlers.clear()
        self._all_handlers.clear()
        self._once_handlers.clear()


# Global event bus instance
bus = EventBus()
