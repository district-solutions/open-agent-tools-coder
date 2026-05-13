"""
Usage tracking for coder2 - aggregates statistics across all sessions.

Tracks lifetime statistics including:
- Total sessions created
- Total prompts (messages) sent
- Total tokens used (prompt + completion) with input/output breakdown
- Token breakdown by session
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from oats.session.session import list_sessions, SessionInfo


@dataclass
class SessionUsageEntry:
    """Per-session usage entry for detailed breakdown."""

    session_id: str
    title: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    message_count: int = 0
    created: Optional[datetime] = None


@dataclass
class UsageStats:
    """Aggregated usage statistics across all sessions."""

    total_sessions: int = 0
    total_prompts: int = 0  # Total user messages across all sessions
    total_tokens: int = 0   # Total tokens across all sessions
    total_input_tokens: int = 0   # Total input (prompt) tokens
    total_output_tokens: int = 0  # Total output (completion) tokens
    earliest_session: Optional[datetime] = None
    latest_session: Optional[datetime] = None
    sessions: list[SessionUsageEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for display."""
        return {
            "total_sessions": self.total_sessions,
            "total_prompts": self.total_prompts,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "earliest_session": self.earliest_session.isoformat() if self.earliest_session else None,
            "latest_session": self.latest_session.isoformat() if self.latest_session else None,
        }


async def get_usage_stats() -> UsageStats:
    """Calculate aggregated usage statistics across all sessions.

    Returns:
        UsageStats with lifetime totals for sessions, prompts, and tokens.
    """
    sessions = await list_sessions()

    if not sessions:
        return UsageStats()

    stats = UsageStats(
        total_sessions=len(sessions),
        total_prompts=0,
        total_tokens=0,
        total_input_tokens=0,
        total_output_tokens=0,
        earliest_session=None,
        latest_session=None,
    )

    for session_info in sessions:
        # Accumulate prompt count (message_count includes all messages)
        stats.total_prompts += session_info.message_count

        # Accumulate token usage
        stats.total_tokens += session_info.total_tokens
        stats.total_input_tokens += getattr(session_info, 'total_input_tokens', 0)
        stats.total_output_tokens += getattr(session_info, 'total_output_tokens', 0)

        # Track per-session breakdown
        stats.sessions.append(SessionUsageEntry(
            session_id=session_info.id,
            title=session_info.title,
            input_tokens=getattr(session_info, 'total_input_tokens', 0),
            output_tokens=getattr(session_info, 'total_output_tokens', 0),
            total_tokens=session_info.total_tokens,
            message_count=session_info.message_count,
            created=session_info.time.created,
        ))

        # Track earliest and latest sessions
        created = session_info.time.created
        if stats.earliest_session is None or created < stats.earliest_session:
            stats.earliest_session = created
        if stats.latest_session is None or created > stats.latest_session:
            stats.latest_session = created

    # Sort sessions by most recent first
    stats.sessions.sort(key=lambda s: s.created or datetime.min, reverse=True)

    return stats


def format_tokens(count: int) -> str:
    """Format token count with K/M/B suffixes for readability."""
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    elif count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    else:
        return str(count)


def format_usage_summary(stats: UsageStats) -> str:
    """Format usage statistics as a human-readable summary."""
    lines = [
        f"  Total Sessions:  {stats.total_sessions}",
        f"  Total Prompts:   {stats.total_prompts}",
        f"  Total Tokens:    {format_tokens(stats.total_tokens)}",
    ]
    if stats.total_input_tokens or stats.total_output_tokens:
        lines.append(f"    Input Tokens:  {format_tokens(stats.total_input_tokens)}")
        lines.append(f"    Output Tokens: {format_tokens(stats.total_output_tokens)}")

    if stats.earliest_session:
        lines.append(f"  First Session:   {stats.earliest_session.strftime('%Y-%m-%d %H:%M')}")
    if stats.latest_session:
        lines.append(f"  Latest Session:  {stats.latest_session.strftime('%Y-%m-%d %H:%M')}")

    return "\n".join(lines)
