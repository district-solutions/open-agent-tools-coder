"""
ID generation utilities using uuid
"""
from __future__ import annotations


from uuid import uuid4


def generate_id() -> str:
    """Generate a new unique ID."""
    return str(uuid4())


def generate_short_id() -> str:
    """Generate a shorter ID (first 8 chars)."""
    return str(generate_id())[:8]
