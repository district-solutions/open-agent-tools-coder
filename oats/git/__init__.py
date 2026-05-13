"""
Git utilities for the coder agent.

This package provides tools for managing Git worktrees, searching commit
history, extracting diffs, and converting repositories to structured data formats.
"""
from oats.git.worktree import WorktreeManager

__all__ = ["WorktreeManager"]
