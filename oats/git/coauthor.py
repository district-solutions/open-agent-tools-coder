"""
Git co-author enforcement for coder.

Ensures all git commits made by coder include a Co-Authored-By trailer
to comply with AI-generated code attribution requirements (Linux Foundation stance).
"""
from __future__ import annotations

import re

CO_AUTHOR_LINE = os.getenv('CO_AUTHOR_COMMITS', "Co-Authored-By: coder <hello@districtsolutions.ai>"


def ensure_coauthor_trailer(command: str) -> str:
    """
    If *command* is a ``git commit`` invocation, ensure the commit message
    includes the coder co-author trailer.

    Returns the (possibly modified) command string.
    """
    if not _is_git_commit(command):
        return command

    # Already has our trailer?
    if CO_AUTHOR_LINE in command or "Co-Authored-By: coder" in command:
        return command

    # Handle -m "message" or -m 'message' style commits
    m_flag_pattern = re.compile(
        r'(-m\s+)(["\'])(.*?)(\2)',
        re.DOTALL,
    )
    match = m_flag_pattern.search(command)
    if match:
        prefix = match.group(1)
        quote = match.group(2)
        msg = match.group(3)
        # Append trailer with blank line separator (git convention)
        if "\n\n" in msg:
            new_msg = f"{msg}\n{CO_AUTHOR_LINE}"
        else:
            new_msg = f"{msg}\n\n{CO_AUTHOR_LINE}"
        replacement = f"{prefix}{quote}{new_msg}{quote}"
        return command[: match.start()] + replacement + command[match.end() :]

    # Handle heredoc style: -m "$(cat <<'EOF'\n...\nEOF\n)"
    heredoc_pattern = re.compile(r"(<<\s*['\"]?EOF['\"]?.*?)(EOF)", re.DOTALL)
    h_match = heredoc_pattern.search(command)
    if h_match:
        before_eof = h_match.group(1)
        if CO_AUTHOR_LINE not in before_eof:
            return (
                command[: h_match.start()]
                + before_eof.rstrip()
                + f"\n\n{CO_AUTHOR_LINE}\n"
                + h_match.group(2)
                + command[h_match.end() :]
            )

    # Fallback: append --trailer flag (works with git >= 2.32)
    if "--trailer" not in command:
        return command.rstrip() + f' --trailer "{CO_AUTHOR_LINE}"'

    return command


def _is_git_commit(command: str) -> bool:
    """Check if a shell command is a git commit invocation."""
    for segment in re.split(r"[;&|]+", command):
        segment = segment.strip()
        if re.match(r"^git\s+commit\b", segment):
            return True
        if "git commit" in segment and "git" in segment:
            return True
    return False
