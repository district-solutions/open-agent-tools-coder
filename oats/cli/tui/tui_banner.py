"""Terminal UI banner and session information display utilities.

Handles printing the startup banner (with git repo detection, model
and provider info), session details, token cost summaries, and file
access logs for the current session.
"""

import os
from rich.console import Console
from oats.cli.tui.tui_consts import SYM_SEP
from oats.session.session import Session
from oats.session.processor import SessionProcessor

def _detect_git_info(cwd: str) -> dict:
    """Detect git repo info for the current directory."""
    info = {"is_git": False, "branch": None, "repo_name": None, "dirty": False}
    from pathlib import Path
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, cwd=cwd, timeout=5,
        )
        if result.returncode != 0:
            return info
        info["is_git"] = True

        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=cwd, timeout=5,
        )
        if branch.returncode == 0:
            info["branch"] = branch.stdout.strip()

        repo_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, cwd=cwd, timeout=5,
        )
        if repo_root.returncode == 0:
            info["repo_name"] = Path(repo_root.stdout.strip()).name

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=cwd, timeout=5,
        )
        if status.returncode == 0:
            info["dirty"] = bool(status.stdout.strip())
    except Exception:
        pass
    return info

def _short_model(model_id: str) -> str:
    """Shorten model ID for display: hosted_vllm/Qwen3.5-27B-AWQ-4bit -> Qwen3.5-27B-AWQ-4bit"""
    if "/" in model_id:
        return model_id.rsplit("/", 1)[-1]
    return model_id


def _print_banner(console: Console, cwd: str, session_id: str, provider_id: str, model_id: str):
    """Print the startup banner."""
    import os
    git = _detect_git_info(cwd)
    short = _short_model(model_id)
    console.print()
    version = os.getenv('CODER_VERSION', '1.2.0')
    console.print(f"  [bold white]coder[/bold white] [dim]v{version}[/dim]  [dim]·[/dim]  [cyan]{short}[/cyan]  [dim]·[/dim]  [dim]{provider_id}[/dim]")
    parts = [f"  [dim]{cwd}[/dim]"]
    if git["is_git"]:
        dirty = " [yellow]*[/yellow]" if git["dirty"] else ""
        parts.append(f"  [dim]git:[/dim] {git['branch']}{dirty} [dim]({git['repo_name']})[/dim]")
    # parts.append(f"  [dim]session:[/dim] {session_id[:8]}")
    console.print("\n".join(parts))
    console.print(f"  [dim]{SYM_SEP * 50}[/dim]")
    console.print(f"  [dim]Enter to send · Alt+Enter for newline · /help for commands[/dim]")
    console.print()

def _print_session_info(console: Console, session: Session, turn_count: int, provider_id: str, model_id: str):
    """Print session details."""
    from oats.cli.tui.tui_consts import _format_tokens
    console.print()
    console.print(f"  [dim]session[/dim]   {session.id[:12]}")
    console.print(f"  [dim]title[/dim]     {session.title}")
    console.print(f"  [dim]messages[/dim]  {len(session.messages)}")
    console.print(f"  [dim]turns[/dim]     {turn_count}")
    console.print(f"  [dim]tokens[/dim]    {_format_tokens(session.info.total_tokens)}")
    console.print(f"  [dim]model[/dim]     {_short_model(model_id)} [dim]({provider_id})[/dim]")
    console.print()

def _print_cost(console: Console, session: Session):
    """Print session token usage."""
    from oats.cli.tui.tui_consts import _format_tokens
    total = session.info.total_tokens
    ctx = int(os.getenv('CODER_CTX_LEN', '262100'))
    pct = (total / ctx * 100) if ctx > 0 else 0

    console.print()
    console.print(f"  [dim]total tokens[/dim]  {_format_tokens(total)}")
    console.print(f"  [dim]context window[/dim] {_format_tokens(ctx)}")
    console.print(f"  [dim]utilization[/dim]   {pct:.0f}%")
    console.print()

def _print_files(console: Console, processor: SessionProcessor):
    """Print files read/written this session."""
    cache = processor._file_cache
    read_files = cache.get_read_files()
    written_files = cache.get_written_files()
    from oats.cli.tui.tui_consts import SYM_TOOL

    console.print()
    if written_files:
        console.print(f"  [bold cyan]written ({len(written_files)})[/bold cyan]")
        for f in sorted(written_files):
            console.print(f"  [yellow]  {SYM_TOOL}[/yellow] {f}")
    if read_files:
        console.print(f"  [bold cyan]read ({len(read_files)})[/bold cyan]")
        for f in sorted(read_files):
            console.print(f"  [dim]  · {f}[/dim]")
    if not read_files and not written_files:
        console.print(f"  [dim]no files accessed yet[/dim]")
    console.print()

