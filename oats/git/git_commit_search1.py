#!/usr/bin/env python3

"""
git_commit_search.py — Search a git repository's commit history for
case-insensitive phrases, or replay a predefined set of git actions
loaded from a JSON file.
REPLACE_USER in repo: REPLACE_REPO_PATH search term: REPLACE_GIT_SEARCH

Usage:
    # Phrase search (builds -G actions automatically)
    python git_commit_search.py -r REPLACE_REPO_PATH -p "REPLACE_GIT_SEARCH"

    # Replay a saved action plan
    python git_commit_search.py -r REPLACE_REPO_PATH -f git_actions.json

    # Both: run file actions first, then phrase search
    python git_commit_search.py -r . -f git_actions.json -p "REPLACE_GIT_SEARCH" -a

    # Verbose output
    python git_commit_search.py -f git_actions.json -v
"""

import argparse
import json
import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field
from oats.log import cl

try:
    from git import InvalidGitRepositoryError, NoSuchPathError, Repo
except ImportError:
    print(
        "gitpython is not installed — run: pip install gitpython",
        file=sys.stderr,
    )
    sys.exit(1)

log = cl('git_search')

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    LOG = "log"
    SHOW = "show"


class GitCommitAction(BaseModel):
    """
    Represents a single git command — either ``git log`` or ``git show``.

    Covers every flag used during the feed-detail-fix1 UUID investigation:

        git log --author=<email> --oneline [-n N] [--skip N]
        git log --all --oneline [--follow] -S <pickaxe> [-- <path>]
        git log --all --oneline [--regexp-ignore-case] -G <phrase> [-- <path>]
        git show <hash> [--stat] [-- <path>]
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    action: ActionType = Field(..., description="Git sub-command: 'log' or 'show'")
    description: Optional[str] = Field(None, description="Human-readable purpose of this action")

    # --- git log options ----------------------------------------------------
    author: Optional[str] = Field(
        None, description="Filter commits by author e-mail or name (--author)"
    )
    all_branches: bool = Field(
        False, description="Include all branches and tags (--all)"
    )
    follow: bool = Field(
        False, description="Follow file renames across history (--follow, requires path_filter)"
    )
    oneline: bool = Field(
        True, description="Condensed single-line output (--oneline)"
    )
    limit: Optional[int] = Field(
        None, description="Maximum number of commits to return (-n)"
    )
    skip: Optional[int] = Field(
        None, description="Skip the first N commits before output (--skip)"
    )
    pickaxe: Optional[str] = Field(
        None,
        description=(
            "Case-sensitive string pickaxe search (-S): finds commits where "
            "the number of occurrences of the string changed"
        ),
    )
    grep_phrase: Optional[str] = Field(
        None,
        description=(
            "Case-insensitive regex search (-G --regexp-ignore-case): "
            "built automatically when phrases are supplied via -p on the CLI"
        ),
    )
    path_filter: Optional[str] = Field(
        None, description="Restrict the log/show to a specific file path (-- <path>)"
    )

    # --- git show options ---------------------------------------------------
    commit_hash: Optional[str] = Field(
        None, description="Commit hash (or short hash) to inspect — used with action='show'"
    )
    stat: bool = Field(
        False, description="Emit diffstat summary instead of full patch (--stat)"
    )


class GitSearchPlan(BaseModel):
    """Top-level wrapper for a git_actions JSON file."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    name: Optional[str] = Field(None, description="Human-readable name for this search plan")
    repo: Optional[str] = Field(".", description="Path to the git repository")
    actions: Optional[list[GitCommitAction]] = Field(
        default_factory=list, description="Ordered list of git actions to execute"
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Parse CLI arguments and return the populated Namespace.

    Short flags:
        -r  path to the git repository (default: current directory)
        -f  path to a git_actions JSON file
        -p  one or more case-insensitive phrases to search for
        -a  search all branches (--all) when using -p phrases
        -n  maximum commits to return per phrase search (default: 50)
        -A  filter by author e-mail or name when using -p phrases
        -P  restrict phrase search to a specific file path
        -v  enable DEBUG-level logging
    """
    log.info("parse_args: start")

    parser = argparse.ArgumentParser(
        prog="git_commit_search",
        description=(
            "Search git history for case-insensitive phrases, or replay "
            "a predefined set of git actions from a JSON file."
        ),
    )
    parser.add_argument(
        "-r", "--repo",
        default=".",
        metavar="REPO",
        help="Path to the git repository (default: current directory)",
    )
    parser.add_argument(
        "-f", "--file",
        default=None,
        metavar="FILE",
        help="Path to a git_actions JSON file to replay",
    )
    parser.add_argument(
        "-p", "--phrases",
        nargs="+",
        default=[],
        metavar="PHRASE",
        help="One or more case-insensitive phrases to search for in commit history",
    )
    parser.add_argument(
        "-a", "--all-branches",
        action="store_true",
        help="Search all branches when running phrase searches via -p",
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Maximum commits returned per phrase search action (default: 50)",
    )
    parser.add_argument(
        "-A", "--author",
        default=None,
        metavar="AUTHOR",
        help="Filter phrase-search results to a specific author e-mail or name",
    )
    parser.add_argument(
        "-P", "--path",
        default=None,
        metavar="PATH",
        help="Restrict phrase searches to a specific file path",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )

    ns = parser.parse_args(argv)

    if ns.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        log.debug("parse_args: DEBUG logging enabled")

    if not ns.file and not ns.phrases:
        parser.error("supply at least one of -f FILE or -p PHRASE [PHRASE ...]")

    log.info(f"parse_args: end  →  {ns}")
    return ns


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def load_search_plan(file_path: str) -> GitSearchPlan:
    """
    Read and validate a search plan from *file_path*.

    Accepts either:
    * a bare JSON array    ``[ {...}, {...} ]``
    * a wrapped object     ``{ "name": "...", "repo": ".", "actions": [ ... ] }``
    """
    log.info(f"load_search_plan: start  path={file_path}")

    raw = Path(file_path).read_text(encoding="utf-8")
    data = json.loads(raw)

    if isinstance(data, list):
        plan = GitSearchPlan(actions=[GitCommitAction(**item) for item in data])
    else:
        plan = GitSearchPlan(**data)

    count = len(plan.actions or [])
    log.info(f"load_search_plan: end  →  {count} action(s) loaded from {file_path}")
    return plan


def build_phrase_actions(
    phrases: list[str],
    *,
    all_branches: bool = False,
    limit: Optional[int] = 50,
    author: Optional[str] = None,
    path_filter: Optional[str] = None,
) -> list[GitCommitAction]:
    """
    Convert a list of plain-text phrases into ``GitCommitAction`` objects
    using case-insensitive ``-G`` grep searches.

    Args:
        phrases:      List of text strings to search for.
        all_branches: If ``True``, pass ``--all`` to git log.
        limit:        Cap on commits returned per phrase.
        author:       Optional author filter.
        path_filter:  Optional file path to restrict search to.

    Returns:
        One :class:`GitCommitAction` per phrase.
    """
    log.info(f"build_phrase_actions: start  phrases={len(phrases)}  all_branches={all_branches}  limit={limit}")

    actions = [
        GitCommitAction(
            action=ActionType.LOG,
            description=f"Case-insensitive phrase search: {phrase!r}",
            grep_phrase=phrase,
            all_branches=all_branches,
            oneline=True,
            limit=limit,
            author=author,
            path_filter=path_filter,
        )
        for phrase in phrases
    ]

    log.info(f"build_phrase_actions: end  →  {len(actions)} action(s) created")
    return actions


# ---------------------------------------------------------------------------
# Git command builders
# ---------------------------------------------------------------------------


def _build_log_parts(action: GitCommitAction) -> list[str]:
    """
    Translate a ``GitCommitAction`` with ``action='log'`` into a list of
    git-log flag strings suitable for ``repo.git.log(*parts)``.

    Supports every flag represented in the feed-detail-fix1 investigation:
        --oneline, --all, --follow, --author, -n, --skip,
        -S (pickaxe), -G --regexp-ignore-case, -- <path>
    """
    log.info(f"_build_log_parts: start  desc={action.description or '(none)'}")

    parts: list[str] = []

    if action.oneline:
        parts.append("--oneline")
    if action.all_branches:
        parts.append("--all")
    if action.follow:
        parts.append("--follow")
    if action.author:
        parts.append(f"--author={action.author}")
    if action.limit is not None:
        parts.extend(["-n", str(action.limit)])
    if action.skip is not None:
        parts.append(f"--skip={action.skip}")
    if action.pickaxe:
        parts.extend(["-S", action.pickaxe])
    if action.grep_phrase:
        parts.append("--regexp-ignore-case")
        parts.extend(["-G", action.grep_phrase])
    if action.path_filter:
        parts.extend(["--", action.path_filter])

    log.info(f"_build_log_parts: end  →  git log {' '.join(parts)}")
    return parts


def _build_show_parts(action: GitCommitAction) -> list[str]:
    """
    Translate a ``GitCommitAction`` with ``action='show'`` into a list of
    argument strings suitable for ``repo.git.show(*parts)``.

    Supports: ``--stat`` and ``-- <path>`` filters.
    """
    log.info(f"_build_show_parts: start  hash={action.commit_hash}  stat={action.stat}  path={action.path_filter}")

    if not action.commit_hash:
        raise ValueError("GitCommitAction with action='show' requires commit_hash")

    parts: list[str] = [action.commit_hash]

    if action.stat:
        parts.append("--stat")
    if action.path_filter:
        parts.extend(["--", action.path_filter])

    log.info(f"_build_show_parts: end  →  git show {' '.join(parts)}")
    return parts


# ---------------------------------------------------------------------------
# Per-action executor
# ---------------------------------------------------------------------------


def execute_action(repo: "Repo", action: GitCommitAction) -> dict[str, Any]:
    """
    Execute a single :class:`GitCommitAction` against *repo* and return a
    result dict with keys:

    * ``action``       – the action type string
    * ``description``  – human-readable label (may be None)
    * ``command``      – the git command line that was run
    * ``output``       – raw text output from git
    * ``lines``        – output split into non-empty lines
    * ``match_count``  – number of output lines (useful for log searches)
    * ``error``        – error message string if the command failed, else None
    """
    log.info(f"execute_action: start  action={action.action}  desc={action.description or '(none)'}")

    result: dict[str, Any] = {
        "action": action.action,
        "description": action.description,
        "command": "",
        "output": "",
        "lines": [],
        "match_count": 0,
        "error": None,
    }

    try:
        if action.action == ActionType.LOG:
            parts = _build_log_parts(action)
            result["command"] = "git log " + " ".join(parts)
            log.info(f"execute_action: running  {result['command']}")
            output = repo.git.log(*parts)

        elif action.action == ActionType.SHOW:
            parts = _build_show_parts(action)
            result["command"] = "git show " + " ".join(parts)
            log.info(f"execute_action: running  {result['command']}")
            output = repo.git.show(*parts)

        else:
            raise ValueError(f"Unsupported action type: {action.action!r}")

        result["output"] = output
        result["lines"] = [ln for ln in output.splitlines() if ln.strip()]
        result["match_count"] = len(result["lines"])

    except Exception as exc:
        log.warning(f"execute_action: git command failed  error={exc}")
        result["error"] = str(exc)

    log.info(f"execute_action: end  action={action.action}  lines={result['match_count']}  error={result['error'] or 'none'}")
    return result


# ---------------------------------------------------------------------------
# Core search runner
# ---------------------------------------------------------------------------


def run_git_search(
    plan: GitSearchPlan,
    *,
    repo_path: str = ".",
) -> list[dict[str, Any]]:
    """
    Open the git repository at *repo_path*, execute every action in *plan*
    in order, and return the collected results.

    Args:
        plan:      The loaded and validated :class:`GitSearchPlan`.
        repo_path: Filesystem path to the git repo root.  The plan's own
                   ``repo`` field is used as the default; the caller's
                   ``repo_path`` argument takes precedence when supplied.

    Returns:
        A list of result dicts, one per action (see :func:`execute_action`).
    """
    effective_path = repo_path if repo_path != "." else (plan.repo or ".")
    log.info(f"run_git_search: start  repo={effective_path}  actions={len(plan.actions or [])}")

    try:
        repo = Repo(effective_path, search_parent_directories=True)
    except (InvalidGitRepositoryError, NoSuchPathError) as exc:
        log.error(f"run_git_search: failed to open repo at {effective_path!r}  error={exc}")
        raise

    branch = repo.active_branch.name if not repo.head.is_detached else "(detached)"
    log.info(f"run_git_search: repo opened  active_branch={branch}  head={repo.head.commit.hexsha[:8]}")

    results: list[dict[str, Any]] = []
    total = len(plan.actions or [])

    for idx, action in enumerate(plan.actions or [], start=1):
        log.info(f"=== action {idx} / {total}  [{action.action}]  {action.description or ''} ===")
        result = execute_action(repo, action)
        results.append(result)
        _print_result(idx, total, result)

    log.info(f"run_git_search: end  total_actions={total}")
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_result(idx: int, total: int, result: dict[str, Any]) -> None:
    """Pretty-print a single action result to stdout."""
    log.info(f"_print_result: start  idx={idx}")

    separator = "─" * 72
    print(f"\n{separator}")
    print(f"[{idx}/{total}]  {result['action'].upper()}  —  {result['description'] or '(no description)'}")
    print(f"  cmd: {result['command']}")
    print(separator)

    if result["error"]:
        print(f"  ERROR: {result['error']}")
    elif result["output"]:
        for line in result["lines"]:
            print(f"  {line}")
        print(f"\n  ({result['match_count']} line(s))")
    else:
        print("  (no output)")

    log.info(f"_print_result: end  idx={idx}  lines={result['match_count']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """Program entry point — parse args, build plan, run search."""
    log.info("main: start")

    args = parse_args(argv)

    actions: list[GitCommitAction] = []

    # 1. Load predefined actions from JSON file when -f is supplied
    if args.file:
        plan = load_search_plan(args.file)
        # CLI -r overrides the repo path in the file
        if args.repo != ".":
            plan.repo = args.repo
        actions.extend(plan.actions or [])
    else:
        plan = GitSearchPlan(repo=args.repo)

    # 2. Append auto-generated actions for any phrases supplied via -p
    if args.phrases:
        phrase_actions = build_phrase_actions(
            args.phrases,
            all_branches=args.all_branches,
            limit=args.limit,
            author=args.author,
            path_filter=args.path,
        )
        actions.extend(phrase_actions)

    plan.actions = actions

    results = run_git_search(plan, repo_path=args.repo)

    # Summary
    total = len(results)
    errors = sum(1 for r in results if r["error"])
    hits = sum(r["match_count"] for r in results if not r["error"])
    print(f"\n{'═' * 72}")
    print(f"  Summary: {total} action(s)  |  {hits} total line(s) returned  |  {errors} error(s)")
    print(f"{'═' * 72}\n")

    log.info(f"main: end  total={total}  hits={hits}  errors={errors}")


if __name__ == "__main__":
    main()
