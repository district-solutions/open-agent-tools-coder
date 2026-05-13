"""AWS-aware helpers for the bash tool.

Three responsibilities, all thin:

1. Classify an `aws ...` command as READ / MUTATE / AUTH / UNKNOWN so the
   caller can decide how much approval it needs.
2. Redact AWS credentials that might leak into command output (access keys,
   session tokens, account IDs in ARNs if requested).
3. Detect commands that need a real TTY (like `aws sso login`) and tell the
   caller to ask the user to run them manually with `! <cmd>`.

This is deliberately not a new tool. AWS flows through `bash` — we just give
`bash` a sharper eye when it sees the `aws` binary.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from enum import Enum


class AwsRisk(str, Enum):
    READ = "read"         # describe / list / get / head — auto-approve safe
    MUTATE = "mutate"     # create / delete / put / update / terminate — confirm
    AUTH = "auth"         # login / logout / configure — may need TTY
    UNKNOWN = "unknown"   # not classified; treat as mutate


@dataclass
class AwsCommandInfo:
    is_aws: bool
    risk: AwsRisk
    service: str | None = None       # s3, ec2, iam, ...
    operation: str | None = None     # ls, cp, rm, describe-instances, ...
    needs_tty: bool = False          # true for sso login / interactive flows
    reason: str = ""                 # short human-readable explanation


_READ_VERBS = {
    "ls", "cat",
    "describe", "describe-instances", "describe-stacks", "describe-vpcs",
    "describe-log-streams", "describe-log-groups", "describe-table",
    "list", "list-buckets", "list-objects", "list-objects-v2", "list-functions",
    "list-stacks", "list-users", "list-roles", "list-policies",
    "get", "get-object", "get-caller-identity", "get-item", "get-function",
    "get-log-events", "get-bucket-location", "get-parameter",
    "head", "head-bucket", "head-object",
    "search", "query",
    "filter-log-events",
}

_MUTATE_VERBS = {
    "rm", "cp", "mv", "sync",
    "create", "delete", "put", "update", "modify",
    "terminate", "stop", "start", "reboot",
    "attach", "detach", "associate", "disassociate",
    "run-instances", "deploy",
    "put-object", "delete-object", "delete-objects",
    "put-item", "delete-item", "batch-write-item",
    "invoke",  # lambda invoke can be destructive
}

_AUTH_VERBS = {"login", "logout", "configure"}

_TTY_HINTS = (
    re.compile(r"\baws\s+sso\s+login\b"),
    re.compile(r"\baws\s+configure\s+sso\b"),
    re.compile(r"\baws\s+configure\b(?!\s+(list|get|import))"),
)

# Secret patterns that should be scrubbed from output
_ACCESS_KEY_RE = re.compile(r"\b(AKIA|ASIA|AIDA|AROA|AGPA|ANPA)[A-Z0-9]{16}\b")
_SECRET_KEY_RE = re.compile(
    r"(?i)(aws_?secret_?access_?key\s*[:=]\s*['\"]?)([A-Za-z0-9/+=]{40})(['\"]?)"
)
_SESSION_TOKEN_RE = re.compile(
    r"(?i)(aws_?session_?token\s*[:=]\s*['\"]?)([A-Za-z0-9/+=%]{100,})(['\"]?)"
)


def _first_aws_tokens(command: str) -> tuple[str, list[str]] | None:
    """Return (binary, args) if command invokes `aws`, else None.

    Uses shlex so we skip over env-var prefixes like `AWS_REGION=us-east-1 aws ...`.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if "=" in tok and not tok.startswith("-") and "/" not in tok:
            i += 1
            continue
        if tok.endswith("aws") or tok == "aws":
            return tok, tokens[i + 1 :]
        return None
    return None


def classify(command: str) -> AwsCommandInfo:
    """Classify an `aws ...` shell command."""
    if "aws" not in command:
        return AwsCommandInfo(is_aws=False, risk=AwsRisk.UNKNOWN)

    parsed = _first_aws_tokens(command)
    if parsed is None:
        return AwsCommandInfo(is_aws=False, risk=AwsRisk.UNKNOWN)

    _, args = parsed
    non_flag = [a for a in args if not a.startswith("-")]
    service = non_flag[0] if non_flag else None
    operation = non_flag[1] if len(non_flag) > 1 else None

    needs_tty = any(p.search(command) for p in _TTY_HINTS)

    if operation is None and service in _AUTH_VERBS:
        return AwsCommandInfo(
            is_aws=True, risk=AwsRisk.AUTH, service=service,
            operation=None, needs_tty=needs_tty,
            reason="auth subcommand — may require interactive prompt",
        )

    if operation in _AUTH_VERBS or service in _AUTH_VERBS:
        return AwsCommandInfo(
            is_aws=True, risk=AwsRisk.AUTH, service=service,
            operation=operation, needs_tty=needs_tty,
            reason="auth subcommand — may require interactive prompt",
        )

    if operation and (
        operation in _READ_VERBS
        or any(operation.startswith(v + "-") for v in ("describe", "list", "get", "head"))
    ):
        return AwsCommandInfo(
            is_aws=True, risk=AwsRisk.READ, service=service,
            operation=operation, needs_tty=needs_tty,
            reason="read-only verb",
        )

    if operation and (
        operation in _MUTATE_VERBS
        or any(operation.startswith(v + "-") for v in ("create", "delete", "put", "update", "modify", "terminate"))
    ):
        return AwsCommandInfo(
            is_aws=True, risk=AwsRisk.MUTATE, service=service,
            operation=operation, needs_tty=needs_tty,
            reason="state-changing verb",
        )

    return AwsCommandInfo(
        is_aws=True, risk=AwsRisk.UNKNOWN, service=service,
        operation=operation, needs_tty=needs_tty,
        reason="unrecognized verb — treat as mutate",
    )


def redact_secrets(text: str) -> tuple[str, int]:
    """Scrub AWS access keys, secret keys, and session tokens from text.

    Returns (redacted_text, num_redactions).
    """
    if not text:
        return text, 0

    count = 0

    def _ak_sub(m: re.Match) -> str:
        nonlocal count
        count += 1
        return f"{m.group(1)}****************"

    def _sk_sub(m: re.Match) -> str:
        nonlocal count
        count += 1
        return f"{m.group(1)}[redacted]{m.group(3)}"

    def _st_sub(m: re.Match) -> str:
        nonlocal count
        count += 1
        return f"{m.group(1)}[redacted]{m.group(3)}"

    text = _ACCESS_KEY_RE.sub(_ak_sub, text)
    text = _SECRET_KEY_RE.sub(_sk_sub, text)
    text = _SESSION_TOKEN_RE.sub(_st_sub, text)
    return text, count
