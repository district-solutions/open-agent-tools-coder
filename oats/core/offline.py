"""Offline-mode helpers.

Coder2's core turn loop (provider → tools → session) does not phone home.
All external traffic is either:

  1. The LLM provider endpoint — user-configured; point at a local vLLM on
     localhost for a fully offline setup.
  2. Explicit user-initiated actions via plugins (corpus fetchers,
     playwright, mattermost relays). These are surfaced in /help and the
     user has to invoke them directly.

When ``CODER_OFFLINE_STRICT=1`` is set, anything that would reach outside
``localhost``/``127.0.0.1``/``.local``/``.internal`` has to consult
:func:`require_network` first. The flag is advisory (we don't patch the
kernel namespace) — it gives plugin authors a single hook to check before
opening a socket, and a single audit point users can grep for.

Typical usage inside a tool implementation::

    from oats.core.offline import require_network
    require_network(label="corpus:github", url=url)  # raises if blocked
"""
from __future__ import annotations

import os
from urllib.parse import urlparse
from oats.log import cl

log = cl(__name__)


class OfflineBlockedError(RuntimeError):
    """Raised when strict offline mode refuses an external request."""


def _is_truthy(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def offline_strict() -> bool:
    """True if ``CODER_OFFLINE_STRICT=1``. Re-read per call so users can
    toggle it without restarting the session."""
    return _is_truthy("CODER_OFFLINE_STRICT", False)


_LOCAL_HOST_SUFFIXES = (".local", ".internal", ".lan")
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def is_local_url(url: str) -> bool:
    """Best-effort classification: does ``url`` target this machine or LAN?"""
    if not url:
        return True
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
    except Exception:
        return False
    if not host:
        return True  # no host means unix socket / relative path
    if host in _LOCAL_HOSTS:
        return True
    if host.endswith(_LOCAL_HOST_SUFFIXES):
        return True
    # Simple private-IPv4 classification; don't want to pull in ipaddress for
    # one cheap check, but still catch the obvious ranges.
    if host.startswith(("10.", "192.168.", "172.16.", "172.17.", "172.18.",
                        "172.19.", "172.2", "172.30.", "172.31.")):
        return True
    return False


def require_network(*, label: str, url: str = "") -> None:
    """Gate an outbound network operation on strict-offline mode.

    Raises :class:`OfflineBlockedError` when ``CODER_OFFLINE_STRICT=1`` AND
    the target (``url``) is not local. Local targets (``localhost``,
    private-range IPs, ``.local``) always pass. Plugins invoking explicit
    user actions can skip this — the point of strict mode is to gate
    *implicit* or accidental egress.
    """
    if not offline_strict():
        return
    if not url or is_local_url(url):
        return
    log.warning(f"offline_strict_blocked label={label} url={url}")
    raise OfflineBlockedError(
        f"CODER_OFFLINE_STRICT=1 refuses external network call "
        f"({label} → {url}). Unset or use a local endpoint."
    )
