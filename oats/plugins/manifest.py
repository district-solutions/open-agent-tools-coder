"""
Plugin manifest schema and discovery.

A plugin lives in a directory containing ``coder.plugin.json``::

    my-plugin/
      coder.plugin.json
      __init__.py           # optional, if entrypoint is a package
      plugin.py             # conventional entrypoint — must export activate(ctx)

Manifest example::

    {
      "id": "my-plugin",
      "name": "My Plugin",
      "version": "0.1.0",
      "description": "Adds a `lint` tool and a pre_tool_use logger",
      "entrypoint": "plugin",
      "enabled_by_default": true,
      "on_features": ["planning"],
      "model_support": ["*"],
      "provides": {
        "toolsets": ["lint"]
      }
    }

Discovery walks two roots by default: ``<data_dir>/plugins`` (user-installed)
and ``<project_dir>/.coder/plugins`` (project-local). Additional roots can
be passed in for testing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from oats.core.config import get_data_dir
from oats.log import cl

log = cl("oats.plugins.manifest")

MANIFEST_FILENAME = "coder.plugin.json"


class PluginProvides(BaseModel):
    """What the plugin is *declared* to register.

    Purely descriptive — used for UI and activation filtering, not enforced
    against what ``activate()`` actually does. Keeps the manifest honest
    without coupling the loader to runtime side effects.
    """
    toolsets: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    hooks: list[str] = Field(default_factory=list)
    slash_commands: list[str] = Field(default_factory=list)


class PluginManifest(BaseModel):
    """Declarative plugin descriptor.

    Kept intentionally small. The loader reads the manifest first (cheap
    JSON parse) so it can filter by feature flags or model before paying
    the cost of importing the plugin's Python module.
    """
    id: str
    name: str = ""
    version: str = "0.0.0"
    description: str = ""

    # Import path *relative to the plugin directory*. "plugin" means the
    # loader will import ``<plugin_dir>/plugin.py`` as a module and call
    # its ``activate(ctx)``.
    entrypoint: str = "plugin"

    enabled_by_default: bool = True

    # Activation gates. If any gate is specified and not satisfied, the
    # plugin is skipped without ever importing its Python.
    on_features: list[str] = Field(default_factory=list)  # requires is_feature_enabled(X) for all
    model_support: list[str] = Field(default_factory=lambda: ["*"])

    provides: PluginProvides = Field(default_factory=PluginProvides)

    # Filled in by the loader — not authored by the plugin itself.
    source_dir: Path | None = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("id")
    @classmethod
    def _id_is_slug(cls, v: str) -> str:
        """Validate that the plugin id is a clean slug with no whitespace or path separators."""
        if not v or any(c in v for c in " \t\n/\\"):
            raise ValueError(f"plugin id must be a slug, got {v!r}")
        return v

    def matches_model(self, model_id: str | None) -> bool:
        """True if ``model_support`` contains ``"*"`` or a prefix of the model id."""
        if "*" in self.model_support:
            return True
        if not model_id:
            return False
        return any(model_id.startswith(m) for m in self.model_support)


def _load_one(manifest_path: Path) -> PluginManifest | None:
    """Load and validate a single plugin manifest from disk.

    Reads the JSON file, validates it against the :class:`PluginManifest`
    schema, and sets ``source_dir`` on the returned manifest. Returns
    ``None`` if the file is unreadable or invalid.

    Args:
        manifest_path: Path to the ``coder.plugin.json`` file.

    Returns:
        A validated :class:`PluginManifest`, or ``None`` on error.
    """
    try:
        raw = json.loads(manifest_path.read_text())
    except Exception as e:
        log.warn(f"plugin_manifest_unreadable path={manifest_path} err={e}")
        return None
    try:
        m = PluginManifest.model_validate(raw)
    except Exception as e:
        log.warn(f"plugin_manifest_invalid path={manifest_path} err={e}")
        return None
    m.source_dir = manifest_path.parent
    return m


def discover_manifests(roots: list[Path] | None = None) -> list[PluginManifest]:
    """Walk the plugin roots and return one validated manifest per plugin dir.

    Default roots (skipped if they don't exist):
      - ``<data_dir>/plugins``
      - ``<cwd>/.coder/plugins``

    Does not import plugin Python. Safe to call multiple times; idempotent.
    Duplicate ids are resolved by first-wins (stable sort on root order).
    """
    if roots is None:
        roots = _default_roots()

    out: list[PluginManifest] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for mf in sorted(root.glob(f"*/{MANIFEST_FILENAME}")):
            m = _load_one(mf)
            if m is None:
                continue
            if m.id in seen:
                log.warn(f"plugin_id_conflict id={m.id} skipped_path={mf}")
                continue
            seen.add(m.id)
            out.append(m)
    return out


def _default_roots() -> list[Path]:
    """Return the default plugin discovery roots.

    Order matters — the first occurrence of a given plugin id wins:

    1. ``<data_dir>/plugins`` — user-installed plugins, highest priority so
       a user can shadow a builtin with a local copy.
    2. ``<cwd>/.coder/plugins`` — project-local plugins.
    3. ``<pkg_dir>/plugins/builtin`` — plugins shipped with coder2 itself.
    """
    roots: list[Path] = []
    try:
        roots.append(get_data_dir() / "plugins")
    except Exception:
        pass
    roots.append(Path.cwd() / ".coder" / "plugins")
    # Builtin plugins ship alongside the package so the sample round-trip
    # works out-of-the-box when the feature flag is on.
    roots.append(Path(__file__).parent / "builtin")
    return roots
