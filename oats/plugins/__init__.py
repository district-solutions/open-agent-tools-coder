"""
Plugin system for oats.

Declarative ``oats.plugin.json`` manifests let third parties (and us) add
tools, toolsets, and hook handlers without editing core. Matches the "one
taxonomy" rule from the improvement doc — plugins use the *existing*
``ToolRegistry``, ``HookEngine``, and ``FeatureProfile`` vocabulary instead
of introducing parallel concepts.

See :mod:`oats.plugins.manifest` for the manifest schema and
:mod:`oats.plugins.loader` for discovery + activation.
"""
from __future__ import annotations

from oats.plugins.manifest import PluginManifest, discover_manifests
from oats.plugins.loader import PluginContext, install, load_all

__all__ = [
    "PluginContext",
    "PluginManifest",
    "discover_manifests",
    "install",
    "load_all",
]
