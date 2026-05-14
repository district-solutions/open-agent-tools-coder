#!/usr/bin/env python3
"""CLI utility to list and check the configuration status of all available providers.

Prints each provider's name, ID, and whether it is currently configured
(has valid credentials and endpoint settings).
"""

import os
from oats.log import cl

log = cl('provider.check')

def check_providers() -> None:
    """List available providers."""
    from rich.console import Console
    from oats.provider.provider import list_providers
    console = Console()
    providers_list = list_providers()
    for provider in providers_list:
        status = "[green]configured[/green]" if provider.is_configured() else "[red]not configured[/red]"
        console.print(f"{provider.name} ({provider.id}): {status}")

if __name__ == '__main__':
    try:
        check_providers()
    except Exception:
        import traceback
        from pathlib import Path
        config_path_home = f'{Path.home()}/.local/share/oats/coder.json'
        log.info(f"### Sorry!! CODER_CONFIG_FILE misconfigured:\n```\n{os.getenv('CODER_CONFIG_FILE', 'Setup cp ' + config_path_home + ' file for you env with default: ./oats/config/coder.json')}\n```\nPlease confirm the settings are correct with error:\n```\n{traceback.format_exc()}\n```\n")
