#!/usr/bin/env python3
"""Get tool choices for a user prompt from the OAT index.

Tries exact-match lookup first, then falls back to BM25 ranking.
"""

import argparse

from oats.get_oat_config import get_oat_config
from oats.models import OatPromptChoices
from oats.log import gl

log = gl('oat.get_choices')

def agent_get_tool_choices_for_prompt(prompt: str, top_k: int = 5, verbose: bool = False) -> OatPromptChoices:
    """Get the best tool choices for a prompt, trying exact match then BM25."""
    oat_config = get_oat_config()
    # multi-tool resolution depending on need/use case
    choices = oat_config.get_prompt_choices(prompt=prompt, verbose=verbose)
    if not choices.status:
        choices = oat_config.get_best_matches_bm25(prompt=prompt, top_k=top_k, verbose=verbose)
    return choices

def parse_args(args=None) -> argparse.Namespace:
    """Parse CLI arguments for the tool-choices CLI."""
    parser = argparse.ArgumentParser(description='Get prompt choices from the OAT index.')
    parser.add_argument('-p', '--prompt', type=str, required=True, help='The prompt text to extract choices for.')
    parser.add_argument('-t', '--top-k', type=int, default=5, help='Number of top results to return when using BM25 (default: 5).')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging which will break jq command line piping for debugging')
    return parser.parse_args(args)

def main(args=None):
    """CLI entry point: look up tool choices and print JSON."""
    args = parse_args(args)
    result = agent_get_tool_choices_for_prompt(args.prompt, top_k=args.top_k, verbose=args.verbose)
    print(result.model_dump_json(indent=2))

if __name__ == '__main__':
    main()
