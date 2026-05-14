#!/usr/bin/env python3
"""
Interactive CLI to generate a coder.json config file.

Prompts the user for vllm-small and t1 provider settings, then writes
a new config file (default: /tmp/coder.json).

Usage:
    python setup_coder_config.py -o /path/to/output.json
    python setup_coder_config.py  # writes to /tmp/coder.json by default
"""

import argparse
import json
import sys

from oats.log import gl
log = gl('setup_coder_config')


DEFAULT_OUTPUT = "/tmp/coder.json"

# --- vllm-small defaults ---
VLLM_SMALL_DEFAULT_URL = "http://0.0.0.0:20773/v1"
VLLM_SMALL_DEFAULT_API_KEY = "CHANGE_PASSWORD"
VLLM_SMALL_DEFAULT_MODEL = "hosted_vllm/chat:latest"
VLLM_SMALL_DEFAULT_MAX_TOKENS = 262100

# --- t1 defaults ---
T1_DEFAULT_URL = "http://0.0.0.0:20700/v1"
T1_DEFAULT_API_KEY = "CHANGE_PASSWORD"
T1_DEFAULT_MODEL = "openai/google/functiongemma-270m-it"


def prompt(msg: str, default: str | None = None) -> str:
    """Prompt the user with an optional default value."""
    if default is not None:
        display = f"❓ {msg} [{default}]: "
    else:
        display = f"{msg}: "
    answer = input(display).strip()
    return answer if answer else (default or "")


def ensure_v1_suffix(url: str) -> str:
    """Append /v1 to the base_url if it's missing."""
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def collect_vllm_small() -> dict:
    """Collect vllm-small provider configuration from the user."""
    log.info(f'\n# vllm-small (chat model)\n')

    base_url = prompt(
        "Where is the vllm-small instance running?\n  ",
        VLLM_SMALL_DEFAULT_URL,
    )
    base_url = ensure_v1_suffix(base_url)

    api_key = prompt("API key for vllm-small\n  ", VLLM_SMALL_DEFAULT_API_KEY)

    model_name = prompt(
        "Model ID name for vllm-small\n  ",
        VLLM_SMALL_DEFAULT_MODEL,
    )

    max_tokens = prompt(
        "maxTokens for this model on vllm-small\n  ",
        str(VLLM_SMALL_DEFAULT_MAX_TOKENS),
    )
    max_tokens = int(max_tokens)

    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "vllm-small",
        "base_url": base_url,
        "api_key": api_key,
        "models": [
            {
                "name": model_name,
                "maxTokens": max_tokens,
            }
        ],
    }


def collect_t1() -> dict:
    """Collect t1 provider (tool-calling) configuration from the user."""
    log.info(f'\n# t1 (tool-calling model)\n')

    base_url = prompt(
        "Where is the t1 vllm instance running?",
        T1_DEFAULT_URL,
    )
    base_url = ensure_v1_suffix(base_url)

    api_key = prompt("API key for t1", T1_DEFAULT_API_KEY)

    model_name = prompt(
        "Model ID name for t1",
        T1_DEFAULT_MODEL,
    )

    return {
        "name": "t1",
        "npm": "@ai-sdk/openai-compatible",
        "base_url": base_url,
        "api_key": api_key,
        "models": [
            {
                "name": model_name,
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a coder.json config file interactively.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output file path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    out_file_path = args.output
    log.info(f'# OATs Coder Config Setup\n\n🎉 🎉 😄 Welcome thanks for checking out the oats coder.😄 🎉 🎉 \n\n---\n\nWe would like to help everyone setup the ``coder`` configuration the same way because it can be **annoying** the first time. Please let us know if there\'s a way to make this easier!!🔧🔧 \n\nIf you hit an issue please reach out so we can help everyone:\nhttps://github.com/district-solutions/open-agent-tools-coder/issues/new\n\n---\n\nBy default the coder requires a ``coder.json`` file that holds the location and credentials to access 1 to many vLLM instances.\nIf you do not have these deployed, please refer to the Readme: https://github.com/district-solutions/open-agent-tools-coder/blob/main/README.md\n\nOnce you have your vLLM running, you can save the ``coder.json`` to a custom location outside the repo for security purposes.\n\nBy default this tool will save the ``coder.json`` file with the vLLM credentials to:\n```\n{out_file_path}\n```\n\n')
    log.info('### Let\'s get started!!\n---\n')
    coder_config_file = prompt(
        f"Do you want to save the coder.json file to another location?\n   - Hit enter to use the default\n  ",
        out_file_path,
    )

    vllm_small = collect_vllm_small()
    t1 = collect_t1()

    config = {
        "provider": {
            "vllm-small": vllm_small,
            "t1": t1,
        }
    }

    with open(out_file_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    log.info(f'\n# Config Written Successfully 🔥🔥 \n\nConfig written to: {out_file_path}\n\n```json\n{json.dumps(config, indent=2)}\n```\n\nTry running the command:\n\n``export CODER_CONFIG_FILE={out_file_path}``\n\n``oat``')


if __name__ == "__main__":
    main()
