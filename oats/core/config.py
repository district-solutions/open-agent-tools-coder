"""
Configuration management with hierarchical loading.

Config priority (highest to lowest):
1. Environment variables set CODER_CONFIG_FILE to your coder.json file
export CODER_CONFIG_FILE=./oats/config/coder.json')
"""
from __future__ import annotations


import json
import os
from pathlib import Path
from typing import Optional, Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from oats.pp import pp
from oats.log import cl

log = cl('config.1')


# Global config instance (lazy loaded)
_config: Config | None = None


class ProviderConfig(BaseModel):
    """Configuration for an AI provider."""

    email: Optional[str] = None
    user_id: Optional[str] = None
    pw: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    enabled: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Configuration for model selection."""

    model_config = {"protected_namespaces": ()}

    provider_id: str = os.getenv('VLLM_PROVIDER_ID', "vllm-small")
    model_id: str = os.getenv('VLLM_MODEL_ID', "hosted_vllm/chat:latest")


class HookEntry(BaseModel):
    """A single hook configuration entry."""

    event: str  # "pre_tool_use", "post_tool_use", "user_prompt_submit", "session_start", "file_changed"
    matcher: Optional[str] = None  # tool name glob pattern (e.g. "bash", "write*")
    command: str  # shell command to execute
    timeout: int = 30  # seconds


class HookConfig(BaseModel):
    """Hook system configuration."""

    hooks: list[HookEntry] = Field(default_factory=list)


class PermissionConfig(BaseModel):
    """Permission system configuration."""

    # Default permission rules
    read: dict[str, str] = Field(default_factory=lambda: {"*": "allow"})
    write: dict[str, str] = Field(default_factory=lambda: {"*": "ask"})
    bash: str = "ask"
    external_directory: dict[str, str] = Field(default_factory=lambda: {"*": "ask"})


class Config(BaseSettings):
    """
    Main application configuration.

    Loads from environment variables and config files.
    """

    # Provider configurations
    provider: dict[str, ProviderConfig] = Field(default_factory=dict)

    # Model selection
    model: ModelConfig = Field(default_factory=ModelConfig)

    # Permission rules
    permission: PermissionConfig = Field(default_factory=PermissionConfig)

    # Hook system
    hooks: HookConfig = Field(default_factory=HookConfig)

    # Data directory
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".local" / "share" / "oats")

    # Project directory (current working directory by default)
    project_dir: Path = Field(default_factory=Path.cwd)

    # Server settings
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    server_username: str | None = None
    server_password: str | None = None

    # Debug mode
    debug: bool = False

    model_config = {
        "env_prefix": "CODER_",
        "env_nested_delimiter": "__",
    }

    @classmethod
    def load(cls, project_dir: Path | None = None) -> "Config":
        """
        Load configuration from all sources.

        Merges configs in priority order.
        """
        # Start with defaults
        config_data: dict[str, Any] = {}

        # Load managed config (lowest priority)
        loaded = False
        config_path = os.getenv('CODER_CONFIG_FILE', './oats/config/coder.json')
        managed_paths = [
            config_path,
            '/opt/ds/oats/oats/config/coder.json',
            "/Library/Application Support/oats/coder.json",
            f'{Path.home()}/.local/share/oats/coder.json',
        ]
        env_persona_config = os.getenv('PERSONA_CONFIG', None)
        for path in managed_paths:
            if os.path.exists(path):
                # log.debug(f'CODER_ACTIVE_CONFIG: {path}')
                config_data = _merge_config(config_data, _load_json(Path(path)))
                loaded = True
                break

        if env_persona_config is not None:
            env_persona_config = str(env_persona_config)
            if os.path.exists(env_persona_config):
                log.debug(f'last_config_env_persona_config: {env_persona_config}')
                config_data = _merge_config(config_data, _load_json(Path(env_persona_config)))
            else:
                log.error(f'missing_env_persona_config: {env_persona_config}')
        else:
            if not loaded:
                example_coder_config = {
                    "provider": {
                        "vllm-small": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "vllm-small",
                        "base_url": "http://0.0.0.0:8111/v1",
                        "api_key": "CHANGE_PASSWORD",
                        "models": [
                            {
                            "name": "hosted_vllm/chat:latest",
                            "maxTokens": 262100
                            }
                        ]
                        },
                        "t1": {
                        "name": "t1",
                        "npm": "@ai-sdk/openai-compatible",
                        "base_url": "http://0.0.0.0:20700/v1",
                        "api_key": "CHANGE_PASSWORD",
                        "models": [
                            {
                            "name": "openai/google/functiongemma-270m-it"
                            }
                        ]
                        }
                    }
                }
                err_msg = f'### Sorry!! Please set the environment variable CODER_CONFIG_FILE to your coder.json with this command:\n```\nexport CODER_CONFIG_FILE=PATH\n```\n\n### Here is an example coder.json file contents found in the repo: ``oats/config/coder.json`` or on github at: ``https://github.com/district-solutions/open-agent-tools-coder/blob/main/oats/config/coder.json`` with contents to configure based on your environment:\n```\n{pp(example_coder_config)}\n```\n\nmanaged_paths:\n```\n{pp(managed_paths)}\n```\n'
                log.error(err_msg)
                import sys
                sys.exit(1)

        # Set project_dir
        config_data["project_dir"] = project_dir

        # Environment variables override everything (handled by pydantic-settings)
        # Also check for provider API keys from environment
        config_data = _load_provider_env_vars(config_data)

        """
        for key in config_data:
            log.debug(key)
            if key not in ['project_dir']:
                log.info(pp(config_data[key]))
            else:
                log.info(config_data[key])
        """

        return cls(**config_data)


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON config file."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to load config from {path}: {e}")
        return {}


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two config dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    return result


def _load_provider_env_vars(config_data: dict[str, Any]) -> dict[str, Any]:
    """Load provider API keys from environment variables."""
    if "provider" not in config_data:
        config_data["provider"] = {}

    env_mappings = {
        "ANTHROPIC_API_KEY": ("anthropic", "api_key"),
        "OPENAI_API_KEY": ("openai", "api_key"),
        "AZURE_OPENAI_API_KEY": ("azure", "api_key"),
        "AZURE_OPENAI_ENDPOINT": ("azure", "base_url"),
        "GOOGLE_API_KEY": ("google", "api_key"),
        "GOOGLE_GENERATIVE_AI_API_KEY": ("google", "api_key"),
        "MISTRAL_API_KEY": ("mistral", "api_key"),
        "GROQ_API_KEY": ("groq", "api_key"),
        "OPENROUTER_API_KEY": ("openrouter", "api_key"),
        "TOGETHER_API_KEY": ("together", "api_key"),
        "COHERE_API_KEY": ("cohere", "api_key"),
    }

    for env_var, (provider, field) in env_mappings.items():
        value = os.environ.get(env_var)
        if value:
            if provider not in config_data["provider"]:
                config_data["provider"][provider] = {}
            config_data["provider"][provider][field] = value

    return config_data


def get_config(project_dir: Path | None = None, reload: bool = False, verbose: bool = False) -> Config:
    """Get the global config instance, loading if necessary."""
    global _config
    if _config is None or reload:
        if project_dir is None:
            project_dir = os.getcwd()
        if verbose:
            log.debug(f'using project_dir: {project_dir}')
        if project_dir is None:
            _config = Config.load()
        else:
            _config = Config.load(project_dir=project_dir)
    return _config


def get_data_dir() -> Path:
    """Get the data directory, creating if needed."""
    data_dir = get_config().data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
