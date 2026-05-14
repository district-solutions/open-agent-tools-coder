"""
Model definitions and registry.
"""
from __future__ import annotations


import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Model:
    """Definition of an AI model.

    Attributes:
        id: Unique model identifier (e.g. "gpt-4o").
        provider_id: The provider this model belongs to (e.g. "openai", "anthropic").
        name: Human-readable display name.
        description: Short description of the model's capabilities.
        context_length: Maximum number of tokens in the input context window.
        max_output_tokens: Maximum number of tokens the model can generate in a single response.
        supports_tools: Whether the model supports function/tool calling.
        supports_vision: Whether the model supports image/vision input.
        supports_streaming: Whether the model supports streaming responses.
        cost_per_input_token: Cost per input token in USD.
        cost_per_output_token: Cost per output token in USD.
        extra: Additional provider-specific metadata.
    """

    id: str
    provider_id: str
    name: str
    description: str = ""
    # AGENT_TAG_MAX_CONTEXT_SIZE
    context_length: int = int(os.getenv('CODER_CTX_LEN', '262100'))
    max_output_tokens: int = int(os.getenv('CODER_CTX_LEN', '262100'))
    supports_tools: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def litellm_model(self) -> str:
        """Get the LiteLLM model identifier.

        Maps the internal provider/model combination to the format expected
        by LiteLLM (e.g. "azure/gpt-4o", "gemini/gemini-2.0-flash").

        Returns:
            The model string to pass to LiteLLM's acompletion().
        """
        # LiteLLM uses provider/model format for some providers
        if self.provider_id == "ow":
            return f"hosted_vllm/{self.id}"
        elif self.provider_id in ['vllm-small']:
            return f"hosted_vllm/{self.id}"
        elif self.provider_id == "anthropic":
            return self.id
        elif self.provider_id == "openai":
            return self.id
        elif self.provider_id == "azure":
            return f"azure/{self.id}"
        elif self.provider_id == "google":
            return f"gemini/{self.id}"
        elif self.provider_id == "mistral":
            return f"mistral/{self.id}"
        elif self.provider_id == "groq":
            return f"groq/{self.id}"
        elif self.provider_id == "openrouter":
            return f"openrouter/{self.id}"
        elif self.provider_id == "together":
            return f"together_ai/{self.id}"
        elif self.provider_id == "ollama":
            return f"ollama/{self.id}"
        elif self.provider_id == "vllm":
            return f"hosted_vllm/{self.id}"
        else:
            return self.id


# Built-in model definitions — pre-configured models for known providers.
BUILTIN_MODELS: list[Model] = [
    # Anthropic models
    Model(
        id="claude-opus-4-20250514",
        provider_id="anthropic",
        name="Claude Opus 4",
        description="Most capable Claude model for complex tasks",
        context_length=200000,
        max_output_tokens=32000,
        supports_tools=True,
        supports_vision=True,
        cost_per_input_token=0.015,
        cost_per_output_token=0.075,
    ),
    Model(
        id="claude-sonnet-4-20250514",
        provider_id="anthropic",
        name="Claude Sonnet 4",
        description="Balanced Claude model for most tasks",
        context_length=200000,
        max_output_tokens=16000,
        supports_tools=True,
        supports_vision=True,
        cost_per_input_token=0.003,
        cost_per_output_token=0.015,
    ),
    Model(
        id="claude-3-5-haiku-20241022",
        provider_id="anthropic",
        name="Claude 3.5 Haiku",
        description="Fast and efficient Claude model",
        context_length=200000,
        max_output_tokens=8192,
        supports_tools=True,
        supports_vision=True,
        cost_per_input_token=0.0008,
        cost_per_output_token=0.004,
    ),
    # OpenAI models
    Model(
        id="gpt-4o",
        provider_id="openai",
        name="GPT-4o",
        description="OpenAI's most capable model",
        context_length=128000,
        max_output_tokens=16384,
        supports_tools=True,
        supports_vision=True,
        cost_per_input_token=0.005,
        cost_per_output_token=0.015,
    ),
    Model(
        id="gpt-4o-mini",
        provider_id="openai",
        name="GPT-4o Mini",
        description="Fast and affordable GPT-4o variant",
        context_length=128000,
        max_output_tokens=16384,
        supports_tools=True,
        supports_vision=True,
        cost_per_input_token=0.00015,
        cost_per_output_token=0.0006,
    ),
    Model(
        id="o1",
        provider_id="openai",
        name="o1",
        description="OpenAI reasoning model",
        context_length=200000,
        max_output_tokens=100000,
        supports_tools=False,
        supports_vision=True,
        cost_per_input_token=0.015,
        cost_per_output_token=0.06,
    ),
    # Google models
    Model(
        id="gemini-2.0-flash",
        provider_id="google",
        name="Gemini 2.0 Flash",
        description="Google's fast multimodal model",
        context_length=1000000,
        max_output_tokens=8192,
        supports_tools=True,
        supports_vision=True,
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
    ),
    Model(
        id="gemini-1.5-pro",
        provider_id="google",
        name="Gemini 1.5 Pro",
        description="Google's most capable model",
        context_length=2000000,
        max_output_tokens=8192,
        supports_tools=True,
        supports_vision=True,
        cost_per_input_token=0.00125,
        cost_per_output_token=0.005,
    ),
    # Mistral models
    Model(
        id="mistral-large-latest",
        provider_id="mistral",
        name="Mistral Large",
        description="Mistral's most capable model",
        context_length=128000,
        max_output_tokens=8192,
        supports_tools=True,
        supports_vision=False,
        cost_per_input_token=0.002,
        cost_per_output_token=0.006,
    ),
    # Groq models
    Model(
        id="llama-3.3-70b-versatile",
        provider_id="groq",
        name="Llama 3.3 70B",
        description="Fast Llama 3.3 on Groq",
        context_length=128000,
        max_output_tokens=32768,
        supports_tools=True,
        supports_vision=False,
        cost_per_input_token=0.00059,
        cost_per_output_token=0.00079,
    ),
]

if os.getenv('CODER_DISABLED_CLOUD_MODELS', '0') == '0':
    BUILTIN_MODELS: list[Model] = []

class ModelRegistry:
    """Registry of available models.

    Maintains a central index of all registered AI models, keyed by
    ``provider_id/model_id``. Initialized with built-in model definitions
    on construction.
    """

    def __init__(self) -> None:
        """Initialize the registry and load built-in models."""
        self._models: dict[str, Model] = {}
        # Load built-in models
        for model in BUILTIN_MODELS:
            self.register(model)

    def register(self, model: Model) -> None:
        """Register a model in the registry.

        Args:
            model: The model definition to register.
        """
        key = f"{model.provider_id}/{model.id}"
        self._models[key] = model

    def get(self, provider_id: str, model_id: str) -> Model | None:
        """Get a model by provider and model ID.

        Args:
            provider_id: The provider identifier (e.g. "openai").
            model_id: The model identifier (e.g. "gpt-4o").

        Returns:
            The Model if found, or None.
        """
        key = f"{provider_id}/{model_id}"
        return self._models.get(key)

    def list(self, provider_id: str | None = None) -> list[Model]:
        """List all models, optionally filtered by provider.

        Args:
            provider_id: If provided, only return models from this provider.

        Returns:
            A list of Model instances.
        """
        models = list(self._models.values())
        if provider_id:
            models = [m for m in models if m.provider_id == provider_id]
        return models

    def list_by_provider(self) -> dict[str, list[Model]]:
        """List models grouped by provider.

        Returns:
            A dict mapping provider_id to a list of Model instances.
        """
        result: dict[str, list[Model]] = {}
        for model in self._models.values():
            if model.provider_id not in result:
                result[model.provider_id] = []
            result[model.provider_id].append(model)
        return result


# Global model registry
_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry.

    Lazily initializes a singleton ModelRegistry on first call.

    Returns:
        The global ModelRegistry instance.
    """
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def get_model(provider_id: str, model_id: str) -> Model | None:
    """Get a model by provider and model ID.

    Args:
        provider_id: The provider identifier (e.g. "openai").
        model_id: The model identifier (e.g. "gpt-4o").

    Returns:
        The Model if found, or None.
    """
    return get_model_registry().get(provider_id, model_id)


def list_models(provider_id: str | None = None) -> list[Model]:
    """List all available models.

    Args:
        provider_id: If provided, only return models from this provider.

    Returns:
        A list of all registered Model instances.
    """
    return get_model_registry().list(provider_id)
