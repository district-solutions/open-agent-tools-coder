"""
AI Provider abstraction using LiteLLM.

Provides:
- Multi-provider LLM calls via LiteLLM
- Retry with exponential backoff and jitter for transient errors
- Text-based tool call parsing for open-source models (Qwen, Hermes)
- Streaming support
"""
from __future__ import annotations

import os
import re
import uuid
import traceback
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from pydantic import BaseModel

from oats.oweb.login import login_to_openwebui
from oats.core.config import get_config, ProviderConfig
from oats.core.bus import bus, Event, EventType
from oats.provider.models import get_model, Model
from oats.core.features import (
    streaming_tool_assembly_enabled,
    strict_tool_schemas_enabled,
)
from oats.pp import pp
from oats.log import cl

log = cl('provider')

# Configure litellm
# litellm.set_verbose = False


# ─── Retry configuration ────────────────────────────────────────────────
# Exponential backoff with jitter for transient errors, rate limits,
# and server errors.

MAX_RETRIES = 3
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0

# Errors worth retrying (status codes and exception substrings)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}
_RETRYABLE_ERROR_SUBSTRINGS = [
    "rate limit", "rate_limit", "overloaded", "timeout",
    "connection", "temporarily unavailable", "server error",
    "internal error", "bad gateway", "service unavailable",
]


def _is_retryable(exc: Exception) -> bool:
    """Decide if an LLM call error is transient and worth retrying."""
    exc_str = str(exc).lower()

    # Check for HTTP status codes embedded in the exception
    for code in _RETRYABLE_STATUS_CODES:
        if str(code) in exc_str:
            return True

    # Check for known transient error messages
    for substr in _RETRYABLE_ERROR_SUBSTRINGS:
        if substr in exc_str:
            return True

    return False


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter: base * 2^attempt + random jitter."""
    import random
    delay = min(INITIAL_BACKOFF_S * (2 ** attempt), MAX_BACKOFF_S)
    jitter = random.uniform(0, delay * 0.25)
    return delay + jitter


# ─── Text-based tool call parser ────────────────────────────────────────
# Critical for Qwen, Hermes, and other open-source models via vLLM that
# embed tool calls in the response text instead of structured tool_calls.

def _strip_non_json_code_blocks(content: str) -> str:
    """
    Remove non-JSON markdown code blocks (```python, ```bash, etc.)
    so we don't parse example code as tool calls. Keep ```json blocks
    because models often wrap real tool calls in them.
    """
    # Strip ```python, ```bash, ```text, etc. but NOT ```json or bare ```
    stripped = re.sub(r'```(?!json)(?:[a-z]+)\n.*?```', '', content, flags=re.DOTALL)
    return stripped


def _strip_hallucination_tokens(content: str) -> str:
    """
    Remove leaked chat template tokens from open-source models.
    Qwen leaks <|im_start|>, <|im_end|>, etc.
    """
    # Truncate at the first <|im_start|> — everything after is hallucinated
    im_start_pos = content.find("<|im_start|>")
    if im_start_pos >= 0:
        content = content[:im_start_pos]
    content = re.sub(r'<\|im_end\|>', '', content)
    content = re.sub(r'<\|endoftext\|>', '', content)
    # Strip <tool_response>...</tool_response> hallucinations
    content = re.sub(r'<tool_response>.*?</tool_response>', '', content, flags=re.DOTALL)
    # Strip bare {"tool_response": ...} hallucinations
    content = re.sub(r'\{"tool_response"\s*:.*?\}', '', content, flags=re.DOTALL)
    return content.strip()


# ── Model-specific special tokens that can leak into tool call arguments ──
# vLLM's streaming tool parsers occasionally leave residual delimiter tokens
# in the assembled JSON argument string. These are model-specific tokens that
# have no meaning in JSON and must be stripped before parsing.
_TOOL_ARG_SPECIAL_TOKENS = re.compile(
    r'<\|"\|>'           # Gemma 4 string delimiter
    r'|<\|tool_call>'    # Gemma 4 tool call start
    r'|<tool_call\|>'    # Gemma 4 tool call end
    r'|<\|tool_response>'  # Gemma 4 tool response
    r'|<tool_response\|>'  # Gemma 4 tool response end
)


def _sanitize_tool_arguments(arguments: str) -> str:
    """Strip leaked model special tokens from tool call argument strings.

    vLLM's streaming tool parsers (especially gemma4) can leave residual
    delimiter tokens like <|"|> in the assembled JSON. This function removes
    them so json.loads succeeds. Only touches known non-JSON tokens — valid
    JSON content is never altered.
    """
    if not arguments or '<' not in arguments:
        return arguments
    return _TOOL_ARG_SPECIAL_TOKENS.sub('', arguments)


def _parse_tool_calls_from_text(content: str, available_tools: list[str] | None = None) -> tuple[list[dict], str]:
    """
    Parse tool calls embedded in text content from models that don't use
    structured tool_calls (e.g., Qwen via vLLM with hermes parser).

    Supports formats:
    - <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    - <tools>{"name": "...", "arguments": {...}}</tools>
    - {"name": "tool_name", "arguments": {...}}  (bare JSON when tool name matches)

    Ignores tool calls inside markdown code blocks (```...```) to avoid
    parsing examples/documentation as real tool calls.

    Returns (parsed_tool_calls, remaining_text).
    """
    import json as _json

    # Strip hallucination tokens first
    content = _strip_hallucination_tokens(content)

    # Strip non-JSON code blocks to avoid parsing examples
    parse_content = _strip_non_json_code_blocks(content)

    tool_calls = []
    remaining = content

    # Pattern 1: <tool_call>...</tool_call> (hermes format)
    tc_pattern = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
    for match in tc_pattern.finditer(parse_content):
        try:
            data = _json.loads(match.group(1))
            if "name" in data:
                tool_calls.append(data)
                remaining = remaining.replace(match.group(0), "").strip()
        except _json.JSONDecodeError:
            continue

    # Pattern 2: <tools>...</tools>
    tools_pattern = re.compile(r'<tools>\s*(.*?)\s*</tools>', re.DOTALL)
    for match in tools_pattern.finditer(parse_content):
        try:
            data = _json.loads(match.group(1))
            if "name" in data:
                tool_calls.append(data)
                remaining = remaining.replace(match.group(0), "").strip()
        except _json.JSONDecodeError:
            continue

    # Pattern 3: bare JSON object with "name" and "arguments" keys
    # Only match the FIRST valid tool call to avoid over-parsing
    if not tool_calls and available_tools:
        bare_json_pattern = re.compile(
            r'(?<!")\{[^{}]*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})[^{}]*\}',
            re.DOTALL,
        )
        for match in bare_json_pattern.finditer(parse_content):
            try:
                data = _json.loads(match.group(0))
                if data.get("name") in available_tools:
                    tool_calls.append(data)
                    remaining = remaining.replace(match.group(0), "").strip()
            except _json.JSONDecodeError:
                continue

    # Deduplicate: same tool name + same arguments = same call
    seen = set()
    deduped = []
    for tc in tool_calls:
        key = (tc["name"], _json.dumps(tc.get("arguments", {}), sort_keys=True))
        if key not in seen:
            seen.add(key)
            deduped.append(tc)

    return deduped, remaining


# Global provider registry
_registry: ProviderRegistry | None = None


class Message(BaseModel):
    """A chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str | list[dict[str, Any]]
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ToolDefinition(BaseModel):
    """Definition of a tool for the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool = False


class CompletionRequest(BaseModel):
    """Request for a completion."""

    messages: list[Message]
    model: str | None = None
    provider_id: str | None = None
    tools: list[ToolDefinition] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    stream: bool = False
    debug_context: dict[str, Any] | None = None


class ToolCall(BaseModel):
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: str  # JSON string


class CompletionResponse(BaseModel):
    """Response from a completion."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    model: str | None = None


class CompletionChunk(BaseModel):
    """A streaming chunk from a completion."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None


@dataclass
class Provider:
    """
    An AI provider (Anthropic, OpenAI, etc.).

    Uses LiteLLM for actual API calls.
    """

    id: str
    name: str
    config: ProviderConfig
    models: list[Model] = field(default_factory=list)

    def is_configured(self) -> bool:
        """Check if the provider has required configuration."""
        if self.id in ['ow']:
            try:
                if self.config.email is None:
                    self.config.email = os.getenv('CODER_CHAT_EMAIL', None)
                if self.config.pw is None:
                    self.config.pw = os.getenv('CODER_CHAT_PASSWORD', None)
                if self.config.base_url is None:
                    self.config.base_url = os.getenv('CODER_CHAT_URL', None)
                if self.config.email is not None or self.config.pw is not None:
                    return False
                if self.config.api_key is None:
                    login_dict = login_to_openwebui(email=self.config.email, password=self.config.pw, base_url=self.config.base_url)
                    if login_dict is not None:
                        self.config.api_key = login_dict.get("token", None)
                        self.config.user_id = login_dict.get('user_id', None)
                        self.config.base_url = f'{self.config.base_url}/openai'
                return self.config.api_key is not None
            except Exception:
                log.info(f'### Sorry!! failed to check open-webui_provider_login_failed with error:\n```\n{traceback.format_exc()}\n```\n')
                return False
        elif self.id in ['ollama']:
            return self.config.api_key is not None or self.id in ["ollama"]
        else:
            return self.config.api_key is not None

    def _resolve_litellm_model(self, model_id: str) -> str:
        """Resolve the LiteLLM model string from provider ID and model ID."""
        model = get_model(self.id, model_id)
        if model is not None:
            return model.litellm_model

        # Provider-specific prefixes for LiteLLM routing
        _prefix_map = {
            "ollama": "ollama/",
            "azure": "azure/",
            "google": "gemini/",
            "mistral": "mistral/",
            "groq": "groq/",
            "openrouter": "openrouter/",
            "together": "together_ai/",
        }
        prefix = _prefix_map.get(self.id, "")
        return f"{prefix}{model_id}"

    def _build_kwargs(self, request: CompletionRequest, litellm_model: str) -> dict[str, Any]:
        """Build the kwargs dict for LiteLLM acompletion call."""
        # Build messages
        messages = []
        for m in request.messages:
            msg_dict = {"role": m.role, "content": m.content}
            if m.name:
                msg_dict["name"] = m.name
            if m.tool_call_id:
                msg_dict["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                msg_dict["tool_calls"] = m.tool_calls
            messages.append(msg_dict)

        kwargs: dict[str, Any] = {
            "model": litellm_model,
            "messages": messages,
        }

        # API key
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        # Base URL
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url

        # Optional parameters
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        # Stop tokens — prevent open-source models from hallucinating
        # past their intended output boundary
        if request.stop:
            kwargs["stop"] = request.stop

        # Tools
        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                        **(
                            {"strict": True}
                            if t.strict and strict_tool_schemas_enabled()
                            else {}
                        ),
                    },
                }
                for t in request.tools
            ]

        return kwargs

    def _parse_tool_calls_from_response(
        self, message: Any, request: CompletionRequest
    ) -> tuple[str | None, list[ToolCall] | None]:
        """
        Extract tool calls from an LLM response message.

        First checks structured tool_calls, then falls back to parsing
        tool calls from text content (for open-source models).

        Returns (content, tool_calls).
        """
        import json as _json

        tool_calls = None
        response_content = message.content

        # Structured tool_calls (standard OpenAI-compatible path)
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=_sanitize_tool_arguments(tc.function.arguments),
                )
                for tc in message.tool_calls
            ]

        # Fallback: parse from text content (Qwen, Hermes, etc.)
        if not tool_calls and response_content and request.tools:
            available_tool_names = [t.name for t in request.tools]
            parsed, remaining_text = _parse_tool_calls_from_text(
                response_content, available_tool_names
            )
            if parsed:
                tool_calls = [
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=tc["name"],
                        arguments=_json.dumps(tc.get("arguments", {})),
                    )
                    for tc in parsed
                ]
                response_content = remaining_text or None
                log.info(f"parsed {len(tool_calls)} tool call(s) from text content")

        return response_content, tool_calls

    async def complete(self, request: CompletionRequest, verbose: bool = True) -> CompletionResponse:
        """
        Get a completion from the provider with retry logic.

        Retries transient errors (rate limits, 5xx, timeouts) with
        exponential backoff + jitter.
        """
        model_id = request.model or get_config().model.model_id
        litellm_model = self._resolve_litellm_model(model_id)
        kwargs = self._build_kwargs(request, litellm_model)
        if verbose:
            log.info(f'## Sending_provider_complete {__name__} request:\n```\n{pp(kwargs)}\n```\n')

        # Publish request event
        await bus.publish(
            Event(
                type=EventType.PROVIDER_REQUEST,
                data={"provider": self.id, "model": litellm_model},
            )
        )

        last_error: Exception | None = None

        from litellm import acompletion
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await acompletion(**kwargs)

                # Parse the response
                choice = response.choices[0]
                response_content, tool_calls = self._parse_tool_calls_from_response(
                    choice.message, request
                )

                result = CompletionResponse(
                    content=response_content,
                    tool_calls=tool_calls,
                    finish_reason=choice.finish_reason,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                    if response.usage
                    else None,
                    model=response.model,
                )
                dbg = request.debug_context or {}
                from oats.session.debug_trace import trace_event
                trace_event(
                    dbg.get("session_id"),
                    "provider.complete",
                    {
                        "iteration": dbg.get("iteration"),
                        "provider_id": request.provider_id,
                        "model": litellm_model,
                        "tool_count": len(request.tools or []),
                        "has_tool_calls": bool(tool_calls),
                        "tool_names": [tc.name for tc in tool_calls or []],
                        "usage": result.usage,
                    },
                )

                # Publish response event
                await bus.publish(
                    Event(
                        type=EventType.PROVIDER_RESPONSE,
                        data={
                            "provider": self.id,
                            "model": litellm_model,
                            "usage": result.usage,
                            "retries": attempt,
                        },
                    )
                )

                return result

            except Exception as e:
                last_error = e
                import asyncio

                if attempt < MAX_RETRIES and _is_retryable(e):
                    delay = _backoff_delay(attempt)
                    log.warn(
                        f"retryable error (attempt {attempt + 1}/{MAX_RETRIES + 1}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                    continue

                # Non-retryable or exhausted retries
                await bus.publish(
                    Event(
                        type=EventType.PROVIDER_ERROR,
                        data={
                            "provider": self.id,
                            "model": litellm_model,
                            "error": str(e),
                            "retries": attempt,
                        },
                    )
                )
                raise

        # Should not reach here, but just in case
        raise last_error  # type: ignore[misc]

    async def stream(self, request: CompletionRequest, verbose: bool = False) -> AsyncIterator[CompletionChunk]:
        """
        Stream a completion from the provider with retry logic.

        Yields chunks as they arrive from the LLM. For tool calls that
        arrive as text content (open-source models), the final chunk
        will contain the parsed tool calls.
        """
        model_id = request.model or get_config().model.model_id
        litellm_model = self._resolve_litellm_model(model_id)
        kwargs = self._build_kwargs(request, litellm_model)
        kwargs["stream"] = True

        await bus.publish(
            Event(
                type=EventType.PROVIDER_REQUEST,
                data={"provider": self.id, "model": litellm_model, "streaming": True},
            )
        )

        last_error: Exception | None = None

        from litellm import acompletion
        for attempt in range(MAX_RETRIES + 1):
            try:
                if verbose:
                    log.info(f'### Provider Completion Request\n\n{__file__}\nkwargs:\n\n```\n{pp(kwargs)}\n```\n')
                response = await acompletion(**kwargs)

                # Accumulate full content for text-based tool call detection
                accumulated_content = ""
                has_structured_tool_calls = False
                structured_calls: dict[int, dict[str, str]] = {}

                async for chunk in response:
                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    delta = choice.delta

                    chunk_tool_calls = None
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        has_structured_tool_calls = True
                        for tc in delta.tool_calls:
                            idx = getattr(tc, "index", None)
                            if idx is None:
                                idx = len(structured_calls)
                            existing = structured_calls.setdefault(
                                idx,
                                {"id": "", "name": "", "arguments": ""},
                            )
                            if getattr(tc, "id", None):
                                existing["id"] = tc.id
                            if getattr(tc, "function", None):
                                fn = tc.function
                                if getattr(fn, "name", None):
                                    existing["name"] = fn.name
                                if getattr(fn, "arguments", None):
                                    existing["arguments"] += fn.arguments

                    # Accumulate text content
                    chunk_content = delta.content if hasattr(delta, "content") else None
                    if chunk_content:
                        accumulated_content += chunk_content

                    yield CompletionChunk(
                        content=chunk_content,
                        tool_calls=chunk_tool_calls,
                        finish_reason=choice.finish_reason,
                    )

                    # On stream end, check for text-based tool calls
                    if choice.finish_reason:
                        if (
                            streaming_tool_assembly_enabled()
                            and has_structured_tool_calls
                            and structured_calls
                        ):
                            finalized = []
                            for idx in sorted(structured_calls):
                                tc = structured_calls[idx]
                                if not tc["name"]:
                                    continue
                                finalized.append(
                                    ToolCall(
                                        id=tc["id"] or f"call_{uuid.uuid4().hex[:12]}",
                                        name=tc["name"],
                                        arguments=_sanitize_tool_arguments(tc["arguments"] or "{}"),
                                    )
                                )
                            if finalized:
                                dbg = request.debug_context or {}
                                from oats.session.debug_trace import trace_event
                                trace_event(
                                    dbg.get("session_id"),
                                    "provider.stream.structured_tool_assembly",
                                    {
                                        "iteration": dbg.get("iteration"),
                                        "provider_id": request.provider_id,
                                        "model": litellm_model,
                                        "tool_names": [tc.name for tc in finalized],
                                        "tool_count": len(finalized),
                                    },
                                )
                                yield CompletionChunk(
                                    content=None,
                                    tool_calls=finalized,
                                    finish_reason="tool_calls",
                                )
                        elif accumulated_content and request.tools:
                            import json as _json
                            available_tool_names = [t.name for t in request.tools]
                            parsed, remaining = _parse_tool_calls_from_text(
                                accumulated_content, available_tool_names
                            )
                            if parsed:
                                text_tool_calls = [
                                    ToolCall(
                                        id=f"call_{uuid.uuid4().hex[:12]}",
                                        name=tc["name"],
                                        arguments=_json.dumps(tc.get("arguments", {})),
                                    )
                                    for tc in parsed
                                ]
                                log.info(
                                    f"stream: parsed {len(text_tool_calls)} tool call(s) from text"
                                )
                                dbg = request.debug_context or {}
                                from oats.session.debug_trace import trace_event
                                trace_event(
                                    dbg.get("session_id"),
                                    "provider.stream.text_tool_parse",
                                    {
                                        "iteration": dbg.get("iteration"),
                                        "provider_id": request.provider_id,
                                        "model": litellm_model,
                                        "tool_names": [tc.name for tc in text_tool_calls],
                                        "tool_count": len(text_tool_calls),
                                    },
                                )
                                yield CompletionChunk(
                                    content=None,
                                    tool_calls=text_tool_calls,
                                    finish_reason="tool_calls",
                                )

                return  # Stream completed successfully

            except Exception as e:
                last_error = e
                import asyncio

                if attempt < MAX_RETRIES and _is_retryable(e):
                    delay = _backoff_delay(attempt)
                    log.warn(
                        f"stream retryable error (attempt {attempt + 1}/{MAX_RETRIES + 1}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                    continue

                await bus.publish(
                    Event(
                        type=EventType.PROVIDER_ERROR,
                        data={
                            "provider": self.id,
                            "model": litellm_model,
                            "error": str(e),
                            "retries": attempt,
                        },
                    )
                )
                raise

        raise last_error  # type: ignore[misc]


class ProviderRegistry:
    """Registry of available providers."""

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}

    def register(self, provider: Provider) -> None:
        """Register a provider."""
        self._providers[provider.id] = provider

    def get(self, provider_id: str) -> Provider | None:
        """Get a provider by ID."""
        return self._providers.get(provider_id)

    def list(self) -> list[Provider]:
        """List all providers."""
        return list(self._providers.values())

    def list_configured(self) -> list[Provider]:
        """List only configured providers."""
        return [p for p in self._providers.values() if p.is_configured()]


def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry, initializing from config if needed."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
        _init_providers(_registry)
    return _registry


def _init_providers(registry: ProviderRegistry) -> None:
    """Initialize providers from configuration."""
    config = get_config()

    # Built-in provider definitions
    builtin_providers = [
        ('vllm-small', 'vllm-small'),
        ('t1', 't1'),
        ('ow', 'ow'),
        ("anthropic", "Anthropic"),
        ("openai", "OpenAI"),
        ("azure", "Azure OpenAI"),
        ("google", "Google AI"),
        ("mistral", "Mistral"),
        ("groq", "Groq"),
        ("openrouter", "OpenRouter"),
        ("together", "Together AI"),
        ("cohere", "Cohere"),
        ("ollama", "Ollama"),
    ]

    for provider_id, name in builtin_providers:
        provider_config = config.provider.get(provider_id, ProviderConfig())
        # if provider_id in ['ow']:
        #     log.info(f'## OpenWebUI Provider Config\n\n```\n{provider_config}\n```\n')
        #     log.info(f'---------')
        registry.register(
            Provider(
                id=provider_id,
                name=name,
                config=provider_config,
            )
        )

def get_provider(provider_id: str | None = None) -> Provider:
    """
    Get a provider by ID, or the default provider.

    Raises ValueError if the provider is not found or not configured.
    """
    if provider_id is None:
        provider_id = get_config().model.provider_id

    provider = get_provider_registry().get(provider_id)

    if provider is None:
        raise ValueError(f"{__name__} - coder_failed_to_find_provider_id '{provider_id}' not found")

    if not provider.is_configured():
        raise ValueError(
            f"{__name__} - Provider '{provider_id}' is not configured. "
            f"Please set the API key in your config or environment."
        )

    return provider

def list_providers() -> list[Provider]:
    """List all available providers."""
    return get_provider_registry().list()
