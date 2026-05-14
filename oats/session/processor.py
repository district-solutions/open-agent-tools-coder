"""
Session processor for handling messages and tool execution.

Handles the core message-processing loop:
- Streaming LLM responses (yields chunks as they arrive)
- Concurrent tool execution for read-only tools
- Retry-aware provider calls (handled in provider layer)
- Hook integration at all lifecycle points
"""
from __future__ import annotations

import os
import traceback
import asyncio
import re as _re
import ujson as json
from pathlib import Path
from typing import Optional, Any, AsyncIterator, Awaitable, Callable

from oats.models import OatPromptChoices
from oats.agent_get_tool_choices_for_prompt import agent_get_tool_choices_for_prompt
from oats.core.config import get_config
from oats.core.bus import bus, Event, EventType
from oats.provider.provider import CompletionRequest, CompletionChunk, Message as ProviderMessage, ToolDefinition, get_provider
from oats.session.session import Session, get_session_storage
from oats.tool.registry import ToolContext, ToolResult, get_tool, list_tools
from oats.session.models import SelectedToolsManifest
from oats.mcp.intent import select_tools_for_prompt, build_mcp_system_context
from oats.call_tool_with_loader1 import run_tool_call
from oats.hook.engine import HookEngine, HookEvent, HookContext
from oats.session.compaction import ConversationCompactor
from oats.session.debug_trace import trace_event
from oats.session.file_cache import FileStateCache
from oats.session.build_system_prompt import build_system_prompt
from oats.session.tool_retention import retain_tool_result
from oats.session.task_budget import SessionTaskBudget
from oats.session.token_budget import SessionTokenBudget, format_budget_guidance
from oats.core.features import (
    reactive_compaction_candidate_enabled,
    token_budget_candidate_enabled,
    task_budget_candidate_enabled,
    result_retention_enabled,
    should_disable_streaming,
)
from oats.pp import pp
from oats.log import gl

log = gl('sprc')

SYSTEM_PROMPT = """# Objective

You are an AI coding assistant. You help users with software engineering tasks including:

- Writing and debugging code
- Explaining code and concepts
- Refactoring and improving code
- Finding and fixing bugs
- Answering technical questions

# Tools

You have access to tools for:

- Reading files
- Writing files
- Editing files
- Running bash commands
- Searching files by pattern (glob)
- Searching file contents (grep)

# Safety

- DO NOT SHARE PPI OR SENSITIVE DATA!

# Guidelines

- Be concise and direct
- Read files before modifying them
- Use the appropriate tool for each task
- Complete all steps the user requested — do not stop after partial progress
- When a task requires multiple steps (e.g. read then edit then run), execute ALL steps
- Explain your actions when helpful
- Ask for clarification when requirements are unclear

Current working directory: REPLACE_WORKING_DIR
"""

# ─── Concurrency-safe tool names ────────────────────────────────────────
# These tools only read state and never mutate files/processes, so they
# can run in parallel when the model requests multiple of them at once.
# Used to decide whether a batch of tool calls can run in parallel.

_CONCURRENCY_SAFE_TOOLS = frozenset({
    "read", "glob", "grep", "webfetch", "websearch",
    "memory_read", "todoread", "question", "plan_status",
    "agent_status", "check_certificate_expiration",
    "get_app_manifest", "get_app_logs", "get_app_credentials",
    "convert_pq_to_json",
})


# ─── Auto-todo reminder cadence (mirrors claude-code) ───────────────────
# After this many assistant turns without a `todowrite`, AND this many
# turns since the previous reminder, inject a system nudge so long tasks
# don't drop incomplete steps. Local models (Qwen3.6, etc.) won't reach
# for `todowrite` on their own — the reminder is the trigger.
TODO_REMINDER_TURNS_SINCE_WRITE = int(os.getenv("CODER_TODO_REMINDER_TURNS_SINCE_WRITE", "10"))
TODO_REMINDER_TURNS_BETWEEN = int(os.getenv("CODER_TODO_REMINDER_TURNS_BETWEEN", "10"))
TODO_REMINDER_TEXT = (
    "<system-reminder>\n"
    "The todowrite tool hasn't been used recently. If you're working on a "
    "multi-step task that would benefit from tracking progress, call todowrite "
    "to create or update a checklist, then check items off as you complete them. "
    "This helps long tasks finish without dropping later steps. Also consider "
    "cleaning up the todo list if it has become stale. Only use it if relevant "
    "to the current work. This is just a gentle reminder — ignore if not "
    "applicable. Do NOT mention this reminder to the user.\n"
    "</system-reminder>"
)


def validate_coder_env(config, provider_id: str, model_id: str, verbose: bool = False) -> bool:
    found_base_url = None
    config_dict = {}
    coder_config_file = os.getenv('CODER_CONFIG_FILE', None)
    if coder_config_file is None:
        err_msg = f'### Sorry!!💥 The environment variable: ``CODER_CONFIG_FILE`` is missing. Please create your own oats/coder.json file outside the repo and then export it with:\n```\nexport CODER_CONFIG_FILE=PATH/coder.json\n```\n'
        log.info(err_msg)
        raise Exception('Please fix the CODER_CONFIG_FILE for the logged error')
    else:
        config_contents = ''
        with open(coder_config_file, 'r') as file:
            config_contents = file.read()
        if config_contents == '':
            err_msg = f'### Sorry!!💥 The CODER_CONFIG_FILE is empty. Please check the environment variable: ``CODER_CONFIG_FILE`` file is at that path and is a valid coder.json file like the ``oats/config/coder.json``\n\nHere is the path to the current config:\n```\nexport CODER_CONFIG_FILE={coder_config_file}\n```\n'
            log.info(err_msg)
            raise Exception('Please fix the CODER_CONFIG_FILE for the logged error')
        try:
            config_dict = json.loads(config_contents)
        except Exception:
            err_msg = f'### Sorry!!💥 The CODER_CONFIG_FILE has ❗💥 ``invalid JSON``💥❗. Please check the environment variable: ``CODER_CONFIG_FILE`` file is at that path and is a valid coder.json file like the ``oats/config/coder.json``\n\nHere is the path to the current config:\n```\nexport CODER_CONFIG_FILE={coder_config_file}\n```\nCurrent CODER_CONFIG_FILE contents:\n```\n{config_contents}\n```\n'
            log.info(err_msg)
            raise Exception('Please fix the CODER_CONFIG_FILE for the logged error')
    if len(config_dict) == 0:
        err_msg = f'### Sorry!!💥 The CODER_CONFIG_FILE does not have any valid model providers. Please check the environment variable: ``CODER_CONFIG_FILE`` file is at that path and is a valid coder.json file like the ``oats/config/coder.json``\n\nHere is the ``coder.js  on`` config dictionary:\n```\n{pp(config_dict)}\n```\n\nHere is the path to the current config:\n```\nexport CODER_CONFIG_FILE={coder_config_file}\n```\n'
        log.info(err_msg)
        raise Exception('Please fix the CODER_CONFIG_FILE for the logged error')
    if 'provider' not in config_dict:
        err_msg = f'### Sorry!!💥 The CODER_CONFIG_FILE is missing the ``provider`` root key. Please check the environment variable: ``CODER_CONFIG_FILE`` file is at that path and is a valid coder.json file like the ``oats/config/coder.json``\n\nHere is the ``coder.js  on`` config dictionary:\n```\n{pp(config_dict)}\n```\n\nHere is the path to the current config:\n```\nexport CODER_CONFIG_FILE={coder_config_file}\n```\n'
        log.info(err_msg)
        raise Exception('Please fix the CODER_CONFIG_FILE for the logged error')
    if provider_id not in config_dict['provider']:
        err_msg = f'### Sorry!!💥 The CODER_CONFIG_FILE is missing the provider_id: ``{provider_id}`` in the providers root key dictionary. Please check the environment variable: ``CODER_CONFIG_FILE`` file is at that path and is a valid coder.json file like the ``oats/config/coder.json``\n\nHere is the ``coder.json`` config dictionary:\n```\n{pp(config_dict)}\n```\n\nHere is the path to the current config:\n```\nexport CODER_CONFIG_FILE={coder_config_file}\n```\n'
        log.info(err_msg)
        raise Exception('Please fix the CODER_CONFIG_FILE for the logged error')
    if 'models' not in config_dict['provider'][provider_id]:
        err_msg = f'### Sorry!!💥 The CODER_CONFIG_FILE is missing the ``models`` list in the provider_id: ``{provider_id}`` definition. Please check the environment variable: ``CODER_CONFIG_FILE`` file is at that path and is a valid coder.json file like the ``oats/config/coder.json``\n\nHere is the ``coder.json`` config dictionary:\n```\n{pp(config_dict)}\n```\n\nHere is the path to the current config:\n```\nexport CODER_CONFIG_FILE={coder_config_file}\n```\n'
        log.info(err_msg)
        raise Exception('Please fix the CODER_CONFIG_FILE for the logged error')
    found_model = False
    if 'base_url' not in config_dict['provider'][provider_id]:
        err_msg = f'### Sorry!!💥 The CODER_CONFIG_FILE is missing a ``base_url`` for the provider_id: ``{provider_id}`` to reach the backend ai service. Please check the environment variable: ``CODER_CONFIG_FILE`` file is at that path and is a valid coder.json file like the ``oats/config/coder.json``\n\nHere is the ``coder.json`` config dictionary:\n```\n{pp(config_dict)}\n```\n\nHere is the path to the current config:\n```\nexport CODER_CONFIG_FILE={coder_config_file}\n```\n'
        log.info(err_msg)
        raise Exception('Please fix the CODER_CONFIG_FILE for the logged error')
    for model_node in config_dict['provider'][provider_id]['models']:
        if 'name' not in model_node:
            err_msg = f'### Sorry!!💥 The CODER_CONFIG_FILE provider_id: ``{provider_id}`` is missing a valid model dictionary in the ``models`` list.\n\nPlease check the environment variable: ``CODER_CONFIG_FILE`` file is at that path and is a valid coder.json file like the ``oats/config/coder.json``\n\nHere is the ``coder.json`` config dictionary:\n```\n{pp(config_dict)}\n```\n\nHere is the path to the current config:\n```\nexport CODER_CONFIG_FILE={coder_config_file}\n```\n'
            log.info(err_msg)
            raise Exception('Please fix the CODER_CONFIG_FILE for the logged error')
        else:
            if model_node['name'] == model_id:
                found_model = True
    if not found_model:
        err_msg = f'### Sorry!!💥 The CODER_CONFIG_FILE provider_id: ``{provider_id}`` does not have the model_id: ``{model_id}`` in the ``models`` list.\n\nPlease check the environment variable: ``CODER_CONFIG_FILE`` file is at that path and is a valid coder.json file like the ``oats/config/coder.json``\n\nHere is the ``coder.json`` config dictionary:\n```\n{pp(config_dict)}\n```\n\nHere is the path to the current config:\n```\nexport CODER_CONFIG_FILE={coder_config_file}\n```\n'
        log.info(err_msg)
        raise Exception('Please fix the CODER_CONFIG_FILE for the logged error')
    if verbose:
        log.info(f'validated CODER_CONFIG_FILE for chat with provider_id: {provider_id} with model_id: {model_id}')
    return True

def _todo_reminder_turn_counts(messages, last_reminder_turn: int) -> tuple[int, int | None, int | None]:
    """Return (next_turn, turns_since_todowrite, turns_since_reminder).

    Gap values are None when the event has never occurred — caller treats
    that as "no nudge yet" rather than "nudge immediately".
    """
    assistant_turn = 0
    last_todowrite_turn: int | None = None
    for msg in messages:
        if msg.role != "assistant":
            continue
        assistant_turn += 1
        for tc in msg.get_tool_calls():
            if tc.tool_name == "todowrite":
                last_todowrite_turn = assistant_turn
    next_turn = assistant_turn + 1
    gap_write = (next_turn - last_todowrite_turn) if last_todowrite_turn is not None else next_turn
    gap_reminder = (next_turn - last_reminder_turn) if last_reminder_turn > 0 else next_turn
    return next_turn, gap_write, gap_reminder


import re

_THINK_RE = re.compile(r'<think>.*?</think>\s*', re.DOTALL)


def _strip_thinking(content: str) -> str:
    """Remove <think>...</think> blocks from model output before storing.

    The model still generates thinking tokens (better decisions), but we
    don't carry them in conversation history across iterations. This saves
    ~500-1500 tokens per agentic turn — critical on a 32K context window.
    """
    if not content or '<think>' not in content:
        return content
    stripped = _THINK_RE.sub('', content).strip()
    return stripped if stripped else content  # fallback: don't store empty


_SPECIAL_TOKEN_RE = _re.compile(r'<[|/]?[\w"|]+[|]?>')


def _parse_tool_args(raw: str) -> dict:
    """Parse tool call arguments JSON with a repair fallback.

    First tries a direct json.loads. If that fails (e.g. residual model tokens
    that slipped past provider sanitization), strips anything that looks like
    an angle-bracket special token and retries. Returns {} as last resort so
    the tool can report a clear error rather than crashing the iteration.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    # Repair: strip residual special tokens and retry
    cleaned = _SPECIAL_TOKEN_RE.sub('', raw)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        log.warn(f"unparseable tool arguments after repair: {repr(raw[:200])}")
        return {}


class ApprovalDecision:
    """Return value from an approval callback.

    ``approved=False`` skips the tool with a structured error fed back to the
    model so it can adapt. ``instructions`` (optional) is appended to that
    error so the user can redirect the agent without typing a new turn.
    ``approve_all=True`` flips the session to auto-approve for the rest of
    the turn (the CLI's ApprovalManager handles persistence across turns).
    """
    __slots__ = ("approved", "instructions", "approve_all")

    def __init__(self, approved: bool, instructions: str | None = None, approve_all: bool = False):
        self.approved = approved
        self.instructions = instructions
        self.approve_all = approve_all


def _all_concurrency_safe(tool_calls: list) -> bool:
    """Check if every tool in a batch is safe for parallel execution."""
    for tc in tool_calls:
        tool = get_tool(tc.name)
        if tool is None:
            return False
        if tool.is_concurrency_safe():
            continue
        # Also check the static set as fallback
        if tc.name in _CONCURRENCY_SAFE_TOOLS:
            continue
        return False
    return True


class SessionProcessor:
    """
    Processes messages in a session, handling LLM interactions and tool execution.
    """

    def __init__(self, session: Session, agent_depth: int = 0) -> None:
        self.session = session
        self._storage = get_session_storage()
        self._agent_depth = agent_depth
        self._active_tool_names: set[str] = set()
        # Index of the assistant turn at which we last injected a todo
        # reminder (mirrors claude-code's TODO_REMINDER_CONFIG cadence).
        self._last_todo_reminder_turn: int = -10**9
        # Fires once, lazily, on the first process_message call — so hooks
        # that need the session to exist (storage, logger, etc.) are safe.
        self._session_start_fired = False

        self.local_tools = []
        self.local_tool_data = {}
        self.local_tool_impls = {}

        self.tool_api_url = os.getenv('TOOL_API_URL', 'http://0.0.0.0:20700/v1')
        self.tool_api_key = os.getenv('TOOL_API_KEY', 'CHANGE_PASSWORD')

        # Initialize file state cache
        self._file_cache = FileStateCache()
        self.stm = SelectedToolsManifest()
        self.oats_repo_uses_tools = OatPromptChoices()

        self.config = get_config()
        # Initialize hook engine from config
        try:
            hook_entries = [h.model_dump() for h in self.config.hooks.hooks] if self.config.hooks.hooks else []
            self._hook_engine = HookEngine(hook_entries)
        except Exception:
            log.warning(f"### Sorry!! hit_error_with_hook_entries\n```\n{traceback.format_exc()}\n```\n")
            self._hook_engine = HookEngine([])

        # Install trajectory logger — idempotent, gated on the feature flag.
        # Done here (not only in interactive.py) so oc run, oweb, and cron all
        # populate the trajectory store when the flag is on.
        try:
            from oats.trajectory.logger import install as _install_traj_logger
            _install_traj_logger()
        except Exception:
            log.warning(f"### Sorry!! hit_error_with_install_traj_logger\n```\n{traceback.format_exc()}\n```\n")
            pass

    async def process_message(
        self,
        content: str,
        auto_approve_tools: bool = False,
        max_tokens: Optional[int] = None,
        images: list[dict[str, str]] | None = None,
        approval_callback: Optional[
            Callable[[str, dict[str, Any], str], Awaitable["ApprovalDecision"]]
        ] = None,
        needs_local_tools: bool = True,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Process a user message and generate responses.

        *images* is an optional list of dicts with keys media_type, data, url
        that will be attached to the user message for vision-capable models.

        Yields events during processing for streaming updates.
        """
        # Fire session_start exactly once per processor lifetime. Deferred to
        # first process_message (rather than __init__) so handlers see a
        # fully-initialized session + storage, and sub-agent processors still
        # get the event even though they're constructed mid-turn.
        # Get provider and model
        config = get_config()
        provider_id = self.session.info.provider_id or config.model.provider_id
        model_id = self.session.info.model_id or config.model.model_id
        validate_coder_env(config=config, provider_id=provider_id, model_id=model_id)

        if not self._session_start_fired:
            self._session_start_fired = True
            try:
                await self._hook_engine.fire(
                    HookEvent.SESSION_START,
                    HookContext(
                        session_id=self.session.id,
                        event=HookEvent.SESSION_START,
                        working_dir=self.session.info.working_dir,
                    ),
                )
            except Exception as e:
                # Never let a misbehaving SESSION_START handler block the session.
                log.info(f"### Sorry!! Failed_process_message_hit_error:\n\n```\n{traceback.format_exc()}\n```\n")
                print(f"### Sorry!! Failed_process_message_hit_error:\n\n```\n{traceback.format_exc()}\n```\n")

        # Fire user_prompt_submit hook
        hook_ctx = HookContext(
            session_id=self.session.id,
            event=HookEvent.USER_PROMPT_SUBMIT,
            user_prompt=content,
            working_dir=self.session.info.working_dir,
        )
        hook_result = await self._hook_engine.fire(HookEvent.USER_PROMPT_SUBMIT, hook_ctx)
        if hook_result.action == "block":
            yield {
                "type": "error",
                "error": f"Blocked by hook: {hook_result.message or 'user_prompt_submit hook denied'}",
            }
            return

        # Add user message (with optional images for vision)
        user_message = self.session.create_user_message(content, images=images)
        await self._save()

        yield {
            "type": "user_message",
            "message_id": user_message.id,
            "content": content,
        }

        tool_provider_id = 't1'

        tool_provider = None
        try:
            provider = get_provider(provider_id)
        except ValueError as e:
            log.info(f'### Sorry!! Failed to get provider_id: {provider_id} - please check your coder.json file')
            yield {"type": "error", "error": str(e)}
            return
        if needs_local_tools:
            try:
                tool_provider = get_provider(tool_provider_id)
            except ValueError as e:
                if verbose:
                    log.info(f'warning - no tool_provider_set: {tool_provider_id}')
                # log.info(f'### Sorry!! Failed to get local_tool server with tool_provider_id: {tool_provider_id} - please check your coder.json file')
                # yield {"type": "error", "error": str(e)}
                # return

        # Build tool definitions — intent-aware selection
        # log.debug('----\nTool Selection Start\n')
        # print(content)
        # print('getting tools:')
        self.local_tool_impls = {}
        # select_tools_for_prompt(content, project_dir=Path(self.session.info.project_dir), needs_local_tools=needs_local_tools)
        self.stm = select_tools_for_prompt(content, project_dir=self.session.info.project_dir, needs_local_tools=needs_local_tools)

        # oat enabled by default
        self.oat_enabled = os.getenv('OATS_ENABLED', '0') == '0'
        self.oats_repo_uses_tools = OatPromptChoices()

        selected_tools = self.stm.core_tools
        selected_tool_names = self.stm.core_tool_names
        selected_local_tools = self.stm.local_tools
        selected_local_tool_names = self.stm.local_tool_names
        selected_mcp_tools = self.stm.mcp_tools
        selected_mcp_tool_names = self.stm.mcp_tool_names

        if self.oat_enabled:
            try:
                self.oats_repo_uses_tools = agent_get_tool_choices_for_prompt(prompt=content, top_k=5, verbose=verbose)
                if verbose:
                    log.info(f'## OAT Repo Tools\n\n```\n{pp(self.oats_repo_uses_tools.model_dump(mode='json'))}\n```\n')
            except Exception:
                log.info(f'### Sorry!! oats_prompt_tool_resolution_failed_with_errorr:\n```\n{traceback.format_exc()}\n```\n')

        local_impls = self.stm.local_impls
        num_core_tools = len(selected_tools)
        num_local_tools = len(selected_local_tools)
        num_mcp_tools = len(selected_mcp_tools)
        if verbose:
            log.info(f'# SelectedToolsManifest\n\nprompt:\n```\n{content[0:200]}\n```\ncore_tools:\n```\n{selected_tool_names}\n```\nlocal_tools:\n```\n{selected_local_tool_names}\n```\nmcp_tools:\n```\n{selected_mcp_tool_names}\n```\n')
        # log.info(f'# SelectedToolsManifest\n\nprompt:\n```\n{content[0:200]}\n```\nnum_core_tools: {num_core_tools}\nnum_local_tools: {num_local_tools}\nnum_mcp_tools: {num_mcp_tools}')
        if num_core_tools == 0 and num_local_tools == 0 and num_mcp_tools == 0:
            err = f'### Sorry!!! Failed_to_find_any_tools_to_process_prompt:\n```\n{content[0:10000]}\n```\n'
            log.info('# ERROR_DETECTED_TOOL_SELECTION')
            log.info(err)
            return
        # log.debug('Tool Selection End\n----')
        tool_definitions = []
        tool_names = []
        tool_model_id = 'openai/google/functiongemma-270m-it'
        if num_core_tools > 0:
            if verbose:
                log.info(f'loading_core_tools: {num_core_tools}')
            for t in selected_tools:
                new_tool_def = None
                # handle coder internal oat mappings
                new_tool_def = ToolDefinition(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters,
                    strict=t.strict,
                )
                tool_definitions.append(new_tool_def)
                self._active_tool_names.add(new_tool_def.name)
                tool_names.append(new_tool_def.name)
            if verbose:
                log.info(f'using_core_tools: {self._active_tool_names} model_id: {provider_id}@{model_id}')
        if num_local_tools > 0:
            if verbose:
                log.debug(f'loading_local_tools: {needs_local_tools} oat_enabled: {self.oat_enabled}')
            """
            for impl in local_impls:
                print(impl)
                print(local_impls[impl])
                print('-----')
            """
            if verbose:
                print(selected_local_tools)
                for lidx, local_tool in enumerate(selected_local_tools):
                    log.debug('local_tool {lidx + 1}/{num_local_tools}:')
                    print(local_tool)
                    print(local_tool.name)
                    print(local_tool.description)
                    print(local_tool.parameters)
                    print(local_tool.strict)
                    print('-----')
                print('\n\n\n')
                log.info('# best tools:')
                for best_tool in self.stm.best_tools:
                    print(best_tool)
                print('\n\n\n')
                log.info('# best tool impls:')
                for best_fn_name in self.stm.best_impls:
                    print(best_fn_name)
                    print(self.stm.best_impls[best_fn_name])
                    print('-----')
            self.local_tool_impls = local_impls
            for t in selected_local_tools:
                new_tool_def = None
                # handle coder internal oat mappings
                new_tool_def = ToolDefinition(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters,
                    strict=t.strict,
                )
                tool_definitions.append(new_tool_def)
                self._active_tool_names.add(new_tool_def.name)
                tool_names.append(new_tool_def.name)
                self.local_tools.append(new_tool_def)
                self.local_tool_data[t.name] = new_tool_def
            if verbose:
                log.info(f'using_local_tools: {self._active_tool_names} model_id: {tool_provider_id}@{tool_model_id} oat_enabled: {self.oat_enabled}')

        trace_event(
            self.session.id,
            "tool_selection",
            {
                "prompt_preview": content[:240],
                "selected_tool_names": tool_names,
                "selected_tool_count": len(tool_names),
            },
        )

        # Initialize conversation compactor
        ctx_len = int(os.getenv('CODER_CTX_LEN', '65536'))
        compactor = ConversationCompactor(
            model_context_length=ctx_len,
            provider_id=provider_id,
            model_id=model_id,
        )
        budget_manager = SessionTokenBudget(context_window=ctx_len)

        # Agent loop - keep processing until no more tool calls
        max_iterations = int(os.getenv("CODER_MAX_ITERATIONS", "150"))
        task_budget = SessionTaskBudget()
        iteration = 0
        _recent_responses: list[str] = []  # Track recent outputs for loop detection
        # Self-improvement metrics — counted across the agent loop, flushed on exit.
        _tool_error_count = 0
        _turn_started_at = asyncio.get_event_loop().time()
        _turn_completed_cleanly = False

        while iteration < max_iterations:
            iteration += 1

            # Check if conversation compaction is needed
            if True and compactor.should_compact(self.session.messages):
                self.session.messages = await compactor.compact(
                    self.session.messages, self.session.id
                )
                await self._save()
                await bus.publish(
                    Event(
                        type=EventType.SESSION_COMPACTED,
                        data={
                            "session_id": self.session.id,
                            "reason": "proactive",
                            "iteration": iteration,
                        },
                    )
                )
                trace_event(
                    self.session.id,
                    "session.compacted",
                    {
                        "reason": "proactive",
                        "iteration": iteration,
                        "message_count": len(self.session.messages),
                    },
                )
                yield {
                    "type": "compaction",
                    "message": "Conversation compacted to stay within context limits",
                }

            # Build messages for LLM
            budget_snapshot = None
            if token_budget_candidate_enabled():
                budget_snapshot = budget_manager.snapshot(
                    self.session.messages,
                    requested_max_tokens=max_tokens,
                )
            task_snapshot = None
            if task_budget_candidate_enabled():
                task_snapshot = task_budget.snapshot(iteration)
                await bus.publish(
                    Event(
                        type=EventType.SESSION_TASK_BUDGET,
                        data={
                            "session_id": self.session.id,
                            "iteration": task_snapshot.iteration,
                            "max_iterations": task_snapshot.max_iterations,
                            "tool_calls": task_snapshot.tool_calls,
                            "max_tool_calls": task_snapshot.max_tool_calls,
                            "repeated_tool_streak": task_snapshot.repeated_tool_streak,
                            "pressure": task_snapshot.pressure,
                            "should_stop": task_snapshot.should_stop,
                        },
                    )
                )
                yield {
                    "type": "task_budget",
                    "iteration": task_snapshot.iteration,
                    "max_iterations": task_snapshot.max_iterations,
                    "tool_calls": task_snapshot.tool_calls,
                    "max_tool_calls": task_snapshot.max_tool_calls,
                    "repeated_tool_streak": task_snapshot.repeated_tool_streak,
                    "pressure": task_snapshot.pressure,
                    "should_stop": task_snapshot.should_stop,
                }
                trace_event(
                    self.session.id,
                    "session.task_budget",
                    {
                        "iteration": task_snapshot.iteration,
                        "tool_calls": task_snapshot.tool_calls,
                        "repeated_tool_streak": task_snapshot.repeated_tool_streak,
                        "pressure": task_snapshot.pressure,
                        "should_stop": task_snapshot.should_stop,
                    },
                )
                if task_snapshot.should_stop and iteration > 1:
                    if not task_budget._committed:
                        task_budget.commit(iteration)
                        task_snapshot = task_budget.snapshot(iteration)
                        yield {
                            "type": "warning",
                            "message": (
                                "Discovery budget exhausted — entering commit mode. "
                                "No more exploration; finalizing with gathered context."
                            ),
                        }
                    else:
                        yield {
                            "type": "warning",
                            "message": "Task budget exhausted after commit extension; stopping.",
                        }
                        await self._save()
                        return

            messages = await self._build_messages(
                budget_snapshot=budget_snapshot,
                task_snapshot=task_snapshot,
            )

            if verbose:
                log.info(f'### Iteration: {iteration} messages: {len(messages)} model_id: {model_id} provider_id: {provider_id}')
            # end of large_model_inference

            # Create completion request
            request = CompletionRequest(
                messages=messages,
                model=model_id,
                provider_id=provider_id,
                tools=tool_definitions,
                temperature=0.2,
                stop=["<|im_start|>", "<|endoftext|>"],
                max_tokens=(
                    budget_snapshot.recommended_max_output_tokens
                    if budget_snapshot is not None
                    else max_tokens
                ),
                debug_context={"session_id": self.session.id, "iteration": iteration},
            )

            if budget_snapshot is not None:
                await bus.publish(
                    Event(
                        type=EventType.SESSION_BUDGET,
                        data={
                            "session_id": self.session.id,
                            "iteration": iteration,
                            "estimated_input_tokens": budget_snapshot.estimated_input_tokens,
                            "context_window": budget_snapshot.context_window,
                            "remaining_tokens": budget_snapshot.remaining_tokens,
                            "pressure": budget_snapshot.pressure,
                            "recommended_max_output_tokens": budget_snapshot.recommended_max_output_tokens,
                        },
                    )
                )
                yield {
                    "type": "budget",
                    "estimated_input_tokens": budget_snapshot.estimated_input_tokens,
                    "context_window": budget_snapshot.context_window,
                    "remaining_tokens": budget_snapshot.remaining_tokens,
                    "pressure": budget_snapshot.pressure,
                    "recommended_max_output_tokens": budget_snapshot.recommended_max_output_tokens,
                }
                trace_event(
                    self.session.id,
                    "session.budget",
                    {
                        "iteration": iteration,
                        "estimated_input_tokens": budget_snapshot.estimated_input_tokens,
                        "remaining_tokens": budget_snapshot.remaining_tokens,
                        "pressure": budget_snapshot.pressure,
                        "recommended_max_output_tokens": budget_snapshot.recommended_max_output_tokens,
                    },
                )

            # ── LLM call ───────────────────────────────────────────
            yield {"type": "llm_request", "iteration": iteration}

            use_non_streaming = should_disable_streaming(model_id)

            if verbose:
                log.info(f'##### start_chat: {len(messages)} model: {provider_id}@{model_id} stream: {not use_non_streaming}')

            if use_non_streaming:
                # ── Non-streaming path ─────────────────────────────
                # For models where vLLM's streaming tool parser corrupts
                # arguments (e.g. Gemma 4 leaking <|"|> tokens). Direct
                # API calls produce valid tool call JSON consistently.
                try:
                    response = await provider.complete(request)
                    response_content = response.content or ""
                    response_tool_calls = response.tool_calls or []
                    response_usage = response.usage

                    # Emit the full text as a single delta so downstream
                    # consumers (UI, pipes) still receive it.
                    if response_content:
                        yield {
                            "type": "assistant_text_delta",
                            "content": response_content,
                        }
                except Exception as e:
                    log.info(f"### Sorry!! Failed_provider.complete(request):\n\n```\n{traceback.format_exc()}\n```\n")
                    if await self._try_reactive_compaction(str(e), compactor):
                        await self._save()
                        yield {
                            "type": "compaction",
                            "message": "Conversation compacted reactively after context overflow",
                        }
                        continue
                    yield {"type": "error", "error": f"LLM error: {e}"}
                    return
            else:
                # ── Streaming path ─────────────────────────────────
                # Stream text chunks as they arrive from the model in
                # real-time (critical for 32B model on single GPU where
                # generation can take 10-30s).
                try:
                    response_content = ""
                    response_tool_calls = []
                    response_usage = None
                    streamed_any = False

                    async for chunk in provider.stream(request):
                        streamed_any = True

                        # Capture usage from the final chunk
                        if chunk is not None and hasattr(chunk, 'usage') and chunk.usage:
                            response_usage = chunk.usage

                        # Yield text chunks in real-time
                        if chunk.content:
                            response_content += chunk.content
                            yield {
                                "type": "assistant_text_delta",
                                "content": chunk.content,
                            }

                        # Collect tool calls
                        if chunk.tool_calls:
                            response_tool_calls.extend(chunk.tool_calls)

                    if not streamed_any:
                        yield {"type": "error", "error": "Empty response from LLM"}
                        return

                except Exception as e:
                    # Fall back to non-streaming on stream failure
                    log.info(f"### Sorry!! streaming_failed_check_auth_errors_trying_non_streaming_http:\n\n```\n{traceback.format_exc()}\n```\n")
                    try:
                        response = await provider.complete(request)
                        response_content = response.content or ""
                        response_tool_calls = response.tool_calls or []
                        response_usage = response.usage
                    except Exception as e2:
                        full_err = str(traceback.format_exc())
                        coder_config = os.getenv('CODER_CONFIG_FILE', None)
                        err_msg = f'### Sorry!! Failed chat with CODER_CONFIG_FILE: {coder_config} and then run:\n```\nexport CODER_CONFIG_FILE=PATH/coder.json\n```\n\nFailed chat request with provider: ``{provider_id}``\nvllm_api_url:\n```\n{provider.config.base_url}\n```\n\nwith error:\n```\n{full_err}\n```\n'
                        if 'Hosted_vllmException - {"error":"Unauthorized"}' in full_err:
                            err_msg = f'### Sorry!! Please set up a local CODER_CONFIG_FILE:\n```{coder_config}\n```\nand then run:\n```\nexport CODER_CONFIG_FILE=PATH/coder.json\n```\n\nFailed chat request with provider: ``{provider_id}``\nvllm_api_url:\n```\n{provider.config.base_url}\n```\n\nNo local vllm models are configured correctly for chat at this time.'
                            if coder_config is not None:
                                err_msg = f'### Sorry!! Please set up a local CODER_CONFIG_FILE:\n```\n{coder_config}\n```\nAfter setting up the CODER_CONFIG_FILE then run:\n```\nexport CODER_CONFIG_FILE=PATH/coder.json\n```\n\nFailed chat request with provider: ``{provider_id}``\nvllm_api_url:\n```\n{provider.config.base_url}\n```\n\nNo local vllm models are configured correctly for chat at this time.'
                        elif 'Cannot connect to host' in str(e2):
                            err_msg = f'### Sorry!! Had connection failure reaching provider_id: ``{provider_id}`` with error:\n```\n{full_err}\n```\n'
                        if await self._try_reactive_compaction(str(e2), compactor):
                            await self._save()
                            yield {
                                "type": "compaction",
                                "message": "Conversation compacted reactively after context overflow",
                            }
                            continue
                        log.info(err_msg)
                        raise Exception('Please review the provider chat error log above to fix this issue')
                        yield {"type": "error", "error": f"LLM error: {e2}"}
                        return

            # ── Strip thinking tokens before storing ────────────
            # Qwen3.5 thinking mode wraps reasoning in <think>...</think>.
            # The model benefits from generating these (better decisions),
            # but storing them in conversation history wastes context across
            # iterations. Strip them so only the actionable content persists.
            stored_content = _strip_thinking(response_content)

            # ── Loop / hallucination detection ───────────────────
            # If the model produces near-identical output 3 times in a
            # row, it's stuck. Break out rather than burn GPU cycles.
            response_sig = f"{stored_content[:200] if stored_content else ''}|{'|'.join(tc.name for tc in response_tool_calls)}"
            _recent_responses.append(response_sig)
            if len(_recent_responses) > 20:
                _recent_responses.pop(0)
            if len(_recent_responses) == 20 and len(set(_recent_responses)) == 1:
                log.warn("loop detected: model produced identical output 3 times, breaking")
                yield {
                    "type": "warning",
                    "message": "Loop detected - model is repeating itself. Stopping.",
                }
                await self._save()
                return

            # Create assistant message
            assistant_message = self.session.create_assistant_message()

            # Add text content if present (stored_content has thinking stripped)
            if stored_content:
                assistant_message.add_text(stored_content)
                yield {
                    "type": "assistant_text",
                    "message_id": assistant_message.id,
                    "content": response_content,  # stream full content to user (with thinking)
                }

            # Track usage
            if response_usage:
                self.session.add_usage(response_usage)
                yield {"type": "usage", "usage": response_usage}

            # Check for tool calls
            if not response_tool_calls:
                if response_content:
                    if verbose:
                       log.info('##### Done Thinking - {tk.tracking_id if tk else ""}\n\n---\n\n{response_content[0:1000]}\n\n---')
                # No more tool calls, we're done
                await self._save()
                _turn_completed_cleanly = True
                self._log_turn_outcome(
                    turn_started_at=_turn_started_at,
                    iterations=iteration,
                    tool_error_count=_tool_error_count,
                    completed=_turn_completed_cleanly,
                    model_id=model_id,
                )
                if response_content:
                    await self._hook_engine.fire(
                        HookEvent.ASSISTANT_RESPONSE,
                        HookContext(
                            session_id=self.session.id,
                            event=HookEvent.ASSISTANT_RESPONSE,
                            assistant_response=response_content,
                            working_dir=self.session.info.working_dir,
                            root_session_id=self.session.info.root_session_id or self.session.id,
                        ),
                    )
                yield {"type": "complete", "message_id": assistant_message.id}
                return

            # Drop exact duplicates the model emitted within the same turn
            # (same name + same normalized args). Some models repeat identical
            # tool calls in parallel, which wastes the iteration/tool budget.
            if len(response_tool_calls) > 1:
                _seen: set[tuple[str, str]] = set()
                _deduped = []
                _dropped = 0
                for tc in response_tool_calls:
                    key = (tc.name, json.dumps(_parse_tool_args(tc.arguments), sort_keys=True, ensure_ascii=True))
                    if key in _seen:
                        _dropped += 1
                        continue
                    _seen.add(key)
                    _deduped.append(tc)
                if _dropped:
                    log.info(f"deduped {_dropped} duplicate tool call(s) in turn")
                    response_tool_calls = _deduped

            # ── Concurrent tool execution ───────────────────────────
            # If ALL tool calls in the batch are concurrency-safe
            # (read-only), run them in parallel. Otherwise run
            # sequentially to avoid conflicting side effects.

            use_concurrent = (
                len(response_tool_calls) > 1
                and _all_concurrency_safe(response_tool_calls)
            )

            if use_concurrent:
                # Register all tool calls on the assistant message first
                for tool_call in response_tool_calls:
                    parsed_args = _parse_tool_args(tool_call.arguments)
                    if verbose:
                        log.info(f'#### Response Tool Call - {tool_call.name}\nTool_args:\n```\n{parsed_args}\n```\n')
                    assistant_message.add_tool_call(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        arguments=parsed_args,
                    )
                    yield {
                        "type": "tool_call",
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.name,
                        "arguments": parsed_args,
                    }

                # Run all concurrently
                log.info(f"running {len(response_tool_calls)} tool(s) concurrently")
                tasks = []
                for tc in response_tool_calls:
                    new_tool_task = self._execute_tool(
                        tool_name=tc.name,
                        tool_call_id=tc.id,
                        arguments=_parse_tool_args(tc.arguments),
                        auto_approve=auto_approve_tools,
                        needs_local_tools=needs_local_tools,
                        verbose=verbose,
                        approval_callback=approval_callback,
                    )
                    tasks.append(new_tool_task)
                for tc in response_tool_calls:
                    if task_budget_candidate_enabled():
                        task_budget.record_tool_call(tc.name, _parse_tool_args(tc.arguments))
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for tool_call, result in zip(response_tool_calls, results):
                    if isinstance(result, Exception):
                        log.info(f"### Sorry!! Hit tool_result_error: {tool_call.name} with error:\n\n```\n{str(result)}\n```\n")
                        result = ToolResult(
                            title=f"Error in {tool_call.name}",
                            output="",
                            error=str(result),
                        )
                    retained = (
                        retain_tool_result(tool_call.name, result)
                        if result_retention_enabled()
                        else None
                    )
                    if retained is not None and retained.metadata.get("retention_applied"):
                        trace_event(
                            self.session.id,
                            "tool_result.retained",
                            {
                                "iteration": iteration,
                                "tool_name": tool_call.name,
                                "original_output_chars": retained.metadata.get("original_output_chars"),
                                "retained_output_chars": retained.metadata.get("retained_output_chars"),
                            },
                        )

                    assistant_message.add_tool_result(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        title=result.title,
                        output=retained.output if retained is not None else result.output,
                        error=result.error,
                        metadata=retained.metadata if retained is not None else result.metadata,
                    )
                    if result.error:
                        _tool_error_count += 1
                    yield {
                        "type": "tool_result",
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.name,
                        "title": result.title,
                        "output": result.output[:500] if result.output else "",
                        "error": result.error,
                        "metadata": result.metadata or {},
                    }
                    tool_definitions = self._maybe_promote_tools(tool_definitions, tool_call.name, result)

            else:
                # Sequential execution (original path)
                for tool_call in response_tool_calls:
                    parsed_args = _parse_tool_args(tool_call.arguments)
                    assistant_message.add_tool_call(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        arguments=parsed_args,
                    )

                    new_tool_dict = {
                        "type": "tool_call",
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.name,
                        "arguments": parsed_args,
                    }

                    yield new_tool_dict

                    set_tools = False
                    if verbose:
                        log.info(f'## Thinking {iteration} - Tool - {tool_call.name} - Start\n\n{tool_call.arguments}\n\n')

                    if task_budget_candidate_enabled():
                        task_budget.record_tool_call(tool_call.name, parsed_args)

                    # Execute the tool
                    result = await self._execute_tool(
                        tool_name=tool_call.name,
                        tool_call_id=tool_call.id,
                        arguments=parsed_args,
                        auto_approve=auto_approve_tools,
                        needs_local_tools=needs_local_tools,
                        verbose=verbose,
                        approval_callback=approval_callback,
                    )
                    retained = (
                        retain_tool_result(tool_call.name, result)
                        if result_retention_enabled()
                        else None
                    )
                    if retained is not None and retained.metadata.get("retention_applied"):
                        trace_event(
                            self.session.id,
                            "tool_result.retained",
                            {
                                "iteration": iteration,
                                "tool_name": tool_call.name,
                                "original_output_chars": retained.metadata.get("original_output_chars"),
                                "retained_output_chars": retained.metadata.get("retained_output_chars"),
                            },
                        )

                    # Add tool result to the assistant message
                    assistant_message.add_tool_result(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        title=result.title,
                        output=retained.output if retained is not None else result.output,
                        error=result.error,
                        metadata=retained.metadata if retained is not None else result.metadata,
                    )
                    if result.error:
                        _tool_error_count += 1
                    if result.output is not None:
                        if verbose:
                            log.info(f'\n## Thinking {iteration} - Tool - {tool_call.name} - Logs\n\n```\n{result.output[-9000:]}\n```\n')
                    if result.error is not None:
                        log.info(f'\n### Thinking {iteration} - Tool - {tool_call.name} - Error :mag: :bomb:\n\n```\n{result.error[-9000:]}\n```\n')

                    yield {
                        "type": "tool_result",
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.name,
                        "title": result.title,
                        "output": result.output[:500] if result.output else "",
                        "error": result.error,
                        "metadata": result.metadata or {},
                    }
                    tool_definitions = self._maybe_promote_tools(tool_definitions, tool_call.name, result)

            await self._save()

            # Publish event
            await bus.publish(
                Event(
                    type=EventType.TOOL_COMPLETE,
                    data={
                        "session_id": self.session.id,
                        "iteration": iteration,
                    },
                )
            )

        # Reached max iterations — turn did not complete cleanly.
        self._log_turn_outcome(
            turn_started_at=_turn_started_at,
            iterations=iteration,
            tool_error_count=_tool_error_count,
            completed=False,
            model_id=model_id,
        )
        yield {
            "type": "warning",
            "message": f"Reached maximum iterations ({max_iterations})",
        }

    async def _try_reactive_compaction(
        self,
        error_text: str,
        compactor: ConversationCompactor,
    ) -> bool:
        """
        Lightweight reactive compact fallback inspired by Claude Code.

        If the provider still rejects the request as too large, compact and
        retry once through the normal loop instead of immediately failing.
        """
        if not reactive_compaction_candidate_enabled():
            return False

        text = error_text.lower()
        too_long_signals = [
            "prompt too long",
            "context length",
            "maximum context",
            "too many tokens",
            "context window",
            "413",
        ]
        if not any(signal in text for signal in too_long_signals):
            return False
        if len(self.session.messages) <= 4:
            return False

        self.session.messages = await compactor.compact(
            self.session.messages,
            self.session.id,
        )
        await bus.publish(
            Event(
                type=EventType.SESSION_COMPACTED,
                data={
                    "session_id": self.session.id,
                    "reason": "reactive",
                },
            )
        )
        trace_event(
            self.session.id,
            "session.compacted",
            {
                "reason": "reactive",
                "message_count": len(self.session.messages),
            },
        )
        log.warn("reactive_compaction_triggered")
        return True

    async def _execute_tool(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
        auto_approve: bool = False,
        needs_local_tools: bool = False,
        verbose: bool = False,
        approval_callback: Optional[
            Callable[[str, dict[str, Any], str], Awaitable[ApprovalDecision]]
        ] = None,
    ) -> ToolResult:
        """Execute a tool and return the result."""
        tool = get_tool(tool_name)

        if tool is None:
            return ToolResult(
                title=f"Unknown tool: {tool_name}",
                output="",
                error=f"Tool '{tool_name}' not found",
            )

        use_provider_id = 'ow'
        if len(self.local_tool_data) > 0:
            if tool.name in self.local_tool_data:
                use_provider_id = 't1'


        # Create tool context
        ctx = ToolContext(
            session_id=self.session.id,
            project_dir=Path(self.session.info.project_dir),
            working_dir=Path(self.session.info.working_dir),
            user_confirmed=auto_approve,
            parent_session_id=self.session.info.parent_session_id,
            root_session_id=self.session.info.root_session_id or self.session.id,
            agent_depth=getattr(self, '_agent_depth', 0),
            file_cache=self._file_cache,
        )

        # Check permissions. If the tool flags a gated operation AND we're not
        # in auto-approve, route through the caller-supplied approval callback.
        # A declined call short-circuits with a structured ToolResult that the
        # model will see on its next turn — so the agent learns the user said
        # no and can adapt, rather than silently retrying.
        permission_needed = tool.requires_permission(arguments, ctx)

        # Plan-mode enforcement. If planning mode is active for this session,
        # any tool that requires permission (write/edit/bash/multiedit/…) is
        # blocked with a structured ToolResult. Read-only tools (read, grep,
        # glob, plan_exit, …) inherit the base `requires_permission -> None`
        # and pass through. This matches the advertised /plan semantics:
        # "you CANNOT modify files". The model sees the block and can adapt
        # — typically by asking the user to exit plan mode.
        if permission_needed:
            try:
                from oats.tool.plan import is_planning_mode
                if await is_planning_mode(self.session.id):
                    log.info(f"plan_mode_block tool={tool_name}")
                    return ToolResult(
                        title=f"Blocked in plan mode: {tool_name}",
                        output="",
                        error=(
                            "Plan mode is active — mutating tools are disabled. "
                            "Use plan_exit or /edit/auto to exit plan mode before "
                            "running this tool."
                        ),
                        metadata={"plan_mode_blocked": True, "tool_name": tool_name},
                    )
            except Exception as e:
                # Defensive — a broken plan subsystem must not block tool use.
                log.info(f"### Sorry!! Hit plan_mode_check_failed with error:\n\n```\n{str(traceback.format_exc())}\n```\n")

        if permission_needed and not auto_approve and approval_callback is not None:
            try:
                decision = await approval_callback(tool_name, arguments, permission_needed)
            except Exception as e:
                log.info(f"### Sorry!! Hit approval_callback_failed_with error:\n\n```\n{str(traceback.format_exc())}\n```\n")
                decision = ApprovalDecision(approved=False)
            if not decision.approved:
                msg = "user declined this tool call"
                if decision.instructions:
                    msg += f". User instructions: {decision.instructions}"
                return ToolResult(
                    title=f"Declined: {tool_name}",
                    output="",
                    error=msg,
                    metadata={"declined_by_user": True},
                )
            # Approved — record on context so tools can introspect if they care.
            ctx.user_confirmed = True

        # Fire pre_tool_use hook
        hook_ctx = HookContext(
            session_id=self.session.id,
            event=HookEvent.PRE_TOOL_USE,
            tool_name=tool_name,
            tool_args=arguments,
            working_dir=self.session.info.working_dir,
            root_session_id=self.session.info.root_session_id or self.session.id,
        )
        hook_result = await self._hook_engine.fire(HookEvent.PRE_TOOL_USE, hook_ctx)
        if hook_result.action == "block":
            return ToolResult(
                title=f"Blocked by hook: {tool_name}",
                output="",
                error=hook_result.message or f"pre_tool_use hook blocked {tool_name}",
            )
        if hook_result.action == "modify" and hook_result.modified_args is not None:
            original_args = arguments
            arguments = hook_result.modified_args
            # Re-approval integrity: if the user manually approved the original
            # args, a hook silently mutating them after approval creates a trust
            # mismatch (user saw X, tool runs Y). Re-prompt with the new args
            # when interactive approval is in play. Under auto-approve, just
            # log loudly — the user already consented to unattended execution.
            args_changed = original_args != arguments
            if args_changed and permission_needed:
                if not auto_approve and approval_callback is not None:
                    try:
                        decision = await approval_callback(
                            tool_name, arguments,
                            f"{permission_needed} (HOOK-MODIFIED — re-approve)",
                        )
                    except Exception as e:
                        log.info(f"### Sorry!! Hit reapproval_callback_failed_error:\n\n```\n{str(traceback.format_exc())}\n```\n")
                        decision = ApprovalDecision(approved=False)
                    if not decision.approved:
                        msg = "user declined modified call"
                        if decision.instructions:
                            msg += f". User instructions: {decision.instructions}"
                        return ToolResult(
                            title=f"Declined (hook-modified): {tool_name}",
                            output="",
                            error=msg,
                            metadata={
                                "declined_by_user": True,
                                "hook_modified": True,
                                "original_args": original_args,
                            },
                        )
                elif auto_approve:
                    log.warn(
                        f"hook_modified_args_under_auto tool={tool_name} "
                        f"from_keys={list(original_args.keys())} to_keys={list(arguments.keys())}"
                    )

        # Execute the tool
        await bus.publish(
            Event(
                type=EventType.TOOL_START,
                data={
                    "session_id": self.session.id,
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                },
            )
        )

        result = None
        try:
            if needs_local_tools and tool_name in self.stm.best_tool_names:
                if verbose:
                    log.info(f'### LocalTool: {tool_name}\ntool_api_url: {self.tool_api_url}\nargs:\n{pp(arguments)}\nself.stm.best_tool_names:\n```\n{self.stm.best_tool_names}\n```\nself.stm.best_tools:\n```\nself.stm.best_tools\n```\nbest_impls:\n```\n{self.stm.best_impls}\n```\n')
                run_tool_status, answer = run_tool_call(
                    prompt=f'{tool_name}({json.dumps(arguments)})',
                    api_base=self.tool_api_url,
                    api_key=self.tool_api_key,
                    tools=self.stm.best_tools,
                    tool_impls=self.stm.best_impls,
                )
                if run_tool_status:
                    if verbose:
                        log.info(f'# Result: {tool_name}\n\n```\n{answer}\n```\n')
                    result = ToolResult(title=f'Response for {tool_name}', output=answer)
                else:
                    result = ToolResult(title=f'Error using LocalTool {tool_name}', output=answer, error=f'Failed tool: {tool_name}')
            else:
                # log.info(f'#### Running_core_tool: {tool_name}')
                # log.info(f'### Running_core_tool: {tool_name}\nargs:\n{pp(arguments)}')
                # log.info(f'### Running_core_tool: {tool_name}\nargs:\n{pp(arguments)}\nself.stm.best_tool_names:\n```\n{self.stm.best_tool_names}\n```\nself.stm.best_tools:\n```\nself.stm.best_tools\n```\nbest_impls:\n```\n{self.stm.best_impls}\n```\n')
                result = await tool.execute(arguments, ctx)
            if verbose:
                log.info(f'# Tool Result:\n\n{result}\n')
        except Exception as e:
            from oats.date import utc
            now = utc().strftime('%Y%m%d%H%M%S')
            local_tool_debug_file = f'/tmp/debug_coder_tool_local_tools_{now}.json'
            err = f"### Sorry!! Failed tool: {tool_name} args: {pp(arguments)} execute with error:\n\n```\n{str(traceback.format_exc())}\n```\nurl:\n```\n{self.tool_api_url}\n```\napi_key:\n```\n{self.tool_api_key}\n```\nLocal Tool Definitions stored in the file:\n```\n{local_tool_debug_file}\n```\n"
            log.info(err)
            result = ToolResult(
                title=f"Error in {tool_name}",
                output="",
                error=err,
            )
            try:
                with open(local_tool_debug_file, 'w') as f:
                    f.write(str(self.local_tool_impls))
            except Exception:
                log.error(f'failed_to_save_local_tool_debug_file: {local_tool_debug_file}')
                pass
            raise e

        # Fire post_tool_use hook
        post_hook_ctx = HookContext(
            session_id=self.session.id,
            event=HookEvent.POST_TOOL_USE,
            tool_name=tool_name,
            tool_args=arguments,
            tool_result_output=result.output[:1000] if result.output else None,
            tool_result_error=result.error,
            working_dir=self.session.info.working_dir,
            root_session_id=self.session.info.root_session_id or self.session.id,
        )
        await self._hook_engine.fire(HookEvent.POST_TOOL_USE, post_hook_ctx)

        # Fire file_changed for file-modifying tools
        if tool_name in ("write", "edit", "multiedit", "patch") and not result.error:
            file_path = arguments.get("file_path") or arguments.get("path")
            if file_path:
                file_hook_ctx = HookContext(
                    session_id=self.session.id,
                    event=HookEvent.FILE_CHANGED,
                    tool_name=tool_name,
                    file_path=str(file_path),
                    working_dir=self.session.info.working_dir,
                    root_session_id=self.session.info.root_session_id or self.session.id,
                )
                await self._hook_engine.fire(HookEvent.FILE_CHANGED, file_hook_ctx)
                await bus.publish(
                    Event(
                        type=EventType.FILE_CHANGED,
                        data={"session_id": self.session.id, "file_path": str(file_path)},
                    )
                )

        return result

    def _log_turn_outcome(
        self,
        *,
        turn_started_at: float,
        iterations: int,
        tool_error_count: int,
        completed: bool,
        model_id: str | None,
    ) -> None:
        """Write the current turn's outcome into turn_metrics. Best-effort."""
        try:
            from oats.trajectory.logger import _counter_lock, _turn_counters
            from oats.trajectory.metrics import log_turn_outcome as _log
            with _counter_lock:
                # The prompt logger already bumped the counter; the turn we
                # just finished is the one before it.
                next_idx = _turn_counters.get(self.session.id, 0)
                turn_idx = max(0, next_idx - 1)
            duration_ms = int((asyncio.get_event_loop().time() - turn_started_at) * 1000)
            _log(
                session_id=self.session.id,
                turn_idx=turn_idx,
                iterations=iterations,
                tool_error_count=tool_error_count,
                completed=completed,
                duration_ms=duration_ms,
                model_id=model_id,
            )
        except Exception as e:
            err = f"### Sorry!! Failed log_turn_outcome with error:\n\n```\n{str(traceback.format_exc())}\n```\n"
            log.info(err)

    async def _build_messages(self, budget_snapshot=None, task_snapshot=None) -> list[ProviderMessage]:
        """Build messages for the LLM, including system prompt.

        Defensively reconciles any orphan tool_calls in the session's
        assistant messages before producing wire format — this catches the
        state left behind when a prior turn was cancelled mid-tool-dispatch
        or when a provider error interrupted the call/result pairing. Without
        this, the provider API would reject the next turn with
        "tool_use block without matching tool_result".
        """
        patched_total = 0
        for msg in self.session.messages:
            if msg.role == "assistant":
                patched_total += self._patch_orphan_tool_calls(msg, reason="recovered_from_prior_turn")
        if patched_total:
            log.info(f"build_messages_patched_orphans count={patched_total}")

        messages: list[ProviderMessage] = []

        # Pull latest user text for downstream selectors (skill matcher, MCP router).
        latest_user_text = ""
        for msg in reversed(self.session.messages):
            if msg.role == "user":
                latest_user_text = msg.get_text_content()
                break

        # Build enhanced system prompt (git status, memories, date, etc.)
        try:
            from oats.session.modes import mode_guidance as _mode_guidance
            system_prompt = await build_system_prompt(
                working_dir=self.session.info.working_dir,
                project_dir=self.session.info.project_dir,
                session_id=self.session.id,
                active_tool_names=sorted(self._active_tool_names),
                budget_guidance=(
                    format_budget_guidance(budget_snapshot)
                    if budget_snapshot is not None
                    else None
                ),
                task_guidance=(
                    task_snapshot.guidance
                    if task_snapshot is not None
                    else None
                ),
                mode_guidance=_mode_guidance(),
                user_prompt=latest_user_text or None,
            )
        except Exception as e:
            err = f"### Sorry!! Failed enhanced_prompt_trying_fallback_after error:\n\n```\n{str(traceback.format_exc())}\n```\n"
            log.warn(err)
            system_prompt = SYSTEM_PROMPT.replace('REPLACE_WORKING_DIR', self.session.info.working_dir)

        # Inject MCP tool context — uses the BM25 index to auto-route
        # the user's prompt to the correct MCP resource
        try:
            proj_dir = Path(self.session.info.project_dir)
            user_text = ""
            for msg in reversed(self.session.messages):
                if msg.role == "user":
                    user_text = msg.get_text_content()
                    break
            mcp_context = ""
            if user_text:
                mcp_context = build_mcp_system_context(
                    user_text,
                    project_dir=proj_dir,
                )
                if mcp_context:
                    system_prompt += mcp_context
            # Also include registry tools if no index match
            if not mcp_context:
                try:
                    from oats.mcp.registry import get_mcp_registry
                    registry = get_mcp_registry()
                    mcp_tools = registry.list_tools()
                    if mcp_tools:
                        mcp_context = build_mcp_system_context(
                            user_text, mcp_tools=mcp_tools, project_dir=proj_dir,
                        )
                        if mcp_context:
                            system_prompt += mcp_context
                except Exception:
                    err = f"### Sorry!! Failed mcp_context_loading_with_error:\n\n```\n{str(traceback.format_exc())}\n```\n"
                    log.info(err)
                    pass
        except Exception:
            err = f"### Sorry!! Failed mcp_loading_with_error:\n\n```\n{str(traceback.format_exc())}\n```\n"
            log.info(err)
            pass

        messages.append(ProviderMessage(role="system", content=system_prompt))

        # Add conversation messages
        for msg in self.session.messages:
            if msg.role == "user":
                if msg.has_images():
                    # Build multimodal content (text + image blocks)
                    messages.append(ProviderMessage(
                        role="user",
                        content=msg._build_multimodal_content(),
                    ))
                else:
                    messages.append(ProviderMessage(role="user", content=msg.get_text_content()))
            elif msg.role == "assistant":
                text_content = msg.get_text_content()
                tool_calls = msg.get_tool_calls()
                tool_results = msg.get_tool_results()
                if tool_calls:
                    # Add assistant message with tool calls
                    messages.append(
                        ProviderMessage(
                            role="assistant",
                            content=text_content or "",
                            tool_calls=[
                                {
                                    "id": tc.tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.tool_name,
                                        "arguments": json.dumps(tc.arguments),
                                    },
                                }
                                for tc in tool_calls
                            ],
                        )
                    )

                    # Add tool results as separate messages
                    for result in tool_results:
                        content = (
                            (result.metadata or {}).get("retained_output")
                            or result.output
                        )
                        if result.error:
                            content = f"Error: {result.error}\n{content}"
                        messages.append(
                            ProviderMessage(
                                role="tool",
                                content=content,
                                tool_call_id=result.tool_call_id,
                            )
                        )
                else:
                    messages.append(
                        ProviderMessage(role="assistant", content=text_content)
                    )

        # Auto-todo reminder: nudge the model to plan with todowrite on long
        # tasks. Sent as a transient system message — not stored in the
        # session — so it doesn't accumulate or get summarized away.
        # Skipped for sub-agents (depth > 0) which handle their own brief.
        if self._agent_depth == 0 and "todowrite" in self._active_tool_names:
            next_turn, gap_write, gap_reminder = _todo_reminder_turn_counts(
                self.session.messages, self._last_todo_reminder_turn
            )
            if gap_write and gap_reminder and (
                next_turn >= TODO_REMINDER_TURNS_SINCE_WRITE
                and gap_write >= TODO_REMINDER_TURNS_SINCE_WRITE
                and gap_reminder >= TODO_REMINDER_TURNS_BETWEEN
            ):
                # Use user-role with <system-reminder> tags — vLLM rejects
                # system-role messages anywhere except position 0. This
                # mirrors claude-code's attachment-style reminders.
                messages.append(ProviderMessage(role="user", content=TODO_REMINDER_TEXT))
                self._last_todo_reminder_turn = next_turn
                log.info(f"todo_reminder_injected at_turn={next_turn} gap_write={gap_write}")

        return messages

    def _maybe_promote_tools(
        self,
        tool_definitions: list[ToolDefinition],
        tool_name: str,
        result: ToolResult,
    ) -> list[ToolDefinition]:
        """Promote deferred tools into the active callable set after tool_search."""
        if tool_name != "tool_search":
            return tool_definitions

        matched = result.metadata.get("matched_tool_names", []) if result.metadata else []
        if not matched:
            return tool_definitions

        existing = {t.name for t in tool_definitions}
        for matched_name in matched:
            tool = get_tool(matched_name)
            if tool is None or tool.name in existing:
                continue
            tool_definitions.append(
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                    strict=tool.strict,
                )
            )
            self._active_tool_names.add(tool.name)
            existing.add(tool.name)
            trace_event(
                self.session.id,
                "tool_search.promoted",
                {
                    "matched_tool_name": tool.name,
                    "active_tool_count": len(self._active_tool_names),
                },
            )
        return tool_definitions

    async def _save(self) -> None:
        """Save the session state."""
        await self._storage.update(self.session)

    @staticmethod
    def _patch_orphan_tool_calls(message, reason: str = "cancelled") -> int:
        """Ensure every tool_call on `message` has a matching tool_result.

        Prevents a broken turn (Ctrl+C mid-tool, provider error after tool_call
        dispatch) from leaving the session in a state where the next
        ``_build_messages`` emits an orphan ``tool_use`` block — which the
        provider API rejects, bricking the whole session.

        Returns the number of synthetic results inserted.
        """
        if message is None:
            return 0
        try:
            calls = message.get_tool_calls()
            existing_result_ids = {r.tool_call_id for r in message.get_tool_results()}
        except Exception:
            log.warning(f"### Sorry!! hit_error_patch_orphan_tool_calls\n```\n{traceback.format_exc()}\n```\n")
            return 0
        patched = 0
        for call in calls:
            if call.tool_call_id in existing_result_ids:
                continue
            message.add_tool_result(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                title=f"Incomplete: {call.tool_name}",
                output="",
                error=f"tool call was interrupted ({reason}) — no result produced",
                metadata={"synthetic_result": True, "reason": reason},
            )
            patched += 1
        if patched:
            log.warn(f"patched_orphan_tool_calls message_id={message.id} count={patched} reason={reason}")
        return patched
