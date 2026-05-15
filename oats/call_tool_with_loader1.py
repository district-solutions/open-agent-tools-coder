#!/usr/bin/env python3
"""
Call tool with loader — dynamically load tools from source and wrap them as LocalTool instances.
"""
from __future__ import annotations

import os
import sys
import traceback
import argparse
import asyncio
from pathlib import Path
from typing import Any, Tuple
import ujson as json
from oats.load_tools_from_source1 import get_best_tools_for_prompt
from oats.tool.registry import Tool, ToolContext, ToolResult
from oats.pp import pp
from oats.log import gl

log = gl(__name__)

DEFAULT_PROMPT = "get utc"

# ---------------------------------------------------------------------------
# LocalTool — a mutable Tool subclass backed by a real callable
# ---------------------------------------------------------------------------

class LocalTool(Tool):
    """A Tool whose attributes are set via setters rather than hard-coded properties."""

    def __init__(self):
        self._name: str = ""
        self._aliases: list[str] = []
        self._keywords: list[str] = []
        self._always_load: bool = False
        self._strict: bool = False
        self._description: str = ""
        self._parameters: dict[str, Any] = {}
        self._requires_permission_msg: str | None = None
        self._output: str = ""
        self.tool_context: ToolContext | None = None
        self._impl: Any = None  # the underlying callable

    # -- setters ----------------------------------------------------------

    def set_name(self, value: str) -> None:
        self._name = value

    def set_aliases(self, value: list[str]) -> None:
        self._aliases = value

    def set_keywords(self, value: list[str]) -> None:
        self._keywords = value

    def set_always_load(self, value: bool) -> None:
        self._always_load = value

    def set_strict(self, value: bool) -> None:
        self._strict = value

    def set_description(self, value: str) -> None:
        self._description = value

    def set_parameters(self, value: dict[str, Any]) -> None:
        self._parameters = value

    def set_requires_permission(self, value: str | None) -> None:
        self._requires_permission_msg = value

    def set_output(self, value: str) -> None:
        self._output = value

    def set_tool_context(self, value: ToolContext) -> None:
        self.tool_context = value

    def set_impl(self, impl: Any) -> None:
        self._impl = impl

    # -- abstract properties (required by Tool ABC) -----------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    # -- optional overrides -----------------------------------------------

    @property
    def aliases(self) -> list[str]:
        return self._aliases

    @property
    def keywords(self) -> list[str]:
        return self._keywords

    @property
    def always_load(self) -> bool:
        return self._always_load

    @property
    def strict(self) -> bool:
        return self._strict

    @property
    def output(self) -> str:
        return self._output

    def requires_permission(self, args: dict[str, Any], ctx: ToolContext) -> str | None:
        return self._requires_permission_msg

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Execute the underlying callable and return a ToolResult."""
        self.tool_context = ctx
        try:
            if self._impl is None:
                return ToolResult(
                    title=self._name,
                    output="",
                    error=f"No implementation set for tool '{self._name}'",
                )
            result = self._impl(**args)
            self._output = str(result)
            return ToolResult(
                title=self._name,
                output=str(result),
            )
        except Exception as e:
            return ToolResult(
                title=f"{self._name} (error)",
                output="",
                error=str(e),
            )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_tools_from_repo_uses_index(prompt: str, file_path: str | None = None, min_score: float = 0.0, verbose: bool = False) -> Tuple[bool, list, dict, list, list[dict], dict, list, list]:
    """
    Import source code using get_best_tools_for_prompt(), then create a
    LocalTool for each matched tool.

    Args:
        file_path: Path to the tool-uses index JSON file (or None for default).
        prompt: The prompt to match tools against.

    Returns:
        A list of LocalTool instances ready for use.
    """
    if verbose:
        log.info(f"load_tools_from_repo_uses_index: file_path={file_path!r}, prompt={prompt[0:2000]}")

    local_tool_impls = []
    local_tool_names = []
    local_tools: list[LocalTool] = []
    found_best_tool, all_tools, all_tool_impls, best_files, best_tools, best_impls = get_best_tools_for_prompt(prompt=prompt, min_score=min_score, tool_schema=file_path, verbose=verbose)

    if verbose:
        log.info(
            f"found_best_tool={found_best_tool}, "
             f"len(best_tools)={len(best_tools)}, "
             f"len(best_impls)={len(best_impls)}, "
             f"best_files={best_files}")

    if not found_best_tool:
        if verbose:
            log.info(f"### Sorry!! No best tools found in file: {__file__} for prompt — returning empty list")
        return found_best_tool, all_tools, all_tool_impls, best_files, best_tools, best_impls, local_tools, local_tool_names

    for tool_schema in best_tools:
        if "function" not in tool_schema:
            continue
        func_info = tool_schema["function"]
        fname = func_info.get("name", "")
        if not fname:
            continue

        impl = best_impls.get(fname)

        lt = LocalTool()
        lt.set_name(fname)
        lt.set_description(func_info.get("description", ""))
        lt.set_parameters(func_info.get("parameters", {}))
        lt.set_aliases([])
        lt.set_keywords([])
        lt.set_always_load(True)
        lt.set_strict(False)
        lt.set_requires_permission(f"Execute tool: {fname}")
        lt.set_output("")
        if impl is not None:
            lt.set_impl(impl)
            if verbose:
                log.info(f"Created LocalTool '{fname}' with impl")
        else:
            log.warning(f"Created LocalTool '{fname}' WITHOUT impl")

        local_tool_names.append(fname)
        local_tools.append(lt)

    if verbose:
        log.info(f"load_tools_from_repo_uses_index: created {len(local_tools)} LocalTool(s)")
        log.debug('# Local Tools\n\n')
        print(all_tools)
        print(all_tool_impls)
        print(best_files)
        print(best_tools)
        print(best_impls)
    return found_best_tool, all_tools, all_tool_impls, best_files, best_tools, best_impls, local_tools, local_tool_names


def run_tool_call(
    prompt: str,
    tools: list,
    tool_impls: dict,
    provider_id: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    verbose: bool = False,
) -> Tuple[bool, str]:
    """Run a two-turn tool-call loop via LiteLLM and return the final answer."""

    if model is None:
        model = 'openai/google/functiongemma-270m-it'
    if api_base is None:
        api_base = 'http://0.0.0.0:20700/v1'
    if prompt == '{}':
        raise Exception(f'# Sorry!! call_tool_with_loader1.run_tool_call requires a valid prompt: {prompt}')
    messages = [{"role": "user", "content": prompt}]
    api_key = os.getenv('TOOL_API_KEY', 'CHANGE_PASSWORD')
    call_kwargs = dict(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        api_base=api_base,
        api_key=api_key,
    )

    if verbose:
        log.debug('-------\nTools - Start\n-------')
        for tool_name in tools:
            print(tool_name)
        log.debug('-------\nTools - End\n-------')
        log.debug('-------\nTools Impls\n-------')
        num_tool_impls = len(tool_impls)
        for tidx, tool_name in enumerate(tool_impls):
            tool_data = tool_impls[tool_name]
            log.info(f'## Tool {tidx + 1}/{num_tool_impls}\nname: **{tool_name}**\nimplementation:\n```\n{tool_data}\n```\n')
            print('---')
        log.debug('-------\nTools Impls - End\n-------')

    # print(f"\n{'='*80}")
    # print(f"Prompt : {prompt}")
    # print(f"Model  : {model}")
    # print(f"API    : {api_base}")
    # print(f"Tools  : {[t['function']['name'] for t in tools]}")
    # print(f"{'='*80}\n")

    if verbose:
        log.info(f'loading litellm')
    import litellm
    if verbose:
        log.info(f'### First turn calling tool with args:\n\n```\n{pp(call_kwargs)}\n```\n')
    resp = litellm.completion(**call_kwargs)
    msg = resp.choices[0].message
    messages.append(msg.model_dump(exclude_none=True))
    if not msg.tool_calls:
        err = f'### Sorry!! {__file__} - no_tool_calls_call_tool_with_loader1 detected. model response:\n```\n{msg.content}\n```\n'
        log.error(err)
        print(err)
        log.error('call_kwargs')
        print(pp(call_kwargs))
        log.error('msg')
        print(msg)
        return False, err

    # print(f"Tool calls:\n{pp([tc.model_dump() for tc in msg.tool_calls])}\n")

    for tc in msg.tool_calls:
        fname = str(tc.function.name)
        try:
            fargs = json.loads(tc.function.arguments or "{}")
        except Exception:
            log.error(f"### Sorry failed to json.loads Tool '{fname}' raised:\n```\n{traceback.format_exc()}\n```\n")
            fargs = {}
        if fname not in tool_impls:
            result = f"Error: tool '{fname}' not found"
            log.error(result)
        else:
            try:
                result = tool_impls[fname](**fargs)
                if verbose:
                    log.info(f"Tool '{fname}' → {result}")
            except Exception as exc:
                result = f"### Sorry!! Error:\n```\n{traceback.format_exc()}\n```\n"
                log.error(f"### Sorry failed to json.loads Tool '{fname}' raised:\n```\n{traceback.format_exc()}\n```\n")

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": fname,
            "content": str(result),
        })

    # print(f"Tool results:\n{pp(messages[-len(msg.tool_calls):])}\n")

    call_kwargs["messages"] = messages
    if verbose:
        log.info("Second turn ...")
    # print(pp(call_kwargs))
    resp2 = litellm.completion(**call_kwargs)
    if resp2 is None:
        err = f'### Sorry!! Failed second_tool_call_turn with error:\n```\n{resp2}\n```\n'
        log.error(err)
        return False, err
    if not hasattr(resp2, 'choices'):
        err = f'### Sorry!! Failed second_tool_call_turn with error_missing_choices:\n```\n{resp2}\n```\n'
        log.error(err)
        return False, err
    answer = str(resp2.choices[0].message.content)
    # messages.append({"role": "assistant", "content": answer})

    if verbose:
        log.info(f"run_tool_call_answers: {answer}\n")
    # print(f"\n{'='*80}")
    # print("Full conversation:")
    # print(f"{'='*80}\n")
    # print(pp(messages))
    return True, answer

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point: load tools from index, wrap as LocalTool, and demo execution."""
    api_base = os.getenv("TOOL_FUNCTION_1", "http://0.0.0.0:20700/v1")

    parser = argparse.ArgumentParser(
        description="Load tools from source index and wrap them as LocalTool instances"
    )
    parser.add_argument(
        "-p", "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Prompt to match tools against (default: {DEFAULT_PROMPT})",
    )
    parser.add_argument(
        "-s", "--schema",
        default=None,
        help="Path to JSON tool-uses index file (default: $CODER_TOOL_USES_INDEX)",
    )
    parser.add_argument(
        "-t", "--top-k",
        type=int,
        default=5,
        help="Number of top candidate files (default: 5)",
    )
    parser.add_argument(
        "-m", "--min-score",
        type=float,
        default=0.0,
        help="Minimum retrieval score threshold (default: 0.0)",
    )
    parser.add_argument(
        "-r", "--rerank",
        action="store_true",
        help="Apply cross-encoder reranker after BM25 retrieval",
    )
    parser.add_argument(
        "--rerank-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model for reranking",
    )
    parser.add_argument(
        "--bm25-model",
        default="bm25",
        help="Retrieval model: bm25 | tfidf | <sentence-transformer> (default: bm25)",
    )
    args = parser.parse_args()

    log.info(f"prompt={args.prompt!r}, schema={args.schema!r}, top_k={args.top_k}")

    found_best_tool, all_tools, all_tool_impls, best_files, best_tools, best_impls, local_tools, local_tool_names = load_tools_from_repo_uses_index(file_path=args.schema, prompt=args.prompt)

    if not found_best_tool:
        log.error("No LocalTools were created — check your prompt and schema.")
        sys.exit(1)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Loaded {len(all_tools)} LocalTool(s) for prompt: {args.prompt!r}")
    print(f"{'='*80}\n")

    for lt in local_tools:
        print(f"  Tool: {lt.name}")
        print(f"    description : {lt.description[:100]}")
        print(f"    parameters  : {lt.parameters}")
        print(f"    aliases     : {lt.aliases}")
        print(f"    keywords    : {lt.keywords}")
        print(f"    always_load : {lt.always_load}")
        print(f"    strict      : {lt.strict}")
        print()

    # Demo: execute the first tool with empty args
    if local_tools:
        ctx = ToolContext(
            session_id="test-session",
            project_dir=Path("./.coder"),
            working_dir=Path("./.coder"),
        )
        lt = local_tools[0]
        log.info(f"Executing LocalTool '{lt.name}' with args={{}}")
        result = asyncio.run(lt.execute({}, ctx))
        print(f"Execution result for '{lt.name}':")
        print(f"  title  : {result.title}")
        print(f"  output : {result.output}")
        print(f"  error  : {result.error}")

    print("\ndone")


if __name__ == "__main__":
    main()
