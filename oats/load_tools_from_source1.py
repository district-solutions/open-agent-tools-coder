#!/usr/bin/env python3

import os
import sys
import traceback
import re
import argparse
import importlib.util
import inspect
import ujson as json
from typing import Tuple
from pydantic import BaseModel
from oats.determine_best_tools1 import determine_best_tools
from oats.agent_get_tool_choices_for_prompt import agent_get_tool_choices_for_prompt
from oats.pp import pp
from oats.log import gl

log = gl('load_tools_from_source1')

SERVED_MODEL_NAME = "functiongemma"
DEFAULT_PROMPT = "get utc str"

_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

class OatRepoUses(BaseModel):
    repo_uses_tool_schema_file: str = 'OAT_NOT_INIT'
    repo_uses_prompts: dict = {}
    repo_uses_tool_schema: dict = {}
    repo_src_files: list = []

# ---------------------------------------------------------------------------
# Dynamic loader
# ---------------------------------------------------------------------------

def _load_module(path: str):
    path = os.path.abspath(path)
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # register so __module__ checks work
    spec.loader.exec_module(mod)
    return mod


def _param_docs(func) -> dict[str, str]:
    """Parse Google-style 'Args:' section from a docstring into {param: desc}."""
    doc = inspect.getdoc(func) or ""
    result: dict[str, str] = {}
    in_args = False
    for line in doc.splitlines():
        stripped = line.strip()
        if re.match(r"^(args|arguments|parameters?|params?)\s*:", stripped, re.I):
            in_args = True
            continue
        if in_args:
            if stripped and not line[:1].isspace():
                break  # next top-level section
            m = re.match(r"(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)", stripped)
            if m:
                result[m.group(1)] = m.group(2).strip()
    return result


def _build_schema(func) -> dict:
    """Build an OpenAI tool-schema dict from a callable via introspection."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or f"Call {func.__name__}."
    description = doc.splitlines()[0]
    param_docs = _param_docs(func)

    properties: dict = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        ann = param.annotation
        json_type = _PY_TO_JSON.get(ann if ann is not inspect.Parameter.empty else str, "string")
        prop: dict = {"type": json_type}
        if pname in param_docs:
            prop["description"] = param_docs[pname]
        properties[pname] = prop
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    schema: dict = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {"type": "object", "properties": properties},
        },
    }
    if required:
        schema["function"]["parameters"]["required"] = required
    return schema


def load_tools(file_paths: list[str], verbose: bool = False) -> tuple[list, dict]:
    """
    Load public functions from each .py file.
    Returns (TOOLS list of OpenAI schemas, TOOL_IMPLS name→callable dict).
    """
    missing_tools = []
    tools: list = []
    impls: dict = {}
    for org_path in file_paths:
        path = org_path
        if org_path.startswith('coder/') or org_path.startswith('/coder/'):
            path = f'/opt/ds/oats/{org_path}'
        else:
            path = f'{org_path}'
            if not os.path.exists(path):
                path = os.getenv('CODER_TOOL_BASE_DIR', f'/opt/ds/oats/{org_path}')
            if not os.path.exists(path):
                path = os.getenv('CODER_TOOL_BASE_DIR', f'/data/ebs2/data/repos/matlok/{org_path}')
            if not os.path.exists(path):
                missing_tools.append(path)
                # err_msg = f'### Sorry!! failed_to_load_tools_with_source:\n```\n{__file__}\n```\ntried to load tools from source org_path:\n```\n{org_path}\n```\n'
                # log.error(err_msg)
                # raise Exception(err_msg)
        try:
            if verbose:
                log.info(f'loading_file: {path}')
            mod = _load_module(path)
        except Exception as exc:
            log.error(f"Could not load path: {path} with error:\n```\n{traceback.format_exc()}\n```\n")
            continue
        mod_name = mod.__name__
        for fname, obj in inspect.getmembers(mod, inspect.isfunction):
            if fname.startswith("_"):
                continue
            if obj.__module__ != mod_name:
                continue  # skip re-exported symbols from other modules
            tools.append(_build_schema(obj))
            impls[fname] = obj
            if verbose:
                log.info(f"Registered tool {fname} from path: {path}")
    if len(missing_tools) > 0:
        err_msg = f'### Sorry!! detected_missing_tools: {len(missing_tools)} with source code paths:\n```\n{pp(missing_tools)}\n```\n'
        log.error(err_msg)
    return tools, impls


# ---------------------------------------------------------------------------
# Tool-call loop
# ---------------------------------------------------------------------------

def run_tool_call(model: str, api_base: str, prompt: str, tools: list, tool_impls: dict) -> Tuple[bool, str, list[dict]]:
    messages = [{"role": "user", "content": prompt}]

    call_kwargs = dict(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        api_base=api_base,
        api_key=os.getenv('TOOL_API_KEY', 'CHANGE_PASSWORD'),
    )

    print(f"\n{'='*80}")
    print(f"Prompt : {prompt}")
    print(f"Model  : {model}")
    print(f"API    : {api_base}")
    print(f"Tools  : {[t['function']['name'] for t in tools]}")
    print(f"{'='*80}\n")

    log.info("First turn ...")
    import litellm
    resp = litellm.completion(**call_kwargs)
    msg = resp.choices[0].message
    messages.append(msg.model_dump(exclude_none=True))

    if not msg.tool_calls:
        err = f'### Sorry!! {__file__} failed to run_tool_call_response_tools\n\nmsg.content:\n```\n{msg.content}\n```\nmsg.tool_calls:\n```\n{msg.tool_calls}\n```'
        log.info(err)
        print(err)
        return False, err, messages

    log.info(f"Tool calls:\n{pp([tc.model_dump() for tc in msg.tool_calls])}\n")

    for tc in msg.tool_calls:
        fname = str(tc.function.name)
        try:
            fargs = json.loads(tc.function.arguments or "{}")
        except Exception:
            fargs = {}

        if fname not in tool_impls:
            result = f"Error: tool '{fname}' not found"
            log.error(result)
            print(pp(tc.model_dump(mode='json')))
        else:
            try:
                result = tool_impls[fname](**fargs)
                log.info(f"Tool '{fname}' → {result}")
            except Exception as exc:
                result = f"Error: {exc}"
                log.error(f"Tool '{fname}' raised:\n```\n{traceback.format_exc()}\n```\n")
                print(pp(tc.model_dump(mode='json')))

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": fname,
            "content": str(result),
        })

    print(f"Tool results:\n{pp(messages[-len(msg.tool_calls):])}\n")

    call_kwargs["messages"] = messages
    log.info("Second turn ...")
    resp2 = litellm.completion(**call_kwargs)
    final = str(resp2.choices[0].message.content)
    messages.append({"role": "assistant", "content": final})

    print(f"Final answer: {final}\n")
    print(f"\n{'='*80}")
    print("Full conversation:")
    print(f"{'='*80}\n")
    print(pp(messages))
    return True, final, messages


# ---------------------------------------------------------------------------
# Auto tool selection
# ---------------------------------------------------------------------------

repo_uses_tool_schema = {}
repo_uses_prompts = {}
repo_src_files = []

def get_oat_repo_uses_tools() -> OatRepoUses:
    global repo_uses_tool_schema
    global repo_uses_prompts
    repo_uses_tool_schema_file = os.getenv("CODER_TOOL_USES_INDEX", "./.ai/AGENT.repo_uses.python.tools.json")
    if repo_uses_prompts is None:
        repo_uses_prompts = {}
    if repo_uses_tool_schema is None:
        repo_uses_tool_schema = {}
    exclude_path_terms = [
        'finetune',
        'app.py',
        'app2.py',
        'run_functiongemma_tool_caller',
        'validate_functiongemma',
        'tools_oweb_direct',
        'jarvis',
        'deploy_app_with_docker_compose',
        'convert_data_to_pq2',
        'form_type_ner1',
        'matlok',
        'mcp_skills2',
        'find_github_repos',
    ]
    if len(repo_uses_tool_schema) == 0:
        log.info(f'loading_oat_repo_uses: {repo_uses_tool_schema_file}')
        with open(repo_uses_tool_schema_file, 'r') as f:
            repo_uses_tool_schema = json.loads(f.read())
        for src_file in repo_uses_tool_schema:
            valid = True
            for ex_term in exclude_path_terms:
                if f'{ex_term}' in src_file:
                    valid = False
                    break
            if not valid:
                continue
            log.debug(f'added_src_file: {src_file}')
            repo_src_files.append(src_file)
            for prompt_category in repo_uses_tool_schema:
                repo_uses_prompts[prompt_category] = {
                    'file': src_file,
                    'prompt': repo_uses_tool_schema[prompt_category],
                }
    oat_repo_uses = OatRepoUses(
        repo_uses_tool_schema_file=repo_uses_tool_schema_file,
        repo_uses_prompts=repo_uses_prompts,
        repo_uses_tool_schema=repo_uses_tool_schema,
        repo_src_files=repo_src_files,
    )
    return oat_repo_uses

def get_best_tools_for_prompt(
    prompt: str,
    tool_schema: str | None = None,
    top_k: int = 3,
    min_score: float = 0.0,
    rerank: bool = False,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    bm25_model: str = "bm25",
    run: bool = True,
    oat_enabled = True,
    verbose: bool = False,
) -> Tuple[bool, list, dict, list, list, dict]:
    """
    Examine the prompt, determine the best tool files via determine_best_tools(),
    then load only those files and run the tool call.

    Args:
        prompt: The user prompt.
        tool_schema: Path to the JSON tool-uses index file.
        top_k: Number of top candidate files to consider.
        min_score: Minimum BM25/retrieval score threshold.
        rerank: Whether to apply cross-encoder reranking.
        rerank_model: Cross-encoder model name for reranking.
        bm25_model: Retrieval model name (bm25, tfidf, or embedding model).
    """
    found_best_tool = False
    all_tools = []
    all_tool_impls = {}

    # Filter tools and impls to only those from the best files.
    # best_files are relative paths from the schema (e.g. "coder/date.py").
    # We need to match them against the loaded tool sources.
    # Build a mapping from file path -> tool schemas and impls.
    best_files: list[str] = []
    best_tools: list[dict] = []
    best_impls: dict = {}

    if tool_schema is None:
        tool_schema = os.getenv("CODER_TOOL_USES_INDEX", "./.ai/AGENT.repo_uses.python.tools.json")

    if oat_enabled:
        if verbose:
            log.info(f'# OAT best tools for prompt:\n```\n{prompt}\n```\nschema:\n```\n{tool_schema}\n```\n')
        oat_choices = agent_get_tool_choices_for_prompt(prompt=prompt, top_k=top_k, verbose=verbose)
        best_files = oat_choices.src_files
    else:
        if verbose:
            log.info(f'# Determining best tools for prompt:\n```\n{prompt}\n```\nschema:\n```\n{tool_schema}\n```\n')
        result = determine_best_tools(
            prompt=prompt,
            schema=tool_schema,
            model=bm25_model,
            top_k=top_k,
            min_score=min_score,
            rerank=rerank,
            rerank_model=rerank_model,
        )

        best_files = result.get("best_files", [])
    """
    if not best_files:
        oat_repo_uses = get_oat_repo_uses_tools()
        repo_uses_tools, repo_uses_tool_impls = load_tools(oat_repo_uses.repo_src_files)
        return found_best_tool, repo_uses_tools, repo_uses_tool_impls, best_files, best_tools, best_impls
    """
    # log.debug('PREVIOUS:')
    # print(pp(best_files))
    # print(pp(result))
    # log.debug('OAT:')
    # oat_tool_defs = oat_choices.tool_data.get('results', [])
    # print(pp(oat_tool_defs))
    # print(pp(oat_choices.model_dump(mode='json')))
    # print(pp(best_files))
    if not best_files:
        if verbose:
            log.info(f"### Sorry!! Unable to find best files found for prompt in file: {__file__} name: {__name__} — falling back to all loaded tools")
        return found_best_tool, all_tools, all_tool_impls, best_files, best_tools, best_impls

    # If best_files is empty, use all tools
    if len(best_files) != 0:
        # Normalize best_files for matching
        best_files_normalized = set()
        for bf in best_files:
            best_files_normalized.add(bf)
            # Also add with and without leading ./
            best_files_normalized.add(bf.lstrip("./"))
            best_files_normalized.add("./" + bf)

        all_tools, all_tool_impls = load_tools(best_files)

        # We need to know which file each tool came from.
        # Re-scan: for each tool in all_tools, check if its impl's source file
        # matches any of the best_files.
        # print(all_tools)
        num_best_tools = len(all_tools)
        if num_best_tools == 0:
            log.info(f'### Sorry!! {__file__} failed_no_best_tools_found')
        for tidx, tool_schema in enumerate(all_tools):
            if 'function' not in tool_schema:
                continue
            if 'name' not in tool_schema['function']:
                continue
            fname = tool_schema["function"]["name"]
            if fname not in all_tool_impls:
                continue
            impl = all_tool_impls[fname]
            source_file = getattr(impl, "__module__", None)
            if source_file is None:
                log.error(f'MISSING_SOURCE: {fname}')
                continue
            # source_file is the module name we registered (basename without .py)
            # We need to match against best_files which are full paths.
            # Check if any best file ends with the module name or contains it.
            matched = False
            for bf in best_files_normalized:
                bf_base = os.path.splitext(os.path.basename(bf))[0]
                if source_file == bf_base or bf in source_file or source_file in bf:
                    matched = True
                    break
            if matched:
                if verbose:
                    log.info(f'added_tool {tidx + 1}/{num_best_tools}: {fname}')
                found_best_tool = True
                best_tools.append(tool_schema)
                best_impls[fname] = impl
            else:
                log.error(f'### Skipped skipped_tool {tidx + 1}/{num_best_tools}: {fname}')
    if verbose:
        log.debug(f'best_impls:')
        for fname in best_impls:
            log.debug(fname)
            print(best_impls[fname])
        log.info(f'# OAT Enabled: {oat_enabled}')
    # print(best_files)
    # import time
    # time.sleep(1000)
    return len(best_tools) != 0, all_tools, all_tool_impls, best_files, best_tools, best_impls

def run_tool_for_prompt(
    prompt: str,
    model: str | None = None,
    api_base: str | None = None,
    schema: str | None = None,
    top_k: int = 5,
    min_score: float = 0.0,
    rerank: bool = False,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    bm25_model: str = "bm25",
    run: bool = True,
) -> Tuple[bool, bool, list, dict, str, list[dict]]:
    """
    Examine the prompt, determine the best tool files via determine_best_tools(),
    then load only those files and run the tool call.

    Args:
        model: The model identifier for litellm.
        api_base: The API base URL.
        prompt: The user prompt.
        schema: Path to the JSON tool-uses index file.
        top_k: Number of top candidate files to consider.
        min_score: Minimum BM25/retrieval score threshold.
        rerank: Whether to apply cross-encoder reranking.
        rerank_model: Cross-encoder model name for reranking.
        bm25_model: Retrieval model name (bm25, tfidf, or embedding model).
        run: if False, skips running tool
    """
    all_tools = []
    all_tool_impls = {}

    if api_base is None:
        api_base = os.getenv("TOOL_FUNCTION_1", "http://0.0.0.0:20700/v1")
    if model is None:
        model = "openai/google/functiongemma-270m-it"
    if schema is None:
        schema = os.getenv("CODER_TOOL_USES_INDEX", "./.ai/AGENT.repo_uses.python.tools.json")

    log.info(f'# Determining best tools for prompt:\n```\n{prompt}\n```\nschema:\n```\n{schema}\n```\n')
    result = determine_best_tools(
        prompt=prompt,
        schema=schema,
        model=bm25_model,
        top_k=top_k,
        min_score=min_score,
        rerank=rerank,
        rerank_model=rerank_model,
    )

    best_files: list[str] = result.get("best_files", [])
    if not best_files:
        log.info(f"### Sorry!! No best files found for prompt — falling back to all loaded tools result: {result}")
        best_files = []
        return False, False, all_tools, all_tool_impls, f'STOPPED_1_{__file__}', []

    # Filter tools and impls to only those from the best files.
    # best_files are relative paths from the schema (e.g. "coder/date.py").
    # We need to match them against the loaded tool sources.
    # Build a mapping from file path -> tool schemas and impls.
    best_tools: list = []
    best_impls: dict = {}

    # If best_files is empty, use all tools
    if best_files:
        # Normalize best_files for matching
        best_files_normalized = set()
        for bf in best_files:
            best_files_normalized.add(bf)
            # Also add with and without leading ./
            best_files_normalized.add(bf.lstrip("./"))
            best_files_normalized.add("./" + bf)

        all_tools, all_tool_impls = load_tools(best_files)

        # We need to know which file each tool came from.
        # Re-scan: for each tool in all_tools, check if its impl's source file
        # matches any of the best_files.
        # print(all_tools)
        num_best_tools = len(all_tools)
        for tidx, tool_schema in enumerate(all_tools):
            fname = tool_schema["function"]["name"]
            if fname not in all_tool_impls:
                continue
            impl = all_tool_impls[fname]
            source_file = getattr(impl, "__module__", None)
            if source_file is None:
                log.error(f'MISSING_SOURCE: {fname}')
                continue
            # source_file is the module name we registered (basename without .py)
            # We need to match against best_files which are full paths.
            # Check if any best file ends with the module name or contains it.
            matched = False
            for bf in best_files_normalized:
                bf_base = os.path.splitext(os.path.basename(bf))[0]
                if source_file == bf_base or bf in source_file or source_file in bf:
                    matched = True
                    break
            if matched:
                log.info(f'added_tool {tidx + 1}/{num_best_tools}: {fname}')
                best_tools.append(tool_schema)
                best_impls[fname] = impl
            else:
                log.error(f'### Skipped skipped_tool {tidx + 1}/{num_best_tools}: {fname}')

    tool_ran = False
    tool_run_status = False
    tool_response_msg = 'NOT_RUN'
    tool_messages = []
    if run:
        if len(best_tools) > 0:
            log.info(f"Selected {len(best_tools)} tools from {len(best_files)} best files for prompt")
            for t in best_tools:
                log.info(f'  - {t["function"]["name"]}')
            tool_run_status, tool_response_msg, tool_messages = run_tool_call(model, api_base, prompt, best_tools, best_impls)
            tool_ran = True
        else:
            log.error(f'### Sorry!! not_running_tool best_tools_detected: {best_tools}\nbest_files:\n```\n{best_files}\n```\n')
    return tool_ran, tool_run_status, all_tools, all_tool_impls, tool_response_msg, tool_messages


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    api_base = os.getenv("TOOL_FUNCTION_1", "http://0.0.0.0:20700/v1")
    model = "openai/google/functiongemma-270m-it"

    parser = argparse.ArgumentParser(description="FunctionGemma inference with dynamically loaded tool files")
    parser.add_argument("-f", "--files", required=False, metavar="FILE1,FILE2,...", help="Comma-separated list of .py files whose public functions become tools")
    parser.add_argument("-p", "--prompt", default=DEFAULT_PROMPT, help="Prompt to send")
    parser.add_argument("--api-base", default=api_base, help=f"OpenAI-compatible endpoint (default: $TOOL_FUNCTION_1 or {api_base})")
    parser.add_argument("--auto", action="store_true", help="Auto-select best tools for the prompt using determine_best_tools (BM25 + optional reranker)")
    parser.add_argument("--schema", default=None, help="Path to JSON tool-uses index file (default: $CODER_TOOL_USES_INDEX)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top candidate files (default: 5)")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum retrieval score threshold (default: 0.0)")
    parser.add_argument("--rerank", action="store_true", help="Apply cross-encoder reranker after BM25 retrieval")
    parser.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Cross-encoder model for reranking")
    parser.add_argument("--bm25-model", default="bm25", help="Retrieval model: bm25 | tfidf | <sentence-transformer> (default: bm25)")
    args = parser.parse_args()

    api_base = args.api_base

    if args.files is not None:
        file_paths = [p.strip() for p in args.files.split(",") if p.strip()]
        tools, tool_impls = load_tools(file_paths)
        if not tools:
            log.error("### No tools loaded - please check -f paths and ensure files export public functions.")
            sys.exit(1)

    run_tool_for_prompt(
        model=model,
        api_base=api_base,
        prompt=args.prompt,
        schema=args.schema,
        top_k=args.top_k,
        min_score=args.min_score,
        rerank=args.rerank,
        rerank_model=args.rerank_model,
        bm25_model=args.bm25_model,
        run=True,
    )

    print("done")

if __name__ == "__main__":
    main()
