#!/usr/bin/env python3

"""
ff -m 'get utc' -n -p vllm-small -z hosted_vllm/MODEL_NAME
"""

from typing import Optional
from pydantic import BaseModel
from pydantic.config import ConfigDict
from oats.tool.registry import Tool
from oats.call_tool_with_loader1 import LocalTool

class SelectedToolsManifest(BaseModel):
    prompt: str = None
    found_best_tool: bool = False
    found_all_tools: list = []
    core_tools: list[Tool] = []
    core_tool_names: set[str] = set()
    core_impls: dict = {}
    mcp_tools: list[Tool] = []
    mcp_tool_names: set[str] = set()
    mcp_impls: dict = {}
    local_tools: list[LocalTool] = []
    local_tool_names: set[str] = set()
    local_impls: dict = {}
    all_tools: list[Tool] = []
    all_tool_names: set[str] = set()
    plan_tools: list[Tool] = []
    plan_tool_names: set[str] = set()
    agent_tools: list[Tool] = []
    agent_tool_names: set[str] = set()
    best_tools: dict = {}
    best_tool_names: set[str] = set()
    best_files: list = []
    best_impls: dict = {}
    all_tools_dict: dict = {}
    provider_tool_map: dict = {}

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        # Allow extra fields if needed
        extra='allow'
    )
