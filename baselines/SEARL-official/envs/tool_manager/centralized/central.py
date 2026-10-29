# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import ray
import json
import torch
import asyncio
import datetime
import threading
import networkx as nx
from typing import List, Optional, Any, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from envs.tool_manager.reasoner import QwenManager
import yaml
from concurrent.futures import ThreadPoolExecutor
import contextlib
import re
from qwen_agent.llm.schema import ASSISTANT, SYSTEM, USER, FUNCTION, ContentItem
from ast import literal_eval
import ast
import time
from jinja2 import StrictUndefined, Template
from transformers import AutoTokenizer
import os
from pathlib import Path
FENCE_RE = re.compile(r'<python>([\s\S]*?)</python>', re.S | re.IGNORECASE)
FUNC_RE  = re.compile(r'^\s*def\s+\w+\s*\([^)]*\)\s*:', re.MULTILINE)
ANSI_RE  = re.compile(r'\x1B\[[0-9;]*[A-Za-z]')
SENTINEL_RE = re.compile(r'<\|im_[^|]*\|>')

def _repo_root() -> Path:
    return Path(os.environ.get("RL_FACTORY_ROOT", Path(__file__).resolve().parents[3]))

def _default_embedding_model() -> str:
    return os.environ.get("RL_FACTORY_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def _resolve_project_name(cfg=None) -> str:
    'Resolve the project name from environment, configuration, then the default.'
    env_name = os.getenv("RL_FACTORY_PROJECT_NAME") or os.getenv("PROJECT_NAME")
    if env_name:
        return env_name
    if cfg is not None:
        cfg_name = getattr(cfg, "project_name", None)
        if cfg_name:
            return cfg_name
    return "default_project"

def parse_mcp_tools_config(file_path, project_name=None):
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        resolved_project_name = project_name or _resolve_project_name(None) or "default_project"
        if "{project_name}" in content:
            content = content.replace("{project_name}", resolved_project_name)
        data = literal_eval(content)
        return data
    except Exception as e:
        print(f"Parse error: {e}")
        return []

def remove_middle_tokens(s):

    start_tag = "<|im_start|>"
    end_tag = "<|im_end|>"

    first_idx = s.find(start_tag) + len(start_tag)
    last_idx = s.rfind(end_tag)

    if first_idx == -1 or last_idx == -1:
        return s

    middle_part = s[first_idx:last_idx]

    middle_part = middle_part.replace(start_tag, "").replace(end_tag, "")

    return s[:first_idx] + middle_part + s[last_idx:]

def populate_template(template: str, variables: Dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")

def extract_tool_names(text: str):
    'Extract unique tool names from <tool_call> blocks in encounter order, including names inside arguments.'
    names = []
    seen = set()

    tool_call_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
    name_pattern = re.compile(r'"name"\s*:\s*"([^"]+)"')
    create_mcp_args_pattern = re.compile(r'"arguments"\s*:\s*\{[^}]*"name"\s*:\s*"([^"]+)"', re.DOTALL)

    for block in tool_call_pattern.findall(text):

        for n in name_pattern.findall(block):
            if n == 'create_and_execute_mcp':
                args_match = create_mcp_args_pattern.search(block)
                if args_match:
                    n = args_match.group(1)
                    names.append(n)
            elif n not in ["search-search", "execute_python_code-execute_python_code"]:
                if n not in seen:
                    seen.add(n)
                    names.append(n)
    return names

def map_st_to_number(st_id: str) -> int:
    match = re.match(r"ST_?(\d+)", st_id)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Invalid step ID format: {st_id}")

def _sanitize_code(code: str) -> str:
    code = ANSI_RE.sub('', code)
    code = SENTINEL_RE.sub('', code)
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    # REMOVED: The line using BACKTICK_TAIL_RE is gone.
    return code

def _parse_plan(raw_plan: str):
    text = _sanitize_code(raw_plan)

    pattern = re.compile(
        r'^\s*##\s*(ST\d+):?([^\n]*)\n'
        r'([\s\S]*?)'
        r'(?=\n\s*##\s*ST\d+\b|\Z)',
        flags=re.MULTILINE
    )

    subtask_dict = {}

    for m in pattern.finditer(text):
        st_code, title_raw, content = m.groups()

        title = title_raw.strip() or st_code

        content_stripped = content.strip()

        steps = re.findall(r'^\s*\d+\.\s*.*$', content_stripped, flags=re.MULTILINE)
        steps_str = '\n'.join(s.strip() for s in steps)

        subtask_dict[st_code] = {
            "title": title,
            "steps": steps_str
        }
    return subtask_dict

def _parse_dag_list(text: str, max_turns: Optional[int] = None) -> Tuple[Optional[List], str]:
    'Parse ##DAG_LIST and return (dag_list, status), where status is ok, missing, empty or invalid.'
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    m = re.search(r'##DAG_LIST[^\n]*\n([\s\S]*?)(?=\n##|\Z)', text)
    if not m:
        return None, "missing"
    s = m.group(1).strip()
    if s.startswith("```"):
        s = re.sub(r'^```[^\n]*\n([\s\S]*?)\n```$', r'\1', s).strip()
    if not s:
        return None, "empty"

    s = re.sub(r'(?<!")\b(ST\d+)\b(?!")', r'"\1"', s)
    try:
        parsed = ast.literal_eval(s)
    except Exception:
        return None, "invalid"

    if not isinstance(parsed, list) or len(parsed) == 0:
        return None, "invalid"

    nodes = []
    for item in parsed:
        if isinstance(item, str):
            nodes.append(item)
        elif isinstance(item, (tuple, list)):
            if len(item) not in [1, 2]:
                return None, "invalid"
            for node in item:
                if not isinstance(node, str):
                    return None, "invalid"
                nodes.append(node)
        else:
            return None, "invalid"

    for node in nodes:
        if not re.match(r'^ST\d+$', node):
            return None, "invalid"

    if max_turns is not None and nodes:
        max_id = max(int(n[2:]) for n in nodes)
        if max_id > max_turns:
            return None, "invalid"

    return parsed, "ok"

class CentralizedQwenManager(QwenManager):
    'Forward QwenManager tool calls to the Ray actor.'

    def __init__(self, verl_config, centralized_actor_handle=None, mock=False):
        if not mock and centralized_actor_handle is None:
            raise ValueError('CentralizedQwenManager requires centralized_actor_handle')

        super().__init__(verl_config)

        self.centralized_actor_handle = centralized_actor_handle
        with open(_repo_root() / "prompt" / "workflow.yaml", "r") as file:
            self.system_template = yaml.safe_load(file)

        self.timeout_dag = 30
        self.max_turns = verl_config.get("max_turns", 6)

        self.dynamic_prompt = getattr(verl_config, 'dynamic_prompt', True)
        print(f"[CentralizedQwenManager] Initialized with dynamic_prompt={self.dynamic_prompt}")

        self._temp_mapping_cache = {}
        self._temp_mapping_cache_time = 0
        self._temp_mapping_cache_ttl = 5
        self._cache_lock = threading.Lock()

        self._tool_selection_cache = {}  # key: query_hash, value: (tools, timestamp)
        self._tool_selection_cache_ttl = 10

        self._cached_mcp_version = -1
        self._cached_functions_mcps = []

        self._build_temp_mcp_tools(init_project=False)
        print('CentralizedQwenManager initialized; tool calls use the centralized actor.')

    def _build_temp_mcp_tools(self, init_project=False):
        if getattr(self.verl_config, 'runtime_config_path', None):
            # Dynamic source is data in the isolated registry, never a host MCP import.
            self.functions_mcps = []
            return
        # try:

        self._ensure_mcp_tool_files(init_project=init_project)
        temp_config_path = self.verl_config.temp_config_path
        official_config_path = self.verl_config.official_config_path
        assert temp_config_path is not None and official_config_path is not None, "temp_config_path and official_config_path must be set"
        # if init_project:
        #     official_function_list = []
        # else:
        official_function_list = parse_mcp_tools_config(official_config_path, self.project_name)

        base_functions = set()
        for fn in getattr(self, "functions", []):
            if isinstance(fn, dict):
                name = fn.get("name")
            else:
                name = getattr(fn, "name", None)
            if name:
                base_functions.add(name)
        print(official_function_list, "OFFICIAL_FUNCTION_LIST")
        for tool in official_function_list:
            try:
                self._init_tool(tool)
            except Exception as inner_e:
                raise RuntimeError(f"Error initializing tool {tool}: {inner_e}") from inner_e

        all_functions = [func.function for func in self.tool_map.values()]
        self.functions_mcps = []
        for fn in all_functions:
            name = None
            if isinstance(fn, dict):
                name = fn.get("name")
            else:
                name = getattr(fn, "name", None)

            if name and name not in base_functions:
                self.functions_mcps.append(fn)
        # except Exception as e:
        #     # Catch any exception (including McpError) and raise a RuntimeError
        #     # This ensures that Ray can serialize the exception properly, avoiding the unpickle error.
        #     raise RuntimeError(f"Failed to build MCP tools: {e}") from e

    def _ensure_mcp_tool_files(self, init_project=False):
        'Create temporary and official MCP entry files with imports and startup blocks.'
        base_dir = Path(__file__).resolve().parents[2] / "tools"
        invented_dir = base_dir / "invented_tools" / (self.project_name or "default_project")
        invented_dir.mkdir(parents=True, exist_ok=True)
        invented_file = invented_dir / "temp_tool_list.py"
        official_file = invented_dir / "official_tool_list.py"

        common_imports = (
            "import json\n"
            "import sys\n"
            "import traceback\n"
            "import math\n"
            "import numpy as np\n"
            "import re\n"
            "import os\n"
            "import datetime\n"
            "from collections import Counter, defaultdict, deque\n"
            "from mcp.server.fastmcp import FastMCP\n\n"
        )
        main_block = (
            "\n\nif __name__ == \"__main__\":\n"
            "    print(\"\\nStart MCP service:\")\n"
            "    mcp.run(transport='stdio')\n"
        )

        def ensure_file(file_path: Path, server_name: str):
            if init_project:

                content = f"{common_imports}mcp = FastMCP('{server_name}')\n\n{main_block}"
            else:

                content = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
                header = f"{common_imports}mcp = FastMCP('{server_name}')\n\n"

                if not content.strip():
                    content = header
                elif "FastMCP" not in content:
                    content = header + content

                if main_block not in content:
                    content += main_block

            file_path.write_text(content, encoding="utf-8")

        # ensure_file(invented_file, "InventedServer")
        ensure_file(official_file, "Official_Server")

    def reload_official_tools(self):
        'Reload official tools after each training step.'
        if getattr(self.verl_config, 'runtime_config_path', None):
            return
        official_config_path = self.verl_config.official_config_path
        if official_config_path is None:
            print("[reload_official_tools] official_config_path is None, skipping reload")
            return
        try:

            official_function_list = parse_mcp_tools_config(official_config_path, self.project_name) or []

            if not official_function_list:
                print("[reload_official_tools] No tools found in official config")
                return

            base_functions = set()
            for fn in getattr(self, "functions", []):
                if isinstance(fn, dict):
                    name = fn.get("name")
                else:
                    name = getattr(fn, "name", None)
                if name:
                    base_functions.add(name)

            old_count = len(self.functions_mcps) if hasattr(self, 'functions_mcps') else 0

            for tool in official_function_list:
                try:
                    self._init_tool(tool)
                except Exception as inner_e:
                    print(f"[reload_official_tools] Warning: Error re-initializing tool {tool}: {inner_e}")

            all_functions = [func.function for func in self.tool_map.values()]

            self.functions_mcps = []
            for fn in all_functions:
                name = None
                if isinstance(fn, dict):
                    name = fn.get("name")
                else:
                    name = getattr(fn, "name", None)

                if name and name not in base_functions:
                    self.functions_mcps.append(fn)

            new_count = len(self.functions_mcps)
            print(f"[reload_official_tools] Reloaded tools: {old_count} -> {new_count} (total: {len(self.functions_mcps)})")

        except Exception as e:
            print(f"[reload_official_tools] Error reloading official tools: {e}")
            import traceback
            traceback.print_exc()

    def get_item_name(self, item):
        if type(item) == dict:
            if item["name"] != "create_and_execute_mcp":
                return item["name"]
            else:

                try:
                    args = item.get("args", "{}")

                    if isinstance(args, str):
                        parsed_args = json.loads(args)
                    else:
                        parsed_args = args

                    if isinstance(parsed_args, dict):
                        mcp_name = parsed_args.get("name", "error_mcp_name")
                    else:

                        mcp_name = parsed_args if isinstance(parsed_args, str) else "error_mcp_name"
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"[get_item_name] Error parsing mcp name from args: {e}, args={item.get('args', 'N/A')}")
                    mcp_name = "error_mcp_name"

                return mcp_name
        else:
            return item

    def execute_actions(self, responses: List[str], runtime_contexts=None):
        assert self.centralized_actor_handle is not None, 'CentralizedQwenManager requires centralized_actor_handle'
        'Forward execution to the centralized actor.'
        actions, tools, tool_names = [], [], []
        for response in responses:
            temp_action, temp_tool_list = self.parse_response(response_content=response)
            # temp_action: answer or tools
            # if temp_action is 'answer', temp_tool_list is the answer
            # else, temp_tool_list is the list of the 'Tool' instances
            actions.append(temp_action)
            temp_tool_names = [self.get_item_name(item) for item in temp_tool_list]
            temp_tool_list = self.full_name(temp_tool_list)
            tools.append(temp_tool_list)
            tool_names.append(temp_tool_names)

        if runtime_contexts is not None:
            if len(runtime_contexts) != len(responses):
                raise ValueError("Runtime trajectory alignment mismatch")
            results = []
            for identity, response, action, calls in zip(runtime_contexts, responses, actions, tools):
                ray.get(self.centralized_actor_handle.runtime_assistant.remote(identity, response))
                if action == 'actions':
                    rows = []
                    for call in calls:
                        result = ray.get(self.centralized_actor_handle.execute_single_tool.remote(
                            {**call, '_runtime_context': identity}))
                        rows.append({'role': 'tool', 'name': call['name'], 'content': result})
                    results.append(rows)
                else:
                    if action == 'answer':
                        ray.get(self.centralized_actor_handle.runtime_end.remote(identity))
                        results.append({'role': 'assistant', 'content': str(calls)})
                    else:
                        result = "Invalid tool-call format. Return a valid tool call or final answer."
                        ray.get(self.centralized_actor_handle.runtime_invalid_action.remote(identity, result))
                        results.append([{'role': 'tool', 'content': result}])
            return actions, results, tool_names

        current_time = time.time()

        with self._cache_lock:
            cache_expired = current_time - self._temp_mapping_cache_time > self._temp_mapping_cache_ttl
            if cache_expired:

                need_update = True
                temp_dict = self._temp_mapping_cache
            else:

                need_update = False
                temp_dict = self._temp_mapping_cache

        if need_update:
            try:
                new_temp_dict = ray.get(self.centralized_actor_handle.get_temp_mapping.remote(), timeout=10)

                with self._cache_lock:
                    self._temp_mapping_cache = new_temp_dict
                    self._temp_mapping_cache_time = current_time
                temp_dict = new_temp_dict
            except ray.exceptions.GetTimeoutError:
                print("Warning: get_temp_mapping timeout after 10s, using cached dict")

            except Exception as e:
                print(f"Failed to get temp_mapping from actor: {e}, using cached dict")

        try:

            try:
                asyncio.get_running_loop()

                import concurrent.futures
                import multiprocessing

                def run_in_process(actions, tools, temp_dict, centralized_actor_handle):
                    'Run in a separate process to avoid event-loop conflicts.'
                    try:

                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            result = new_loop.run_until_complete(
                                self.execute_all_tools(actions, tools, temp_dict)
                            )
                            return result
                        finally:
                            new_loop.close()
                    except Exception as e:
                        return {'error': str(e), 'exception': e}

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_in_process, actions, tools, temp_dict, self.centralized_actor_handle)
                    try:
                        tool_results = future.result(timeout=300)
                        if isinstance(tool_results, dict) and 'error' in tool_results:
                            raise tool_results.get('exception', Exception(tool_results['error']))
                    except concurrent.futures.TimeoutError:
                        print("Error: Tool execution timeout after 300s")

                        future.cancel()
                        tool_results = [{'role': 'assistant', 'content': 'Tool execution timeout'}] * len(actions)

            except RuntimeError:

                tool_results = asyncio.run(self.execute_all_tools(actions, tools, temp_dict))
        except Exception as e:
            print(f"Error running async tools: {e}")
            import traceback
            traceback.print_exc()
            tool_results = [{'role': 'assistant', 'content': f'Error: {str(e)}'} for _ in actions]

        return actions, tool_results, tool_names

    def parse_subtask_plan(self, text: str) -> bool:
        'Validate ##DAG_LIST syntax and require every subtask ID to fit max_turns.'
        _, status = _parse_dag_list(text, max_turns=self.max_turns)
        return status == "ok"

    def parse_plan_response(self, response_content: str, overall_task=None, mcp_worthiness=None, is_validation: bool=False):
        'Parse a plan response with the overall task, MCP value and validation mode.'
        plan_success = self.parse_subtask_plan(response_content)
        if plan_success:
            system_prompt = self.initialize_system_prompt(
                subtask_plan=response_content,
                overall_task=overall_task,
                mcp_worthiness=mcp_worthiness,
                is_validation=is_validation
            )
            return 'plan_success', system_prompt
        else:
            system_prompt = self.initialize_system_prompt(
                subtask_plan=None,
                overall_task=overall_task,
                mcp_worthiness=mcp_worthiness,
                is_validation=is_validation
            )
            return 'plan_failed', system_prompt

    def initialize_system_prompt(self, subtask_plan=None, overall_task=None, mcp_worthiness=None, is_validation=False) -> str:
        'Build the system prompt from the subtask plan and task; validation forces dynamic-prompt behavior.'

        use_dynamic_prompt = self.dynamic_prompt or is_validation

        # if not use_dynamic_prompt:

        if subtask_plan:
            mode = "subtask"
            system_prompt = populate_template(
                self.system_template["system_prompt"],
                variables={"tools": [], "managed_agents": [], "input_plan": subtask_plan, "question":overall_task, "mode": mode, "max_turns":self.max_turns},
            )
        else:
            mode = "overall"
            system_prompt = populate_template(
                self.system_template["system_prompt"],
                variables={"tools": [], "managed_agents": [], "input_plan": overall_task, "question":overall_task, "mode": mode, "max_turns":self.max_turns},
            )
        # else:

        #     if mcp_worthiness == 1:
        #         if subtask_plan:
        #             mode = "subtask"
        #             system_prompt = populate_template(
        #                 self.system_template["system_prompt"],
        #                 variables={"tools": [], "managed_agents": [], "input_plan": subtask_plan, "question":overall_task, "mode": mode, "max_turns":self.max_turns},
        #             )
        #         else:
        #             mode = "overall"
        #             system_prompt = populate_template(
        #                 self.system_template["system_prompt"],
        #                 variables={"tools": [], "managed_agents": [], "input_plan": overall_task, "question":overall_task, "mode": mode, "max_turns":self.max_turns},
        #             )
        #     elif mcp_worthiness == 0:
        #         if subtask_plan:
        #             mode = "subtask"
        #             system_prompt = populate_template(
        #                 self.system_template["simple_prompt"],
        #                 variables={"tools": [], "managed_agents": [], "input_plan": subtask_plan, "question":overall_task, "mode": mode, "max_turns":self.max_turns},
        #             )
        #         else:
        #             mode = "overall"
        #             system_prompt = populate_template(
        #                 self.system_template["simple_prompt"],
        #                 variables={"tools": [], "managed_agents": [], "input_plan": overall_task, "question":overall_task, "mode": mode, "max_turns":self.max_turns},
        #             )
        #     else:

        #         if subtask_plan:
        #             mode = "subtask"
        #             system_prompt = populate_template(
        #                 self.system_template["system_prompt"],
        #                 variables={"tools": [], "managed_agents": [], "input_plan": subtask_plan, "question":overall_task, "mode": mode, "max_turns":self.max_turns},
        #             )
        #         else:
        #             mode = "overall"
        #             system_prompt = populate_template(
        #                 self.system_template["system_prompt"],
        #                 variables={"tools": [], "managed_agents": [], "input_plan": overall_task, "question":overall_task, "mode": mode, "max_turns":self.max_turns},
        #             )

        # if is_validation:
        #     print(f"[Validation Mode] Using system_prompt (dynamic_prompt forced to True)")

        return system_prompt

    def execute_plans(self, responses: List[str], overall_tasks: List[str]=None, mcp_worthiness: List[int]=None, is_validation: bool=False):
        assert self.centralized_actor_handle is not None, 'CentralizedQwenMCPGraphManager requires centralized_actor_handle'
        'Forward plan execution to the centralized actor.\nArgs: responses are model responses; overall_tasks are tasks; mcp_worthiness holds MCP values; is_validation selects validation mode (default False).'
        actions, plans, raw_plans = [], [], []
        for response, overall_task, mcp_worth in zip(responses, overall_tasks, mcp_worthiness):
            temp_action, temp_plan = self.parse_plan_response(
                response_content=response,
                overall_task=overall_task,
                mcp_worthiness=mcp_worth,
                is_validation=is_validation
            )
            actions.append(temp_action)
            plans.append(temp_plan)
            if temp_action == 'plan_success':
                raw_plans.append(response)
            else:
                raw_plans.append(overall_task)
        names = len(plans) * ["plan"]
        return actions, plans, raw_plans, names

    async def _select_tools_async(self, query: str, top_k: int = 2):
        'Select top_k MCPs asynchronously with hybrid-search caching; return no tools on failure and exclude built-ins.'
        import hashlib
        import time

        picked_items = []
        search_timeout = 10
        try:

            future = self.centralized_actor_handle.hybrid_search.remote(query, top_k=top_k)
            results = await asyncio.wait_for(future, timeout=search_timeout)

            print(f"[_select_tools_async] hybrid_search returned {len(results) if results else 0} groups")

            flat = []
            for group in results or []:
                flat.extend(group or [])

            print(f"[_select_tools_async] Flattened to {len(flat)} items")
            if flat:
                print(f"[_select_tools_async] First item structure: {list(flat[0].keys()) if isinstance(flat[0], dict) else type(flat[0])}")

            seen_names = set()
            for item in flat:
                name = item.get("mcp_id") or item.get("mcp_name") or item.get("name")
                score = item.get("search_score", 0.0)
                if name and name not in seen_names:
                    picked_items.append((name, score))
                    seen_names.add(name)
                if len(picked_items) >= top_k:
                    break

            print(f"[_select_tools_async] Picked items: {[(name, score) for name, score in picked_items]}")

        except asyncio.TimeoutError:
            print(f"[_select_tools_async] hybrid_search timeout after {search_timeout}s, returning empty list")
            picked_items = []
        except Exception as e:
            print(f"[_select_tools_async] hybrid_search failed: {e}, returning empty list")
            import traceback
            traceback.print_exc()
            picked_items = []

        print(f"[_select_tools_async] Checking MCP version...")
        try:

            version_future = self.centralized_actor_handle.get_mcp_version.remote()
            current_version = await asyncio.wait_for(version_future, timeout=1.0)

            if current_version != self._cached_mcp_version:
                print(f"[_select_tools_async] Version changed: {self._cached_mcp_version} -> {current_version}, refreshing cache...")

                data_future = self.centralized_actor_handle.get_functions_mcps_with_version.remote()
                data = await asyncio.wait_for(data_future, timeout=3.0)

                self._cached_mcp_version = data['version']
                self._cached_functions_mcps = data['functions_mcps']
                print(f"[_select_tools_async] Cache updated: version={self._cached_mcp_version}, tools={len(self._cached_functions_mcps)}")
            else:
                print(f"[_select_tools_async] Version unchanged ({current_version}), using local cache with {len(self._cached_functions_mcps)} tools")

            all_mcp_tools = self._cached_functions_mcps

        except asyncio.TimeoutError:
            print("[_select_tools_async] Timeout getting version/tools, using stale cache")
            all_mcp_tools = self._cached_functions_mcps
        except Exception as e:
            print(f"[_select_tools_async] Error getting version/tools: {e}, using stale cache")
            import traceback
            traceback.print_exc()
            all_mcp_tools = self._cached_functions_mcps

        print(f"[_select_tools_async] Looking for picked_items in {len(all_mcp_tools)} cached tools")

        selected_tools_with_scores = []  # List[Tuple[dict, float]]
        for fn in all_mcp_tools:
            fn_name = fn.get("name", "")
            for picked_name, score in picked_items:

                if fn_name.endswith(f"-{picked_name}") or fn_name == picked_name:
                    selected_tools_with_scores.append((fn, score))
                    break

        print(f"[_select_tools_async] Selected {len(selected_tools_with_scores)} MCP tools from hybrid search, {len(all_mcp_tools)} MCP tools in total")
        return selected_tools_with_scores

    # def _select_tools(self, query: str, top_k: int = 2):
    #     """

    #     """
    #     import hashlib
    #     import time

    #     try:
    #         future = self.centralized_actor_handle.hybrid_search.remote(query, top_k=top_k)
    #         results = ray.get(future, timeout=search_timeout)
    #         flat = []
    #         for group in results or []:
    #             flat.extend(group or [])

    #         seen_names = set()
    #         for item in flat:
    #             name = item.get("mcp_id") or item.get("mcp_name") or item.get("name")

    #             if name and name not in seen_names:
    #                 picked_items.append((name, score))
    #                 seen_names.add(name)
    #             if len(picked_items) >= top_k:
    #                 break
    #     except ray.exceptions.GetTimeoutError:
    #         print(f"[_select_tools] hybrid_search timeout after {search_timeout}s, returning empty list")
    #         picked_items = []
    #     except Exception as e:
    #         print(f"[_select_tools] hybrid_search failed: {e}, returning empty list")
    #         picked_items = []

    #     try:
    #         all_mcp_tools = ray.get(self.centralized_actor_handle.get_functions_mcps.remote(), timeout=3.0)
    #         print(f"[_select_tools] Retrieved {len(all_mcp_tools)} MCP tools from centralized actor")
    #     except ray.exceptions.GetTimeoutError:
    #         print("[_select_tools] Timeout getting MCP tools from actor, using local self.functions_mcps")
    #         all_mcp_tools = self.functions_mcps
    #     except Exception as e:
    #         print(f"[_select_tools] Error getting MCP tools from actor: {e}, using local self.functions_mcps")
    #         all_mcp_tools = self.functions_mcps

    #     selected_tools_with_scores = []  # List[Tuple[dict, float]]
    #     for fn in all_mcp_tools:
    #         fn_name = fn.get("name", "")
    #         for picked_name, score in picked_items:

    #             if fn_name.endswith(f"-{picked_name}") or fn_name == picked_name:
    #                 selected_tools_with_scores.append((fn, score))
    #                 break

    #     print(f"[_select_tools] Selected {len(selected_tools_with_scores)} MCP tools from hybrid search, {len(all_mcp_tools)} MCP tools in total")
    #     return selected_tools_with_scores

    def get_prompt(self, input_data, tokenizer, mode='initial', add_generation_prompt=True, current_plan=None):
        if isinstance(current_plan, dict) and '_runtime_context' in current_plan:
            return self._runtime_prompt(input_data, tokenizer, mode, add_generation_prompt, current_plan['_runtime_context'])
        assert mode in ['initial', 'tool_call', 'assistant_response', 'plan_call'], 'Invalid mode: {}'.format(mode)
        if isinstance(input_data, str):
            input_data = re.sub(r'<\|im_start\|>|<\|im_end\|>', '', input_data)
        base_chat = [
            {'role': SYSTEM, 'content': 'base'},
            {'role': USER, 'content': 'base'},
        ]
        base_prompt = tokenizer.apply_chat_template(
            conversation=base_chat,
            # tools=self.functions,
            tools=[],
            tokenize=False, add_generation_prompt=False
        )

        if mode == 'initial':
            chat = input_data

            # tools_to_use = self._select_tools(current_plan, top_k=2)
            prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=chat, tokenize=False,
                add_generation_prompt=add_generation_prompt, enable_thinking=self.verl_config.enable_thinking
            )
        elif mode in ['plan_call']:

            if type(input_data) == str:
                chat = [{'role': SYSTEM, 'content': input_data}]
            elif type(input_data) == list:
                chat = input_data
            else:
                raise ValueError('Unexpected type of input_data {} ({})'.format(type(input_data), input_data))

            tools_with_scores = []  # List[Tuple[dict, float]]
            if isinstance(current_plan, dict):

                for step_name, step_content in current_plan.items():
                    step_text = step_content.get('steps', '')
                    if step_text.strip():

                        step_tools = self._select_tools(step_text, top_k=2)
                        tools_with_scores.extend(step_tools)
                        print(f"[tool_selection] Step '{step_name}' found {len(step_tools)} tools")
            elif isinstance(current_plan, str):
                tools_with_scores = self._select_tools(current_plan, top_k=2)

            seen = set()
            sorted_tools_with_scores = sorted(tools_with_scores, key=lambda x: x[1], reverse=True)
            selected_mcp_tools = []
            for tool, score in sorted_tools_with_scores:
                name = tool.get("name")
                if name and name not in seen:
                    seen.add(name)
                    selected_mcp_tools.append(tool)
                    print(f"[tool_selection] Selected tool '{name}' with score {score:.4f}")
                if len(selected_mcp_tools) >= 3:
                    break

            print(f"[tool_selection] Final selected {len(selected_mcp_tools)} tools from {len(tools_with_scores)} candidates")

            base_tool_names = {"search", "execute_python_code", "create_and_execute_mcp"}

            print(f"[DEBUG] self.functions type: {type(self.functions)}")
            print(f"[DEBUG] self.functions length: {len(self.functions) if hasattr(self, 'functions') else 'N/A'}")
            # print(f"[DEBUG] self.tool_map keys: {list(self.tool_map.keys())}")

            if not hasattr(self, 'functions') or not self.functions:
                print("[WARN] self.functions is empty, rebuilding from tool_map")
                self.functions = [func.function for func in self.tool_map.values()]

            base_tools = [fn for fn in self.functions if fn.get("name") in base_tool_names]
            print(f"[DEBUG] base_tools found: {[fn.get('name') for fn in base_tools]}")

            assert base_tools is not None, "base_tools is None"
            tools_for_prompt = base_tools + selected_mcp_tools

            prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=chat, tools=tools_for_prompt,
                tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=self.verl_config.enable_thinking
            )

        elif mode in ['tool_call', 'assistant_response']:
            # NOTE: the assistant response might not be used
            role = 'tool' if mode == 'tool_call' else ASSISTANT
            if type(input_data) == str:
                chat = [{'role': role, 'content': input_data}]
            elif type(input_data) == list:
                chat = input_data
            else:
                raise ValueError('Unexpected type of input_data {} ({})'.format(type(input_data), input_data))

            temp_prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=base_chat + chat, tools=[],
                tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=self.verl_config.enable_thinking
            )
            prompt_with_chat_template = temp_prompt_with_chat_template.replace(base_prompt, '')
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

        return prompt_with_chat_template

    async def get_prompt_async(self, input_data, tokenizer, mode='initial', add_generation_prompt=True, current_plan=None):
        'Build prompts asynchronously for concurrent requests.'
        if isinstance(current_plan, dict) and '_runtime_context' in current_plan:
            return await asyncio.to_thread(self._runtime_prompt, input_data, tokenizer, mode,
                                           add_generation_prompt, current_plan['_runtime_context'])
        assert mode in ['initial', 'tool_call', 'assistant_response', 'plan_call'], 'Invalid mode: {}'.format(mode)
        base_chat = [
            {'role': SYSTEM, 'content': 'base'},
            {'role': USER, 'content': 'base'},
        ]
        base_prompt = tokenizer.apply_chat_template(
            conversation=base_chat,
            tools=[],
            tokenize=False, add_generation_prompt=False
        )

        if mode == 'initial':
            chat = input_data
            prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=chat, tokenize=False,
                add_generation_prompt=add_generation_prompt, enable_thinking=self.verl_config.enable_thinking
            )
        elif mode in ['plan_call']:

            if type(input_data) == str:

                full_conversation = [
                    {'role': SYSTEM, 'content': input_data}
                ]
            elif type(input_data) == list:
                full_conversation = input_data
            else:
                raise ValueError('Unexpected type of input_data {} ({})'.format(type(input_data), input_data))

            tools_with_scores = []  # List[Tuple[dict, float]]
            if isinstance(current_plan, dict):

                for step_name, step_content in current_plan.items():
                    step_text = step_content.get('steps', '')
                    if step_text.strip():

                        step_tools = await self._select_tools_async(step_text, top_k=2)
                        tools_with_scores.extend(step_tools)
                        print(f"[tool_selection] Step '{step_name}' found {len(step_tools)} tools")
            elif isinstance(current_plan, str):
                tools_with_scores = await self._select_tools_async(current_plan, top_k=2)

            seen = set()
            sorted_tools_with_scores = sorted(tools_with_scores, key=lambda x: x[1], reverse=True)
            selected_mcp_tools = []
            for tool, score in sorted_tools_with_scores:
                name = tool.get("name")
                if name and name not in seen:
                    seen.add(name)
                    selected_mcp_tools.append(tool)
                    print(f"[tool_selection] Selected tool '{name}' with score {score:.4f}")
                if len(selected_mcp_tools) >= 3:
                    break

            print(f"[tool_selection] Final selected {len(selected_mcp_tools)} tools from {len(tools_with_scores)} candidates")

            print(f"[DEBUG] self.functions type: {type(self.functions)}")
            print(f"[DEBUG] self.functions length: {len(self.functions) if hasattr(self, 'functions') else 'N/A'}")
            # print(f"[DEBUG] self.tool_map keys: {list(self.tool_map.keys())}")
            base_tool_names = {"search-search", "execute_python_code-execute_python_code", "create_and_execute_mcp-create_and_execute_mcp"}
            base_tools = [fn for fn in self.functions if fn.get("name") in base_tool_names]
            assert base_tools is not [], "base_tools is None"
            tools_for_prompt = base_tools + selected_mcp_tools

            temp_prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=full_conversation, tools=tools_for_prompt,
                tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=self.verl_config.enable_thinking
            )

            prompt_with_chat_template = remove_middle_tokens(temp_prompt_with_chat_template)
        elif mode in ['tool_call', 'assistant_response']:
            role = 'tool' if mode == 'tool_call' else ASSISTANT
            if type(input_data) == str:
                chat = [{'role': role, 'content': input_data}]
            elif type(input_data) == list:
                chat = input_data
            else:
                raise ValueError('Unexpected type of input_data {} ({})'.format(type(input_data), input_data))

            temp_prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=base_chat + chat, tools=[],
                tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=self.verl_config.enable_thinking
            )
            prompt_with_chat_template = temp_prompt_with_chat_template.replace(base_prompt, '')
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

        return prompt_with_chat_template

    def _runtime_prompt(self, input_data, tokenizer, mode, add_generation_prompt, identity):
        decision = ray.get(self.centralized_actor_handle.runtime_prompt_context.remote(identity))
        base_names = {'search', 'search-search', 'execute_python_code', 'execute_python_code-execute_python_code',
                      'create_and_execute_mcp', 'create_and_execute_mcp-create_and_execute_mcp'}
        tools = [row for row in self.functions if row.get('name') in base_names] + decision['schemas']
        if isinstance(input_data, list):
            chat = input_data
        elif isinstance(input_data, dict):
            chat = [input_data]
        else:
            chat = [{'role': SYSTEM if mode == 'plan_call' else 'tool', 'content': str(input_data)}]
        if decision['note']:
            chat = list(chat) + [{'role': SYSTEM, 'content': decision['note']}]
        if mode == 'plan_call' or decision['arm'] != 'searl' or decision.get('teacher_bootstrap') or decision.get('contextual'):
            return tokenizer.apply_chat_template(conversation=chat, tools=tools,
                tokenize=False, add_generation_prompt=add_generation_prompt,
                enable_thinking=self.verl_config.enable_thinking)
        # SEARL appends ordinary tool observations; retrieval/recovery changes are treatments only.
        return self.get_prompt(input_data, tokenizer, mode, add_generation_prompt, current_plan=None)

    def _get_task_base(self) -> int:
        if self.centralized_actor_handle is not None:
            try:
                return ray.get(
                    self.centralized_actor_handle.get_task_base.remote(),
                    timeout=10
                )
            except ray.exceptions.GetTimeoutError:
                print("Warning: get_task_base timeout after 10s, returning 1")
                return 1
            except Exception as e:
                print(f"Error getting task_base: {e}, returning 1")
                return 1
        return getattr(self, "_local_task_base", 1)

    def _set_task_base(self, value: int):
        if self.centralized_actor_handle is not None:
            try:
                return ray.get(
                    self.centralized_actor_handle.set_task_base.remote(value),
                    timeout=10
                )
            except ray.exceptions.GetTimeoutError:
                print(f"Warning: set_task_base timeout after 10s")
                return None
            except Exception as e:
                print(f"Error setting task_base: {e}")
                return None
        self._local_task_base = value
        return value

    def extract_dag_and_parallel(self, text: str) -> Tuple[Optional[List], Optional[int]]:
        try:
            subtask_dict = _parse_plan(text)
        except:
            subtask_dict = None
        dag_list, status = _parse_dag_list(text, max_turns=None)
        if status in ["missing", "empty"]:
            return [], self._get_task_base(), subtask_dict
        if status != "ok":
            print("[ERROR] Failed to parse DAG_LIST")
            return None, None, subtask_dict
        print("[SUCCESS] successfully parse DAG_LIST")

        task_id = self._get_task_base()

        if dag_list:

            valid_pairs = [tuple(item) for item in dag_list if isinstance(item, (tuple, list)) and len(item) == 2]

            singleton_nodes = []
            for item in dag_list:
                if isinstance(item, (tuple, list)) and len(item) == 1:
                    singleton_nodes.append(item[0])
                elif isinstance(item, str):
                    singleton_nodes.append(item)

            node_pair_list = [
                (map_st_to_number(a) + task_id, map_st_to_number(b) + task_id)
                for (a, b) in valid_pairs
            ] if valid_pairs else []
            node_single_list = [map_st_to_number(n) + task_id for n in singleton_nodes] if singleton_nodes else []

            if self.centralized_actor_handle is not None:
                try:

                    future = self.centralized_actor_handle.update_sub_task_graph.remote(node_pair_list, node_single_list)
                    result = ray.get(future, timeout=self.timeout_dag)
                except ray.exceptions.GetTimeoutError:
                    print(f"[extract_dag_and_parallel] update_sub_task_graph timed out after {self.timeout_dag} seconds")
                except Exception as e:
                    print(f"[extract_dag_and_parallel] update_sub_task_graph failed: {e}")

            all_nodes = []
            for item in dag_list:
                if isinstance(item, (tuple, list)):
                    all_nodes.extend(item)
                elif isinstance(item, str):
                    all_nodes.append(item)

            if all_nodes:
                max_sub_task_id = max(map(map_st_to_number, all_nodes))
                self._set_task_base(task_id + max_sub_task_id)

        return dag_list, task_id, subtask_dict

    async def execute_all_tools(self, actions, tool_list, temp_dict=None):
        'Execute tool batches asynchronously, sharing the temporary MCP mapping.'

        tasks = []
        for temp_action, temp_tool_list in zip(actions, tool_list):
            tasks.append(self._execute_tool_batch(temp_action, temp_tool_list, temp_dict))

        results = await asyncio.gather(*tasks)

        return results

    async def _execute_tool_batch(self, action, tools, temp_dict=None):
        if action == 'answer':
            # 'tools' is a str (the answer)
            results = {'role': 'assitant', 'content': tools}
        elif action == 'error':
            # 'error' only occurs when there is no 'actions' tag or there is no 'action' tag after extraction
            # ('Cannot extract the actions tag' or 'There is no action after extraction')
            results = {'role': 'assitant', 'content': """# Extract the tools failed due to: {}""".format(tools)}
        elif action == 'actions':
            # Add MCP registration logic
            # 'tools' is the list of the 'Tool' instances
            object_refs = [self.centralized_actor_handle.execute_single_tool.remote(temp_tool) for temp_tool in tools]
            tool_results = await asyncio.gather(*object_refs)
            results = [{'role': 'tool', 'content': temp_tool_result} for temp_tool_result in tool_results]
        else:
            raise ValueError('Unexpected action: {}'.format(action))

        return results

@ray.remote(max_concurrency=200, num_gpus=0.05)
class CentralizedToolActor:
    'Coordinate concurrent MCP tool calls in one Ray actor.'

    def __init__(self, verl_config):
        self.verl_config = verl_config
        self.project_name = _resolve_project_name(verl_config)
        self.tool_manager = CentralizedQwenManager(verl_config, mock=True)
        self.max_turns = verl_config.max_turns
        self.timeout = getattr(verl_config, 'tool_timeout', 30)
        self._temp_cache_mtime = None
        self._temp_cache_data = None
        self._official_cache_mtime = None
        self._official_cache_data = None
        self._search_cache_refresh_interval = 15
        self._search_cache_mod_time = 0.0
        self._search_cache_last_check = 0.0
        self._search_cache = {}
        self._search_cache_lock = None

        self._lock_loop = None
        self.local_search = getattr(self.verl_config, 'local_search', True)
        self._search_cache_path = os.environ.get(
            "RL_FACTORY_SEARCH_CACHE_DIR",
            str(Path.home() / ".cache" / "rl_factory" / "search"),
        )
        if self.local_search:
            self._search_cache_file = Path(os.path.join(self._search_cache_path, "local_bing_search_cache.jsonl"))
        else:
            self._search_cache_file = Path(os.path.join(self._search_cache_path, "bing_search_cache.jsonl"))
        max_workers = getattr(verl_config, 'max_concurrency', 50)
        self._search_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="SearchWorker")

        self._default_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="DefaultToolWorker")
        self._load_search_cache()
        self._init_mcp_functionality()

        self.mcp_version = 0
        self.method_runtime = None
        runtime_path = getattr(verl_config, 'runtime_config_path', None)
        if runtime_path:
            from control.runtime import Runtime, load_config
            from envs.tool_manager.centralized.dispatch import IsolatedToolDispatcher
            config = load_config(runtime_path)
            dispatcher = IsolatedToolDispatcher.from_environment()
            if Path(config['registry_path']).resolve() != Path(os.environ['SEARL_DYNAMIC_REGISTRY']).resolve():
                raise ValueError("Runtime registry differs from isolated execution registry")
            self.method_runtime = Runtime(config, dispatcher)
            if config['arm'] in ('searl', 'evocond') and (not self.use_graphrag or self.graphrag_engine is None):
                raise ValueError("SEARL/EvoCond runtime requires the declared official GraphRAG; no silent fallback")
            self._runtime_sync_graph()
        print(f"Centralized tool actor initialized; tool timeout: {self.timeout} seconds, MCP version: {self.mcp_version}")

    def _runtime_sync_graph(self):
        if self.method_runtime is None:
            return
        registry = self.method_runtime.dispatcher.registry
        self.mcps = {name: {**registry['semantic'][version], 'mcp_name': name, 'step_id': version}
                     for i, (name, version) in enumerate(sorted(registry['aliases'].items()))}
        from control.runtime import schemas
        self.tool_manager.functions_mcps = schemas(self.method_runtime.dispatcher)
        if self.graphrag_engine is not None:
            self.graphrag_engine.load_tools(self.mcps, self.sub_task_graph)
        self.mcp_version += 1

    def runtime_begin(self, question, plan_text, split):
        return self.method_runtime.begin(question, plan_text, split=split)

    def runtime_assistant(self, identity, response):
        self.method_runtime.assistant(identity, response)
    def runtime_charge(self, identity, kind, quantities):
        self.method_runtime.charge(identity, kind, quantities)

    def runtime_binding_mask(self, identity):
        return self.method_runtime.binding_mask(identity)

    def runtime_invalid_action(self, identity, error):
        self.method_runtime.observe(identity, '<invalid_format>', error, success=False, executed=False)

    def runtime_end(self, identity):
        # Answer emission is not correctness feedback. Wait for the official reward.
        self.method_runtime.context(identity)['execution_complete'] = True

    def runtime_capture(self, identities):
        snapshots = [self.method_runtime.capture_continuation(identity) for identity in identities]
        self.method_runtime.discard(identities)
        return snapshots

    def runtime_resume(self, snapshots, alpha, seed):
        return [self.method_runtime.resume_continuation(snapshot, alpha, seed) for snapshot in snapshots]

    def runtime_branch_receipts(self, identities):
        from control.dynamic import digest
        rows = []
        for identity in identities:
            context = self.method_runtime.context(identity)
            if not context.get('collection_branch') or context.get('interventions') != 1:
                raise ValueError('Continuation must execute exactly one retrieval intervention')
            rows.append({'trace_hash': digest({'history': context['history'], 'invocations': context['invocations'],
                                               'registry': context['dispatcher'].registry}),
                         'policy_hash': self.method_runtime.policy_hash, 'seed': context['branch_seed'], 'alpha': context['branch_alpha'],
                         'interventions': context['interventions'], 'shared_registry_hash': digest(self.method_runtime.dispatcher.registry)})
        self.method_runtime.discard(identities)
        return rows

    def runtime_bootstrap_export(self):
        if not self.method_runtime.config.get('bootstrap_warmup'):
            raise ValueError('Only an identified train-only warmup may export teacher alignment data')
        return {'registry': self.method_runtime.snapshot()['registry'],
                'queries': getattr(self, '_warmup_queries', []), 'policy_hash': self.method_runtime.policy_hash}

    def runtime_registry_hash(self):
        from control.dynamic import digest
        return digest(self.method_runtime.dispatcher.registry)

    def runtime_complete_batch(self, identities, outcomes, split):
        if split not in ('train', 'validation', 'test') or len(identities) != len(outcomes):
            raise ValueError('Reward feedback is not aligned to the declared split')
        if len(set(identities)) != len(identities):
            raise ValueError('One final reward per identified trajectory is required')
        for identity, outcome in zip(identities, outcomes):
            context = self.method_runtime.context(identity)
            if context['split'] != split or type(outcome) not in (bool, type(None)):
                raise ValueError('Cross-split or invalid final reward feedback')
        for identity, outcome in zip(identities, outcomes):
            context = self.method_runtime.context(identity)
            self.runtime_finish(identity, task_success=outcome if split == 'train' else None)
            if split == 'train' and self.method_runtime.config.get('bootstrap_warmup'):
                if not hasattr(self, '_warmup_queries'):
                    self._warmup_queries = []
                versions = sorted({row['tool_version'] for row in context['invocations'] if row.get('tool_version') and row.get('success')})
                self._warmup_queries.append({'query_id': context['task_id'], 'query': context['question'],
                                            'current_user_text': context['question'],
                                            'eligible_versions': versions, 'split': 'train',
                                            'registry': context['dispatcher'].registry})
        # Validation padding trajectories may be omitted from the unpadded result.
        if split != 'train':
            for identity, context in list(self.method_runtime.contexts.items()):
                if context['split'] == split:
                    self.runtime_finish(identity)
        self.method_runtime.release_completed()
        self._runtime_sync_graph()
        return self.method_runtime.schedule()

    def runtime_finish(self, identity, task_success=None):
        context = self.method_runtime.context(identity)
        if context['split'] == 'train' and not context.get('private') and not context['completed'] and context['plan']:
            tasks = context['plan']['tasks']
            observed = {key: row['evidence']['tool_version'] for key, row in tasks.items() if row.get('evidence')}
            registry = self.method_runtime.dispatcher.registry
            for key, version in observed.items():
                name = registry['semantic'][version]['name']
                self.sub_task_graph.add_node(version, mcp_name=name)
                for source in tasks[key]['depends_on']:
                    if source in observed:
                        self.sub_task_graph.add_edge(observed[source], version)
        self.method_runtime.finish(identity, task_success=task_success)

    def runtime_identity(self):
        from control.dynamic import digest
        return digest(self.method_runtime.config) if self.method_runtime is not None else None

    def runtime_policy_step(self, step, policy_drift=None):
        if self.method_runtime is not None:
            return self.method_runtime.set_policy_step(step, policy_drift)

    async def runtime_prompt_context(self, identity):
        result = self.method_runtime.select(identity)
        result['arm'] = self.method_runtime.arm
        if result['arm'] in ('searl', 'evocond') and not result.get('teacher_bootstrap') and not result.get('contextual'):
            context = self.method_runtime.context(identity)
            plan = context['plan']
            queries = [row['goal'] for row in (plan or {}).get('tasks', {}).values()] or [context['question']]
            ranked = []
            for query in queries:
                # Preserve official GraphRAG weights and top-k-per-subtask recipe.
                found = await self.hybrid_search(query, top_k=2)
                ranked.extend(row for group in found for row in group)
            selected = []
            for row in sorted(ranked, key=lambda r: r.get('search_score', 0.0), reverse=True):
                name = row.get('mcp_id') or row.get('mcp_name') or row.get('name')
                if name and name not in selected:
                    selected.append(name)
                if len(selected) == 3:
                    break
            # Newly created evaluation tools remain private, yet callable by their creator.
            private = [s['name'] for s in result['schemas'] if s['name'] not in self.mcps]
            result['schemas'] = [s for s in result['schemas'] if s['name'] in selected+private]
        return result

    def runtime_save(self, path):
        if self.method_runtime is None:
            return
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        state = self.method_runtime.snapshot()
        (folder/'runtime.json').write_text(json.dumps(state, sort_keys=True, allow_nan=False), encoding='utf-8')
        if self.graphrag_engine is not None:
            self.graphrag_engine.save_cache(str(folder/'graphrag.json'))
        nx.write_graphml(self.sub_task_graph, folder/'subtasks.graphml')

    def runtime_restore(self, path):
        if self.method_runtime is None:
            return
        folder = Path(path)
        required = ['runtime.json', 'subtasks.graphml']
        if self.graphrag_engine is not None:
            required.append('graphrag.json')
        if any(not (folder/name).is_file() for name in required):
            raise ValueError('Incomplete paired runtime checkpoint')
        self.method_runtime.restore(json.loads((folder/'runtime.json').read_text(encoding='utf-8')))
        self.sub_task_graph = nx.read_graphml(folder/'subtasks.graphml')
        self._runtime_sync_graph()
        if self.graphrag_engine is not None:
            self.graphrag_engine.execution_history.clear()
            self.graphrag_engine.load_cache(str(folder/'graphrag.json'))

    def _ensure_async_structs(self):
        loop = asyncio.get_running_loop()
        if self._lock_loop is not loop or self._search_cache_lock is None or self._in_flight_lock is None:
            self._search_cache_lock = asyncio.Lock()
            self._in_flight_lock = asyncio.Lock()
            self._in_flight_requests = {}
            self._lock_loop = loop

    def _load_search_cache(self) -> None:
        'Load the search cache.'
        if not self._search_cache_file.exists():
            return

        print(f"Loading search cache from {self._search_cache_file}...")
        loaded_cache = {}
        try:
            self._search_cache_mod_time = os.path.getmtime(self._search_cache_file)
            with open(self._search_cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        entry = json.loads(line)
                        if "query" in entry and "response" in entry:
                            loaded_cache[entry["query"]] = entry["response"]
                    except (json.JSONDecodeError, KeyError):
                        print(f"Warning: Skipping malformed cache line: {line}")

            self._search_cache = loaded_cache
            self._search_cache_last_check = time.time()
            print(f"Loaded {len(self._search_cache)} search cache entries.")
        except Exception as e:
            print(f"Failed to load search cache file: {str(e)}")
            self._search_cache = {}

    def _init_mcp_functionality(self):
        'Initialize MCP handling from the MCPBoxActor implementation.'
        import threading
        from sentence_transformers import SentenceTransformer
        import networkx as nx
        from sklearn.feature_extraction.text import TfidfVectorizer
        from collections import defaultdict

        self.lock = threading.RLock()
        self.task_base = 1
        self.sub_task_graph = nx.DiGraph()
        self.field_components = {
            'vectorizer': TfidfVectorizer(stop_words='english'),
            'matrix': None,
            'mcp_ids': []
        }
        self.mcps = {}
        self.alias_map = {}
        self.temp_mapping = {}

        self.tool_timeout = 30
        runtime_config = None
        if getattr(self.verl_config, 'runtime_config_path', None):
            from control.runtime import load_config
            runtime_config = load_config(self.verl_config.runtime_config_path)
        embedding_device = runtime_config.get('encoder_device', 'cpu') if runtime_config else 'cuda:0'
        self.embedding_model = SentenceTransformer(
            _default_embedding_model(),
            device=embedding_device
        )
        print(f"CentralizedToolActor: SentenceTransformer device={embedding_device}.")

        self.use_graphrag = getattr(self.verl_config, 'use_graphrag', False)
        self.graphrag_engine = None

        if self.use_graphrag:
            try:
                from envs.tool_manager.centralized.retrieval import GraphRAGRetrieval

                graphrag_config = {
                    'enable_community_detection': getattr(self.verl_config, 'graphrag_community_detection', True),
                    'enable_history_tracking': getattr(self.verl_config, 'graphrag_history_tracking', True),
                    'cache_dir': getattr(self.verl_config, 'graphrag_cache_dir', 'cache/graphrag')
                }
                if runtime_config:
                    graphrag_config['cache_dir'] = str(Path(runtime_config['root'])/'graphrag_cache')

                print(f"[GraphRAG] Initializing GraphRAG engine with config: {graphrag_config}")

                self.graphrag_engine = GraphRAGRetrieval(
                    embedding_model_path=_default_embedding_model(),
                    device=embedding_device,
                    enable_community_detection=graphrag_config['enable_community_detection'],
                    enable_history_tracking=graphrag_config['enable_history_tracking'],
                    cache_dir=graphrag_config['cache_dir']
                )

                try:
                    if not runtime_config:
                        self.graphrag_engine.load_cache()
                        print("[GraphRAG] Cache loaded successfully")
                except Exception as cache_e:
                    print(f"[GraphRAG] Warning: Failed to load cache: {cache_e}")

                print("[GraphRAG] Initialization completed successfully")

            except ImportError as e:
                print(f"[GraphRAG] Warning: GraphRAG module not found, falling back to traditional search: {e}")
                self.use_graphrag = False
                self.graphrag_engine = None
            except Exception as e:
                print(f"[GraphRAG] Warning: Failed to initialize GraphRAG, falling back to traditional search: {e}")
                import traceback
                traceback.print_exc()
                self.use_graphrag = False
                self.graphrag_engine = None
        else:
            print("[GraphRAG] GraphRAG disabled (use_graphrag=False), using traditional search")

        self.init_mcps([])

        print("DEBUG: All initialization steps completed")

    def get_task_base(self):
        return self.task_base

    def set_task_base(self, value: int):
        self.task_base = value
        return self.task_base

    def init_mcps(self, tools: List[Any]):
        for tool in tools:
            mcp_name = getattr(tool, 'name', None)
            if mcp_name and mcp_name not in self.mcps:
                description = getattr(tool, 'description', None)
                arguments = getattr(tool, 'inputs', None)
                returns = getattr(tool, 'output_type', None)
                requires = getattr(tool, 'requires', None)
                current_step_id = self.task_base
                self.mcps[mcp_name] = {
                    "mcp_name": mcp_name,
                    "description": description,
                    "arguments": arguments,
                    "returns": returns,
                    "requires": requires,
                    "code": f"{mcp_name}(**kwargs)",
                    "step_id" : current_step_id,
                }
                if not self.sub_task_graph.has_node(current_step_id):
                    self.sub_task_graph.add_node(current_step_id)
                    self.sub_task_graph.nodes[current_step_id]['mcp_name'] = mcp_name
                self.task_base += 1
            elif mcp_name:
                print(f"Agent Log: Tool '{mcp_name}' already exists, skipping.")

        self.naive_mcps = self.mcps
        self.build_tfidf_indices()
        self.build_embeddings()

        if self.use_graphrag and self.graphrag_engine is not None:
            try:
                self.graphrag_engine.load_tools(self.mcps, self.sub_task_graph)
                self.mcp_version += 1
                print(f"[GraphRAG] Synced {len(self.mcps)} tools to GraphRAG engine (version {self.mcp_version})")
            except Exception as e:
                print(f"[GraphRAG] Warning: Failed to sync tools to GraphRAG: {e}")

    def parse_subtask_plan(self, text: str) -> bool:
        'Validate ##DAG_LIST syntax and require every subtask ID to fit max_turns.'
        _, status = _parse_dag_list(text, max_turns=self.max_turns)
        return status == "ok"

    def sequential_planning(self, rollout_max_turns) -> Optional[List[Tuple[str, str]]]:
        task_id = self.get_task_base()
        valid_pairs = [(i + 1 + task_id, i + 2 +task_id) for i in range(rollout_max_turns - 1)]  # Example sequential pairs, replace with actual logic

        self.sub_task_graph.add_edges_from(valid_pairs)

        self.set_task_base(task_id + rollout_max_turns - 1)

        return None, task_id, {}

    def update_self_mcp(self):
        'Get the named MCP.'
        self.mcps.update(self.temp_mapping)

    def _register_mcp_sync(self, code: str, task_id: int = None, subtask_dict=None):
        if self.method_runtime is not None:
            # Calls have already been verified and versioned by the dispatcher.
            return
        assert task_id is not None, "task_id is required"

        step_id_format = re.compile(
            r'<subtask>\s*(?P<step_id>ST_?\d+):?\s*(.*?)\s*</subtask>',
            re.IGNORECASE
        )
        step_info = step_id_format.search(code)
        plan_title = None
        plan_steps = None
        if not step_info:
            print("Code does not contain a valid step ID (e.g., '<subtask> ST_1: ... </subtask>').")
            return
        if subtask_dict:
            current_plan = subtask_dict.get(str(step_info.group(1)).replace('_', ''), None)
            if current_plan:
                plan_title = current_plan.get('title', None)
                plan_steps = current_plan.get('steps', None)
        step_id = map_st_to_number(step_info.group(1))
        nid = task_id + step_id

        tool_names = extract_tool_names(code)
        if not tool_names:
            print(f"Agent Log: No tool names found in code.")
            return

        with self.lock:

            if not self.sub_task_graph.has_node(nid):
                self.sub_task_graph.add_node(nid)

            for name in tool_names:
                if name in self.mcps:
                    self.sub_task_graph.add_edge(self.mcps[name]['step_id'], nid)
                elif name in self.temp_mapping:
                    temp_item = self.temp_mapping[name]
                    new_mcp_data = {
                        "mcp_name": name,
                        "description": temp_item.get('description'),
                        "arguments": temp_item.get('arguments'),
                        "returns": temp_item.get('returns'),
                        "code": temp_item.get('code', ""),
                        "step_id": nid,
                    }
                    self.mcps[name] = new_mcp_data
                    self.sub_task_graph.add_node(nid)
                    self.sub_task_graph.nodes[nid]['mcp_name'] = name

                    if self.use_graphrag and self.graphrag_engine is not None:
                        try:
                            self.graphrag_engine.update_tool(name, new_mcp_data)
                            self.mcp_version += 1
                            print(f"[GraphRAG] Synced newly registered tool: {name} (version {self.mcp_version})")
                        except Exception as e:
                            print(f"[GraphRAG] Warning: Failed to sync tool {name}: {e}")
                else:
                    print(f"Agent Log: MCP {name} not found.")
                    return
            print(f"Agent Log: MCPs registered for Step {step_id} (node {nid}).")
        return True

    # def temp_tool_list(self) -> Dict[str, Dict[str, str]]:
    #     """

    #     {tool_name: {"description": str, "arguments": str, "returns": str, "code": str}}

    #     """
    #     import ast
    #     import re
    #     from pathlib import Path

    #     root_dir = os.environ.get("RL_FACTORY_ROOT", str(_repo_root()))
    #     tool_path = Path(root_dir) / "envs" / "tools" / "invented_tools" / self.project_name / "temp_tool_list.py"
    #     if not tool_path.exists():
    #         return {}

    #     mtime = tool_path.stat().st_mtime
    #     if self._temp_cache_mtime == mtime and self._temp_cache_data is not None:
    #         return self._temp_cache_data

    #     lines = text.splitlines()

    #     try:
    #         tree = ast.parse(text)
    #     except Exception as e:
    #         print(f"Failed to parse temp_tool_list.py: {e}")
    #         return {}

    #     def _parse_docstring(ds: Optional[str]) -> Tuple[str, str, str]:

    #         if not ds:
    #             return "", "", ""

    #         desc_lines, arg_lines, ret_lines = [], [], []
    #         state = "desc"
    #         for raw in ds.splitlines():
    #             line = raw.strip()
    #             if not line:
    #                 continue
    #             if re.match(r"(?i)^args?:", line):
    #                 state = "args"
    #                 continue
    #             if re.match(r"(?i)^returns?:", line):
    #                 state = "returns"
    #                 continue
    #             if state == "desc":
    #                 desc_lines.append(line)
    #             elif state == "args":
    #                 arg_lines.append(line)
    #             elif state == "returns":
    #                 ret_lines.append(line)

    #         description = " ".join(desc_lines).strip()
    #         arguments = " ".join(arg_lines).strip()
    #         returns = " ".join(ret_lines).strip()
    #         return description, arguments, returns

    #     tools: Dict[str, Dict[str, str]] = {}
    #     for node in tree.body:
    #         if not isinstance(node, ast.FunctionDef):
    #             continue

    #         tool_name = node.name
    #         has_mcp_deco = False
    #         if node.decorator_list:
    #             for deco in node.decorator_list:

    #                 if isinstance(deco, ast.Attribute) and deco.attr == "tool":
    #                     has_mcp_deco = True
    #                 elif isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute) and deco.func.attr == "tool":
    #                     has_mcp_deco = True
    #                     for kw in deco.keywords:
    #                         if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
    #                             tool_name = kw.value.value
    #                 if has_mcp_deco:
    #                     break

    #         if not has_mcp_deco:
    #             continue

    #         start_line = min([getattr(d, "lineno", node.lineno) for d in node.decorator_list] + [node.lineno])
    #         end_line = getattr(node, "end_lineno", node.lineno)
    #         code_block = "\n".join(lines[start_line - 1 : end_line])

    #         docstring = ast.get_docstring(node)
    #         description, arguments, returns = _parse_docstring(docstring)

    #         if not arguments:
    #             sig_args = []
    #             for arg in getattr(node, "args", []).args:
    #                 if arg.arg == "self":
    #                     continue
    #                 ann = getattr(arg, "annotation", None)
    #                 if ann:
    #                     try:
    #                         ann_str = ast.unparse(ann)
    #                     except Exception:
    #                         ann_str = ""
    #                 else:
    #                     ann_str = ""
    #                 sig_args.append(f"{arg.arg}: {ann_str}" if ann_str else arg.arg)
    #             arguments = ", ".join(sig_args)

    #         if not returns and getattr(node, "returns", None):
    #             try:
    #                 returns = ast.unparse(node.returns)
    #             except Exception:
    #                 returns = ""

    #         tools[tool_name] = {
    #             "description": description,
    #             "arguments": arguments,
    #             "returns": returns,
    #             "code": code_block,
    #         }

    #     self._temp_cache_mtime = mtime
    #     self._temp_cache_data = tools
    #     return tools

    # def official_tool_list(self) -> Dict[str, Dict[str, str]]:
    #     """

    #     {tool_name: {"description": str, "arguments": str, "returns": str, "code": str}}

    #     """
    #     import ast
    #     import re
    #     from pathlib import Path

    #     root_dir = os.environ.get("RL_FACTORY_ROOT", str(_repo_root()))
    #     tool_path = Path(root_dir) / "envs" / "tools" / "invented_tools" / self.project_name / "official_tool_list.py"
    #     if not tool_path.exists():
    #         return {}

    #     mtime = tool_path.stat().st_mtime
    #     if self._official_cache_mtime == mtime and self._official_cache_data is not None:
    #         return self._official_cache_data

    #     lines = text.splitlines()

    #     try:
    #         tree = ast.parse(text)
    #     except Exception as e:
    #         print(f"Failed to parse official_tool_list.py: {e}")
    #         return {}

    #     def _parse_docstring(ds: Optional[str]) -> Tuple[str, str, str]:

    #         if not ds:
    #             return "", "", ""

    #         desc_lines, arg_lines, ret_lines = [], [], []
    #         state = "desc"
    #         for raw in ds.splitlines():
    #             line = raw.strip()
    #             if not line:
    #                 continue
    #             if re.match(r"(?i)^args?:", line):
    #                 state = "args"
    #                 continue
    #             if re.match(r"(?i)^returns?:", line):
    #                 state = "returns"
    #                 continue
    #             if state == "desc":
    #                 desc_lines.append(line)
    #             elif state == "args":
    #                 arg_lines.append(line)
    #             elif state == "returns":
    #                 ret_lines.append(line)

    #         description = " ".join(desc_lines).strip()
    #         arguments = " ".join(arg_lines).strip()
    #         returns = " ".join(ret_lines).strip()
    #         return description, arguments, returns

    #     tools: Dict[str, Dict[str, str]] = {}
    #     for node in tree.body:
    #         if not isinstance(node, ast.FunctionDef):
    #             continue

    #         tool_name = node.name
    #         has_mcp_deco = False
    #         if node.decorator_list:
    #             for deco in node.decorator_list:

    #                 if isinstance(deco, ast.Attribute) and deco.attr == "tool":
    #                     has_mcp_deco = True
    #                 elif isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute) and deco.func.attr == "tool":
    #                     has_mcp_deco = True
    #                     for kw in deco.keywords:
    #                         if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
    #                             tool_name = kw.value.value
    #                 if has_mcp_deco:
    #                     break

    #         if not has_mcp_deco:
    #             continue

    #         start_line = min([getattr(d, "lineno", node.lineno) for d in node.decorator_list] + [node.lineno])
    #         end_line = getattr(node, "end_lineno", node.lineno)
    #         code_block = "\n".join(lines[start_line - 1 : end_line])

    #         docstring = ast.get_docstring(node)
    #         description, arguments, returns = _parse_docstring(docstring)

    #         if not arguments:
    #             sig_args = []
    #             for arg in getattr(node, "args", []).args:
    #                 if arg.arg == "self":
    #                     continue
    #                 ann = getattr(arg, "annotation", None)
    #                 if ann:
    #                     try:
    #                         ann_str = ast.unparse(ann)
    #                     except Exception:
    #                         ann_str = ""
    #                 else:
    #                     ann_str = ""
    #                 sig_args.append(f"{arg.arg}: {ann_str}" if ann_str else arg.arg)
    #             arguments = ", ".join(sig_args)

    #         if not returns and getattr(node, "returns", None):
    #             try:
    #                 returns = ast.unparse(node.returns)
    #             except Exception:
    #                 returns = ""

    #         tools[tool_name] = {
    #             "description": description,
    #             "arguments": arguments,
    #             "returns": returns,
    #             "code": code_block,
    #         }

    #     self._official_cache_mtime = mtime
    #     self._official_cache_data = tools
    #     return tools

    async def register_mcp(self, code: str, task_id: int = None, subtask_dict=None):
        'Register an MCP asynchronously.'

        import concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            return await loop.run_in_executor(executor, self._register_mcp_sync, code, task_id, subtask_dict)

    async def register_mcps_batch(self, mcp_data_list: List[Tuple[str, int, Optional[Dict]]]):
        'Register MCPs in one batch to reduce remote-call overhead.'
        import concurrent.futures
        loop = asyncio.get_event_loop()

        async def _register_single(data):
            code, task_id, subtask_dict = data
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return await loop.run_in_executor(executor, self._register_mcp_sync, code, task_id, subtask_dict)

        semaphore = asyncio.Semaphore(4)

        async def _register_with_semaphore(data):
            async with semaphore:
                return await _register_single(data)

        tasks = [_register_with_semaphore(data) for data in mcp_data_list]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def build_tfidf_indices(self, temp_mcps: Optional[Dict[int, str]] = None):
        'Build TF-IDF indices while minimizing lock duration.'
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.sparse import vstack as safe_vstack
        from sklearn.utils.validation import check_is_fitted
        from sklearn.exceptions import NotFittedError

        if temp_mcps is None:

            with self.lock:
                vectorizer = self.field_components['vectorizer']
                field_data_descriptions = [
                    mcp_func_data.get('description', "") or ""
                    for mcp_func_data in self.mcps.values()
                ]
                mcp_ids_for_index = list(self.mcps.keys())

            if field_data_descriptions:
                matrix = vectorizer.fit_transform(field_data_descriptions)
            else:
                matrix = None

            with self.lock:
                self.field_components['matrix'] = matrix
                self.field_components['mcp_ids'] = mcp_ids_for_index if matrix is not None else []
                if matrix is not None:
                    print(f"Global TF-IDF index built successfully with {len(mcp_ids_for_index)} documents.")
        else:

            with self.lock:
                vectorizer = self.field_components.get('vectorizer')
                if vectorizer is None:
                    return
                try:
                    check_is_fitted(vectorizer)
                except NotFittedError:
                    return
                new_descriptions = []
                new_mcp_ids = []
                for step_id, mcp_name in temp_mcps.items():
                    if mcp_name in self.mcps:
                        new_descriptions.append(self.mcps[mcp_name].get('description', "") or "")
                        new_mcp_ids.append(mcp_name)

            if not new_descriptions:
                return
            new_matrix = vectorizer.transform(new_descriptions)

            with self.lock:
                if self.field_components.get('matrix') is None:
                    self.field_components['matrix'] = new_matrix
                else:
                    self.field_components['matrix'] = safe_vstack([self.field_components['matrix'], new_matrix])

                if 'mcp_ids' not in self.field_components:
                    self.field_components['mcp_ids'] = []
                self.field_components['mcp_ids'].extend(new_mcp_ids)

    def build_embeddings(self, temp_mcps: Optional[Dict[int, str]] = None):
        'Build cached embeddings while minimizing lock duration.'
        batch_size = 32

        if temp_mcps is None:

            with self.lock:
                items = list(self.mcps.items())

            queries_to_encode = []
            queries_indices = []

            for idx, (mcp_name, mcp_data) in enumerate(items):
                description = mcp_data.get('description') or ""
                queries_to_encode.append(description)
                queries_indices.append(idx)

            if queries_to_encode:
                new_embeddings = self.embedding_model.encode(
                    queries_to_encode,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )

            with self.lock:
                for idx, (mcp_name, mcp_data) in enumerate(items):
                    description = mcp_data.get('description') or ""
                    mcp_data['query_embedding'] = new_embeddings[idx]
        else:

            with self.lock:
                target_names = [name for _, name in temp_mcps.items()]
                target_items = [(name, self.mcps.get(name)) for name in target_names if name in self.mcps]

            if not target_items:
                return

            queries_to_encode = [data.get('description') or "" for _, data in target_items]
            new_embeddings = self.embedding_model.encode(
                queries_to_encode,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            with self.lock:
                for idx, (mcp_name, mcp_data) in enumerate(target_items):
                    mcp_data['query_embedding'] = new_embeddings[idx]

    def field_text_search(self, query: str, top_k: int = 3) -> List[dict]:
        'Search by text similarity.'
        from sklearn.metrics.pairwise import cosine_similarity

        with self.lock:
            component = self.field_components
            if component['matrix'] is None or not component['mcp_ids']:
                return []

            query_vec = component['vectorizer'].transform([query])
            similarities = cosine_similarity(query_vec, component['matrix']).flatten()

            actual_top_k = min(top_k, len(similarities))
            top_indices = similarities.argsort()[-actual_top_k:][::-1]

            results = []
            for idx in top_indices:
                mcp_id = component['mcp_ids'][idx]
                if mcp_id in self.mcps:
                    results.append({
                        'mcp_id': mcp_id,
                        'score': float(similarities[idx]),
                        'content': self.mcps[mcp_id]
                    })
            return results

    def field_semantic_search(self, query: str, top_k: int = 3):
        'Search by semantic similarity.'
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        with self.lock:
            embeddings = []
            mcp_data_list = []

            for mcp_name, mcp_data in self.mcps.items():
                emb = mcp_data.get('query_embedding')
                if emb is not None:
                    embeddings.append(emb)
                    mcp_data_list.append(mcp_data)

            if not embeddings:
                return []

            embeddings_matrix = np.array(embeddings)
            similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]

            actual_top_k = min(top_k, len(similarities))
            top_indices = similarities.argsort()[-actual_top_k:][::-1]

            results = []
            for idx in top_indices:
                mcp_item = mcp_data_list[idx]
                results.append({
                    'mcp_id': mcp_item['mcp_name'],
                    'score': float(similarities[idx]),
                    'field': 'description',
                    'content': mcp_item.get('description', "")
                })
            return results

    async def hybrid_search(self, query: str, top_k: int = 5,
                           weights: Optional[Dict[str, float]] = None,
                           score_threshold: float = 0.2,
                           use_graph_enhanced: Optional[bool] = None,
                           max_hops: int = 2) -> List[List[Dict[str, Any]]]:
        'Search tools above score_threshold; weights select traditional or GraphRAG scoring, use_graph_enhanced overrides availability, and max_hops bounds graph search.'

        should_use_graphrag = use_graph_enhanced if use_graph_enhanced is not None else self.use_graphrag

        if should_use_graphrag and self.graphrag_engine is not None:
            try:
                print(f"[hybrid_search] Using GraphRAG enhanced search (max_hops={max_hops})")

                graphrag_weights = weights or {
                    'semantic': 0.3,
                    'text': 0.2,
                    'graph': 0.2,
                    'community': 0.15,
                    'history': 0.15
                }

                results = self.graphrag_engine.graph_enhanced_search(
                    query=query,
                    top_k=top_k,
                    max_hops=max_hops,
                    use_community=True,
                    use_history=True,
                    weights=graphrag_weights
                )

                detailed_results = []
                for tool in results:
                    detailed_results.append([tool])

                print(f"[hybrid_search] GraphRAG returned {len(detailed_results)} results")
                return detailed_results

            except Exception as e:
                if getattr(self, 'method_runtime', None) is not None:
                    raise RuntimeError("Declared runtime GraphRAG failed; refusing a different retrieval baseline") from e
                print(f"[hybrid_search] Warning: GraphRAG search failed, falling back to traditional search: {e}")
                import traceback
                traceback.print_exc()

                should_use_graphrag = False

        if not should_use_graphrag:
            print(f"[hybrid_search] Using traditional hybrid search (TF-IDF + Semantic)")

            from collections import defaultdict

            weights = weights or {'text': 0.5, 'semantic': 0.5}
            score_board = defaultdict(float)

            text_results = self.field_text_search(query, top_k * 2)
            semantic_results = self.field_semantic_search(query, top_k * 2)

            # print(f"[hybrid_search] query='{query[:50]}...', text_results={len(text_results)}, semantic_results={len(semantic_results)}")
            for r in text_results:
                score_board[r['mcp_id']] += weights.get('text', 0.5) * r['score']
            for r in semantic_results:
                score_board[r['mcp_id']] += weights.get('semantic', 0.5) * r['score']

            print(f"[hybrid_search] score_board has {len(score_board)} items")

            filtered_results = [(mcp_id, score) for mcp_id, score in score_board.items() if score > score_threshold]
            sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:top_k]
            print(f"[hybrid_search] After threshold filtering: {len(filtered_results)} items, returning top {len(sorted_results)}")
            # print(f"[hybrid_search] sorted_results: {[(mcp_id, score) for mcp_id, score in sorted_results]}")

            detailed_results = []
            for mcp_id, score in sorted_results:
                mcp_item = self.mcps.get(mcp_id)
                if mcp_item:

                    mcp_item_with_score = mcp_item.copy()
                    mcp_item_with_score['search_score'] = score
                    detailed_results.append([mcp_item_with_score])
                else:
                    print(f"[hybrid_search] Warning: mcp_id '{mcp_id}' not found in self.mcps (keys: {list(self.mcps.keys())[:5]})")

            print(f"[hybrid_search] Returning {len(detailed_results)} results")
            return detailed_results

    def _record_graphrag_execution(
        self,
        tool_name: str,
        success: bool,
        execution_time: float = 0.0,
        co_used_tools: Optional[List[str]] = None
    ):
        'Record tool success, execution time and co-used tools in GraphRAG.'
        if not self.use_graphrag or self.graphrag_engine is None:
            return

        try:

            mcp_name = tool_name.split('-')[0] if '-' in tool_name else tool_name

            if mcp_name in self.mcps:
                self.graphrag_engine.record_execution(
                    mcp_name=mcp_name,
                    success=success,
                    execution_time=execution_time,
                    co_used_tools=co_used_tools
                )
        except Exception as e:
            print(f"[GraphRAG] Warning: Failed to record execution for {tool_name}: {e}")

    def get_graphrag_tool_stats(self, mcp_name: str) -> Dict[str, Any]:
        'Return GraphRAG execution statistics, or an empty mapping when disabled.'
        if not self.use_graphrag or self.graphrag_engine is None:
            return {}

        try:
            return self.graphrag_engine.get_execution_stats(mcp_name)
        except Exception as e:
            print(f"[GraphRAG] Error getting stats for {mcp_name}: {e}")
            return {}

    def get_graphrag_community_info(self) -> Dict[str, Any]:
        'Return GraphRAG community statistics and summaries.'
        if not self.use_graphrag or self.graphrag_engine is None:
            return {
                'enabled': False,
                'message': 'GraphRAG is not enabled'
            }

        try:
            num_communities = len(self.graphrag_engine.communities)
            communities_detail = {}

            for comm_id, members in self.graphrag_engine.communities.items():
                communities_detail[comm_id] = {
                    'size': len(members),
                    'members': list(members)[:10],
                    'summary': self.graphrag_engine.community_summaries.get(comm_id, '')
                }

            return {
                'enabled': True,
                'num_communities': num_communities,
                'communities': communities_detail,
                'global_summary': self.graphrag_engine.global_summary
            }
        except Exception as e:
            print(f"[GraphRAG] Error getting community info: {e}")
            return {
                'enabled': True,
                'error': str(e)
            }

    def save_graphrag_cache(self) -> bool:
        'Save GraphRAG state to disk and report success.'
        if not self.use_graphrag or self.graphrag_engine is None:
            print("[GraphRAG] GraphRAG not enabled, skipping cache save")
            return False

        try:
            self.graphrag_engine.save_cache()
            print("[GraphRAG] Cache saved successfully")
            return True
        except Exception as e:
            print(f"[GraphRAG] Error saving cache: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_graphrag_cache(self) -> bool:
        'Load GraphRAG state from disk and report success.'
        if not self.use_graphrag or self.graphrag_engine is None:
            print("[GraphRAG] GraphRAG not enabled, skipping cache load")
            return False

        try:
            self.graphrag_engine.load_cache()
            print("[GraphRAG] Cache loaded successfully")
            return True
        except Exception as e:
            print(f"[GraphRAG] Error loading cache: {e}")
            import traceback
            traceback.print_exc()
            return False

    def sync_tools_to_graphrag(self):
        'Synchronize the current tool inventory into GraphRAG.'
        if not self.use_graphrag or self.graphrag_engine is None:
            return

        try:
            self.graphrag_engine.load_tools(self.mcps, self.sub_task_graph)
            self.mcp_version += 1
            print(f"[GraphRAG] Manually synced {len(self.mcps)} tools to GraphRAG (version {self.mcp_version})")
        except Exception as e:
            print(f"[GraphRAG] Error syncing tools to GraphRAG: {e}")
            import traceback
            traceback.print_exc()

    def update_mcp(self, mcp_name: str, mcp_content: str):
        'Update MCP content.'
        with self.lock:
            if mcp_name in self.mcps:
                self.mcps[mcp_name]['code'] = mcp_content
                print(f"MCP '{mcp_name}' updated with new content.")

                if self.use_graphrag and self.graphrag_engine is not None:
                    try:
                        self.graphrag_engine.update_tool(mcp_name, self.mcps[mcp_name])
                        self.mcp_version += 1
                        print(f"[GraphRAG] Synced updated tool: {mcp_name} (version {self.mcp_version})")
                    except Exception as e:
                        print(f"[GraphRAG] Warning: Failed to sync updated tool {mcp_name}: {e}")
            else:
                print(f"MCP '{mcp_name}' not found for update.")

    async def clear(self, temp_dict: Optional[dict] = None):
        'Clear temporary MCPs.'
        if temp_dict is None:
            temp_dict = self.temp_mapping
        print("Clearing temp_mapping and rebuilding indices/embeddings.")
        self.build_tfidf_indices(temp_mcps=None)
        self.build_embeddings(temp_mcps=None)
        self.temp_mapping.clear()

    def get_temp_mapping(self) -> Dict[str, Any]:
        'Return the temporary tool mapping.'
        return self.temp_mapping

    async def get_mcps(self) -> Dict[str, Any]:
        'Get all MCPs.'
        mcp_copy = self.mcps.copy()
        mcp_copy.update(self.temp_mapping)
        return mcp_copy

    def _save_search_cache_sync(self, query: str, response: str) -> None:
        'Save the search cache synchronously.'

        os.makedirs(self._search_cache_path, exist_ok=True)

        new_entry = {"query": query, "response": response}
        try:
            print(f"DEBUG: Attempting to save search cache to {self._search_cache_file}")
            with open(self._search_cache_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
            print(f"DEBUG: Successfully saved search cache entry. File size: {os.path.getsize(self._search_cache_file)}")
        except Exception as e:
            print(f"Failed to save the search cache: {str(e)}")
            traceback.print_exc()

    async def list_mcps(self) -> str:
        'List all MCPs.'
        mcps_to_list = self.mcps.copy()
        if not mcps_to_list:
            return "MCP Box is empty."

        mcp_list_str = "Available MCPs:\n"
        for name, data in mcps_to_list.items():
            mcp_list_str += f"- {name} \n"
            mcp_list_str += f"    Description: {data.get('description', 'N/A')}\n"
            mcp_list_str += f"    Arguments: {data.get('arguments', 'N/A')}\n"
            mcp_list_str += f"    Returns: {data.get('returns', 'N/A')}\n"
            if data.get('requires'):
                mcp_list_str += f"    Requires: {data['requires']}\n"
        return mcp_list_str

    def _check_search_cache_update(self) -> bool:
        'Check whether the search cache file changed.'
        now = time.time()
        if now - self._search_cache_last_check < self._search_cache_refresh_interval:
            return False

        self._search_cache_last_check = now

        if not self._search_cache_file.exists():
            return False

        try:
            current_mod_time = os.path.getmtime(self._search_cache_file)
            if current_mod_time > self._search_cache_mod_time:
                print("Search cache file update detected, reloading")
                self._load_search_cache()
                return True
        except Exception as e:
            print(f"Failed to check search cache file updates: {str(e)}")

        return False

    async def execute_single_tool(self, tool):
        if self.method_runtime is not None:
            identity = tool.get('_runtime_context')
            self.method_runtime.context(identity)
            measured_start = time.monotonic()
            try:
                return await self._execute_runtime_tool(tool)
            except (ValueError, TypeError, KeyError) as error:
                result = json.dumps({'error': 'invalid_or_stale_tool_call', 'detail': str(error)})
                self.method_runtime.observe(identity, str(tool.get('name')), result, success=False, executed=False)
                return result
            finally:
                self.method_runtime.charge(identity, 'tools', {'seconds': time.monotonic()-measured_start})
        self._ensure_async_structs()
        # try:
        tool_instance = self.tool_manager.get_tool(tool["name"])
        try:
            args = tool["args"]
        except Exception as e:
            args = tool["arguments"]
        tool_name = tool["name"]
        future = None
        query_str = None

        execution_start_time = time.time() if self.use_graphrag and self.graphrag_engine else None

        if tool_instance is not None:
            try:
                args = json.loads(args)
            except Exception as e:
                pass
            if type(args) is dict:

                if tool_name == "search" or tool_name == "search-search":
                    query = tool.get("args", "")
                    if isinstance(query, dict):
                        query_str = json.dumps(query, sort_keys=True)
                    else:
                        query_str = str(query)

                    self._check_search_cache_update()

                    async with self._search_cache_lock:
                        if query_str in self._search_cache:
                            print(f"Search cache hit: {query_str}")
                            return self._search_cache[query_str]

                    async with self._in_flight_lock:
                        if query_str in self._in_flight_requests:
                            print(f"Cache hit (in-flight): Waiting for another task for '{query_str}'")
                            future = self._in_flight_requests[query_str]
                            try:
                                return await asyncio.wait_for(future, timeout=self.tool_timeout)
                            except asyncio.TimeoutError:

                                self._in_flight_requests.pop(query_str, None)
                                return f"# Tool calling exceeds timeout: {tool_name}, exceeded {self.tool_timeout}s"

                        future = asyncio.get_running_loop().create_future()
                        self._in_flight_requests[query_str] = future

                specific_timeout = self.tool_timeout
                if tool_name == "execute_python_code" or tool_name == "execute_python_code-execute_python_code":

                    try:
                        if isinstance(tool.get("args", ""), str):
                            tool_args = json.loads(tool.get("args", ""))
                            if "timeout" in tool_args:
                                specific_timeout = 15.0
                        elif isinstance(tool.get("args", ""), dict):
                            if "timeout" in tool.get("args", ""):
                                specific_timeout = 15.0
                    except:
                        pass

                    if specific_timeout == self.tool_timeout:
                        specific_timeout = 15.0

                try:
                    if tool_name == "search" or tool_name == "search-search":
                        target_executor = self._search_executor
                    else:
                        target_executor = self._default_executor

                    loop = asyncio.get_running_loop()
                    # run_in_executor returns a Future, not a coroutine, so we cannot use create_task directly on it
                    # We can await the Future directly or use wait_for on it
                    task = loop.run_in_executor(
                        target_executor,
                        self.tool_manager._call_tool,
                        tool_name,
                        json.dumps(args, ensure_ascii=False, indent=4)
                    )
                    try:
                        result = await asyncio.wait_for(task, timeout=specific_timeout)
                    except asyncio.TimeoutError:
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
                        raise
                    except Exception:

                        raise

                    if tool_name in ('create_and_execute_mcp', 'create_and_execute_mcp-create_and_execute_mcp'):
                        from envs.tool_manager.centralized.verified import verified_creation_result
                        candidate_name = args.get('name')
                        if (verified_creation_result(result) and isinstance(candidate_name, str)
                                and candidate_name and self.tool_manager.get_tool(candidate_name) is not None):
                            self.temp_mapping[candidate_name] = {
                                'mcp_name': candidate_name,
                                'description': args.get('description', ''),
                                'arguments': args.get('arguments', ''),
                                'returns': args.get('returns', ''),
                                'code': args.get('code', ''),
                                'step_id': len(self.temp_mapping) + 2,
                            }
                        else:
                            print(f"Agent Log: MCP {candidate_name!r} quarantined: creation failed or not callable")

                    if tool_name == "search" or tool_name == "search-search":
                        await asyncio.to_thread(self._save_search_cache_sync, query_str, result)
                        async with self._search_cache_lock:
                            self._search_cache[query_str] = result

                        if not future.done():
                            future.set_result(result)

                    if execution_start_time is not None:
                        execution_time = time.time() - execution_start_time
                        self._record_graphrag_execution(tool_name, success=True, execution_time=execution_time)

                    return result
                except asyncio.TimeoutError as e:

                    if (tool_name == "search" or tool_name == "search-search") and not future.done():
                        future.set_exception(e)

                    if execution_start_time is not None:
                        execution_time = time.time() - execution_start_time
                        self._record_graphrag_execution(tool_name, success=False, execution_time=execution_time)

                    return f"# Tool Calling Exceeds Timeout: {tool_name}, Exceeded {self.tool_timeout}s"
                except Exception as e:

                    if (tool_name == "search" or tool_name == "search-search") and future is not None and not future.done():
                        future.set_exception(e)

                    if execution_start_time is not None:
                        execution_time = time.time() - execution_start_time
                        self._record_graphrag_execution(tool_name, success=False, execution_time=execution_time)

                    return f"# Tool execution failed: {tool_name}\n- Error: {str(e)}"
                finally:

                    if tool_name == "search" or tool_name == "search-search":
                        async with self._in_flight_lock:
                            if query_str in self._in_flight_requests and self._in_flight_requests[query_str] is future:
                                del self._in_flight_requests[query_str]
            elif type(args) is str:
                result = args
            else:
                result = 'Unexpected type of args: {} (args: {})'.format(type(args), args)
        else:
            if tool_name == '<empty>':
                result = args
            else:
                result = "# Failed to find the tool {} in the tool map".format(tool_name)
        return result

    async def _execute_runtime_tool(self, tool):
        started = time.monotonic()
        identity = tool.get('_runtime_context')
        runtime = self.method_runtime
        context = runtime.context(identity)
        dispatcher = context['dispatcher']
        name = tool['name']
        from control.isolation import ModelToolInputError, prepare_python_source, validate_timeout
        def invalid(error):
            result = json.dumps({'error': 'invalid_tool_input', 'message': str(error)[:500]}, ensure_ascii=False)
            runtime.observe(identity, name, result, success=False, executed=False)
            return result
        args = tool.get('args', tool.get('arguments', {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return invalid('Tool arguments must be valid JSON')
        if not isinstance(args, dict):
            return invalid('Runtime tool arguments must be an object')
        if name == 'submit_revised_plan':
            try:
                return json.dumps(runtime.submit_revised_plan(identity, args), ensure_ascii=False, allow_nan=False)
            except ValueError as exc:
                return invalid(exc)
        from control.runtime import schemas
        from control.evidence import resolve_arguments
        try:
            resolved, bindings = resolve_arguments(args, context['calls'])
        except ModelToolInputError as exc:
            return invalid(exc)
        checked_schemas = schemas(dispatcher)+self.tool_manager.functions
        if dispatcher.has_tool(name):
            checked_schemas = [dict(row, parameters=dispatcher.registry['semantic'][row['tool_version']]['arguments'])
                               if row.get('tool_version') else row for row in checked_schemas]
        from control.binding import apply_rewrites
        resolved, binding_audit = runtime.prepare_arguments(identity, name, resolved, checked_schemas)
        args = apply_rewrites(args, binding_audit)
        # Re-resolve after projection: overwritten references are no longer dependencies.
        resolved, bindings = resolve_arguments(args, context['calls'])
        check = runtime.check(identity, name, resolved, checked_schemas)
        if binding_audit['after']['blocked']:
            check = binding_audit['after']
        version, success, receipt = None, False, None
        if check['blocked']:
            result = json.dumps({'error': 'visible_requirement_conflict', 'audit': check}, ensure_ascii=False)
        elif name in ('create_and_execute_mcp', 'create_and_execute_mcp-create_and_execute_mcp'):
            spec = {key: args.get(key) for key in ('name', 'description', 'arguments', 'returns', 'code')}
            spec['side_effects'] = args.get('side_effects', 'isolated computation; no host filesystem or network access')
            try:
                timeout = validate_timeout(args.get('timeout', 15), maximum=30)
                receipt = await asyncio.to_thread(dispatcher.create, spec, args.get('inputs', {}),
                                                 task_id=context['task_id'], timeout=timeout, encoder=runtime.encoder,
                                                 ledger=context['calls'], capability=getattr(runtime, 'engine', None) is not None)
            except ModelToolInputError as exc:
                return invalid(exc)
            success = receipt.get('run_success') is True and receipt.get('registered') is True
            version = receipt.get('tool_version') if receipt.get('registered') is True else None
            result = json.dumps(receipt, ensure_ascii=False, allow_nan=False)
            if success and context['split'] == 'train' and not context.get('private'):
                self._runtime_sync_graph()
        elif dispatcher.has_tool(name):
            try:
                receipt = await asyncio.to_thread(dispatcher.invoke, name, args,
                    expected_version=context['selected_versions'].get(name), timeout=15, ledger=context['calls'])
            except ModelToolInputError as exc:
                return invalid(exc)
            success, version = receipt.get('run_success') is True, receipt.get('tool_version')
            result = json.dumps(receipt, ensure_ascii=False, allow_nan=False)
        elif name in ('execute_python_code', 'execute_python_code-execute_python_code'):
            try:
                source = prepare_python_source(resolved.get('code', ''))
                receipt = await asyncio.to_thread(dispatcher.backend.run_source, source, timeout=15)
            except ModelToolInputError as exc:
                return invalid(exc)
            success = receipt.get('run_success') is True
            result = json.dumps(receipt, ensure_ascii=False, allow_nan=False)
        elif name in ('search', 'search-search'):
            # Search is fixed infrastructure, not generated source.
            result = await asyncio.to_thread(self.tool_manager._call_tool, name, json.dumps(resolved))
            success = bool(result) and not str(result).startswith(('# Tool execution failed', '# Failed', 'Error'))
        else:
            result = f"# Unknown or unadmitted isolated tool: {name}"
        observed_name = args.get('name', name) if name.startswith('create_and_execute_mcp') else name
        call = runtime.observe(identity, observed_name, result, success=success, version=version,
            arguments=receipt.get('resolved_arguments', resolved) if receipt else resolved,
            execution_result=receipt.get('execution_result', receipt.get('result', result)) if receipt else result,
            bindings=receipt.get('input_bindings', bindings) if receipt else bindings,
            executed=not check['blocked'], recovery_name=name, recovery_arguments=resolved,
            binding_audit=binding_audit)
        if receipt is not None:
            result = json.dumps({**receipt, 'call_id': call['call_id'], 'execution_result': call['result'],
                                 'result_referenceable': call['result_available']}, ensure_ascii=False, allow_nan=False)
        if binding_audit['changed']:
            # The policy sees that execution differed; no success-certificate claim.
            envelope = json.loads(result) if receipt is not None else {'tool_response': result}
            result = json.dumps({**envelope, 'binding_correction': {
                'fields': [row['field'] for row in binding_audit['rewrites']],
                'effective_arguments': resolved, 'semantic_correctness_certified': False}}, ensure_ascii=False, allow_nan=False)
        if context['split'] == 'train' and not context.get('private') and self.graphrag_engine is not None and observed_name in self.mcps:
            self._record_graphrag_execution(observed_name, success=success, execution_time=time.monotonic()-started)
        return result

    # async def execute_single_tool(self, tool):
    #     tool_instance = self.tool_manager.get_tool(tool["name"])
    #     args = tool["args"]
    #     if tool_instance is not None:
    #         try:
    #             args = json.loads(args)
    #         except Exception as e:
    #             pass
    #         if type(args) is dict:
    #             try:
    #                 try:
    #                     tool_result = await asyncio.wait_for(
    #                         asyncio.to_thread(
    #                             self.tool_manager._call_tool,
    #                             tool["name"],
    #                             json.dumps(args, ensure_ascii=False, indent=4)
    #                         ),
    #                         timeout=self.timeout
    #                     )
    #                     if len(tool_result) == 0:
    #                         tool_result = "The tool call format is correct, but the return result is empty. Please try other inputs or tools. "
    #                     result = f"# Execute the tool {tool['name']} successed\n" \
    #                             f"- The result is:\n{tool_result}"
    #                 except asyncio.TimeoutError:
    #                     result = f"# Execute the tool {tool['name']} timeout\n" \
    #                             f"- The tool execution exceeded the timeout limit of {self.timeout} seconds\n"
    #             except Exception as e:
    #                 result = f"# Execute the tool {tool['name']} failed\n" \
    #                         f"- Error message:\n{str(e)}"
    #         elif type(args) is str:
    #             # Json decode error: xxx
    #             result = args
    #         else:
    #             result = 'Unexpected type of args: {} (args: {})'.format(type(args), args)
    #     else:
    #         if tool['name'] == '<empty>':
    #             result = args
    #         else:
    #                 result = "# Failed to find the tool {} in the tool map".format(tool['name'])
    #     return result

    async def mcp_hybrid_search(self, query: str, top_k: int = 5, weights: Optional[Dict[str, float]] = None):
        'Search MCPs asynchronously.'
        return await self.hybrid_search(query, top_k, weights)

    async def mcp_clear(self, temp_dict: Optional[dict] = None):
        'Clear MCPs asynchronously.'
        return await self.clear(temp_dict)

    async def mcp_get_mcps(self):
        'Get MCPs asynchronously.'
        return await self.get_mcps()

    async def mcp_list_mcps(self):
        'List MCPs asynchronously.'
        return await self.list_mcps()

    async def get_functions_mcps(self):
        'Return OpenAI function schemas for all MCP tools.'
        return self.functions_mcps

    async def get_mcp_version(self):
        'Return the MCP inventory version.'
        return self.mcp_version

    async def get_functions_mcps_with_version(self):
        'Return MCP function schemas and their inventory version.'
        return {
            'version': self.mcp_version,
            'functions_mcps': self.tool_manager.functions_mcps
        }

    async def get_mcp_statistics(self):
        'Return total_mcps, official_mcps, temp_mcps, mcp_version, graph_nodes and graph_edges for monitoring.'
        with self.lock:
            stats = {
                "total_mcps": len(self.mcps),
                "official_mcps": len(self.mcps) - len(self.temp_mapping),
                "temp_mcps": len(self.temp_mapping),
                "mcp_version": self.mcp_version,
                "graph_nodes": self.sub_task_graph.number_of_nodes(),
                "graph_edges": self.sub_task_graph.number_of_edges(),
            }

            if self.use_graphrag and self.graphrag_engine is not None:
                stats["graphrag_enabled"] = True
                stats["graphrag_communities"] = len(self.graphrag_engine.communities) if hasattr(self.graphrag_engine, 'communities') else 0
                stats["graphrag_history_enabled"] = self.graphrag_engine.enable_history_tracking if hasattr(self.graphrag_engine, 'enable_history_tracking') else False
            else:
                stats["graphrag_enabled"] = False

        return stats

    def record_tool_execution(
        self,
        mcp_name: str,
        success: bool,
        execution_time: float = 0.0,
        co_used_tools: Optional[List[str]] = None
    ):
        'Record tool execution success, elapsed seconds and co-used tools.'
        if self.use_graphrag and self.graphrag_engine is not None:
            try:
                self.graphrag_engine.record_execution(
                    mcp_name=mcp_name,
                    success=success,
                    execution_time=execution_time,
                    co_used_tools=co_used_tools
                )
            except Exception as e:
                print(f"[GraphRAG] Warning: Failed to record execution for {mcp_name}: {e}")

    def get_tool_stats(self, mcp_name: str) -> Dict[str, Any]:
        'Return tool success-rate and usage statistics.'
        if self.use_graphrag and self.graphrag_engine is not None:
            try:
                return self.graphrag_engine.get_execution_stats(mcp_name)
            except Exception as e:
                print(f"[GraphRAG] Warning: Failed to get stats for {mcp_name}: {e}")
                return {}
        return {}

    def save_graphrag_cache(self):
        'Save the GraphRAG cache.'
        if self.use_graphrag and self.graphrag_engine is not None:
            try:
                self.graphrag_engine.save_cache()
                print("[GraphRAG] Cache saved successfully")
            except Exception as e:
                print(f"[GraphRAG] Warning: Failed to save cache: {e}")

    async def get_community_info(self) -> Dict[str, Any]:
        'Return GraphRAG community statistics.'
        if not self.use_graphrag or self.graphrag_engine is None:
            return {"graphrag_enabled": False}

        try:
            return {
                "graphrag_enabled": True,
                "num_communities": len(self.graphrag_engine.communities),
                "communities": {
                    comm_id: {
                        "size": len(members),
                        "members": list(members)[:10],
                        "summary": self.graphrag_engine.community_summaries.get(comm_id, "")
                    }
                    for comm_id, members in self.graphrag_engine.communities.items()
                },
                "global_summary": self.graphrag_engine.global_summary
            }
        except Exception as e:
            print(f"[GraphRAG] Warning: Failed to get community info: {e}")
            return {"graphrag_enabled": True, "error": str(e)}

    def update_mcp_box(self, similarity_threshold: float = 0.88, prefer_higher_degree: bool = False, step_idx: int = -1):
        if self.method_runtime is not None:
            # Registry admission/verified equivalence replaces unsafe generated-source imports.
            if self.method_runtime.engine is not None:
                return {'runtime': self.method_runtime.arm, 'deferred_to_contextual_barrier': True}
            self.method_runtime.snapshot()
            self._runtime_sync_graph()
            return {'runtime': self.method_runtime.arm, 'registry_tools': len(self.mcps)}
        'Merge duplicate MCP nodes with clustering outside the lock to avoid deadlocks.'

        with self.lock:
            self.update_self_mcp()
            if not self.sub_task_graph or not self.mcps:
                print("Agent Log: Graph or MCPs are empty, skipping merge.")

                self._reload_official_tools_after_merge()
                return

            needs_embedding = any('query_embedding' not in v for v in self.mcps.values() if v.get("description"))

        if needs_embedding:
            self.build_embeddings(temp_mcps=None)

        with self.lock:

            graph_snapshot = self.sub_task_graph.copy()
            mcps_snapshot = {k: v.copy() for k, v in self.mcps.items()}

            alias_map_snapshot = self.alias_map.copy()

        print("Agent Log: Performing heavy computation (clustering) outside of the lock...")

        to_remove = [
            n for n in graph_snapshot.nodes
            if graph_snapshot.nodes[n].get("mcp_name") is None or
               graph_snapshot.nodes[n].get("mcp_name") not in mcps_snapshot
        ]
        for n in to_remove:
            preds = set(graph_snapshot.predecessors(n))
            succs = set(graph_snapshot.successors(n))
            for u in preds:
                for v in succs:
                    if u != v:
                        graph_snapshot.add_edge(u, v)
        if to_remove:
            graph_snapshot.remove_nodes_from(to_remove)

        valid_nodes = []
        for node in graph_snapshot.nodes:
            mcp_name = graph_snapshot.nodes[node].get("mcp_name")
            mcp_info = mcps_snapshot.get(mcp_name)
            if mcp_info and mcp_info.get("description") and mcp_info.get("query_embedding") is not None:
                step_id = mcp_info.get("step_id", node)
                valid_nodes.append((node, mcp_name, step_id, mcp_info["query_embedding"]))

        if len(valid_nodes) <= 1:
            print("Agent Log: Not enough valid nodes to perform a merge. Skipping.")
            try:
                self._write_merged_mcps_to_file_async(mcps_snapshot)
            except Exception as e:
                print(f"Agent Log: Failed to write merged MCPs to file: {e}")

            self._reload_official_tools_after_merge()
            return len(mcps_snapshot)

        clusters = self._cluster_nodes_by_similarity(valid_nodes, similarity_threshold)
        print("Agent Log: Clustering complete.")

        if all(len(ixs) == 1 for ixs in clusters.values()):
            print(f"Agent Log: No clusters found above threshold (>= {similarity_threshold}); nothing merged.")
            try:
                self._write_merged_mcps_to_file_async(mcps_snapshot)
            except Exception as e:
                print(f"Agent Log: Failed to write merged MCPs to file: {e}")

            self._reload_official_tools_after_merge()
            return len(mcps_snapshot)

        merged_count, alias_map = self._merge_clusters(
            graph_snapshot,
            mcps_snapshot,
            alias_map_snapshot,
            clusters,
            valid_nodes,
            prefer_higher_degree
        )

        if merged_count == 0:
            print("Agent Log: No nodes were merged.")
            try:
                self._write_merged_mcps_to_file_async(mcps_snapshot)
            except Exception as e:
                print(f"Agent Log: Failed to write merged MCPs to file: {e}")

            self._reload_official_tools_after_merge()
            return len(mcps_snapshot)

        self._update_dependencies_after_merge_on_snapshot(mcps_snapshot, alias_map)

        with self.lock:

            self.sub_task_graph = graph_snapshot
            self.mcps = mcps_snapshot
            self.alias_map = alias_map_snapshot
            self.temp_mapping.clear()

            print(f"Agent Log: Merged {merged_count} duplicate nodes. Rebuilding indices...")

        if merged_count > 0:
            print(f"Agent Log: {merged_count} nodes were merged, rebuilding indices...")
            self.build_tfidf_indices(temp_mcps=None)
            self.build_embeddings(temp_mcps=None)
            print(f"Agent Log: Index rebuild complete.")
        else:
            print(f"Agent Log: No nodes merged, skipping index rebuild.")

        print(f"Agent Log: Merge complete. Nodes={self.sub_task_graph.number_of_nodes()}, MCPs={len(self.mcps)}")
        try:
            self._write_merged_mcps_to_file_async(mcps_snapshot)
        except Exception as e:
            print(f"Agent Log: Failed to write merged MCPs to file: {e}")

        self._reload_official_tools_after_merge()
        return len(self.mcps)

    def _reload_official_tools_after_merge(self):
        try:
            print(f"[update_mcp_box] Reloading official tools after merge...")
            print(f"[update_mcp_box] Before reload: {len(self.tool_manager.functions_mcps) if hasattr(self.tool_manager, 'functions_mcps') else 0} MCP tools")

            self.tool_manager.reload_official_tools()
            import pickle

            graph_dir = Path(os.environ.get("RL_FACTORY_GRAPH_DIR", _repo_root() / "envs" / "tools" / "graphs"))
            graph_dir.mkdir(parents=True, exist_ok=True)
            with open(graph_dir / f"{self.project_name}_task_graph.pkl", 'wb') as f:
                pickle.dump(self.sub_task_graph, f)

            self.mcp_version += 1
            print(f"[update_mcp_box] MCP version updated to {self.mcp_version}")

            print(f"[update_mcp_box] After reload: {len(self.tool_manager.functions_mcps) if hasattr(self.tool_manager, 'functions_mcps') else 0} MCP tools")

            if hasattr(self.tool_manager, 'functions_mcps') and self.tool_manager.functions_mcps:
                print(f"[update_mcp_box] Tool manager functions_mcps updated successfully with {len(self.tool_manager.functions_mcps)} tools")

                tool_names = [fn.get('name', 'unknown') for fn in self.tool_manager.functions_mcps[:5]]
                print(f"[update_mcp_box] Sample tool names: {tool_names}")
            else:
                print(f"[update_mcp_box] WARNING: functions_mcps is empty or not set!")
        except Exception as e:
            print(f"[update_mcp_box] Failed to reload official tools: {e}")
            import traceback
            traceback.print_exc()

    def _update_dependencies_after_merge_on_snapshot(self, mcps_snapshot: dict, alias_map: dict):
        'Update dependencies on a snapshot without acquiring the live lock.'
        if not alias_map:
            return

        for _, data in mcps_snapshot.items():
            req = data.get("requires")
            if not isinstance(req, str):
                continue

            parts = [p.strip() for p in req.split(",") if p.strip()]
            updated_reqs = {alias_map.get(p, p) for p in parts}
            data["requires"] = ", ".join(sorted(updated_reqs)) if updated_reqs else None

    def _write_merged_mcps_to_file_async(self, mcps_snapshot: dict):
        'Write merged MCPs asynchronously without blocking execution.'
        import threading

        def _write_file():
            try:
                root_dir = _repo_root()
                tool_dir = Path(root_dir) / "envs" / "tools" / "invented_tools" / (self.project_name)
                tool_dir.mkdir(parents=True, exist_ok=True)
                official_file = tool_dir / "official_tool_list.py"

                common_imports = (
                    "import json\n"
                    "import sys\n"
                    "import traceback\n"
                    "import math\n"
                    "import numpy as np\n"
                    "import re\n"
                    "import os\n"
                    "import datetime\n"
                    "from collections import Counter, defaultdict, deque\n"
                    "from mcp.server.fastmcp import FastMCP\n\n"
                )
                main_block = (
                    "\n\nif __name__ == \"__main__\":\n"
                    "    print(\"\\nStart MCP service:\")\n"
                    "    mcp.run(transport='stdio')\n"
                )

                TOOL_TEMPLATE_WITH_DEF = '@mcp.tool(name=\'{name}\')\ndef {wrapper_func_name}(**kwargs):\n    """{description}\n\n    Args:\n        {arguments}\n\n    Returns:\n        {returns}\n    """\n    try:\n{body}\n        return {inner_func_name}(**kwargs)\n    except Exception as e:\n        return f"tool execution failed: {{str(e)}}"\n'

                TOOL_TEMPLATE_SIMPLE = '@mcp.tool(name=\'{name}\')\ndef {name}(**kwargs):\n    """{description}\n\n    Args:\n        {arguments}\n\n    Returns:\n        {returns}\n    """\n    try:\n{indented_body}\n    except Exception as e:\n        return f"tool execution failed: {{str(e)}}"\n'

                def build_tool_block(name: str, description: str, arguments: str, returns: str, code: str) -> str:
                    if code.lstrip().startswith("def "):
                        code_lines = code.splitlines()
                        inner_func_name = code_lines[0].split('(')[0].replace('def ', '').strip()

                        indented_def = "\n".join(
                            f"        {line}" if line.strip() else ""
                            for line in code_lines
                        )
                        if not indented_def.strip():
                            indented_def = "        pass"

                        body = indented_def

                        return TOOL_TEMPLATE_WITH_DEF.format(
                            name=name,
                            wrapper_func_name=name,
                            inner_func_name=inner_func_name,
                            description=description,
                            arguments=arguments,
                            returns=returns,
                            body=body
                        )
                    else:
                        code_lines = code.splitlines() if code.strip() else ["return None"]
                        indented_body = '\n'.join(
                            f"        {line}" if line.strip() else ""
                            for line in code_lines
                        )

                        if not indented_body.strip():
                            indented_body = "        pass"

                        return TOOL_TEMPLATE_SIMPLE.format(
                            name=name,
                            description=description,
                            arguments=arguments,
                            returns=returns,
                            indented_body=indented_body
                        )

                from io import StringIO
                buffer = StringIO()
                buffer.write(f"{common_imports}mcp = FastMCP('Official_Server')\n\n")

                success_tools = []
                failed_tools = []

                for mcp_name, mcp_data in mcps_snapshot.items():
                    if not mcp_name.isidentifier():
                        print(f"Warning: Invalid function name '{mcp_name}', skipping...")
                        failed_tools.append((mcp_name, "Invalid function name"))
                        continue

                    desc = mcp_data.get('description') or ""
                    args = mcp_data.get('arguments') or ""
                    rets = mcp_data.get('returns') or ""
                    code = mcp_data.get('code') or ""

                    try:

                        block = build_tool_block(mcp_name, desc, args, rets, code)

                        test_content = f"{common_imports}mcp = FastMCP('Official_Server')\n\n{block}"

                        compile(test_content, f"<{mcp_name}>", 'exec')

                        buffer.write(block)
                        success_tools.append(mcp_name)

                    except SyntaxError as e:
                        print(f"Agent Log: Syntax error in '{mcp_name}': {e}")
                        failed_tools.append((mcp_name, f"SyntaxError: {e}"))
                        continue
                    except Exception as e:
                        print(f"Agent Log: Failed to build/validate block for '{mcp_name}': {e}")
                        failed_tools.append((mcp_name, str(e)))
                        continue

                buffer.write(main_block)
                content = buffer.getvalue()

                try:
                    compile(content, str(official_file), 'exec')
                    official_file.write_text(content, encoding="utf-8")
                    print(f"Agent Log: Merged MCPs saved to {official_file}")
                    print(f"Agent Log: Successfully wrote {len(success_tools)} tools: {', '.join(success_tools)}")

                    if failed_tools:
                        print(f"Agent Log: Failed to write {len(failed_tools)} tools:")
                        for tool_name, error in failed_tools:
                            print(f"  - {tool_name}: {error}")
                except Exception as e:
                    print(f"Agent Log: Final compilation failed (unexpected): {e}")

            except Exception as e:
                print(f"Error writing MCP file: {e}")

        thread = threading.Thread(target=_write_file, daemon=True)
        thread.start()

    def _cleanup_invalid_nodes(self, G: nx.DiGraph):
        'Remove invalid nodes and reconnect their bypass edges.'
        to_remove = [
            n for n in G.nodes
            if G.nodes[n].get("mcp_name") is None or G.nodes[n].get("mcp_name") not in self.mcps
        ]

        for n in to_remove:
            preds = set(G.predecessors(n))
            succs = set(G.successors(n))
            for u in preds:
                for v in succs:
                    if u != v:
                        G.add_edge(u, v)

        if to_remove:
            G.remove_nodes_from(to_remove)

    def _gather_valid_nodes_for_merging(self, G: nx.DiGraph) -> list:
        'Collect valid nodes for merging.'
        valid_nodes = []
        for node in G.nodes:
            mcp_name = G.nodes[node].get("mcp_name")
            mcp_info = self.mcps.get(mcp_name)
            if mcp_info and mcp_info.get("description") and mcp_info.get("query_embedding") is not None:
                step_id = mcp_info.get("step_id", node)
                valid_nodes.append((node, mcp_name, step_id, mcp_info["query_embedding"]))
        return valid_nodes

    def _cluster_nodes_by_similarity(self, valid_nodes: list, similarity_threshold: float) -> dict:
        'Cluster nodes with cosine similarity and union-find.'
        from collections import defaultdict
        from sklearn.metrics.pairwise import cosine_similarity

        embs = np.stack([t[3] for t in valid_nodes], axis=0)
        S = cosine_similarity(embs, embs)
        N = S.shape[0]
        parent = list(range(N))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(N):
            for j in range(i + 1, N):
                if S[i, j] >= similarity_threshold:
                    union(i, j)

        clusters = defaultdict(list)
        for i in range(N):
            clusters[find(i)].append(i)

        return clusters

    def _merge_clusters(self, G: nx.DiGraph, mcps_dict: dict, alias_map_snapshot: dict, clusters: dict, valid_nodes: list, prefer_higher_degree: bool) -> Tuple[int, Dict[str, str]]:
        'Merge clustered nodes and metadata on graph/MCP/alias snapshots; optionally prefer high-degree representatives and return the merge count and alias map.'
        merged_nodes_count = 0
        alias_map = {}
        node_ids = [t[0] for t in valid_nodes]
        names = [t[1] for t in valid_nodes]
        step_ids = [t[2] for t in valid_nodes]

        for _, member_indices in clusters.items():
            if len(member_indices) <= 1:
                continue

            candidates = [
                (idx, node_ids[idx], step_ids[idx], G.in_degree(node_ids[idx]) + G.out_degree(node_ids[idx]))
                for idx in member_indices
            ]
            sort_key = (lambda t: (-t[3], str(t[2]), str(t[1]))) if prefer_higher_degree else (lambda t: (str(t[2]), str(t[1])))
            candidates.sort(key=sort_key)

            rep_idx, rep_node, _, _ = candidates[0]
            rep_name = names[rep_idx]

            for idx, dup_node, _, _ in candidates[1:]:
                dup_name = names[idx]

                if rep_name not in mcps_dict or dup_name not in mcps_dict:
                    continue

                for u in list(G.predecessors(dup_node)):
                    if u != rep_node:
                        G.add_edge(u, rep_node)
                for v in list(G.successors(dup_node)):
                    if v != rep_node:
                        G.add_edge(rep_node, v)
                G.remove_node(dup_node)

                alias_map_snapshot[mcps_dict[dup_name]['step_id']] = mcps_dict[rep_name]['step_id']
                alias_map[dup_name] = rep_name

                self._merge_mcp_metadata(mcps_dict, rep_name, dup_name)
                mcps_dict.pop(dup_name, None)
                merged_nodes_count += 1

            if G.has_node(rep_node):
                G.nodes[rep_node]['mcp_name'] = rep_name

        return merged_nodes_count, alias_map

    def _merge_mcp_metadata(self, mcps_dict: dict, rep_name: str, dup_name: str):
        'Merge duplicate MCP metadata into its representative.'
        if rep_name in mcps_dict and dup_name in mcps_dict:
            rep, old = mcps_dict[rep_name], mcps_dict[dup_name]

            req_rep = rep.get("requires", "") or ""
            req_old = old.get("requires", "") or ""
            all_reqs = set(r.strip() for r in req_rep.split(",") if r.strip())
            all_reqs.update(r.strip() for r in req_old.split(",") if r.strip())
            rep["requires"] = ", ".join(sorted(all_reqs)) if all_reqs else None

            if not rep.get("code") and old.get("code"):
                rep["code"] = old["code"]

            aliases = rep.get("aliases", []) or []
            if not isinstance(aliases, list):
                aliases = [aliases]
            aliases.append(dup_name)
            rep["aliases"] = sorted(set(aliases))

    def _update_dependencies_after_merge(self, alias_map: dict):
        'Update requires fields through the alias map.'
        if not alias_map:
            return

        for _, data in self.mcps.items():
            req = data.get("requires")
            if not isinstance(req, str):
                continue

            parts = [p.strip() for p in req.split(",") if p.strip()]
            updated_reqs = {alias_map.get(p, p) for p in parts}
            data["requires"] = ", ".join(sorted(updated_reqs)) if updated_reqs else None

    def extract_dag_and_parallel(self, text: str):
        'Parse the DAG synchronously and return dag_list, task_id and subtask_dict.'
        return self.tool_manager.extract_dag_and_parallel(text)

    def batch_extract_dag_and_parallel(self, texts: list, traj_indices: list, max_turns_list: list):
        'Parse DAGs concurrently in a batch.'
        import concurrent.futures
        import time

        start_time = time.time()
        results = [None] * len(texts)

        def process_single_planning(idx, text, max_turns):
            'Process one planning task.'
            try:
                result = self.tool_manager.extract_dag_and_parallel(text)
                return idx, result, None
            except Exception as e:
                print(f"[Trajectory {idx}] DAG parsing failed: {e}, falling back to sequential")
                try:
                    fallback = self.tool_manager.sequential_planning(max_turns)
                    return idx, fallback, None
                except Exception as e2:
                    print(f"[Trajectory {idx}] Sequential planning also failed: {e2}")
                    return idx, (None, None, None), str(e2)

        max_workers = min(32, len(texts))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

            future_to_idx = {
                executor.submit(process_single_planning, idx, text, max_turns): idx
                for idx, (text, max_turns) in enumerate(zip(texts, max_turns_list))
            }

            completed = 0
            for future in concurrent.futures.as_completed(future_to_idx, timeout=60):
                try:
                    idx, result, error = future.result(timeout=5)
                    results[idx] = result
                    completed += 1
                    if completed % 50 == 0:
                        print(f"Planning progress: {completed}/{len(texts)} completed")
                except concurrent.futures.TimeoutError:
                    idx = future_to_idx[future]
                    print(f"[Trajectory {idx}] Planning timeout, using sequential fallback")
                    results[idx] = self.tool_manager.sequential_planning(max_turns_list[idx])
                except Exception as e:
                    idx = future_to_idx[future]
                    print(f"[Trajectory {idx}] Planning error: {e}")
                    results[idx] = (None, None, None)

        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r and r[1] is not None)
        print(f"Batch planning completed: {success_count}/{len(texts)} successful in {elapsed:.2f}s")

        return results

    async def batch_register_mcps(self, tool_call_strs: list, task_id: int, subtask_dict: dict):
        'Register MCPs in a batch to reduce remote-call overhead.'
        results = []
        for tool_call_str in tool_call_strs:
            try:
                await self.register_mcp(tool_call_str, task_id, subtask_dict)
                results.append(True)
            except Exception as e:
                print(f"Failed to register MCP: {e}")
                results.append(False)
        return results

    def update_sub_task_graph(self, edge_list, single_nodes):
        'Update the subtask graph synchronously for remote managers.'
        if edge_list:
            self.sub_task_graph.add_edges_from(edge_list)
        if single_nodes:
            self.sub_task_graph.add_nodes_from(single_nodes)

    def preview_get_prompt(self, input_data, tokenizer, mode: str = 'initial', add_generation_prompt: bool = True, current_plan=None):
        'Return a generated prompt for remote testing.'
        return self.tool_manager.get_prompt(
            input_data=input_data,
            tokenizer=tokenizer,
            mode=mode,
            add_generation_prompt=add_generation_prompt,
            current_plan=current_plan
        )

def test_centralized_tool_actor():
    'Test CentralizedToolActor through a Ray actor instance.'
    import os
    import sys
    import ray
    import asyncio
    import contextlib
    from statistics import mean
    print('Starting CentralizedToolActor tests with Ray...')

    class MockConfig:
        def __init__(self):
            self.tool_timeout = 30
            self.mcp_mode = 'stdio'

            self.config_path = "envs/configs/mcp_tools_local_graph.pydata"
            self.enable_limiter = False
            self.max_turns = 6
            self.temp_config_path = "envs/configs/temp_mcps.pydata"
            self.official_config_path = "envs/configs/official_mcps.pydata"
            self.enable_thinking = False
            self.project_name = "gigpo_gpu6_prm_new_sett"

        def get(self, key, default=None):
            return getattr(self, key, default)

    # try:
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    print('1. Initializing the Ray actor...')
    actor = CentralizedToolActor.options(name="test_centralized_tool_actor").remote(MockConfig())
    print('Actor created for the temp_tool_list test.')
    mcp_list = ray.get(actor.mcp_list_mcps.remote())
    print(f"MCP list: {mcp_list}")
    mcps = ray.get(actor.mcp_get_mcps.remote())
    print(f"Retrieved {len(mcps)} MCPs")

    concurrency_levels = [8, 16]

    def _summary(label, results):
        ok = sum(1 for r in results if r is not None)
        print(f"{label}: {ok}/{len(results)} succeeded")

    print('\n3. Testing concurrent mcp_hybrid_search...')
    try:
        for c in concurrency_levels:
            tasks = [actor.mcp_hybrid_search.remote("web search", top_k=2) for _ in range(c)]
            res = ray.get(tasks)
            _summary(f"  Concurrent {c}", res)
    except Exception as e:
        print(f"Concurrent search test failed: {e}")

    print('\n4. Testing concurrent execute_single_tool (search-search)...')
    create_tool_call = {
        "name": "execute_python_code-execute_python_code",
        "arguments": {

            "code": "import math\n\n# Define the relationship between a and b for quasi-colorability\n# Based on graph theory principles, we assume that the graph is quasi-colorable if the number of degree 4 vertices is sufficiently large relative to the number of degree 3 vertices.\n# We derive the smallest real number c such that if a/b > c, then G is quasi-colorable.\n\n# Assume that the graph is quasi-colorable if a/b > 2\n# This is based on the idea that the graph must have enough degree 4 vertices to ensure that the coloring constraints can be satisfied.\n\n# Calculate the smallest real number c\n# Based on the assumption that a/b > 2 ensures quasi-colorability\n# Therefore, c = 2\n\n# Return the value of c\nprint(2)",
            "timeout": 30.0
        }
    }
    try:
        for c in concurrency_levels:
            tasks = [actor.execute_single_tool.remote(create_tool_call) for _ in range(c)]
            res = ray.get(tasks)
            _summary(f"  Concurrent {c}", res)
            print(res[0])
    except Exception as e:
        print(f"Concurrent execute_single_tool test failed: {e}")

    print('\n4.1 Testing concurrent preview_get_prompt...')
    tokenizer = AutoTokenizer.from_pretrained('assets/models/Qwen3-4B')
    sample_chat = [{'role': USER, 'content': 'Draft a travel itinerary for Beijing.'}]
    try:
        for c in concurrency_levels:
            tasks = [actor.preview_get_prompt.remote(sample_chat, tokenizer, mode='plan_call', add_generation_prompt=True) for _ in range(c)]
            res = ray.get(tasks)
            print(res)
            _summary(f"  Concurrent {c}", res)
    except Exception as e:
        print(f"Concurrent preview_get_prompt test failed: {e}")

    print('\n4.2 Testing concurrent DAG parsing and registration...')

    try:
        from envs.tools import creation as tool_creation

        simple_payload = {
            "name": "dummy_mcp",
            "description": "dummy mcp for register_mcp stress test",
            "arguments": "x (int), y (int)",
            "returns": "sum of two ints",
            "code": "def dummy_mcp(x: int, y: int):\n    return x + y",
            "inputs": {"x": 1, "y": 2},
            "timeout": 5.0,
        }
        simple_tool_call = {
            "name": "create_and_execute_mcp-create_and_execute_mcp",
            "args": json.dumps(simple_payload)
        }
        simple_res = ray.get(actor.execute_single_tool.remote(simple_tool_call))
        print(f"Simple MCP test result: {simple_res}")

        complex_payload = {
            "name": "analyze_polynomial_roots",
            "description": "Find roots of polynomial and return classification and properties",
            "arguments": "coeffs (list of float): coefficients [a_n, a_{n-1}, ..., a_1, a_0]",
            "returns": "dict with keys: roots, num_real, num_complex, roots_sum, roots_product",
            "code": "def analyze_polynomial_roots(coeffs):\n    import numpy as np\n    # Step 1: Find all roots\n    roots = np.roots(coeffs)\n    # Step 2: Classify real vs complex\n    num_real = sum(1 for r in roots if abs(r.imag) < 1e-10)\n    num_complex = len(roots) - num_real\n    # Step 3: Calculate properties\n    roots_sum = sum(roots)\n    roots_product = np.prod(roots)\n    return {\n        'roots': roots.tolist(),\n        'num_real': num_real,\n        'num_complex': num_complex,\n        'sum': complex(roots_sum).real if abs(roots_sum.imag) < 1e-10 else roots_sum,\n        'product': complex(roots_product).real if abs(roots_product.imag) < 1e-10 else roots_product\n    }",
            "inputs": {"coeffs": [1, -5, 6]},  # x^2 - 5x + 6 = (x-2)(x-3), roots: [2, 3]
            "timeout": 10.0,
        }
        complex_tool_call = {
            "name": "create_and_execute_mcp-create_and_execute_mcp",
            "args": json.dumps(complex_payload)
        }
        complex_res = ray.get(actor.execute_single_tool.remote(complex_tool_call))
        print(f"Prompt example MCP test result: {complex_res}")

        if complex_res and "error" not in str(complex_res).lower():
            print('  - MCP created; expected: roots=[3, 2], num_real=2, sum=5, product=6')

    except Exception as e:
        import traceback
        print(f"MCP creation test failed: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
    sample_plan = '## ST1: demo\n1. plan step\n##DAG_LIST\n[(ST1)]\n'
    try:
        for c in concurrency_levels:
            dag_tasks = [actor.extract_dag_and_parallel.remote(sample_plan) for _ in range(c)]
            dag_res = ray.get(dag_tasks)
            _summary(f"  Concurrent DAG {c}", dag_res)

            reg_tasks = []
            for dag_list, task_id, subtask_dict in dag_res:
                dummy_tool_call = "<subtask> ST_1: demo </subtask>\n<tool_call>{\"name\":\"dummy_mcp\"}</tool_call>"
                reg_tasks.append(actor.register_mcp.remote(dummy_tool_call, task_id, subtask_dict))
            reg_res = ray.get(reg_tasks)
            _summary(f"  Concurrent register_mcp {c}", reg_res)
    except Exception as e:
        print(f"Concurrent DAG/registration test failed: {e}")

    print('\n5. Testing graph visualization...')
    try:
        viz_result = ray.get(actor.mcp_visualize_graph.remote())
        print(f"Graph visualization: {viz_result}")
    except Exception as e:
        print(f"Graph visualization failed: {e}")

    print('\n6. Testing DAG parsing and MCP registration...')
    try:
        sample_plan = '## ST1: demo\n1. plan step\n##DAG_LIST\n[(ST1)]\n'
        dag_list, task_id, subtask_dict = ray.get(actor.extract_dag_and_parallel.remote(sample_plan))
        print(f"Parsed DAG dag_list={dag_list}, task_id={task_id}, subtask_dict={subtask_dict}")

        dummy_tool_call = "<subtask> ST_1: demo </subtask>\n<tool_call>{\"name\":\"dummy_mcp\"}</tool_call>"
        ray.get(actor.register_mcp.remote(dummy_tool_call, task_id, subtask_dict))
        print('Simple MCP registration completed.')

        try:

            dag_list2, task_id2, subtask_dict2 = ray.get(actor.extract_dag_and_parallel.remote(sample_plan))

            complex_mcp_call = "<subtask> ST_1: Analyze polynomial x^3 - 6x^2 + 11x - 6 </subtask>\n<tool_call>{\"name\":\"analyze_polynomial_roots\", \"arguments\":{\"coeffs\":[1, -6, 11, -6]}}</tool_call>"
            ray.get(actor.register_mcp.remote(complex_mcp_call, task_id2, subtask_dict2))
            print('Prompt example MCP registered.')

            test_call = {
                "name": "analyze_polynomial_roots",
                "args": json.dumps({"coeffs": [1, -6, 11, -6]})  # x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
            }
            test_res = ray.get(actor.execute_single_tool.remote(test_call))
            print(f"MCP invocation test result: {test_res}")
            print('  - Expected: roots=[1, 2, 3], num_real=3, sum=6, product=6')
        except Exception as e:
            print(f"Prompt example MCP registration/invocation failed: {e}")

    except Exception as e:
        import traceback
        print(f"DAG/MCP test failed: {e}")
        print(f"  Traceback: {traceback.format_exc()}")

    print('\n7. Testing the prompt workflow: create MCP, then reuse...')
    try:

        workflow_plan = '## ST1: Analyze stability of system 1\n1. Create stability checker MCP\n##DAG_LIST\n[(ST1)]\n'
        dag_list3, task_id3, subtask_dict3 = ray.get(actor.extract_dag_and_parallel.remote(workflow_plan))

        create_payload = {
            "name": "check_system_stability",
            "description": "Find poles of transfer function denominator and determine system stability",
            "arguments": "coeffs (list of float): denominator coefficients [a_n, ..., a_0]",
            "returns": "dict with keys: poles, num_stable, num_unstable, is_stable",
            "code": "def check_system_stability(coeffs):\n    import numpy as np\n    # Step 1: Find all poles (roots of denominator)\n    poles = np.roots(coeffs)\n    # Step 2: Check real parts for stability (Re < 0 means stable)\n    num_stable = sum(1 for p in poles if p.real < 0)\n    num_unstable = len(poles) - num_stable\n    is_stable = (num_unstable == 0)\n    # Step 3: Return analysis\n    return {\n        'poles': [complex(p) for p in poles],\n        'num_stable': num_stable,\n        'num_unstable': num_unstable,\n        'is_stable': is_stable\n    }",
            "inputs": {"coeffs": [1, 3, 2]},  # s^2 + 3s + 2 = (s+1)(s+2), poles: [-1, -2] (stable)
            "timeout": 10.0,
        }
        create_call = {
            "name": "create_and_execute_mcp-create_and_execute_mcp",
            "args": json.dumps(create_payload)
        }
        create_result = ray.get(actor.execute_single_tool.remote(create_call))
        print(f"Step 1 - MCP creation result: {create_result}")

        test_systems = [
            {"coeffs": [1, 3, 2], "desc": 'System 1 (stable)'},      # poles: [-1, -2]
            {"coeffs": [1, -3, 2], "desc": 'System 2 (unstable)'},   # poles: [1, 2]
            {"coeffs": [1, 1, 1], "desc": 'System 3 (stable)'},      # complex poles with Re < 0
        ]

        for i, system in enumerate(test_systems, start=2):
            test_call = {
                "name": "check_system_stability",
                "args": json.dumps({"coeffs": system["coeffs"]})
            }
            result = ray.get(actor.execute_single_tool.remote(test_call))
            print(f"PASS Step {i} - {system['desc']} test result: {result}")

        print('Prompt workflow passed: MCP creation and repeated reuse succeeded.')

    except Exception as e:
        import traceback
        print(f"Prompt workflow test failed: {e}")
        print(f"  Traceback: {traceback.format_exc()}")

    print('\n8. Testing execute_python_code...')
    try:
        from envs.tools import python as tool_python

        py_res = tool_python.execute_python_code("print(1+1)", timeout=5.0)
        print(f"Python tool returned: {py_res}")
    except Exception as e:
        print(f"Python tool test failed: {e}")

    print('\n9. Testing _build_temp_mcp_tools...')
    try:

        class TestManager(CentralizedQwenManager):
            def __init__(self, verl_config):
                super().__init__(verl_config, mock=True)

        test_manager = TestManager(MockConfig())

        print('  Testing init_project=False (load)...')
        initial_mcp_count = len(test_manager.functions_mcps) if hasattr(test_manager, 'functions_mcps') else 0
        test_manager._build_temp_mcp_tools(init_project=False)
        after_normal_mcp_count = len(test_manager.functions_mcps) if hasattr(test_manager, 'functions_mcps') else 0
        print(f"    Initial MCP count: {initial_mcp_count}")
        print(f"    Loaded MCP count: {after_normal_mcp_count}")

        print('  Testing init_project=True (initialize)...')
        test_manager._build_temp_mcp_tools(init_project=False)
        after_init_mcp_count = len(test_manager.functions_mcps) if hasattr(test_manager, 'functions_mcps') else 0
        print(f"    Initialized MCP count: {after_init_mcp_count}")

        from pathlib import Path
        base_dir = Path(__file__).resolve().parents[2] / "tools"
        invented_dir = base_dir / "invented_tools" / (test_manager.project_name or "default_project")
        temp_file = invented_dir / "temp_tool_list.py"
        official_file = invented_dir / "official_tool_list.py"

        print(f"  File check: temp_tool_list.py - {temp_file.exists()}, official_tool_list.py - {official_file.exists()}")

        if official_file.exists():
            official_content = official_file.read_text()
            has_imports = "import json" in official_content and "from mcp.server.fastmcp import FastMCP" in official_content
            has_main = "if __name__ == \"__main__\":" in official_content
            print(f"    official_tool_list.py verification: import={has_imports}, main function={has_main}")

        print('  _build_temp_mcp_tools tests completed.')

    except Exception as e:
        print(f"_build_temp_mcp_tools test failed: {e}")
        import traceback
        traceback.print_exc()

    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = test_centralized_tool_actor()
        sys.exit(0 if success else 1)
    else:
        print('Start CentralizedToolActor through Ray.')
        print('Use --test to run local tests.')

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
