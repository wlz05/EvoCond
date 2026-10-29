# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import re
import copy
import json
import json5
import asyncio
import traceback
import time
import os
from pathlib import Path
from ast import literal_eval
from omegaconf import OmegaConf
from envs.tool_manager.manager import ToolManager
# from envs.utils.manager import MCPManager, BaseTool
from typing import Union, List, Tuple, Optional, Any, Dict
from envs.utils.util import ToolServiceError, DocParserError
from envs.utils.manager import MCPManager as SSEMCPManager
from qwen_agent.tools import TOOL_REGISTRY, MCPManager, BaseTool
from qwen_agent.llm.schema import ASSISTANT, SYSTEM, USER, FUNCTION, ContentItem
from envs.utils.limiter import ConcurrencyLimiter
from envs.utils.asynchronous import AsyncMCPManager

def parse_mcp_tools_config(file_path, project_name):
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        if "{project_name}" in content:
            content = content.replace("{project_name}", project_name or "default_project")

        data = literal_eval(content)
        return data
    except Exception as e:
        print(f"Parse error: {e}")
        return None

class QwenManager(ToolManager):
    def __init__(self, verl_config):
        if isinstance(verl_config, dict):
            verl_config = OmegaConf.create(verl_config)
        super().__init__(verl_config)

        self.generate_cfg = {
            'fncall_prompt_type': 'nous',
            'function_choice': 'auto',
            'parallel_function_calls': False,
            'lang': 'en',
            'max_input_tokens': 10000
        }

        if self.verl_config.enable_limiter:
            global_limit = getattr(verl_config, 'max_concurrency', 100)
            self._limiter = ConcurrencyLimiter(global_limit=global_limit)
        else:
            self._limiter = None

        self._search_cache = {}
        self._search_cache_lock = asyncio.Lock()
        self._in_flight_requests = {}
        self._in_flight_lock = asyncio.Lock()
        self._search_cache_refresh_interval = 15
        self._search_cache_mod_time = 0.0
        self._search_cache_last_check = 0.0
        self._search_cache_path = os.environ.get(
            "RL_FACTORY_SEARCH_CACHE_DIR",
            str(Path.home() / ".cache" / "rl_factory" / "search"),
        )

        self.local_search = getattr(self.verl_config, 'local_search', False)
        print(f"Search Cache Config: local_search={self.local_search}")

        if self.local_search:
            self._search_cache_file = Path(os.path.join(self._search_cache_path, "local_bing_search_cache.jsonl"))
        else:
            self._search_cache_file = Path(os.path.join(self._search_cache_path, "bing_search_cache.jsonl"))

        print(f"Search Cache File: {self._search_cache_file}")
        self._load_search_cache()

    def get_tool(self, name_or_short_name: str):
        'Find a tool by name or alias; return None when absent.'
        name_or_short_name = str(name_or_short_name)
        return self.tool_map.get(name_or_short_name, None)

    @property
    def all_tools(self):
        'List all tools.'
        return self.tool_map

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

    def _build_tools(self):
        config_path = self.verl_config.config_path
        if config_path is not None:
            function_list = parse_mcp_tools_config(config_path, self.project_name)
            if function_list:
                for tool in function_list:
                    self._init_tool(tool)

            self.functions = [func.function for func in self.tool_map.values()]
        else:
            print("The config_path is None!")
            self.functions = []

    async def execute_all_tools_with_limiter(self, actions, tools, temp_dict=None):
        'Execute tool calls in parallel.'
        async def execute_single_tool(tool):
            'Execute one tool call.'
            try:
                tool_name = tool.get("name", "default")

                # Add MCP registration logic
                if tool_name == 'create_and_execute_mcp' and temp_dict is not None:
                    try:
                        args_content = tool.get("args", "")
                        if isinstance(args_content, str):
                            args = json.loads(args_content)
                        elif isinstance(args_content, dict):
                            args = args_content
                        else:
                            args = {}

                        if 'name' in args and 'code' in args:
                            temp_dict[args['name']] = {
                                "mcp_name": args['name'],
                                "description": args.get('description', ''),
                                "arguments": args.get('arguments', ''),
                                "returns": args.get('returns', ''),
                                "code": args.get('code', ''),
                                "step_id": len(temp_dict) + 2 # Simple step_id logic
                            }
                            print(f"Agent Log: MCP {args['name']} registered in temp_dict")
                    except Exception as e:
                        print(f"Error registering MCP tool in temp_dict: {e}")

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
                            print(f"DEBUG: In-flight hit: Waiting for query '{query_str}'")
                            return await self._in_flight_requests[query_str]

                        future = asyncio.get_running_loop().create_future()
                        self._in_flight_requests[query_str] = future

                async with self._limiter.limit(tool_name):
                    try:

                        result = await self._call_tool(tool_name, tool.get("args", ""))

                        if tool_name == "search" or tool_name == "search-search":
                            await asyncio.to_thread(self._save_search_cache_sync, query_str, result)
                            async with self._search_cache_lock:
                                self._search_cache[query_str] = result

                            if not future.done():
                                future.set_result(result)

                        return result
                    except Exception as e:

                        if (tool_name == "search" or tool_name == "search-search") and not future.done():
                            future.set_exception(e)
                        raise e
                    finally:

                        if tool_name == "search" or tool_name == "search-search":
                            async with self._in_flight_lock:
                                if query_str in self._in_flight_requests and self._in_flight_requests[query_str] is future:
                                    del self._in_flight_requests[query_str]

            except Exception as e:
                print(f"Tool call failed: {e}")
                return f"# Tool call failed: {str(e)}"

        tasks = []
        for action, tool_list in zip(actions, tools):
            if action == "actions":
                for tool in tool_list:
                    tasks.append(execute_single_tool(tool))

        results = await asyncio.gather(*tasks)
        return results

    async def execute_all_tools(self, actions, tool_list, temp_dict=None):
        'Execute tool batches asynchronously, sharing the temporary MCP mapping.'

        tasks = []
        for temp_action, temp_tool_list in zip(actions, tool_list):
            tasks.append(self._execute_tool_batch(temp_action, temp_tool_list, temp_dict))

        results = await asyncio.gather(*tasks)

        return results

    async def _execute_tool_batch(self, action, tools, temp_dict=None):
        'Execute one tool batch asynchronously with the temporary MCP mapping.'
        async def execute_single_tool(tool):
            try:
                tool_name = tool.get("name", "default")

                # Add MCP registration logic
                if tool_name == 'create_and_execute_mcp' and temp_dict is not None:
                    try:
                        args_content = tool.get("args", "")
                        if isinstance(args_content, str):
                            args = json.loads(args_content)
                        elif isinstance(args_content, dict):
                            args = args_content
                        else:
                            args = {}

                        if 'name' in args and 'code' in args:
                            temp_dict[args['name']] = {
                                "mcp_name": args['name'],
                                "description": args.get('description', ''),
                                "arguments": args.get('arguments', ''),
                                "returns": args.get('returns', ''),
                                "code": args.get('code', ''),
                                "step_id": len(temp_dict) + 2
                            }
                            print(f"Agent Log: MCP {args['name']} registered in temp_dict")
                    except Exception as e:
                        print(f"Error registering MCP tool in temp_dict: {e}")

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
                            print(f"DEBUG: In-flight hit: Waiting for query '{query_str}'")
                            return await self._in_flight_requests[query_str]

                        future = asyncio.get_running_loop().create_future()
                        self._in_flight_requests[query_str] = future

                tool_instance = self.get_tool(tool_name)
                args = tool.get("args", "")
                if tool_instance is not None:
                    try:
                        args = json.loads(args)
                    except Exception as e:
                        pass

                    if type(args) is dict:
                        try:

                            print(f"DEBUG: Executing tool {tool_name}...")
                            tool_result = await asyncio.to_thread(
                                self._call_tool,
                                tool_name, json.dumps(args, ensure_ascii=False, indent=4)
                            )
                            print(f"DEBUG: Tool execution finished. Result length: {len(str(tool_result))}")

                            if tool_name == "search" or tool_name == "search-search":
                                print("DEBUG: Triggering search cache save...")
                                if isinstance(tool.get("args", ""), dict):
                                    query_str = json.dumps(tool.get("args", ""), sort_keys=True)
                                else:
                                    query_str = str(tool.get("args", ""))

                                await asyncio.to_thread(self._save_search_cache_sync, query_str, tool_result)
                                async with self._search_cache_lock:
                                    self._search_cache[query_str] = tool_result

                                if not future.done():
                                    future.set_result(tool_result)

                                print("DEBUG: Search cache save task completed.")

                            result = '# Execute the tool {} successed\n  - The result is:\n{}'.format(tool_name, tool_result)

                        except Exception as e:

                            if (tool_name == "search" or tool_name == "search-search") and not future.done():
                                future.set_exception(e)

                            result = '# Execute the tool {} failed\n  - Error message:\n{}'.format(tool_name, str(e))
                        finally:

                            if tool_name == "search" or tool_name == "search-search":
                                async with self._in_flight_lock:
                                    if query_str in self._in_flight_requests and self._in_flight_requests[query_str] is future:
                                        del self._in_flight_requests[query_str]

                    elif type(args) is str:
                        # Json decode error: xxx
                        result = 'parse json failed, argument is: {}'.format(args)
                    else:
                        result = 'Unexpected type of args: {} (args: {})'.format(type(args), args)
                else:
                    if tool_name == '<empty>':
                        result = 'toolname is empty, argument is: '.format(args)
                    else:
                        result = "# Failed to find the tool {} in the tool map".format(tool_name)

                return result
            except Exception as e:
                print(f"Tool call failed: {e}")
                return f"# Tool call failed: {str(e)}"

        if action == 'answer':
            # 'tools' is a str (the answer)
            results = {'role': 'assitant', 'content': tools}
        elif action == 'error':
            # 'error' only occurs when there is no 'actions' tag or there is no 'action' tag after extraction
            # ('Cannot extract the actions tag' or 'There is no action after extraction')
            results = {'role': 'assitant', 'content': """# Extract the tools failed due to: {}""".format(tools)}
        elif action == 'actions':
            # 'tools' is the list of the 'Tool' instances
            tasks = [execute_single_tool(temp_tool) for temp_tool in tools]
            tool_results = await asyncio.gather(*tasks)
            results = [{'role': 'tool', 'content': temp_tool_result} for temp_tool_result in tool_results]
        else:
            raise ValueError('Unexpected action: {}'.format(action))

        return results

    def _init_tool(self, tool: Union[str, BaseTool]):
        print(f'tool: {tool}')
        if isinstance(tool, BaseTool):
            tool_name = tool.name
            self.tool_map[tool_name] = tool
        elif isinstance(tool, dict) and 'mcpServers' in tool:
            print(f'MCP is using {self.verl_config.mcp_mode} mode')
            if self.verl_config.mcp_mode == 'sse':
                if self.verl_config.parallel_sse_tool_call.is_enabled:
                    tools = AsyncMCPManager(num_instances=self.verl_config.parallel_sse_tool_call.num_instances).initConfig(tool)
                else:
                    tools = SSEMCPManager().initConfig(tool)
            elif self.verl_config.mcp_mode == 'stdio':
                tools = MCPManager().initConfig(tool)
            else:
                raise ValueError(f"Unexpected mcp mode: {self.verl_config.mcp_mode}")

            for tool in tools:
                tool_name = tool.name
                self.tool_map[tool_name] = tool
                print(f'register tool: {tool_name} --> {tool}')
        else:
            if isinstance(tool, dict):
                tool_name = tool['name']
                tool_cfg = tool
            else:
                tool_name = tool
                tool_cfg = None
            if tool_name not in TOOL_REGISTRY:
                raise ValueError(f'Tool {tool_name} is not registered.')

            self.tool_map[tool_name] = TOOL_REGISTRY[tool_name](tool_cfg)

        # select used tools within one sse link before tool learning
        if self.verl_config.mcp_mode == 'sse' and len(self.verl_config.tool_name_selected) != 0:
            try:
                self.tool_map = {each_tool_name: self.tool_map[each_tool_name] for each_tool_name in self.verl_config.tool_name_selected}
            except:
                raise ValueError('Selected tool names are not valid or available sse tool list error. Available tool names: {}'.format(self.tool_map.keys()))

    def execute_actions(self, responses: List[str], temp_dict: Optional[Dict] = None):
        actions, tools = [], []
        for response in responses:
            temp_action, temp_tool_list = self.parse_response(response_content=response)
            # temp_action: answer or tools
            # if temp_action is 'answer', temp_tool_list is the answer
            # else, temp_tool_list is the list of the 'Tool' instances
            actions.append(temp_action)
            temp_tool_list = self.full_name(temp_tool_list)
            tools.append(temp_tool_list)

        try:
            if self.verl_config.enable_limiter:
                assert self._limiter is not None, "Limiter is not enabled"
                tool_results = asyncio.run(self.execute_all_tools_with_limiter(actions, tools, temp_dict))
            else:
                tool_results = asyncio.run(self.execute_all_tools(actions, tools, temp_dict))
        except RuntimeError:

            loop = asyncio.get_event_loop()
            if self.verl_config.enable_limiter:
                assert self._limiter is not None, "Limiter is not enabled"
                tool_results = loop.run_until_complete(self.execute_all_tools_with_limiter(actions, tools, temp_dict))
            else:
                tool_results = loop.run_until_complete(self.execute_all_tools(actions, tools, temp_dict))

        return actions, tool_results

    def parse_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        'Parse action name and arguments; return None for missing or malformed action tags.'

        if_answer, answer = self.parse_end_flag(response_content)
        if if_answer:
            return 'answer', answer

        tools = self.parse_tools(response_content)
        if type(tools) == list:
            return 'actions', tools
        else:
            assert type(tools) == str
            # if the response is not a tool call, it is an answer
            return 'answer', tools

    def parse_end_flag(self, response_content: str) -> tuple[bool, str]:
        'Parse boxed answers and supported answer tags.'
        answer = None

        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, response_content, re.DOTALL)

        if answer_match:

            answer_content = answer_match.group(1).strip()

            if r'\boxed{' in answer_content:
                return True, answer_content

            boxed_pattern = r'\\boxed\{([^}]+)\}'
            boxed_match = re.search(boxed_pattern, answer_content)
            if boxed_match:
                return True, boxed_match.group(1)

            return True, answer_content

        return False, None

    def parse_tools(self, response: str):
        parsed_tools = []
        i = response.find('<tool_call>')
        # If no function call:
        if i < 0:
            j = response.find('</tool_call>')
            if j < 0:
                return response
            else:
                parsed_tools.append({
                        "name": "<error>",
                        "args": "# Extract the tool name failed"
                    })
                return parsed_tools

        # split tool-call to separate assistant msg
        tool_call_list = response.split('<tool_call>')
        pre_thought = tool_call_list[0].strip()
        for txt in tool_call_list[1:]:
            if not txt.strip():
                continue

            if '</tool_call>' not in txt:
                # incomplete </tool_call>: This is to better represent incomplete tool calls in streaming output
                fn_name = '<empty>'
                fn_args = """# Extract the tool name failed"""
                parsed_tools.append(
                    {
                        "name": fn_name,
                        "args": fn_args,
                    }
                )
            else:
                one_tool_call_txt = txt.split('</tool_call>')
                try:

                    if not one_tool_call_txt[0].strip():
                        raise ValueError("Empty tool call content")

                    fn = json5.loads(one_tool_call_txt[0].strip())

                    if type(fn) is not dict or 'name' not in fn or 'arguments' not in fn:
                        raise KeyError("Missing required fields")

                    # Ensure name is always a string to handle cases where it might be an int
                    parsed_tools.append({
                        "name": str(fn['name']),
                        "args": json.dumps(fn['arguments'], ensure_ascii=False, indent=4),
                    })

                except (IndexError, KeyError, ValueError) as e:

                    parsed_tools.append({
                        "name": "<empty>",
                        "args": f"# Extract the tool name failed: {str(e)}"
                    })

        if len(parsed_tools) == 0 :
            # <tool_call> is last token
            fn_name = '<empty>'
            fn_args = """# Extract the tool name failed"""
            parsed_tools.append(
                {
                    "name": fn_name,
                    "args": fn_args,
                })

        return parsed_tools

    def get_prompt(self, input_data, tokenizer, mode='initial', add_generation_prompt=True):
        assert mode in ['initial', 'tool_call', 'assistant_response'], 'Invalid mode: {}'.format(mode)
        base_chat = [
            {'role': SYSTEM, 'content': 'base'},
            {'role': USER, 'content': 'base'},
        ]
        base_prompt = tokenizer.apply_chat_template(
            conversation=base_chat,
            tools=self.functions,
            tokenize=False, add_generation_prompt=False
        )

        if mode == 'initial':
            chat = input_data
            prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=chat, tokenize=False, tools=self.functions,
                add_generation_prompt=add_generation_prompt, enable_thinking=self.verl_config.enable_thinking
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
                conversation=base_chat + chat, tools=self.functions,
                tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=self.verl_config.enable_thinking
            )
            prompt_with_chat_template = temp_prompt_with_chat_template.replace(base_prompt, '')
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

        return prompt_with_chat_template

if __name__ == "__main__":
    import os
    import sys

    def test_qwen_manager():
        'Test basic QwenManager behavior.'
        print('Starting QwenManager tests...')

        test_config = {
            'config_path': 'envs/configs/mcp_tools.pydata',
            'mcp_mode': 'stdio',
            'enable_limiter': False,
            'tool_name_selected': [],
            'use_storage_manager': False,
            'enable_thinking': False,
            'local_search': False,
            'project_name': "baseline_gpu6_grpo_prm"
        }

        try:
            print('1. Initializing QwenManager...')
            manager = QwenManager(test_config)
            print('QwenManager initialized.')

            print('\n2. Checking tool mappings...')
            print(f"   Available tools: {len(manager.tool_map)}")
            for tool_name, tool in manager.tool_map.items():
                print(f"   - {tool_name}: {type(tool).__name__}")
            print('Tool mappings checked.')

            print('\n3. Testing tool lookup...')
            if manager.tool_map:
                first_tool_name = list(manager.tool_map.keys())[0]
                tool = manager.get_tool(first_tool_name)
                if tool:
                    print(f"Tool retrieved: {first_tool_name}")
                else:
                    print(f"Tool lookup failed: {first_tool_name}")
            else:
                print('No tools available.')

            print('\n4. Testing tool execution...')

            import time
            unique_query = f"What is artificial intelligence?"
            test_response = f"""<tool_call>\n{{\n    "name": "search",\n    "arguments": {{\n        "query": "{unique_query}",\n        "topk": 2\n    }}\n}}\n</tool_call>"""

            try:
                action, tools = manager.parse_response(test_response)
                print(f"   Parsed action: {action}")
                if action == 'actions':
                    print(f"   Tool count: {len(tools)}")
                    for i, tool in enumerate(tools):
                        print(f"   Tool{i+1}: {tool}")

                    if tools:
                        print('   Executing the tool...')
                        actions_list, tool_results = manager.execute_actions([test_response])
                        print('   Execution result:')
                        for i, result in enumerate(tool_results):
                            print(f"   Result{i+1}: {result[:200]}..." if len(str(result)) > 200 else f"   Result{i+1}: {result}")
                else:
                    print(f"   Response content: {tools}")
                print('Tool execution tests completed.')

            except Exception as e:
                print(f"Tool execution test failed: {str(e)}")
                traceback.print_exc()

            print('\n5. Testing the Python tool...')

            python_test_response = '<tool_call>\n{\n    "name": "execute_python_code",\n    "arguments": {\n        "code": "print(\'Hello from test!\')\\nresult = 2 + 3\\nprint(f\'2 + 3 = {result}\')",\n        "timeout": 10.0\n    }\n}\n</tool_call>'

            try:
                action, tools = manager.parse_response(python_test_response)
                print(f"   Parsed Python action: {action}")
                if action == 'actions' and tools:
                    print('   Executing the Python tool...')
                    actions_list, tool_results = manager.execute_actions([python_test_response])
                    print('   Python tool result:')
                    for i, result in enumerate(tool_results):
                        print(f"   Result{i+1}: {result[:300]}..." if len(str(result)) > 300 else f"   Result{i+1}: {result}")

                    result_str = str(tool_results[0]) if tool_results else ""
                    if "Hello from test!" in result_str or "2 + 3 = 5" in result_str:
                        print('Python tool result is correct.')
                    else:
                        print('Python tool result may be incorrect.')
                else:
                    print('Python tool parsing failed.')
                print('Python tool tests completed.')
            except Exception as e:
                print(f"Python tool test failed: {str(e)}")
                traceback.print_exc()

            print('\n6. Testing answer parsing...')
            test_answers = [
                "<answer>\\boxed{[2, 1]}</answer>",
                "<answer>42</answer>",
                "<answer>\\boxed{42}</answer>",
                "Some text <answer>\\boxed{[2, 1]}</answer> more text"
            ]

            for test_answer in test_answers:
                try:
                    found, parsed = manager.parse_end_flag(test_answer)
                    print(f"   Input: {test_answer}")
                    print(f"   Parsed result: found={found}, parsed='{parsed}'")
                except Exception as e:
                    print(f"   Parsing failed: {str(e)}")

            print('Answer parsing tests completed.')

            print('\n7. Testing the search cache...')
            if hasattr(manager, '_search_cache'):
                print(f"   Cache entries: {len(manager._search_cache)}")
                print('Search cache checked.')
            else:
                print('Search cache is not initialized.')

            return True

        except Exception as e:
            print(f"Test failed: {str(e)}")
            traceback.print_exc()
            return False

    def test_config_parsing():
        'Test configuration parsing.'
        print('\nTesting configuration parsing...')

        config_path = 'envs/configs/mcp_tools.pydata'
        if not os.path.exists(config_path):
            # Fallback to local path if absolute path doesn't exist (e.g. different environment)
            config_path = str(Path(__file__).resolve().parents[2] / 'envs/configs/mcp_tools.pydata')

        print(f"   Checking config path: {config_path}")

        if os.path.exists(config_path):
            try:
                function_list = parse_mcp_tools_config(config_path, "baseline_gpu6_grpo_prm")
                if function_list is None:
                    print('Configuration parser returned None.')
                    return False

                print(f"Configuration parsed; found {len(function_list)} tool configurations")
                for i, tool_config in enumerate(function_list):
                    print(f"   Configuration{i+1}: {tool_config}")
                return True
            except Exception as e:
                print(f"Configuration parsing failed: {str(e)}")
                return False
        else:
            print(f"Configuration file not found: {config_path}")
            return False

    class DummyTool:
        'Provide an offline test tool without external dependencies.'
        def __init__(self, name: str = "dummy"):
            self.name = name

        def call(self, args, **kwargs):
            try:
                payload = json.loads(args) if isinstance(args, str) else args
            except Exception:
                payload = args
            return {"echo": payload}

    def test_full_pipeline_without_external_deps():
        'Test parsing, execution and response using local stub tools.'
        print('\n8. Testing parsing, execution and response...')
        cfg = {
            'config_path': None,
            'mcp_mode': 'stdio',
            'enable_limiter': False,
            'tool_name_selected': [],
            'use_storage_manager': False,
            'enable_thinking': False,
            'local_search': False
        }

        manager = QwenManager(cfg)

        manager.tool_map.clear()
        dummy = DummyTool()
        manager.tool_map[dummy.name] = dummy

        tool_args = {"msg": "hello", "val": 42}
        test_response = f"""<tool_call>\n{{\n    "name": "{dummy.name}",\n    "arguments": {json.dumps(tool_args, ensure_ascii=False)}\n}}\n</tool_call>"""

        action, tools = manager.parse_response(test_response)
        manager._build_tools()
        assert action == 'actions', f"Failed to parse action; got {action}"
        assert len(tools) == 1 and tools[0]['name'] == dummy.name, 'Unexpected tool parsing result'

        actions_list, tool_results = manager.execute_actions([test_response])
        assert actions_list == ['actions'], f"Unexpected action list: {actions_list}"
        assert tool_results and isinstance(tool_results[0], list), 'Unexpected result format'
        result_item = tool_results[0][0]
        assert isinstance(result_item, dict) and result_item.get('role') == 'tool', 'Tool result has no role field'
        assert 'echo' in str(result_item.get('content', '')), 'Tool result has no echoed field'

        print('End-to-end offline test passed.')
        return True

    print("=" * 60)
    print('QwenManager diagnostics')
    print("=" * 60)

    config_test_passed = test_config_parsing()

    manager_test_passed = test_qwen_manager()

    full_pipeline_passed = test_full_pipeline_without_external_deps()

    print("\n" + "=" * 60)
    print('Test summary:')
    print(f"Configuration parsing: {'PASS' if config_test_passed else 'FAIL'}")
    print(f"Manager functions: {'PASS' if manager_test_passed else 'FAIL'}")
    print(f"Offline end-to-end test: {'PASS' if full_pipeline_passed else 'FAIL'}")

    if config_test_passed and manager_test_passed and full_pipeline_passed:
        print('All tests passed.')
        sys.exit(0)
    else:
        print('Some tests failed; check configuration and services.')
        sys.exit(1)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
