# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import re
import copy
import json
import json5
import asyncio
import traceback
from ast import literal_eval
from omegaconf import OmegaConf
from envs.tool_manager.manager import ToolManager
# from envs.utils.manager import MCPManager, BaseTool
from typing import Union, List, Tuple, Optional, Any, Dict
from envs.utils.util import ToolServiceError, DocParserError
from qwen_agent.tools import TOOL_REGISTRY, MCPManager, BaseTool
from qwen_agent.llm.schema import ASSISTANT, SYSTEM, USER, FUNCTION, ContentItem

def parse_mcp_tools_config(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        data = literal_eval(content)
        return data
    except Exception as e:
        print(f"\u89e3\u6790\u9519\u8bef: {e}")
        return None

class Qwen25Manager(ToolManager):
    def __init__(self, verl_config):
        if isinstance(verl_config, dict):
            verl_config = OmegaConf.to_container(verl_config)
        super().__init__(verl_config)
        self.generate_cfg = {
            'fncall_prompt_type': 'nous',
            'function_choice': 'auto',
            'parallel_function_calls': False,
            'lang': 'en',
            'max_input_tokens': 10000
        }

    def get_tool(self, name_or_short_name: str):
        '\u901a\u8fc7\u540d\u79f0\u6216\u7b80\u5199\u83b7\u53d6\u5de5\u5177\n        \n        Args:\n            name_or_short_name: \u5de5\u5177\u540d\u79f0\u6216\u7b80\u5199\n            \n        Returns:\n            \u627e\u5230\u7684\u5de5\u5177\uff0c\u5982\u679c\u6ca1\u627e\u5230\u5219\u8fd4\u56deNone\n        '
        name_or_short_name = str(name_or_short_name)
        return self.tool_map.get(name_or_short_name, None)

    @property
    def all_tools(self):
        '\u83b7\u53d6\u6240\u6709\u5de5\u5177\n        \n        Returns:\n            \u6240\u6709\u5de5\u5177\u7684\u5217\u8868\n        '
        return self.tool_map

    def _build_tools(self):
        config_path = self.verl_config.config_path
        if config_path is not None:
            function_list = parse_mcp_tools_config(config_path)

            if function_list:
                for tool in function_list:
                    self._init_tool(tool)

            self.functions = [func.function for func in self.tool_map.values()]
        else:
            print("The config_path is None!")
            self.functions = []

    async def execute_all_tools(self, actions, tool_list):
        '\u5f02\u6b65\u5e76\u884c\u6267\u884c\u6240\u6709\u5de5\u5177\u5217\u8868\n        \n        Args:\n            tool_list: \u5de5\u5177\u5217\u8868\u7684\u5217\u8868\n            \n        Returns:\n            \u6240\u6709\u5de5\u5177\u6267\u884c\u7ed3\u679c\u7684\u5217\u8868\n        '

        tasks = []
        for temp_action, temp_tool_list in zip(actions, tool_list):
            tasks.append(self._execute_tool_batch(temp_action, temp_tool_list))

        results = await asyncio.gather(*tasks)

        return results

    async def _execute_tool_batch(self, action, tools):
        '\u5f02\u6b65\u5e76\u884c\u6267\u884c\u4e00\u6279\u5de5\u5177\n        \n        Args:\n            tools: \u5de5\u5177\u5217\u8868\n            \n        Returns:\n            \u5de5\u5177\u6267\u884c\u7ed3\u679c\u7684\u5217\u8868\n        '
        async def execute_single_tool(tool):
            tool_instance = self.get_tool(tool["name"])
            args = tool["args"]
            if tool_instance is not None:
                try:
                    args = json.loads(args)
                except Exception as e:
                    pass

                if type(args) is dict:
                    try:

                        tool_result = await asyncio.to_thread(
                            self._call_tool,
                            tool["name"], json.dumps(args, ensure_ascii=False, indent=4)
                        )
                        result = '# Execute the tool {} successed\n  - The args are: {}\n  - The result is:\n{}'.format(tool["name"], args, tool_result)
                    except Exception as e:
                        result = '# Execute the tool {} failed\n  - The original args are: {}\n  - Error message:\n{}'.format(tool["name"], args, str(e))
                elif type(args) is str:
                    # Json decode error: xxx
                    result = args
                else:
                    result = 'Unexpected type of args: {} (args: {})'.format(type(args), args)
            else:
                if tool['name'] == '<empty>':
                    result = args
                else:
                    result = "# Failed to find the tool {} in the tool map".format(tool['name'])

            return result

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
        if isinstance(tool, BaseTool):
            tool_name = tool.name
            self.tool_map[tool_name] = tool
        elif isinstance(tool, dict) and 'mcpServers' in tool:
            tools = MCPManager().initConfig(tool)
            for tool in tools:
                tool_name = tool.name
                self.tool_map[tool_name] = tool
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

    def execute_actions(self, responses: List[str]):
        actions, tools = [], []
        for response in responses:
            temp_action, temp_tool_list = self.parse_response(response_content=response)
            # temp_action: answer or tools
            # if temp_action is 'answer', temp_tool_list is the answer
            # else, temp_tool_list is the list of the 'Tool' instances
            actions.append(temp_action)
            tools.append(temp_tool_list)

        try:
            tool_results = asyncio.run(self.execute_all_tools(actions, tools))
        except RuntimeError:

            loop = asyncio.get_event_loop()
            tool_results = loop.run_until_complete(self.execute_all_tools(actions, tools))

        return actions, tool_results

    def parse_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        '\u6267\u884c\u52a8\u4f5c\n        \n        Args:\n            response_content: \u54cd\u5e94\u6587\u672c\n            \n        Returns:\n            \u89e3\u6790\u540e\u7684\u52a8\u4f5c\u4fe1\u606f\uff0c\u5305\u542bname\u548cargs\uff08\u5982\u679c\u5b58\u5728\uff09\n            \u5982\u679c\u6ca1\u6709\u52a8\u4f5c\u6807\u7b7e\u6216\u683c\u5f0f\u4e0d\u6b63\u786e\uff0c\u8fd4\u56deNone\n        '

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

    def parse_end_flag(self, response_content: str) -> bool:
        answer = None
        answer_section = re.findall(r'(<answer>.*?</answer>)', response_content, re.DOTALL)
        if len(answer_section) > 0:
            answer = answer_section[-1]
            return True, answer
        else:
            return False, None

    def parse_tools(self, response: str):
        parsed_tools = []
        i = response.find('<tool_call>')
        # If no function call:
        if i < 0:
            return response

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

                    parsed_tools.append({
                        "name": fn['name'],
                        "args": json.dumps(fn['arguments'], ensure_ascii=False, indent=4),
                    })

                except (IndexError, KeyError, ValueError) as e:

                    parsed_tools.append({
                        "name": "<empty>",
                        "args": "# Extract the tool name failed"
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
                add_generation_prompt=add_generation_prompt
            )
        elif mode in ['tool_call', 'assistant_response']:
            # NOTE: the assistant response might not be used
            role = 'tool' if mode == 'tool_call' else ASSISTANT
            if type(input_data) == str:
                chat = {'role': role, 'content': input_data}
            elif type(input_data) == list:
                chat = input_data
            else:
                raise ValueError('Unexpected type of input_data {} ({})'.format(type(input_data), input_data))

            temp_prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=base_chat + chat, tools=self.functions,
                tokenize=False, add_generation_prompt=add_generation_prompt
            )
            prompt_with_chat_template = temp_prompt_with_chat_template.replace(base_prompt, '')
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

        return prompt_with_chat_template

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
