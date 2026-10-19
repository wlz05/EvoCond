# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import re
import sys
import json
import inspect
import asyncio
import importlib.util
from bs4 import BeautifulSoup
from typing import Any, Dict, Optional, List
from envs.tool_manager.manager import ToolManager

class ConfigParser:
    '\u914d\u7f6e\u89e3\u6790\u5668\uff0c\u7528\u4e8e\u8bfb\u53d6\u548c\u89e3\u6790\u914d\u7f6e\u6587\u4ef6'

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        '\u52a0\u8f7d\u914d\u7f6e\u6587\u4ef6\n        \n        Args:\n            config_path: \u914d\u7f6e\u6587\u4ef6\u8def\u5f84\n            \n        Returns:\n            \u914d\u7f6e\u5b57\u5178\n        '
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        '\u4fdd\u5b58\u914d\u7f6e\u5230\u6587\u4ef6\n        \n        Args:\n            config: \u914d\u7f6e\u5b57\u5178\n            config_path: \u914d\u7f6e\u6587\u4ef6\u8def\u5f84\n        '
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

class ToolParam:
    '\u5de5\u5177\u53c2\u6570\u7c7b'

    def __init__(self, param_config: Dict[str, Any]):
        '\u521d\u59cb\u5316\u5de5\u5177\u53c2\u6570\n        \n        Args:\n            param_config: \u53c2\u6570\u914d\u7f6e\u5b57\u5178\n        '
        self.id = param_config.get("id", "")
        self.name = param_config.get("name", "")
        self.type = param_config.get("type", "string")
        self.description = param_config.get("description", "")
        self.required = param_config.get("required", False)
        self.default = param_config.get("default", None)

    def convert_value(self, value: Any) -> Any:
        '\u6839\u636e\u53c2\u6570\u7c7b\u578b\u8f6c\u6362\u503c\n        \n        Args:\n            value: \u8f93\u5165\u503c\n            \n        Returns:\n            \u8f6c\u6362\u540e\u7684\u503c\n        '
        try:
            if self.type == "string":
                return str(value)
            elif self.type == "number":
                return float(value)
            elif self.type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ("true", "yes", "1", "t", "y")
                return bool(value)
            elif self.type == "array":
                if isinstance(value, str):
                    return json.loads(value)
                return list(value)
            elif self.type == "object":
                if isinstance(value, str):
                    return json.loads(value)
                return dict(value)
            else:
                return value
        except Exception as e:
            print(f"\u8f6c\u6362\u53c2\u6570 {self.name} \u5931\u8d25: {e}")
            return value

class Tool:
    '\u5de5\u5177\u7c7b\uff0c\u8868\u793a\u5185\u90e8\u5de5\u5177\u6216\u5916\u90e8\u5de5\u5177'

    def __init__(self, tool_config: Dict[str, Any]):
        '\u521d\u59cb\u5316\u5de5\u5177\n        \n        Args:\n            tool_config: \u5de5\u5177\u914d\u7f6e\u5b57\u5178\n        '
        self.id = tool_config.get("id", "")
        self.name = tool_config.get("name", "")
        self.short_name = tool_config.get("shortName", "")
        self.type = tool_config.get("type", "")
        self.description = tool_config.get("description", "")
        self.python_code = tool_config.get("pythonCode", "")
        self._func = None

        if self.type == "external":

            self.params = [ToolParam(param) for param in tool_config.get("params", [])]

            self.param_map = {param.name: param for param in self.params}
        else:

            self.params = []
            self.param_map = {}

        if self.type == "external" and self.python_code:
            self._compile_function()

    def _compile_function(self):
        '\u7f16\u8bd1Python\u4ee3\u7801\u4e3a\u51fd\u6570'
        try:

            module_name = f"tool_{self.id}"
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module

            exec(self.python_code, module.__dict__)

            for name, obj in module.__dict__.items():
                if inspect.isfunction(obj):
                    self._func = obj
                    break

            if not self._func:
                raise ValueError(f"\u5de5\u5177 {self.name} \u7684Python\u4ee3\u7801\u4e2d\u6ca1\u6709\u627e\u5230\u51fd\u6570\u5b9a\u4e49")

        except Exception as e:
            print(f"\u7f16\u8bd1\u5de5\u5177 {self.name} \u5931\u8d25: {e}")

    def __call__(self, *args, **kwargs):
        '\u8c03\u7528\u5de5\u5177\u51fd\u6570'
        if not self._func:
            raise ValueError(f"\u5de5\u5177 {self.name} \u6ca1\u6709\u53ef\u8c03\u7528\u7684\u51fd\u6570")

        processed_kwargs = {}
        if self.type == "external":

            for key, value in kwargs.items():
                if key in self.param_map:
                    param = self.param_map[key]
                    processed_kwargs[key] = param.convert_value(value)
        else:

            processed_kwargs = kwargs

        return self._func(*args, **processed_kwargs)

    def get_param(self, name: str) -> Optional[ToolParam]:
        '\u901a\u8fc7\u540d\u79f0\u83b7\u53d6\u53c2\u6570\n        \n        Args:\n            name: \u53c2\u6570\u540d\u79f0\n            \n        Returns:\n            \u627e\u5230\u7684\u53c2\u6570\uff0c\u5982\u679c\u6ca1\u627e\u5230\u5219\u8fd4\u56deNone\n        '
        if self.type != "external":
            return None

        return self.param_map.get(name)

class ConfigManager(ToolManager):
    def get_tool(self, name_or_short_name: str):
        '\u901a\u8fc7\u540d\u79f0\u6216\u7b80\u5199\u83b7\u53d6\u5de5\u5177\n        \n        Args:\n            name_or_short_name: \u5de5\u5177\u540d\u79f0\u6216\u7b80\u5199\n            \n        Returns:\n            \u627e\u5230\u7684\u5de5\u5177\uff0c\u5982\u679c\u6ca1\u627e\u5230\u5219\u8fd4\u56deNone\n        '
        name_or_short_name = str(name_or_short_name)
        return self.tool_map.get(name_or_short_name, None)

    @property
    def all_tools(self):
        '\u83b7\u53d6\u6240\u6709\u5de5\u5177\n        \n        Returns:\n            \u6240\u6709\u5de5\u5177\u7684\u5217\u8868\n        '
        return self.internal_tools + self.external_tools

    def _build_tools(self):
        config_path = self.verl_config.config_path
        self.config = ConfigParser.load_config(config_path)

        self.id = self.config.get("id", "")
        self.name = self.config.get("name", "")
        self.description = self.config.get("description", "")

        self.system_prompt_template = self.config.get("systemPrompt", "")
        self.user_prompt_template = self.config.get("userPrompt", "")

        self.internal_tools = self._create_tools(self.config.get("internalTools", []))
        self.external_tools = self._create_tools(self.config.get("externalTools", []))

        all_params = self.config.get("params", [])
        user_params = self.config.get("userParams", [])

        self.env_fields = {}
        self.template_fields = {}
        self.user_fields = {}

        for param in all_params:
            param_name = param.get("name", "")
            if not param_name:
                continue

            category = param.get("category", "")

            if category == '\u6a21\u677f\u5b57\u6bb5':
                self.template_fields[param_name] = param
            elif category == '\u73af\u5883\u5b57\u6bb5':
                self.env_fields[param_name] = param

                param_default = param.get("default")
                setattr(self, param_name, param_default if param_default is not None else None)

        for param in user_params:
            param_name = param.get("name", "")
            if not param_name:
                continue

            self.user_fields[param_name] = param

        self.tool_map = {}
        for tool in self.internal_tools + self.external_tools:
            self.tool_map[tool.name] = tool
            if tool.short_name:
                self.tool_map[tool.short_name] = tool

    def _create_tools(self, tool_configs):
        '\u4ece\u914d\u7f6e\u521b\u5efa\u5de5\u5177\u5217\u8868\n        \n        Args:\n            tool_configs: \u5de5\u5177\u914d\u7f6e\u5217\u8868\n            \n        Returns:\n            \u5de5\u5177\u5bf9\u8c61\u5217\u8868\n        '
        tools = []
        for tool_config in tool_configs:
            tool = Tool(tool_config)
            tools.append(tool)

        return tools

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
            tool_instance = self.tool_manager.get_tool(tool["name"])
            args = tool["args"]
            if tool_instance is not None:
                if type(args) is dict:
                    try:

                        tool_result = await asyncio.to_thread(tool_instance, **args)
                        result = '# Execute the tool {} successed\n  - The args are: {}\n  - The result is:\n{}'.format(tool["name"], args, tool_result)
                    except Exception as e:
                        result = '# Execute the tool {} failed\n  - The original args are: {}\n  - Error message:\n{}'.format(tool["name"], args, str(e))
                elif type(args) is str:
                    # Json decode error: xxx
                    result = args
                else:
                    raise ValueError('Unexpected type of args: {}'.format(type(args)))
            else:
                if tool['name'] == '<empty>':
                    result = args
                else:
                    result = "# Failed to find the tool {} in the tool map".format(tool['name'])

            return result

        if action == 'answer':
            # 'tools' is a str (the answer)
            result_str = tools
        elif action == 'error':
            # 'error' only occurs when there is no 'actions' tag or there is no 'action' tag after extraction
            # ('Cannot extract the actions tag' or 'There is no action after extraction')
            result_str = """# Extract the actions failed due to: {}""".format(tools)
        elif action == 'actions':
            # 'tools' is the list of the 'Tool' instances
            tasks = [execute_single_tool(temp_tool) for temp_tool in tools]
            results = await asyncio.gather(*tasks)
            result_str = '\n'.join(results)
        else:
            raise ValueError('Unexpected action: {}'.format(action))

        return result_str

    def execute_actions(self, responses: List[str]):
        # action: answer, xxx or error
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

        if_tools, tools = self.parse_tools(response_content)
        if if_tools:
            return 'actions', tools

        return None

    def parse_end_flag(self, response_content: str) -> bool:
        answer = None
        answer_section = re.findall(r'(<answer>.*?</answer>)', response_content, re.DOTALL)
        if len(answer_section) > 0:
            answer = answer_section[-1]

        return True, answer

    def parse_tools(self, response_content: str):
        actions_section = re.findall(r'(<actions>.*?</actions>)', response_content, re.DOTALL)
        if len(actions_section) > 0:
            soup = BeautifulSoup(actions_section[-1], 'html.parser')

            # 'error' only occurs when there is no 'actions' tag or there is no 'action' tag after extraction
            ext_actions = soup.find_all('actions')[-1]
            if not ext_actions:
                return 'error', 'Cannot extract the actions tag'

            ext_action_list = ext_actions.find_all('action')
            if not ext_action_list:
                return 'error', 'There is no action after extraction'

            tool_list = []
            for ext_action in ext_action_list:

                name_tag = ext_action.find('name')
                if name_tag is not None:
                    action_name = name_tag.text.strip()

                    args_tag = ext_action.find('args')
                    if args_tag:
                        args_text = args_tag.text.strip()
                        try:
                            args_dict = json.loads(args_text)
                        except json.JSONDecodeError as e:
                            args_dict = f"Json decode error: {e}"
                else:
                    action_name = '<empty>'
                    args_dict = """# Extract the tool name failed"""

                tool = {
                    "name": action_name,
                    "args": args_dict
                }
                tool_list.append(tool)

            return True, tool_list
        else:
            return False, None

    def get_prompt(self, input_data, tokenizer, mode='initial', add_generation_prompt=True):
        assert mode in ['initial', 'tool_call', 'assistant_response'], 'Invalid mode: {}'.format(mode)
        base_chat = [
            {'role': 'system', 'content': 'base'},
            {'role': 'user', 'content': 'base'},
        ]
        base_prompt = tokenizer.apply_chat_template(
            conversation=base_chat,
            tools=self.functions,
            tokenize=False
        )

        if mode == 'initial':
            chat = input_data
            prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=chat, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        elif mode in ['tool_call', 'assistant_response']:
            # NOTE: the assistant response might not be used
            role = 'tool' if mode == 'tool_call' else 'assistant'
            chat = {'role': role, 'content': input_data}
            temp_prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=base_chat + [chat], tokenize=False, add_generation_prompt=add_generation_prompt
            )
            prompt_with_chat_template = temp_prompt_with_chat_template.replace(base_prompt, '')
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

        return prompt_with_chat_template

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
