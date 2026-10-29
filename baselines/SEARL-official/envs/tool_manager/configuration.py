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
    'Read and parse tool configuration files.'

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        'Read a configuration mapping from disk.'
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        'Write a configuration mapping to disk.'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

class ToolParam:
    'Describe a tool parameter.'

    def __init__(self, param_config: Dict[str, Any]):
        'Initialize a tool parameter from its configuration.'
        self.id = param_config.get("id", "")
        self.name = param_config.get("name", "")
        self.type = param_config.get("type", "string")
        self.description = param_config.get("description", "")
        self.required = param_config.get("required", False)
        self.default = param_config.get("default", None)

    def convert_value(self, value: Any) -> Any:
        'Convert an input value to the declared parameter type.'
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
            print(f"Failed to convert argument {self.name}: {e}")
            return value

class Tool:
    'Represent an internal or external tool.'

    def __init__(self, tool_config: Dict[str, Any]):
        'Initialize a tool from its configuration.'
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
        'Compile Python source into a function.'
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
                raise ValueError(f"Tool {self.name} has no Python function definition")

        except Exception as e:
            print(f"Compiling tool {self.name}: {e}")

    def __call__(self, *args, **kwargs):
        'Invoke the tool function.'
        if not self._func:
            raise ValueError(f"Tool {self.name} has no callable function")

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
        'Find a parameter by name; return None when absent.'
        if self.type != "external":
            return None

        return self.param_map.get(name)

class ConfigManager(ToolManager):
    def get_tool(self, name_or_short_name: str):
        'Find a tool by name or alias; return None when absent.'
        name_or_short_name = str(name_or_short_name)
        return self.tool_map.get(name_or_short_name, None)

    @property
    def all_tools(self):
        'Return all tools.'
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

            if category == 'template' or re.fullmatch(r'\u6a21\u677f\u5b57\u6bb5', category):
                self.template_fields[param_name] = param
            elif category == 'environment' or re.fullmatch(r'\u73af\u5883\u5b57\u6bb5', category):
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
        'Build tool objects from configuration entries.'
        tools = []
        for tool_config in tool_configs:
            tool = Tool(tool_config)
            tools.append(tool)

        return tools

    async def execute_all_tools(self, actions, tool_list):
        'Execute all tool batches asynchronously and return their results.'

        tasks = []
        for temp_action, temp_tool_list in zip(actions, tool_list):
            tasks.append(self._execute_tool_batch(temp_action, temp_tool_list))

        results = await asyncio.gather(*tasks)

        return results

    async def _execute_tool_batch(self, action, tools):
        'Execute one tool batch asynchronously and return its results.'
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
        'Parse action name and arguments; return None for missing or malformed action tags.'

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

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
