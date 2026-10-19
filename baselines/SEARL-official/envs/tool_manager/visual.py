# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import re
import copy
import json
import json5, sys
import asyncio
import traceback
import base64
from PIL import Image
from ast import literal_eval
from omegaconf import OmegaConf
import os

from envs.tool_manager.multimodal import ToolManager

from envs.utils.manager import MCPManager
from typing import Union, List, Tuple, Optional, Any, Dict
from envs.utils.util import ToolServiceError, DocParserError
# from qwen_agent.tools import TOOL_REGISTRY, MCPManager, BaseTool  # TODO
from qwen_agent.tools import TOOL_REGISTRY, BaseTool
from qwen_agent.llm.schema import ASSISTANT, SYSTEM, USER, FUNCTION, ContentItem
import json
import io
import base64

from copy import deepcopy
from envs.storage.manager.storage import create_config_storage_manager
from envs.utils.util import ToolServiceError, DocParserError
import torch
import torch.distributed as dist

def print_rank_0(message):
    '\n    \u5982\u679c rank \u4e3a 0\uff0c\u5219\u6253\u5370\u6d88\u606f\u3002\n    '

    if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
        print(message, file=sys.stderr, flush=True)

def parse_mcp_tools_config(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        if "{project_name}" in content:
            content = content.replace("{project_name}", os.environ.get("PROJECT_NAME") or "default_project")

        data = literal_eval(content)
        return data
    except Exception as e:
        print(f"\u89e3\u6790\u9519\u8bef: {e}",file=sys.stderr, flush=True)
        return None

class Qwen25VLManager(ToolManager):
    'Qwen2.5-VL\u5de5\u5177\u7ba1\u7406\u5668\n    \u4f7f\u7528 asyncio \u5b9e\u73b0\u5de5\u5177\u7684\u5e76\u53d1\u6267\u884c\uff0c\u63d0\u9ad8\u6548\u7387\n    \u5185\u7f6e\u56fe\u50cf\u7f16\u89e3\u7801\u529f\u80fd\u652f\u6301\u89c6\u89c9\u4efb\u52a1\n    \u652f\u6301\u591a\u79cd\u54cd\u5e94\u683c\u5f0f\u89e3\u6790\uff0c\u5305\u62ec\u5de5\u5177\u8c03\u7528\u548c\u76f4\u63a5\u56de\u7b54\n    \u5b8c\u5584\u7684\u5f02\u5e38\u6355\u83b7\u548c\u9519\u8bef\u4fe1\u606f\u8fd4\u56de\u673a\u5236\n    \u652f\u6301\u591a\u79cd\u5de5\u5177\u6ce8\u518c\u65b9\u5f0f\u548c\u914d\u7f6e\u52a0\u8f7d\n    \n    '
    def __init__(self, verl_config):
        'Qwen2.5-VL\u5de5\u5177\u7ba1\u7406\u5668\n        \n        Args:\n            verl_config: \u914d\u7f6e\u6587\u4ef6\n        '

        if isinstance(verl_config, dict):
            verl_config = OmegaConf.to_container(verl_config)
        super().__init__(verl_config)
        self.verl_config = verl_config

        self.generate_cfg = {
            'fncall_prompt_type': 'nous',
            'function_choice': 'auto',
            'parallel_function_calls': False,
            'lang': 'en',
            'max_input_tokens': 10000
        }

        self.tokenizer = None
        self.processor = None

    def modify(self, name):
        '\u622a\u53d6\u5b57\u7b26\u4e32\u7684\u524d\u4e00\u534a\uff0c\u7528\u4e8e\u751f\u6210\u51fd\u6570\u540d\u79f0\n        \n        Args:\n            name: \u51fd\u6570\u540d\u79f0\n            \n        Returns:\n            \u622a\u53d6\u540e\u7684\u5b57\u7b26\u4e32\n        '
        length = len(name)//2
        return name[0:length]

    def _load_custom_chat_template(self, tokenizer):
        '\u52a0\u8f7d\u81ea\u5b9a\u4e49\u7684\u804a\u5929\u6a21\u677f\n        \n        Args:\n            tokenizer: \u6a21\u578b\u5206\u8bcd\u5668\n            \n        Returns:\n            \u52a0\u8f7d\u540e\u7684\u5206\u8bcd\u5668\n        '
        self.chat_template_path = self.verl_config.get('load_custom_chat_template', None)
        if self.chat_template_path:
            print_rank_0(f"load chat template from {self.chat_template_path}")
            with open(self.chat_template_path, "r") as f:
                chat_template_from_file = f.read()
            tokenizer.chat_template = chat_template_from_file
        return tokenizer

    def get_tool(self, name_or_short_name: str):
        '\u901a\u8fc7\u540d\u79f0\u6216\u7b80\u5199\u83b7\u53d6\u5de5\u5177\n        \n        Args:\n            name_or_short_name: \u5de5\u5177\u540d\u79f0\u6216\u7b80\u5199\n            \n        Returns:\n            \u627e\u5230\u7684\u5de5\u5177\uff0c\u5982\u679c\u6ca1\u627e\u5230\u5219\u8fd4\u56deNone\n        '
        name_or_short_name = str(name_or_short_name)
        return self.tool_map.get(name_or_short_name, None)

    @property
    def all_tools(self):
        '\u83b7\u53d6\u6240\u6709\u5de5\u5177\n        \n        Returns:\n            \u6240\u6709\u5de5\u5177\u7684\u5217\u8868\n        '
        return self.tool_map

    def _build_tools(self):
        '\n        \u6784\u5efa\u5de5\u5177\u5217\u8868\n        \u8bfb\u53d6\u914d\u7f6e\u6587\u4ef6\uff0c\u521d\u59cb\u5316\u5de5\u5177\uff0c\u5e76\u5c06\u5176\u6dfb\u52a0\u5230\u5de5\u5177\u6620\u5c04\u4e2d\n        \u751f\u6210\u5de5\u5177\u51fd\u6570\u5217\u8868\u4f9b\u6a21\u578b\u8c03\u7528\n        '
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

    async def execute_all_tools(self, actions, tool_list, image_data: List[List[Image.Image]]):
        '\u5f02\u6b65\u5e76\u884c\u6267\u884c\u6240\u6709\u5de5\u5177\u5217\u8868\n        \n        Args:\n            tool_list: \u5de5\u5177\u5217\u8868\u7684\u5217\u8868\n            \n        Returns:\n            \u6240\u6709\u5de5\u5177\u6267\u884c\u7ed3\u679c\u7684\u5217\u8868\n        '

        tasks = []
        for temp_action, temp_tool_list, temp_image_data in zip(actions, tool_list, image_data):
            tasks.append(self._execute_tool_batch(temp_action, temp_tool_list, temp_image_data))

        results = await asyncio.gather(*tasks)

        return results

    async def _execute_tool_batch(self, action, tools, image_data: List[Image.Image]):
        '\u5f02\u6b65\u5e76\u884c\u6267\u884c\u4e00\u6279\u5de5\u5177\n        \n        Args:\n            tools: \u5de5\u5177\u5217\u8868\n            \n        Returns:\n            \u5de5\u5177\u6267\u884c\u7ed3\u679c\u7684\u5217\u8868\n        '
        async def execute_single_tool(tool, image_data:Image.Image):
            tool_instance = self.get_tool(tool["name"])
            args = tool["args"]
            if tool_instance is not None:
                try:
                    args = json.loads(args)
                    original_args = deepcopy(args)
                    args["img_base64"] = self.pil_to_base64(image_data)
                except Exception as e:
                    pass
                if type(args) is dict:
                    try:

                        image_result = await asyncio.to_thread(
                            self._call_tool,
                            tool["name"], json.dumps(args, ensure_ascii=False, indent=4)
                        )
                        assert isinstance(image_result,  Image.Image), f"tool_result is not a PIL.Image.Image instance, got {type(image_result)}"
                        text_result = [
                            {"type": "text", "text": f"Execute the tool {tool['name']} successed. The args are: {original_args}. The image result is: "},
                            {"type": "image"}
                        ]
                        return (text_result, image_result)

                    except Exception as e:
                        result = (f"Execute the tool {tool['name']} failed. The original args are: {original_args}. Error message: {str(e)}", None)
            else:
                result = (f"Failed to find the tool {tool['name']} in the tool map.", None)

            if isinstance(result, str):
                return ([{"type": "text", "text": result}], None)

            else:
                return result

        if action == 'answer':
            # 'tools' is a str (the answer)

            results = [({'role': 'assistant', 'content': tools}, None)]
        elif action == 'error':
            # 'error' only occurs when there is no 'actions' tag or there is no 'action' tag after extraction
            # ('Cannot extract the actions tag' or 'There is no action after extraction')
            results = [({'role': 'assistant', 'content': """# Extract the tools failed due to: {}""".format(tools)}, None)]
        elif action == 'actions':
            # 'tools' is the list of the 'Tool' instances
            tasks = [execute_single_tool(temp_tool, image_data[-1]) for temp_tool in tools]
            tool_results = await asyncio.gather(*tasks)

            results = [({'role': 'tool', 'content': temp_tool_result[0]}, temp_tool_result[1]) for temp_tool_result in tool_results]

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

    def execute_actions(self, responses: List[str], image_data: List[List[Image.Image]]):

        actions, tools = [], []
        for response in responses:
            temp_action, temp_tool_list = self.parse_response(response_content=response)
            actions.append(temp_action)
            tools.append(temp_tool_list)

        try:
            tool_results = asyncio.run(self.execute_all_tools(actions, tools, image_data))
        except RuntimeError:

            loop = asyncio.get_event_loop()
            tool_results = loop.run_until_complete(self.execute_all_tools(actions, tools, image_data))

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

        # If no function call, return original response
        if i < 0:
            return response

        # Split tool-call to separate assistant msg
        tool_call_list = response.split('<tool_call>')
        pre_thought = tool_call_list[0].strip()

        # Only process the first valid tool call
        for txt in tool_call_list[1:]:
            if not txt.strip():
                continue

            if '</tool_call>' not in txt:
                # Incomplete tool call
                parsed_tools.append({
                    "name": "<empty>",
                    "args": "Extract the tool name failed"
                })
                break  # Stop after the first tool call (even if incomplete)
            else:
                one_tool_call_txt = txt.split('</tool_call>')
                try:
                    if not one_tool_call_txt[0].strip():
                        raise ValueError("Empty tool call content")

                    # Try to parse JSON
                    fn = json5.loads(one_tool_call_txt[0].strip())

                    # Only append the first valid tool

                    if type(fn) is not dict or 'name' not in fn or 'arguments' not in fn:
                        raise KeyError("Missing required fields")
                    parsed_tools.append({
                        "name": fn['name'],
                        "args": json.dumps(fn['arguments'], ensure_ascii=False, indent=4),
                    })
                    break  # Stop after the first valid tool call

                except (IndexError, KeyError, ValueError) as e:
                    parsed_tools.append({
                        "name": "<empty>",
                        "args": "Extract the tool name failed"
                    })
                    break  # Stop after the first error

        return parsed_tools

    def get_prompt(self, input_data, tokenizer, mode='initial', add_generation_prompt=True):
        if self.tokenizer is None:
            self.tokenizer = self._load_custom_chat_template(tokenizer)
        tokenizer = self.tokenizer
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
            elif type(input_data) == dict:
                chat = [input_data]
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

    def pil_to_base64(self, image):
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return img_base64

    def base64_to_pil(self, base64_str):
        img_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(img_data))
        assert isinstance(image, Image.Image), "image is not an Image.Image"
        return image

    def is_base64(self, s):
        head = base64.b64decode(s[:100], validate=True)
        return head.startswith((b'\xff\xd8\xff', b'\x89PNG\r\n\x1a\n', b'GIF87a', b'GIF89a', b'BM'))

    def _extract_image_data(self, data):
            '\n            \u4ece\u6570\u636e\u4e2d\u63d0\u53d6\u7b2c\u4e00\u4e2a\u56fe\u50cf\u6570\u636e\n            \n            Args:\n                data: \u5305\u542b\u56fe\u50cf\u7684\u6570\u636e\u7ed3\u6784\n                \n            Returns:\n                PIL.Image: \u63d0\u53d6\u51fa\u7684\u7b2c\u4e00\u4e2a\u56fe\u50cf\uff0c\u5982\u679c\u6ca1\u6709\u627e\u5230\u5219\u8fd4\u56deNone\n            '
            if isinstance(data, Image.Image):
                return data
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, Image.Image):
                        return item
                    elif isinstance(item, dict):

                        if 'image' in item and isinstance(item['image'], Image.Image):
                            return item['image']

                        nested_image = self._extract_image_data(item)
                        if nested_image is not None:
                            return nested_image
                    elif isinstance(item, list):

                        nested_image = self._extract_image_data(item)
                        if nested_image is not None:
                            return nested_image
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, Image.Image):
                        return value
                    elif key in ['image', 'image_url', 'img'] and isinstance(value, Image.Image):
                        return value
                    elif isinstance(value, (list, dict)):

                        nested_image = self._extract_image_data(value)
                        if nested_image is not None:
                            return nested_image

            return None

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
