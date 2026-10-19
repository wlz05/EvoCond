# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from typing import Any

import torch

def set_basic_config(level):
    '\n    This function sets the global logging format and level. It will be called when import verl\n    '
    logging.basicConfig(format="%(levelname)s:%(asctime)s:%(message)s", level=level)

def log_to_file(string):
    print(string)
    if os.path.isdir("logs"):
        with open(f"logs/log_{torch.distributed.get_rank()}", "a+") as f:
            f.write(string + "\n")

def error_print(message: Any, *args, **kwargs) -> None:
    '\n    \u7edf\u4e00\u7684\u9519\u8bef\u8f93\u51fa\u51fd\u6570\uff0c\u6240\u6709\u6d88\u606f\u90fd\u8f93\u51fa\u5230stderr\u5e76\u7acb\u5373\u5237\u65b0\n    \n    Args:\n        message: \u8981\u6253\u5370\u7684\u6d88\u606f\n        *args: \u989d\u5916\u7684\u4f4d\u7f6e\u53c2\u6570\n        **kwargs: \u989d\u5916\u7684\u5173\u952e\u5b57\u53c2\u6570\uff08\u4f1a\u88ab\u5ffd\u7565\uff0c\u56e0\u4e3a\u6211\u4eec\u5f3a\u5236\u4f7f\u7528stderr\u548cflush=True\uff09\n    '

    print(message, *args, file=sys.stderr, flush=True)

def format_error(error: Exception, context: str = "") -> None:
    '\n    \u683c\u5f0f\u5316\u9519\u8bef\u8f93\u51fa\n    \n    Args:\n        error: \u5f02\u5e38\u5bf9\u8c61\n        context: \u9519\u8bef\u4e0a\u4e0b\u6587\u63cf\u8ff0\n    '
    if context:
        error_print(f"{context}: {error}")
    else:
        error_print(f"\u89e3\u6790\u9519\u8bef: {error}")

def unified_print(message: Any, *args, **kwargs) -> None:
    '\n    \u7edf\u4e00\u7684\u6253\u5370\u51fd\u6570\uff0c\u5f3a\u5236\u8f93\u51fa\u5230stderr\n    '
    print(message, *args, file=sys.stderr, flush=True)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
