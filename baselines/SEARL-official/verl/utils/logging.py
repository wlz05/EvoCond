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
    'Print to stderr and flush immediately; additional stream/flush keyword arguments are ignored.'

    print(message, *args, file=sys.stderr, flush=True)

def format_error(error: Exception, context: str = "") -> None:
    'Format an exception with its error context.'
    if context:
        error_print(f"{context}: {error}")
    else:
        error_print(f"Parse error: {error}")

def unified_print(message: Any, *args, **kwargs) -> None:
    'Print through the unified stderr stream.'
    print(message, *args, file=sys.stderr, flush=True)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
