# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

# Copyright 2023-2025 SGLang Team
# Copyright Amazon.com, Inc. or its affiliates.
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
from typing import Any, Optional
from uuid import uuid4

from verl.utils.reward_score import geometry as geo3k
from verl.utils.trace import rollout_trace_op

from .base import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class Geo3kTool(BaseTool):
    'A demo tool for calculating the reward of geo3k.\n    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.\n    - `create`: create a tool instance for a trajectory.\n    - `execute`: execute the tool.\n    - `calc_reward`: calculate the reward respect to tool state.\n    - `release`: release the tool instance.\n    '

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        '\n        _tool_schema = OpenAIFunctionToolSchema.model_validate({\n            "type": "function",\n            "function": {\n                "name": "calc_geo3k_reward",\n                "description": "A tool for calculating the reward of geo3k",\n                "parameters": {\n                    "type": "object",\n                    "properties": {\n                        "answer": {\n                            "type": "string",\n                            "description": "The answer to the question, enclosed in \\boxed{}",\n                        },\n                    },\n                    "required": ["answer"],\n                },\n            }\n        })\n        '
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id, None

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        answer = parameters.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)
        self._instance_dict[instance_id]["response"] = answer
        reward = await self.calc_reward(instance_id)
        # penalty for non improved answer submission
        tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
        # update the reward
        self._instance_dict[instance_id]["reward"] = reward
        return f"Current parsed {answer=} {reward=}", tool_reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return geo3k.compute_score(
            self._instance_dict[instance_id]["response"],
            self._instance_dict[instance_id]["ground_truth"],
            use_boxed=False,
            format_score=0.0,
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
