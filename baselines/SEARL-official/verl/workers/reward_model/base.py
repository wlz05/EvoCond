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
'\nThe base class for reward model\n'

from abc import ABC, abstractmethod

from verl import DataProto

class BasePPORewardModel(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def compute_reward(self, data: DataProto) -> DataProto:
        'Computing reward given input_ids. The transformers should output a tensor with shape\n           [batch_size, sequence_length], and the value at [EOS] mask should be gathered.\n\n        Args:\n            data: must contain keys "input_ids", "attention_mask" and "position_ids".\n                - input_ids: [batch_size, sequence_length]\n                - attention_mask: [batch_size, sequence_length]\n                - position_ids: [batch_size, sequence_length]\n\n        Returns: a data pass protocol containing "reward". Only the [EOS] position contains the reward.\n            Other position should have zero reward. Note that this may change in the future if we use\n            dense reward. So, we leave the interface for general case.\n            - reward: [batch_size, sequence_length].\n\n        '
        pass

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
