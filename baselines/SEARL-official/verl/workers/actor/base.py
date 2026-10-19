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
'\nThe base class for Actor\n'

from abc import ABC, abstractmethod

import torch

from verl import DataProto

__all__ = ["BasePPOActor"]

class BasePPOActor(ABC):
    def __init__(self, config):
        'The base class for PPO actor\n\n        Args:\n            config (DictConfig): a config passed to the PPOActor. We expect the type to be\n                DictConfig (https://omegaconf.readthedocs.io/), but it can be any namedtuple in general.\n        '
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        'Compute logits given a batch of data.\n\n        Args:\n            data (DataProto): a batch of data represented by DataProto. It must contain key ```input_ids```,\n                ```attention_mask``` and ```position_ids```.\n\n        Returns:\n            DataProto: a DataProto containing the key ```log_probs```\n\n\n        '
        pass

    @abstractmethod
    def update_policy(self, data: DataProto) -> dict:
        'Update the policy with an iterator of DataProto\n\n        Args:\n            data (DataProto): an iterator over the DataProto that returns by\n                ```make_minibatch_iterator```\n\n        Returns:\n            Dict: a dictionary contains anything. Typically, it contains the statistics during updating the model\n            such as ```loss```, ```grad_norm```, etc,.\n\n        '
        pass

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
