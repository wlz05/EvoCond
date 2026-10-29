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

from dataclasses import dataclass, field
from typing import Optional

from verl.base import BaseConfig

@dataclass
class KLControlConfig(BaseConfig):
    'Configuration for KL control.\n\n    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.\n\n    Args:\n        type (str): Type of KL control. Can be "fixed" or "adaptive".\n        kl_coef (float): Initial coefficient for KL penalty.\n        horizon (int): Horizon value for adaptive controller.\n        target_kl (float): Target KL divergence for adaptive controller.\n    '

    _frozen_fields = ["type", "kl_coef", "horizon", "target_kl"]
    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1

@dataclass
class PFPPOConfig(BaseConfig):
    'Configuration for preference feedback PPO.\n\n    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.\n\n    Args:\n        reweight_method (str): Method for reweighting samples. Can be "pow", "max_min", or "max_random".\n        weight_pow (float): Power used for weight scaling in "pow" method.\n    '

    _frozen_fields = ["reweight_method", "weight_pow"]
    reweight_method: str = "pow"
    weight_pow: float = 2.0

@dataclass
class FilterGroupsConfig(BaseConfig):
    'Configuration for filter groups (used in DAPO and Entropy).\n\n    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.\n\n    Args:\n        enable (bool): Whether to enable filter groups.\n        metric (Optional[str]): Metric to use for filtering: "acc", "score", "seq_reward", "seq_final_reward", etc.\n        max_num_gen_batches (int): Non-positive values mean no upper limit.\n    '

    _frozen_fields = ["enable", "metric", "max_num_gen_batches"]

    enable: bool = False
    metric: Optional[str] = None
    max_num_gen_batches: int = 0

@dataclass
class AlgoConfig(BaseConfig):
    'Configuration for the algorithm.\n\n    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.\n\n    Args:\n        gamma (float): Discount factor for future rewards.\n        lam (float): Trade-off between bias and variance in the GAE estimator.\n        adv_estimator (str): Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.\n        norm_adv_by_std_in_grpo (bool): Whether to normalize advantages by std (specific to GRPO).\n        use_kl_in_reward (bool): Whether to enable in-reward KL penalty.\n        kl_penalty (str): How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full".\n        kl_ctrl (KLControlConfig): KL control configuration.\n        use_pf_ppo (bool): Whether to enable preference feedback PPO.\n        pf_ppo (Optional[PFPPOConfig]): Preference feedback PPO settings.\n        filter_groups (Optional[FilterGroupsConfig]): Filter groups configuration, used in DAPO and Entropy\n    '

    _frozen_fields = [
        "gamma",
        "lam",
        "adv_estimator",
        "norm_adv_by_std_in_grpo",
        "use_kl_in_reward",
        "kl_penalty",
        "use_pf_ppo",
    ]

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: Optional[PFPPOConfig] = None
    filter_groups: Optional[FilterGroupsConfig] = None

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
