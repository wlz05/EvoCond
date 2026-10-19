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
from typing import Any, Optional

from verl.base import BaseConfig

@dataclass
class CriticConfig(BaseConfig):
    'Configuration for critic model training.\n\n    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.\n\n    Args:\n        rollout_n (int): Number of rollouts per update (mirrors actor rollout_n).\n        strategy (str): Strategy used for critic model training (fsdp, fsdp2, megatron).\n        optim (Dict[str, Any]): Optimizer configuration including lr, weight_decay, etc.\n        model (Dict[str, Any]): Model configuration including path, tokenizer_path, etc.\n        ppo_mini_batch_size (int): PPO mini-batch size per update.\n        ppo_micro_batch_size (Optional[int]): Global micro batch size (deprecated).\n        ppo_micro_batch_size_per_gpu (Optional[int]): Local per-GPU micro batch size.\n        use_dynamic_bsz (bool): Whether to automatically adjust batch size at runtime.\n        ppo_max_token_len_per_gpu (int): Max tokens per GPU in one PPO batch.\n        forward_max_token_len_per_gpu (int): Max token length per GPU in forward pass.\n        ppo_epochs (int): Number of PPO epochs per batch.\n        shuffle (bool): Shuffle training data across PPO epochs.\n        cliprange_value (float): PPO value function clipping range.\n        loss_agg_mode (str): Loss aggregation mode.\n        checkpoint (Dict[str, Any]): Checkpoint configuration.\n        profiler (Dict[str, Any]): Profiler configuration.\n    '

    # For legacy reason configs related to batch_size are mutated in each role
    # In the future they will be added to frozen fields instead
    _frozen_fields = [
        "rollout_n",
        "strategy",
        "use_dynamic_bsz",
        "ppo_max_token_len_per_gpu",
        "forward_max_token_len_per_gpu",
        "ppo_epochs",
        "shuffle",
        "cliprange_value",
        "loss_agg_mode",
    ]

    rollout_n: int = 1
    strategy: str = "fsdp"
    optim: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    ppo_mini_batch_size: int = 1
    ppo_micro_batch_size: Optional[int] = None
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 32768
    forward_max_token_len_per_gpu: int = 32768
    ppo_epochs: int = 1
    shuffle: bool = True
    cliprange_value: float = 0.5
    loss_agg_mode: str = "token-mean"
    checkpoint: dict[str, Any] = field(default_factory=dict)
    profiler: dict[str, Any] = field(default_factory=dict)

@dataclass
class MegatronCriticConfig(CriticConfig):
    'Configuration for Megatron-based critic model training.\n\n    The inheritance from CriticConfig provides all base critic configuration plus Megatron-specific settings.\n\n    Args:\n        nccl_timeout (int): NCCL timeout in seconds for distributed operations.\n        megatron (Dict[str, Any]): Megatron-specific parallelism settings.\n        load_weight (bool): Whether to load initial weights.\n        data_loader_seed (Optional[int]): Seed for data loader.\n    '

    _frozen_fields = CriticConfig._frozen_fields + [
        "nccl_timeout",
        "load_weight",
        "data_loader_seed",
    ]

    strategy: str = "megatron"
    nccl_timeout: int = 600
    megatron: dict[str, Any] = field(default_factory=dict)
    load_weight: bool = True
    data_loader_seed: Optional[int] = None

@dataclass
class FSDPCriticConfig(CriticConfig):
    'Configuration for FSDP-based critic model training.\n\n    The inheritance from CriticConfig provides all base critic configuration plus FSDP-specific settings.\n\n    Args:\n        forward_micro_batch_size (int): Forward-only batch size during inference (global).\n        forward_micro_batch_size_per_gpu (int): Forward-only batch size during inference (per GPU).\n        ulysses_sequence_parallel_size (int): Sequence parallelism size for Ulysses-style model parallelism.\n        grad_clip (float): Gradient clipping for critic updates.\n    '

    _frozen_fields = CriticConfig._frozen_fields + [
        "ulysses_sequence_parallel_size",
        "grad_clip",
    ]

    strategy: str = "fsdp"
    forward_micro_batch_size: int = 1
    forward_micro_batch_size_per_gpu: int = 1
    ulysses_sequence_parallel_size: int = 1
    grad_clip: float = 1.0

@dataclass
class BaseModelConfig(BaseConfig):
    "Base configuration for a model.\n    Contains core settings for loading and initializing a pretrained model checkpoint.\n\n    Args:\n        path (str): Path to pretrained model weights.\n        tokenizer_path (Optional[str]): Tokenizer path (defaults to actor's model path if not set).\n        override_config (dict): Hugging Face config override.\n        external_lib (Optional[str]): External model implementation (optional).\n        trust_remote_code (bool): Whether to trust remote code from Hugging Face models.\n    "

    path: str = "~/models/deepseek-llm-7b-chat"
    tokenizer_path: Optional[str] = None
    override_config: dict[str, Any] = field(default_factory=dict)
    external_lib: Optional[str] = None
    trust_remote_code: bool = False

@dataclass
class CheckpointConfig(BaseConfig):
    "Configuration for model checkpointing.\n\n    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.\n\n    Args:\n        save_contents (list[str]): What to include in saved checkpoints.\n            Options: 'model', 'optimizer', 'extra', 'hf_model'.\n        load_contents (list[str]): Contents to load from checkpoint. Defaults to same as save_contents.\n        async_save (bool): Whether to save checkpoints asynchronously. Only implemented for Megatron as of now.\n    "

    save_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    load_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    async_save: bool = False

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
