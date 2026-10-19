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

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base import BaseConfig

__all__ = ["FSDPEngineConfig", "McoreEngineConfig"]

@dataclass
class McoreEngineConfig(BaseConfig):
    'Configuration for Megatron parallelism.\n\n    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.\n\n    Args:\n        param_offload (bool): Whether to offload parameters to CPU.\n        grad_offload (bool): Whether to offload gradients to CPU.\n        optimizer_offload (bool): Whether to offload optimizer states to CPU.\n        tensor_model_parallel_size (int): Tensor model parallel size.\n        expert_model_parallel_size (int): Expert model parallel size for MoE models.\n        expert_tensor_parallel_size (Optional[int]): Expert tensor parallel size for MoE models.\n        pipeline_model_parallel_size (int): Pipeline model parallel size.\n        virtual_pipeline_model_parallel_size (Optional[int]): Virtual pipeline model parallel size\n            for interleaved scheduling.\n        context_parallel_size (int): Context parallel size for long sequences.\n        sequence_parallel (bool): Whether to enable sequence parallelism.\n        use_distributed_optimizer (bool): Whether to use distributed optimizer.\n        use_dist_checkpointing (bool): Whether to use distributed checkpointing.\n        dist_checkpointing_path (Optional[str]): Path for distributed checkpointing.\n        seed (int): Random seed for reproducibility.\n        override_ddp_config (dict[str, Any]): Override configuration for DDP.\n        override_transformer_config (dict[str, Any]): Override configuration for transformer.\n        use_mbridge (bool): Whether to use MBridge for communication.\n    '

    # sequence_parallel is not listed as a frozen field for auto-correction purpose
    _mutable_fields = BaseConfig._mutable_fields | {"sequence_parallel"}

    param_offload: bool = False
    grad_offload: bool = False
    optimizer_offload: bool = False
    tensor_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: Optional[int] = None
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: int = 1
    sequence_parallel: bool = True
    use_distributed_optimizer: bool = True
    use_dist_checkpointing: bool = False
    dist_checkpointing_path: Optional[str] = None
    seed: int = 42
    override_ddp_config: dict[str, Any] = field(default_factory=dict)
    override_transformer_config: dict[str, Any] = field(default_factory=dict)
    use_mbridge: bool = False

    def __post_init__(self) -> None:
        """config validation logics go here"""
        if self.tensor_model_parallel_size == 1:
            warnings.warn("set sequence parallel to false as TP size is 1", stacklevel=2)
            self.sequence_parallel = False

@dataclass
class FSDPEngineConfig(BaseConfig):
    'Configuration for FSDP (Fully Sharded Data Parallel).\n\n    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.\n\n    Args:\n        wrap_policy (Dict[str, Any]): Configuration for FSDP wrap policy.\n        param_offload (bool): Whether to offload parameters to CPU, default False\n        optimizer_offload (bool): Whether to offload optimizer states to CPU, default False\n        offload_policy (bool): Whether to offload policy model parameters, default False\n        reshard_after_forward (bool): Whether to reshard parameters after forward pass, default True\n        fsdp_size (int): FSDP group size. -1 means use all available GPUs.\n        forward_prefetch (bool): Whether to prefetch parameters for next forward pass, default False\n        model_dtype (str): Model data type used to initialize the transformers model. default "fp32"\n        use_orig_params (bool): Whether to use original parameters when initialize FSDP1, default False\n    '

    wrap_policy: dict[str, Any] = field(default_factory=dict)
    param_offload: bool = False
    optimizer_offload: bool = False
    offload_policy: bool = False
    reshard_after_forward: bool = True
    fsdp_size: int = -1
    forward_prefetch: bool = False
    model_dtype: str = "fp32"
    use_orig_params: bool = False

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
