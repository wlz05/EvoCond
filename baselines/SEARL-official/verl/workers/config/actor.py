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

from omegaconf import MISSING

from verl.base import BaseConfig
from verl.trainer.config import CheckpointConfig

from .engine import FSDPEngineConfig, McoreEngineConfig
from .optimizer import OptimizerConfig

__all__ = ["PolicyLossConfig", "ActorConfig", "FSDPActorConfig", "McoreActorConfig"]

@dataclass
class PolicyLossConfig(BaseConfig):
    "Configuration for policy loss computation.\n\n    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.\n\n    Args:\n        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg'.\n        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.\n        clip_cov_lb (float): Lower bound for clip-cov loss.\n        clip_cov_ub (float): Upper bound for clip-cov loss.\n        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.\n        ppo_kl_coef (float): KL divergence penalty coefficient.\n    "

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1

@dataclass
class ActorConfig(BaseConfig):
    "Configuration for actor model training.\n\n    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.\n\n    Args:\n        strategy (str): Training strategy. Must be specified.\n        ppo_mini_batch_size (int): Mini-batch size for PPO training.\n        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.\n            If None, uses ppo_micro_batch_size_per_gpu.\n        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.\n        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.\n        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.\n        clip_ratio (float): PPO clipping ratio for policy loss.\n        clip_ratio_low (float): Lower bound for PPO clipping ratio.\n        clip_ratio_high (float): Upper bound for PPO clipping ratio.\n        policy_loss (PolicyLossConfig): Configuration for policy loss computation.\n        clip_ratio_c (float): Clipping ratio for critic loss.\n        loss_agg_mode (str): Loss aggregation mode. Options: 'token-mean', 'sample-mean'.\n        entropy_coeff (float): Entropy coefficient for regularization.\n        use_kl_loss (bool): Whether to use KL divergence loss.\n        use_torch_compile (bool): Whether to use torch.compile for optimization.\n        kl_loss_coef (float): KL divergence loss coefficient.\n        kl_loss_type (str): Type of KL loss to use.\n        ppo_epochs (int): Number of PPO epochs per training step.\n        shuffle (bool): Whether to shuffle data during training.\n        checkpoint (CheckpointConfig): Configuration for checkpointing.\n        optim (OptimizerConfig): Configuration for optimizer.\n        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).\n    "

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    entropy_coeff: float = 0
    use_kl_loss: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )

@dataclass
class McoreActorConfig(ActorConfig):
    "Configuration for Megatron actor models.\n\n    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.\n\n    Args:\n        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.\n        data_loader_seed (Optional[int]): Seed for data loader. If None, uses global seed.\n        load_weight (bool): Whether to load model weights from checkpoint.\n        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.\n        profile (dict[str, Any]): Configuration for profiling settings.\n    "

    strategy: str = "megatron"
    data_loader_seed: Optional[int] = None
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)

@dataclass
class FSDPActorConfig(ActorConfig):
    "Configuration for FSDP actor models.\n\n    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.\n\n    Args:\n        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.\n        grad_clip (float): Gradient clipping threshold.\n        ulysses_sequence_parallel_size (int): Ulysses sequence parallel size for long sequences.\n        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits\n            with chunking for memory efficiency.\n        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.\n        fsdp_config (dict[str, Any]): Configuration for FSDP settings.\n        use_remove_padding (bool): Whether to remove padding tokens in inputs during training\n    "

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
