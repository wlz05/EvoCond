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

import os

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",

        "PYTHONWARNINGS": "ignore",
        "LOGURU_LEVEL": "ERROR",

        "MCP_LOG_LEVEL": "ERROR",
    },
}

def get_ppo_ray_runtime_env():
    '\n    A filter function to return the PPO Ray runtime environment.\n    To avoid repeat of some environment variables that are already set.\n    '
    runtime_env = {"env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy()}
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    # Explicitly forward this experiment's endpoints/paths even when Ray's
    # daemon was started before this arm. Never forward arbitrary secrets.
    for key in ("SEARL_ISOLATION_CONFIG", "SEARL_ISOLATION_RECEIPT", "SEARL_DYNAMIC_REGISTRY",
                "SEARL_SEARCH_URL", "RL_FACTORY_EMBEDDING_MODEL", "RL_FACTORY_ROOT",
                "PYTHONPATH", "WANDB_DIR", "WANDB_MODE"):
        if key in os.environ:
            runtime_env["env_vars"][key] = os.environ[key]
    return runtime_env

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
