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
from torch.distributed.device_mesh import init_device_mesh

from verl.utils.device import get_device_name

def create_device_mesh(world_size, fsdp_size):
    '\n    Create a device mesh for distributed training based on the world size and FSDP size.\n\n    Args:\n        world_size (int): Total number of processes in the distributed training setup.\n        fsdp_size (int): Size of the Fully Sharded Data Parallel (FSDP) group.\n\n    Returns:\n        torch.distributed.device_mesh.DeviceMesh: The initialized device mesh.\n    '
    device_name = get_device_name()
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh

def get_sharding_strategy(device_mesh):
    '\n    Determine the appropriate sharding strategy based on the number of dimensions of the device mesh.\n\n    Args:\n        device_mesh (torch.distributed.device_mesh.DeviceMesh): The device mesh used for distributed training.\n\n    Returns:\n        torch.distributed.fsdp.ShardingStrategy: The sharding strategy to be used with FSDP.\n\n    Raises:\n        NotImplementedError: If the number of dimensions of the device mesh is neither 1 nor 2.\n    '
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
