# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

import logging

import torch

logger = logging.getLogger(__name__)

def is_torch_npu_available() -> bool:
    """Check the availability of NPU"""
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False

is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()

def get_visible_devices_keyword() -> str:
    "Function that gets visible devices keyword name.\n    Returns:\n        'CUDA_VISIBLE_DEVICES' or `ASCEND_RT_VISIBLE_DEVICES`\n    "
    return "CUDA_VISIBLE_DEVICES" if is_cuda_available else "ASCEND_RT_VISIBLE_DEVICES"

def get_device_name() -> str:
    'Function that gets the torch.device based on the current machine.\n    This currently only supports CPU, CUDA, NPU.\n    Returns:\n        device\n    '
    if is_cuda_available:
        device = "cuda"
    elif is_npu_available:
        device = "npu"
    else:
        device = "cpu"
    return device

def get_torch_device() -> any:
    'Return the corresponding torch attribute based on the device type string.\n    Returns:\n        module: The corresponding torch device namespace, or torch.cuda if not found.\n    '
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda

def get_device_id() -> int:
    'Return current device id based on the device type.\n    Returns:\n        device index\n    '
    return get_torch_device().current_device()

def get_nccl_backend() -> str:
    'Return nccl backend type based on the device type.\n    Returns:\n        nccl backend type string.\n    '
    if is_cuda_available:
        return "nccl"
    elif is_npu_available:
        return "hccl"
    else:
        raise RuntimeError(f"No available nccl backend found on device type {get_device_name()}.")

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
