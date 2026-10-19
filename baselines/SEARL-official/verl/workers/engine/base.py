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
'\nThe abstract base class defining the interface for model training engines.\n'

from typing import Callable

import torch

from verl import DataProto

class BaseEngine:
    '\n    Abstract base class defining the interface for model training engines.\n\n    Engine implementations must subclass BaseEngine and provide concrete behavior for all methods.\n    '

    def __init__(self, config):
        '\n        Initialize the BaseEngine.\n\n        Args:\n            config: Configuration object containing parameters for engine setup.\n        '
        raise NotImplementedError

    def init_model(self):
        '\n        Instantiate or load the model, optimizer, and learning rate scheduler.\n\n        Should prepare all components necessary for training or evaluation.\n        '
        raise NotImplementedError

    def train_mode(self):
        '\n        Context manager entry for switching the engine and model into training mode.\n\n        Usage:\n            with engine.train_mode():\n                # runs in training mode\n        '
        raise NotImplementedError

    def eval_mode(self):
        '\n        Context manager entry for switching the engine and model into evaluation mode.\n\n        Usage:\n            with engine.eval_mode():\n                # runs in evaluation mode\n        '
        raise NotImplementedError

    def infer_batch(
        self,
        data: DataProto,
        post_fn: Callable[[DataProto, torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ) -> dict[str, torch.Tensor]:
        '\n        Perform inference on a mini batch of data.\n\n        Args:\n            data: The input data for inference, typically containing tensors and metadata.\n            post_fn: A post-processing function that takes a micro-batch and predictions as input,\n                     and returns a tuple containing processed predictions and a dictionary of outputs.\n\n        Returns:\n            dict[str, torch.Tensor]: A dictionary containing the predictions for the entire batch.\n        '
        raise NotImplementedError

    def train_batch(
        self,
        data: DataProto,
        loss_fn: Callable[[DataProto, torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ) -> dict[str, torch.Tensor]:
        '\n        Perform a training step on a mini-batch of data.\n\n        Args:\n            data (DataProto): The input data for training, typically containing tensors and metadata.\n            loss_fn (Callable): A function that computes the loss and metrics given a micro-batch and predictions.\n\n        Returns:\n            dict[str, torch.Tensor]: A dictionary containing the aggregated training metrics for the mini-batch.\n        '
        raise NotImplementedError

    def optimizer_zero_grad(self):
        '\n        Zero out gradients of all parameters before starting a new backward pass.\n        '
        raise NotImplementedError

    def optimizer_step(self):
        '\n        Perform an optimization step to update model parameters based on accumulated gradients.\n\n        Returns:\n            grad_norm (float): The norm of the gradients before clipping or update.\n        '
        raise NotImplementedError

    def lr_scheduler_step(self):
        '\n        Advance the learning rate scheduler by one step.\n\n        Returns:\n            current_lr (float or list[float]): Updated learning rate(s).\n        '
        raise NotImplementedError

    def shard_data(self, data):
        '\n        Shard or partition data for distributed training or parallel execution.\n\n        Args:\n            data: Data structure to be sharded across devices/workers.\n\n        Returns:\n            Sharded data in the same format as input.\n        '
        raise NotImplementedError

    def unshard_data(self, data):
        '\n        Reconstruct or gather sharded data back to a unified format.\n\n        Args:\n            data: Sharded data structure to reconstruct.\n\n        Returns:\n            Unsharded, combined data.\n        '
        raise NotImplementedError

    def to(self, device: str, model: bool = True, optimizer: bool = True):
        '\n        Move model parameters, optimizer states, or both to the specified device.\n\n        Args:\n            device: Target device identifier.\n            model: If True, move the model.\n            optimizer: If True, move the optimizer states.\n        '
        raise NotImplementedError

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        '\n        Save model, optimizer, and scheduler states to a checkpoint.\n\n        Args:\n            local_path: Local filesystem path to save checkpoint.\n            hdfs_path: Optional HDFS path to copy checkpoint.\n            global_step: Integer training step number for naming.\n            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.\n        '
        raise NotImplementedError

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        '\n        Load model, optimizer, and scheduler states from a checkpoint.\n\n        Args:\n            local_path: Local filesystem path of the checkpoint.\n            hdfs_path: Optional HDFS path where checkpoint is stored.\n            del_local_after_load: Whether to delete local copy after loading.\n        '
        raise NotImplementedError

class EngineRegistry:
    '\n    A registry for managing and instantiating different types of training engines.\n\n    This class uses a dictionary to store engine classes, mapping a string key to each class.\n    It provides a decorator `register` to add new engines to the registry and a `new` method\n    to create an instance of a registered engine.\n    '

    _engines = {}

    @classmethod
    def register(cls, key):
        '\n        A class method decorator that registers an engine class with a given key.\n\n        This allows for dynamic instantiation of engine classes by their registered key.\n\n        Args:\n            key (str): The identifier to associate with the engine class.\n\n        Returns:\n            A decorator function that takes an engine class and registers it.\n        '

        def decorator(engine_class):
            assert issubclass(engine_class, BaseEngine)
            cls._engines[key] = engine_class
            return engine_class

        return decorator

    @classmethod
    def new(cls, key, *args, **kwargs):
        '\n        Function to create a new training engine instance based on the provided config.\n        Args:\n            key: A configuration object containing the engine key and other settings.\n            *args: Variable length argument list.\n            **kwargs: Arbitrary keyword arguments.\n        Returns:\n            engine: An instance of the training engine corresponding to the config.\n        Raises:\n            NotImplementedError: If the engine key in the config does not match any known engines.\n        '
        if key in cls._engines:
            return cls._engines[key](*args, **kwargs)
        else:
            raise NotImplementedError(f"Unknown engine: {key}")

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
