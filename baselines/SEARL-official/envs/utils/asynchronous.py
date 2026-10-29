# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import json
import logging
from typing import Union
from envs.utils.manager import MCPManager as SSEMCPManager, BaseTool
import threading
import time

logger = logging.getLogger(__name__)

class LoadBalancedTool(BaseTool):
    'Wrap a tool with load balancing.'
    def __init__(self, register_name: str, register_client_id: str, tool_name: str, tool_desc: str, tool_parameters: dict, config: dict, manager_classes: list):
        self.name = register_name
        self.description = tool_desc
        self.parameters = tool_parameters
        self.client_id = register_client_id
        self.tool_name = tool_name
        self.instances = []
        self._lock = threading.Lock()
        self._instance_states = []  # [(active_requests, last_used), ...]
        self._config = config

        for manager_class in manager_classes:
            try:

                manager = manager_class()
                if len(manager.clients) == 0:
                    tools = manager.initConfig(self._config)
                else:
                    tools = manager.tools

                for tool in tools:
                    if tool.name == self.name:
                        self.instances.append(tool)
                        self._instance_states.append([0, time.time()])
                        break
            except Exception as e:
                logger.error(f"Failed to create instance: {e}")
                continue

        if not self.instances:
            raise RuntimeError("Failed to create any working instances")

    def call(self, params: Union[str, dict], **kwargs) -> str:
        'Invoke a load-balanced tool.'
        if not self.instances:
            raise RuntimeError("No available instances")

        with self._lock:
            min_load = float('inf')
            selected_idx = 0

            for idx, (active_requests, last_used) in enumerate(self._instance_states):
                time_factor = 1.0 / (time.time() - last_used + 1)
                load = active_requests + time_factor
                if load < min_load:
                    min_load = load
                    selected_idx = idx

            self._instance_states[selected_idx][0] += 1
            self._instance_states[selected_idx][1] = time.time()

        try:

            tool_args = params if isinstance(params, dict) else json.loads(params)
            return self.instances[selected_idx].call(json.dumps(tool_args), **kwargs)
        finally:
            with self._lock:
                self._instance_states[selected_idx][0] -= 1

class AsyncMCPManager(SSEMCPManager):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(AsyncMCPManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, num_instances=3):
        if not hasattr(self, '_initialized'):
            super().__init__()
            self.num_instances = num_instances
            self._config = None
            self._initialized = True

            self.manager_classes = [
                type(f'SSEMCPManager_{i}', (SSEMCPManager,), {'_instance': None})
                for i in range(num_instances)
            ]

    def initConfig(self, config: dict):
        'Store the configuration during initialization.'
        self._config = config
        return super().initConfig(config)

    def create_tool_class(self, register_name: str, register_client_id: str, tool_name: str, tool_desc: str, tool_parameters: dict):
        'Create a load-balanced tool class.'
        if not self._config:
            raise ValueError("Configuration not initialized. Please call initConfig first.")

        return LoadBalancedTool(
            register_name=register_name,
            register_client_id=register_client_id,
            tool_name=tool_name,
            tool_desc=tool_desc,
            tool_parameters=tool_parameters,
            config=self._config,
            manager_classes=self.manager_classes
        )

    async def _init_client_with_retry(self, client, server_name, server, max_retries=10):
        'Initialize with retries through the parent implementation.'
        return await super()._init_client_with_retry(client, server_name, server, max_retries)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
