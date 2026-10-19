# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import asyncio
import warnings
from typing import Dict
from contextlib import asynccontextmanager

class ConcurrencyLimiter:
    '\u7b80\u5355\u7684\u5e76\u53d1\u9650\u5236\u5668\uff0c\u4fee\u590d\u4e8b\u4ef6\u5faa\u73af\u5207\u6362\u5bfc\u81f4\u7684\u6b7b\u9501\u95ee\u9898'

    def __init__(self, global_limit: int = 100):
        self._loop = None
        self._global_limit = global_limit

        self._tool_semaphores: Dict[str, asyncio.Semaphore] = {}

        self._global_semaphore = None
        self._semaphore_waiters = 0

    def _ensure_loop(self):
        '\n        \u786e\u4fdd\u4fe1\u53f7\u91cf\u7ed1\u5b9a\u5f53\u524d\u4e8b\u4ef6\u5faa\u73af\u3002\n        \u5982\u679c\u5207\u6362\u5230\u65b0\u7684\u4e8b\u4ef6\u5faa\u73af\uff08\u4f8b\u5982\u591a\u6b21 asyncio.run\uff09\uff0c\u9700\u8981\u91cd\u5efa\u4fe1\u53f7\u91cf\u3002\n        '
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:

            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        if self._loop is not loop:

            if self._semaphore_waiters > 0:
                warnings.warn(
                    f"Event loop changed while {self._semaphore_waiters} tasks are waiting on semaphore. "
                    f"This may cause deadlocks. Consider finishing all async tasks before switching loops.",
                    RuntimeWarning
                )

            self._loop = loop
            self._global_semaphore = asyncio.Semaphore(self._global_limit)
            self._tool_semaphores = {}
            self._semaphore_waiters = 0

    def get_tool_semaphore(self, tool_name: str, limit: int = 10) -> asyncio.Semaphore:
        '\u83b7\u53d6\u6216\u521b\u5efa\u5de5\u5177\u7684\u4fe1\u53f7\u91cf'
        self._ensure_loop()
        if tool_name not in self._tool_semaphores:
            self._tool_semaphores[tool_name] = asyncio.Semaphore(limit)
        return self._tool_semaphores[tool_name]

    @asynccontextmanager
    async def limit(self, tool_name: str):
        '\u4f7f\u7528\u4e0a\u4e0b\u6587\u7ba1\u7406\u5668\u63a7\u5236\u5e76\u53d1'
        self._ensure_loop()
        sem = self.get_tool_semaphore(tool_name)

        self._semaphore_waiters += 1
        try:
            async with self._global_semaphore:
                async with sem:
                    yield
        finally:

            self._semaphore_waiters -= 1

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
