# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import asyncio
import warnings
from typing import Dict
from contextlib import asynccontextmanager

class ConcurrencyLimiter:
    'Limit concurrency while handling event-loop changes safely.'

    def __init__(self, global_limit: int = 100):
        self._loop = None
        self._global_limit = global_limit

        self._tool_semaphores: Dict[str, asyncio.Semaphore] = {}

        self._global_semaphore = None
        self._semaphore_waiters = 0

    def _ensure_loop(self):
        'Recreate the semaphore when the active event loop changes.'
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
        'Get or create the tool semaphore.'
        self._ensure_loop()
        if tool_name not in self._tool_semaphores:
            self._tool_semaphores[tool_name] = asyncio.Semaphore(limit)
        return self._tool_semaphores[tool_name]

    @asynccontextmanager
    async def limit(self, tool_name: str):
        'Limit concurrency within a context manager.'
        self._ensure_loop()
        sem = self.get_tool_semaphore(tool_name)

        self._semaphore_waiters += 1
        try:
            async with self._global_semaphore:
                async with sem:
                    yield
        finally:

            self._semaphore_waiters -= 1

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
