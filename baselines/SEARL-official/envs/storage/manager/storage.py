# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import argparse
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from ..cache.base import CacheMode, EvictionPolicy
from ..cache.cache import CacheBoxCache
from ..persist.disk import DiskPersist

class StorageManager:
    def __init__(self, cache_mode: CacheMode = CacheMode.SINGLE,
                 enable_persist: bool = False,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
                 max_cache_size: int = 1000,
                 persist_dir: str = "cache_data",
                 persist_interval: int = 300,
                 sync_interval: int = 600):
        '\u521d\u59cb\u5316\u5b58\u50a8\u7ba1\u7406\u5668\n        \n        Args:\n            cache_mode: \u7f13\u5b58\u6a21\u5f0f\n            enable_persist: \u662f\u5426\u542f\u7528\u6301\u4e45\u5316\n            eviction_policy: \u7f13\u5b58\u6dd8\u6c70\u7b56\u7565\n            max_cache_size: \u6700\u5927\u7f13\u5b58\u6761\u76ee\u6570\n            persist_dir: \u6301\u4e45\u5316\u5b58\u50a8\u76ee\u5f55\n            persist_interval: \u6301\u4e45\u5316\u95f4\u9694\uff08\u79d2\uff09\n            sync_interval: \u540c\u6b65\u95f4\u9694\uff08\u79d2\uff09\n        '
        self.cache_mode = cache_mode
        self.enable_persist = enable_persist
        self.persist_interval = persist_interval
        self.sync_interval = sync_interval

        self.cache = CacheBoxCache(
            max_size=max_cache_size,
            mode=cache_mode,
            eviction_policy=eviction_policy
        )

        self.persist = DiskPersist(base_dir=persist_dir) if enable_persist else None

        self.persist_buffer: Dict[str, Dict[str, Any]] = {}
        self.last_persist_time = datetime.now()

        if self.enable_persist and self.persist:
            asyncio.create_task(self._start_persist_task())
            asyncio.create_task(self._start_sync_task())

    async def _start_persist_task(self):
        '\u542f\u52a8\u5b9a\u65f6\u6301\u4e45\u5316\u4efb\u52a1'
        while True:
            await asyncio.sleep(self.persist_interval)
            await self._flush_persist_buffer()

    async def _start_sync_task(self):
        '\u542f\u52a8\u5b9a\u65f6\u540c\u6b65\u4efb\u52a1'
        while True:
            await asyncio.sleep(self.sync_interval)
            await self._sync_from_persist()

    async def _flush_persist_buffer(self):
        '\u5c06\u7f13\u51b2\u533a\u6570\u636e\u5199\u5165\u6301\u4e45\u5316\u5b58\u50a8'
        if not self.persist_buffer:
            return

        try:
            for method_name, data in self.persist_buffer.items():
                await asyncio.to_thread(self.persist.save, data, method_name)
            self.persist_buffer.clear()
            self.last_persist_time = datetime.now()
        except Exception as e:
            print(f"Failed to flush persist buffer: {e}")

    async def _sync_from_persist(self):
        '\u4ece\u6301\u4e45\u5316\u5b58\u50a8\u540c\u6b65\u6570\u636e\u5230\u7f13\u5b58'
        if not self.persist:
            return

        try:

            cache_files = self.persist.list_cache_files()
            for file_path in cache_files:
                data = await asyncio.to_thread(self.persist.load, file_path)
                for cache_key, value in data.items():

                    self.cache.set(cache_key, value)
        except Exception as e:
            print(f"Failed to sync from persist: {e}")

    async def get(self, method_name: str, params: dict) -> Optional[Any]:
        '\u83b7\u53d6\u7f13\u5b58\u6570\u636e\n        \n        Args:\n            method_name: \u65b9\u6cd5\u540d\n            params: \u53c2\u6570\u5b57\u5178\n            \n        Returns:\n            Optional[Any]: \u7f13\u5b58\u6570\u636e\n        '
        cache_key = self.cache._hash_key(method_name, params)

        if self.cache.has(cache_key):
            return self.cache.get(cache_key)

        if self.enable_persist and self.persist:
            try:

                cache_files = self.persist.list_cache_files(method_name)
                for file_path in cache_files:
                    data = await asyncio.to_thread(self.persist.load, file_path)
                    if cache_key in data:
                        value = data[cache_key]

                        self.cache.set(cache_key, value)
                        return value
            except Exception as e:
                print(f"Failed to load from persist: {e}")

        return None

    async def set(self, method_name: str, params: dict, value: Any, ttl: Optional[int] = None) -> None:
        '\u8bbe\u7f6e\u7f13\u5b58\u6570\u636e\n        \n        Args:\n            method_name: \u65b9\u6cd5\u540d\n            params: \u53c2\u6570\u5b57\u5178\n            value: \u8981\u7f13\u5b58\u7684\u503c\n            ttl: \u8fc7\u671f\u65f6\u95f4\uff08\u79d2\uff09\n        '
        cache_key = self.cache._hash_key(method_name, params)

        self.cache.set(cache_key, value, ttl)

        if self.enable_persist and self.persist:
            if method_name not in self.persist_buffer:
                self.persist_buffer[method_name] = {}
            self.persist_buffer[method_name][cache_key] = value

            current_time = datetime.now()
            if (len(self.persist_buffer) > 1000 or
                    (current_time - self.last_persist_time).total_seconds() > self.persist_interval):
                await self._flush_persist_buffer()

    def get_stats(self) -> Dict[str, Any]:
        '\u83b7\u53d6\u5b58\u50a8\u7edf\u8ba1\u4fe1\u606f\n        \n        Returns:\n            Dict[str, Any]: \u7edf\u8ba1\u4fe1\u606f\n        '
        stats = {
            "cache_mode": self.cache_mode.value,
            "enable_persist": self.enable_persist,
            "eviction_policy": self.cache.get_eviction_policy().value,
            "cache_stats": self.cache.get_stats(),
            "persist_buffer_size": len(self.persist_buffer) if self.enable_persist else 0,
            "last_persist_time": self.last_persist_time.isoformat() if self.enable_persist else None
        }
        return stats

def parse_args():
    '\u89e3\u6790\u547d\u4ee4\u884c\u53c2\u6570'
    parser = argparse.ArgumentParser(description="Storage Manager")
    parser.add_argument("-cache", choices=["single", "multi"], default="single",
                        help="Cache mode: single (default) or multi")
    parser.add_argument("-persist", action="store_true",
                        help="Enable persistence")
    parser.add_argument("-eviction", choices=["lru", "lfu", "fifo", "ttl"], default="lru",
                        help="Cache eviction policy")
    parser.add_argument("-max-size", type=int, default=1000,
                        help="Maximum cache size")
    parser.add_argument("-persist-dir", default="cache_data",
                        help="Persistence directory")
    return parser.parse_args()

def create_storage_manager(args) -> StorageManager:
    '\u521b\u5efa\u5b58\u50a8\u7ba1\u7406\u5668\n    \n    Args:\n        args: \u547d\u4ee4\u884c\u53c2\u6570\n        \n    Returns:\n        StorageManager: \u5b58\u50a8\u7ba1\u7406\u5668\u5b9e\u4f8b\n    '
    cache_mode = CacheMode.SINGLE if args.cache == "single" else CacheMode.MULTI
    eviction_policy = EvictionPolicy(args.eviction)

    return StorageManager(
        cache_mode=cache_mode,
        enable_persist=args.persist,
        eviction_policy=eviction_policy,
        max_cache_size=args.max_size,
        persist_dir=args.persist_dir
    )

def create_config_storage_manager(verl_config) -> StorageManager:
    '\u521b\u5efa\u5b58\u50a8\u7ba1\u7406\u5668\n\n    Args:\n        verl_config: \u914d\u7f6e\u6587\u4ef6\n\n    Returns:\n        StorageManager: \u5b58\u50a8\u7ba1\u7406\u5668\u5b9e\u4f8b\n    '
    cache_mode = CacheMode.SINGLE if verl_config.cache == "single" else CacheMode.MULTI
    eviction_policy = EvictionPolicy(verl_config.eviction)

    return StorageManager(
        cache_mode=cache_mode,
        enable_persist=verl_config.persist,
        eviction_policy=eviction_policy,
        max_cache_size=verl_config.max_size,
        persist_dir=verl_config.persist_dir
    )

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
