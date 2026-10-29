# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import hashlib
import json
import time
from typing import Any, Optional, Dict
from cachebox import Cache
from .base import CacheBase, CacheMode, EvictionPolicy

class CacheBoxCache(CacheBase):
    def __init__(self, max_size: int = 1000, mode: CacheMode = CacheMode.SINGLE,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        'Initialize the cache capacity, mode and eviction policy.'
        self.cache = Cache(maxsize=max_size)
        self.mode = mode
        self.eviction_policy = eviction_policy
        self.stats = {
            "hits": 0,
            "misses": 0,
            "size": 0,
            "evictions": 0,
            "ttl_expirations": 0
        }
        self.ttl_map = {}

    def _hash_key(self, method_name: str, params: dict) -> str:
        'Hash a method name and parameter mapping into a cache key.'
        key_str = f"{method_name}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:

        if key in self.ttl_map:
            if time.time() > self.ttl_map[key]:
                del self.ttl_map[key]
                self.stats["ttl_expirations"] += 1
                return None

        value = self.cache.get(key)
        if value is not None:
            self.stats["hits"] += 1
        else:
            self.stats["misses"] += 1
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self.cache.set(key, value)
        self.stats["size"] = len(self.cache)

        if ttl is not None:
            self.ttl_map[key] = time.time() + ttl

    def delete(self, key: str) -> None:
        self.cache.delete(key)
        if key in self.ttl_map:
            del self.ttl_map[key]
        self.stats["size"] = len(self.cache)

    def clear(self) -> None:
        self.cache.clear()
        self.ttl_map.clear()
        self.stats["size"] = 0

    def has(self, key: str) -> bool:
        if key in self.ttl_map and time.time() > self.ttl_map[key]:
            del self.ttl_map[key]
            return False
        return key in self.cache

    def get_mode(self) -> CacheMode:
        return self.mode

    def get_stats(self) -> Dict[str, Any]:
        return self.stats

    def get_eviction_policy(self) -> EvictionPolicy:
        return self.eviction_policy

    def set_eviction_policy(self, policy: EvictionPolicy) -> None:
        self.eviction_policy = policy

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
