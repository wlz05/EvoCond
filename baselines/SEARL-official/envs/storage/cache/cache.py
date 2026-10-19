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
        '\u521d\u59cb\u5316\u7f13\u5b58\n        \n        Args:\n            max_size: \u6700\u5927\u7f13\u5b58\u6761\u76ee\u6570\n            mode: \u7f13\u5b58\u6a21\u5f0f\n            eviction_policy: \u7f13\u5b58\u6dd8\u6c70\u7b56\u7565\n        '
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
        '\u751f\u6210\u7f13\u5b58\u952e\n        \n        Args:\n            method_name: \u65b9\u6cd5\u540d\n            params: \u53c2\u6570\u5b57\u5178\n            \n        Returns:\n            str: \u54c8\u5e0c\u540e\u7684\u952e\n        '
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

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
