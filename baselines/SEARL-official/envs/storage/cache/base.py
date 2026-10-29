# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from enum import Enum

class CacheMode(Enum):
    SINGLE = "single"
    MULTI = "multi"

class EvictionPolicy(Enum):
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"

class CacheBase(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        'Read a cached value.'
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        'Cache a value with a TTL in seconds; None disables expiration.'
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        'Delete a cached value.'
        pass

    @abstractmethod
    def clear(self) -> None:
        'Clear the cache.'
        pass

    @abstractmethod
    def has(self, key: str) -> bool:
        'Check whether the key exists.'
        pass

    @abstractmethod
    def get_mode(self) -> CacheMode:
        'Return the cache mode.'
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        'Return cache statistics.'
        pass

    @abstractmethod
    def get_eviction_policy(self) -> EvictionPolicy:
        'Return the eviction policy.'
        pass

    @abstractmethod
    def set_eviction_policy(self, policy: EvictionPolicy) -> None:
        'Set the eviction policy.'
        pass

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
