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
        '\u83b7\u53d6\u7f13\u5b58\u503c'
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        '\u8bbe\u7f6e\u7f13\u5b58\u503c\n        \n        Args:\n            key: \u7f13\u5b58\u952e\n            value: \u7f13\u5b58\u503c\n            ttl: \u8fc7\u671f\u65f6\u95f4\uff08\u79d2\uff09\uff0cNone\u8868\u793a\u6c38\u4e0d\u8fc7\u671f\n        '
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        '\u5220\u9664\u7f13\u5b58\u503c'
        pass

    @abstractmethod
    def clear(self) -> None:
        '\u6e05\u7a7a\u7f13\u5b58'
        pass

    @abstractmethod
    def has(self, key: str) -> bool:
        '\u68c0\u67e5\u952e\u662f\u5426\u5b58\u5728'
        pass

    @abstractmethod
    def get_mode(self) -> CacheMode:
        '\u83b7\u53d6\u7f13\u5b58\u6a21\u5f0f'
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        '\u83b7\u53d6\u7f13\u5b58\u7edf\u8ba1\u4fe1\u606f'
        pass

    @abstractmethod
    def get_eviction_policy(self) -> EvictionPolicy:
        '\u83b7\u53d6\u7f13\u5b58\u6dd8\u6c70\u7b56\u7565'
        pass

    @abstractmethod
    def set_eviction_policy(self, policy: EvictionPolicy) -> None:
        '\u8bbe\u7f6e\u7f13\u5b58\u6dd8\u6c70\u7b56\u7565'
        pass

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
