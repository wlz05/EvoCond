# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from .cache.base import CacheBase
from .persist.base import PersistBase

__all__ = ['CacheBase', 'PersistBase']

CACHE_REGISTRY = {
    'cache': CacheBase,
    'persist': PersistBase
}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
