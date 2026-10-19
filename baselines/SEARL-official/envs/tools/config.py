# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Deployment endpoints; never contact the original authors' private hosts."""
import os
LOCAL_SEARCH_HOST = os.environ.get("SEARL_SEARCH_HOST", "127.0.0.1")
LOCAL_SEARCH_URL = os.environ.get("SEARL_SEARCH_URL", "")
SANDBOX_HOST = os.environ.get("SEARL_SANDBOX_HOST", "127.0.0.1")
REDIS_HOST = os.environ.get("SEARL_REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("SEARL_REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("SEARL_REDIS_DB", "0"))

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
