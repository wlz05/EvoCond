# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from .base import Env as BaseEnv
from .mmbase import MMEnv
from .search import SearchEnv
from .vision import VisionEnv
from .example import RewardRolloutEnv
from .bases import MCPBaseEnv
from .baseline import BaselineEnv
# Define public interface for the module
# Specifies which classes will be imported when using "from module import *"
__all__ = ['BaseEnv', 'SearchEnv', 'RewardRolloutEnv', 'VisionEnv', 'MMEnv', 'BaselineEnv']

# Environment registry mapping - connects environment names to their corresponding classes
# Facilitates dynamic environment creation by referencing names as strings
TOOL_ENV_REGISTRY = {
    'base': BaseEnv,
    'mmbase': MMEnv,
    'search': SearchEnv,
    'reward_rollout': RewardRolloutEnv,
    'vision': VisionEnv,
    'mcp_base': MCPBaseEnv,
    'baseline': BaselineEnv,
}

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
