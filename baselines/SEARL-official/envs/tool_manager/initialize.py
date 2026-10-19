# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from .language import Llama3Manager
from .configuration import ConfigManager
from .reasoner import QwenManager
from .reasoning import Qwen25Manager
from .visual import Qwen25VLManager
from .centralized.central import CentralizedQwenManager
from .baseline import QwenManagerBaseline

__all__ = ['ConfigManager', 'QwenManager', 'Qwen25Manager','Qwen25VLManager', 'Llama3Manager', 'CentralizedQwenManager', 'Qwen3BaselineManager']

TOOL_MANAGER_REGISTRY = {
    'config': ConfigManager,
    'qwen3': QwenManager,
    'qwen2_5': Qwen25Manager,
    'qwen2_5_vl': Qwen25VLManager,
    'llama3' : Llama3Manager,
    'centralized_qwen3': CentralizedQwenManager,
    'qwen3_baseline': QwenManagerBaseline,
}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
