# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import json
import gzip
from datetime import datetime

class PersistBase(ABC):
    @abstractmethod
    def save(self, data: Dict[str, Any], method_name: str) -> str:
        'Save a data mapping under the method name and return its path.'
        pass

    @abstractmethod
    def load(self, file_path: str) -> Dict[str, Any]:
        'Load a data mapping from disk.'
        pass

    @abstractmethod
    def merge(self, file_paths: list[str], output_path: str) -> str:
        'Merge cache files into output_path and return its path.'
        pass

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
