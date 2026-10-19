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
        '\u4fdd\u5b58\u6570\u636e\u5230\u78c1\u76d8\n        \n        Args:\n            data: \u8981\u4fdd\u5b58\u7684\u6570\u636e\u5b57\u5178\n            method_name: \u65b9\u6cd5\u540d\u79f0\uff0c\u7528\u4e8e\u751f\u6210\u6587\u4ef6\u540d\n            \n        Returns:\n            str: \u4fdd\u5b58\u7684\u6587\u4ef6\u8def\u5f84\n        '
        pass

    @abstractmethod
    def load(self, file_path: str) -> Dict[str, Any]:
        '\u4ece\u78c1\u76d8\u52a0\u8f7d\u6570\u636e\n        \n        Args:\n            file_path: \u6587\u4ef6\u8def\u5f84\n            \n        Returns:\n            Dict[str, Any]: \u52a0\u8f7d\u7684\u6570\u636e\u5b57\u5178\n        '
        pass

    @abstractmethod
    def merge(self, file_paths: list[str], output_path: str) -> str:
        '\u5408\u5e76\u591a\u4e2a\u7f13\u5b58\u6587\u4ef6\n        \n        Args:\n            file_paths: \u8981\u5408\u5e76\u7684\u6587\u4ef6\u8def\u5f84\u5217\u8868\n            output_path: \u8f93\u51fa\u6587\u4ef6\u8def\u5f84\n            \n        Returns:\n            str: \u5408\u5e76\u540e\u7684\u6587\u4ef6\u8def\u5f84\n        '
        pass

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
