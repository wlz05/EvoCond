# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import os
import json
import gzip
from datetime import datetime
from typing import Any, Dict, Optional
from .base import PersistBase

class DiskPersist(PersistBase):
    def __init__(self, base_dir: str = "cache_data"):
        '\u521d\u59cb\u5316\u6301\u4e45\u5316\u5b58\u50a8\n        \n        Args:\n            base_dir: \u7f13\u5b58\u6587\u4ef6\u5b58\u50a8\u7684\u57fa\u7840\u76ee\u5f55\n        '
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _get_file_path(self, method_name: str, partition: Optional[int] = None) -> str:
        '\u751f\u6210\u6587\u4ef6\u8def\u5f84\n        \n        Args:\n            method_name: \u65b9\u6cd5\u540d\u79f0\n            partition: \u5206\u533a\u53f7\uff0c\u7528\u4e8e\u5206\u5e03\u5f0f\u573a\u666f\n            \n        Returns:\n            str: \u6587\u4ef6\u8def\u5f84\n        '
        date_str = datetime.now().strftime("%Y%m%d")
        method_hash = hashlib.md5(method_name.encode()).hexdigest()
        partition_str = f"_p{partition}" if partition is not None else ""
        file_name = f"{method_hash}_{date_str}{partition_str}.cache"
        return os.path.join(self.base_dir, file_name)

    def save(self, data: Dict[str, Any], method_name: str, partition: Optional[int] = None) -> str:
        '\u4fdd\u5b58\u6570\u636e\u5230\u78c1\u76d8\n        \n        Args:\n            data: \u8981\u4fdd\u5b58\u7684\u6570\u636e\u5b57\u5178\n            method_name: \u65b9\u6cd5\u540d\u79f0\n            partition: \u5206\u533a\u53f7\uff0c\u7528\u4e8e\u5206\u5e03\u5f0f\u573a\u666f\n            \n        Returns:\n            str: \u4fdd\u5b58\u7684\u6587\u4ef6\u8def\u5f84\n        '
        file_path = self._get_file_path(method_name, partition)

        with gzip.open(file_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f)

        return file_path

    def load(self, file_path: str) -> Dict[str, Any]:
        '\u4ece\u78c1\u76d8\u52a0\u8f7d\u6570\u636e\n        \n        Args:\n            file_path: \u6587\u4ef6\u8def\u5f84\n            \n        Returns:\n            Dict[str, Any]: \u52a0\u8f7d\u7684\u6570\u636e\u5b57\u5178\n        '
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load cache file {file_path}: {e}")
            return {}

    def merge(self, file_paths: list[str], output_path: str) -> str:
        '\u5408\u5e76\u591a\u4e2a\u7f13\u5b58\u6587\u4ef6\n        \n        Args:\n            file_paths: \u8981\u5408\u5e76\u7684\u6587\u4ef6\u8def\u5f84\u5217\u8868\n            output_path: \u8f93\u51fa\u6587\u4ef6\u8def\u5f84\n            \n        Returns:\n            str: \u5408\u5e76\u540e\u7684\u6587\u4ef6\u8def\u5f84\n        '
        merged_data = {}
        for file_path in file_paths:
            data = self.load(file_path)
            merged_data.update(data)

        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            json.dump(merged_data, f)

        return output_path

    def list_cache_files(self, method_name: Optional[str] = None) -> list[str]:
        '\u5217\u51fa\u7f13\u5b58\u6587\u4ef6\n        \n        Args:\n            method_name: \u65b9\u6cd5\u540d\u79f0\uff0c\u5982\u679c\u6307\u5b9a\u5219\u53ea\u5217\u51fa\u8be5\u65b9\u6cd5\u7684\u7f13\u5b58\u6587\u4ef6\n            \n        Returns:\n            list[str]: \u7f13\u5b58\u6587\u4ef6\u8def\u5f84\u5217\u8868\n        '
        files = []
        for file_name in os.listdir(self.base_dir):
            if not file_name.endswith('.cache'):
                continue

            if method_name is not None:
                method_hash = hashlib.md5(method_name.encode()).hexdigest()
                if not file_name.startswith(method_hash):
                    continue

            files.append(os.path.join(self.base_dir, file_name))

        return files

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
