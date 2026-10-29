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
        'Initialize persistent storage under base_dir.'
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _get_file_path(self, method_name: str, partition: Optional[int] = None) -> str:
        'Build a cache path from the method name and distributed partition.'
        date_str = datetime.now().strftime("%Y%m%d")
        method_hash = hashlib.md5(method_name.encode()).hexdigest()
        partition_str = f"_p{partition}" if partition is not None else ""
        file_name = f"{method_hash}_{date_str}{partition_str}.cache"
        return os.path.join(self.base_dir, file_name)

    def save(self, data: Dict[str, Any], method_name: str, partition: Optional[int] = None) -> str:
        'Persist a data mapping for the method and partition; return its path.'
        file_path = self._get_file_path(method_name, partition)

        with gzip.open(file_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f)

        return file_path

    def load(self, file_path: str) -> Dict[str, Any]:
        'Load a data mapping from disk.'
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load cache file {file_path}: {e}")
            return {}

    def merge(self, file_paths: list[str], output_path: str) -> str:
        'Merge cache files into output_path and return its path.'
        merged_data = {}
        for file_path in file_paths:
            data = self.load(file_path)
            merged_data.update(data)

        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            json.dump(merged_data, f)

        return output_path

    def list_cache_files(self, method_name: Optional[str] = None) -> list[str]:
        'List cached paths, optionally restricted to one method.'
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

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
