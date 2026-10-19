# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
class Initializers(importlib.abc.MetaPathFinder):
    def __init__(self, root, entries):
        self.root = Path(root).resolve()
        self.entries = {}
        for entry in entries:
            path = self.root/entry['path']
            if not path.resolve().is_relative_to(self.root) or not path.is_file():
                raise ValueError('Package initializer leaves the source tree or is missing')
            self.entries.setdefault(entry['module'], []).append((path, entry['package']))
    def find_spec(self, fullname, path=None, target=None):
        # Retain package exports without special source filenames.
        search = {os.path.normcase(str(Path(item or '.').resolve())) for item in (path if path is not None else sys.path)}
        for source, package in self.entries.get(fullname, ()):
            parent = source.parent.parent if package else source.parent
            if os.path.normcase(str(parent.resolve())) in search:
                return importlib.util.spec_from_file_location(fullname, source,
                    submodule_search_locations=[str(source.parent)] if package else None)
        return None
def install(root=None):
    root = Path(root or Path(__file__).resolve().parent).resolve()
    manifest = root/'configs/packages.json'
    if not manifest.is_file():
        return None
    for finder in sys.meta_path:
        if isinstance(finder, Initializers) and finder.root == root:
            return finder
    data = json.loads(manifest.read_text(encoding='utf-8'))
    if data.get('format') != 'evocond_package_entries_v1':
        raise ValueError('Unknown package initialization manifest')
    finder = Initializers(root, data['entries'])
    sys.meta_path.insert(0, finder)
    return finder

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
