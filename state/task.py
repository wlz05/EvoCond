# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
from dataclasses import dataclass
from state.data import canonical
from state.schema import Snapshot, label
@dataclass(frozen=True)
class Terminal:
    key: str
    values: tuple[str, ...]
    def __post_init__(self) -> None:
        key = label(canonical(self.key))
        values = tuple(dict.fromkeys(label(item) for item in self.values if label(item)))
        if not key or not values:
            raise ValueError("Terminal predicate must not be empty")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "values", values)
    def __call__(self, snapshot: Snapshot) -> bool:
        if snapshot.terminal():
            return True
        values = dict(snapshot.facts)
        values.update({"status": snapshot.status, "step": snapshot.step, "cursor": snapshot.cursor})
        index = {label(key): value for key, value in values.items()}
        return self.key in index and label(index[self.key]) in self.values

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
