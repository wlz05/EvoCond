# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
from state.data import decode
def local(value: str) -> Path:
    text = str(value).strip()
    path = Path(text)
    if not text or path.is_absolute() or path.drive or path.name in {"", ".", ".."}:
        raise ValueError("File paths must be relative")
    return path
def rows(value: str) -> list[Mapping[str, Any]]:
    path = local(value)
    result: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = decode(line)
            if not isinstance(item, Mapping):
                raise ValueError(f"Reference row {number} must be an object")
            result.append(item)
    if not result:
        raise ValueError("Reference data must not be empty")
    return result
def document(value: str) -> Mapping[str, Any]:
    path = local(value)
    item = decode(path.read_text(encoding="utf-8"))
    if not isinstance(item, Mapping):
        raise ValueError("Configuration data must be an object")
    return item
class Journal:
    def __init__(self, value: str):
        self.path = local(value)
    def write(self, item: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(item, ensure_ascii=True, separators=(",", ":")) + "\n")

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
