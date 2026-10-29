# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Decompose normalized tasks into ordered subtasks."""
from hashlib import blake2b
import re
from core.embed import fold
class Decomposer:
    """Provide explicit decomposition with a deterministic fallback."""
    def __init__(self, limit: int = 8):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("subtask limit is invalid")
        self.limit = limit
    def split(self, text: str, parts: list[str] | None = None) -> list[str]:
        clean = fold(text)
        if not clean:
            raise ValueError("task must not be empty")
        if parts is not None and not isinstance(parts, list):
            raise TypeError("subtasks must be a list")
        values = [fold(item) for item in parts] if parts is not None else [fold(item) for item in re.split(r"(?:[;.!?]+|\bthen\b|\bnext\b|\bfinally\b|\bafterwards\b)", clean)]
        values = list(dict.fromkeys(value for value in values if value))
        return (values or [clean])[:self.limit]
    def key(self, text: str, nonce: float | None = None) -> str:
        clean = fold(text)
        if not clean:
            raise ValueError("task identity must not be empty")
        raw = clean if nonce is None else f"{clean}|{nonce:.12g}"
        return blake2b(raw.encode("utf-8"), digest_size=12).hexdigest()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
