# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Provide a persistent monotonic clock."""
from math import isfinite
class Clock:
    """Order updates without wall clock rollback."""
    def __init__(self, value: float = 0.0):
        if not isfinite(value) or value < 0.0:
            raise ValueError("clock value is invalid")
        self.value = float(value)
    def tick(self, stamp: float | None = None) -> float:
        if stamp is not None and (not isfinite(stamp) or stamp < 0.0):
            raise ValueError("stamp is invalid")
        candidate = self.value + 1.0 if stamp is None else float(stamp)
        self.value = max(self.value + 1.0, candidate)
        return self.value
    def observe(self, stamp: float) -> None:
        if not isfinite(stamp) or stamp < 0.0:
            raise ValueError("stamp is invalid")
        self.value = max(self.value, float(stamp))
    def dump(self) -> dict:
        return {"value": self.value}
    @classmethod
    def load(cls, data: dict | None) -> "Clock":
        return cls(float((data or {}).get("value", 0.0)))

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
