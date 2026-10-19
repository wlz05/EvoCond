# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
from typing import Any
from model.client import Model
PROMPT = "Return one JSON object with width, forbid, due, source, and attempt. Never add task actions, goals, plans, assumptions, or guarantees. Keep width and due no larger than the supplied limits. Use source model and attempt one."
class Monitor:
    def __init__(self, model: Model):
        self.model = model
    def revise(self, suspect: list[str], sharpness: float, shrink: float, step: int, remaining: int, timeout: int) -> dict[str, Any]:
        width = sharpness * shrink / 2.0
        due = min(timeout, max(1, remaining))
        data = self.model.generate(PROMPT, {"kind": "audit", "suspect": list(suspect), "limits": {"width": width, "due": due}, "step": step})
        if not isinstance(data, dict):
            raise TypeError("Model challenger must return a mapping")
        result = dict(data)
        result.update({"source": "model", "attempt": 1})
        return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
