# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
from math import isfinite
from typing import Mapping
from control.audit import Health
from state.schema import Evidence, Move, Scale
DECAY = 0.5
class Ledger:
    def __init__(self, decay: float = DECAY):
        value = float(decay)
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Credit decay must lie inside zero and one")
        self.decay = value
    def heat(self, scale: Scale, evidence: Mapping[Scale, Evidence], channels: str = "vdh") -> float:
        item = evidence.get(scale)
        if item is None or not item.due:
            return 0.0
        forecast = max(0.0, item.difference) if "d" in channels and item.supported() else 0.0
        violation = item.violation if "v" in channels else 0.0
        return min(1.0, max(violation, forecast))
    def support(self, move: Move, scale: Scale | None, evidence: Mapping[Scale, Evidence], health: Mapping[Scale, Health], channels: str = "vdh") -> float:
        local = self.heat(Scale.LOCAL, evidence, channels)
        bridge = self.heat(Scale.BRIDGE, evidence, channels)
        global_ = self.heat(Scale.GLOBAL, evidence, channels)
        if move is Move.CONTINUE:
            return 1.0 - max(local, bridge, global_)
        if move is Move.REPAIR:
            return local
        if move is Move.BRIDGE:
            return max(bridge, self.decay * local)
        if move is Move.GLOBAL:
            return max(global_, self.decay * bridge, self.decay ** 2 * local)
        if scale is None:
            return 0.0
        return 1.0 - health.get(scale, Health(scale)).score if "h" in channels else 0.0

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
