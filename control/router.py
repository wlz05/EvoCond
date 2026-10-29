# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence
ACTIONS = ("continue", "repair:ll", "bridge:ll", "bridge:gl", "global:gl", "global:gg", "audit:ll", "audit:gl", "audit:gg")
SIZE = 12
def vector(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != SIZE or not all(isfinite(value) for value in result):
        raise ValueError("Router features must be finite and complete")
    return result
@dataclass(frozen=True)
class Router:
    labels: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    @classmethod
    def build(cls, data: Mapping[str, Any]) -> "Router":
        labels = tuple(str(item).strip().lower() for item in data.get("labels", ()))
        mean = vector(data.get("mean", ()))
        scale = vector(data.get("scale", ()))
        rawweights = data.get("weights", ())
        bias = tuple(float(item) for item in data.get("bias", ()))
        if labels != ACTIONS or not isinstance(rawweights, (list, tuple)) or len(rawweights) != len(labels) or len(bias) != len(labels):
            raise ValueError("Router model shape is invalid")
        weights = tuple(vector(row) for row in rawweights)
        if any(value <= 0.0 for value in scale) or not all(isfinite(value) for value in bias):
            raise ValueError("Router normalization is invalid")
        return cls(labels, mean, scale, weights, bias)
    def scores(self, features: Sequence[float]) -> dict[str, float]:
        values = vector(features)
        normalized = tuple((value - center) / width for value, center, width in zip(values, self.mean, self.scale))
        return {label: offset + sum(weight * value for weight, value in zip(row, normalized)) for label, row, offset in zip(self.labels, self.weights, self.bias)}

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
