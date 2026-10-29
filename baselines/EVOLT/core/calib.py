# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Calibrate bounded retrieval scores."""
from math import exp, isfinite, log
class Calib:
    """Fit a small validation-only logistic map."""
    def __init__(self, limit: int = 256):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 2:
            raise ValueError("calibration limit is invalid")
        self.limit = limit
        self.temperature = 1.0
        self.bias = 0.0
        self.fitted = False
        self.reference: list[float] = []
        self.recent: list[float] = []
    def logit(self, value: float) -> float:
        value = max(1e-6, min(1.0 - 1e-6, value))
        return log(value / (1.0 - value))
    def map(self, value: float, temperature: float | None = None, bias: float | None = None) -> float:
        if not isfinite(value):
            raise ValueError("calibration score must be finite")
        scale = self.temperature if temperature is None else temperature
        shift = self.bias if bias is None else bias
        if not isfinite(scale) or scale <= 0.0 or not isfinite(shift):
            raise ValueError("calibration parameters are invalid")
        zed = max(-30.0, min(30.0, (self.logit(value) + shift) / scale))
        return 1.0 / (1.0 + exp(-zed))
    def fit(self, scores: list[float], labels: list[int]) -> float:
        if len(scores) != len(labels) or len(scores) < 2:
            raise ValueError("calibration samples are invalid")
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in scores):
            raise ValueError("calibration scores are invalid")
        if any(label not in (0, 1) for label in labels) or len(set(labels)) < 2:
            raise ValueError("calibration labels are invalid")
        best = (float("inf"), 1.0, 0.0)
        for temperature in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0):
            for bias in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
                values = [self.map(value, temperature, bias) for value in scores]
                loss = sum((value - label) ** 2 for value, label in zip(values, labels)) / len(values)
                best = min(best, (loss, temperature, bias))
        self.temperature = best[1]
        self.bias = best[2]
        self.fitted = True
        self.reference = list(scores[-self.limit:])
        self.recent = []
        return best[0]
    def predict(self, value: float) -> float:
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("calibration score is invalid")
        self.recent.append(value)
        if len(self.recent) > self.limit:
            self.recent.pop(0)
        return self.map(value) if self.fitted else value
    def drift(self) -> float:
        if len(self.reference) < 2 or len(self.recent) < 2:
            return 0.0
        left = sum(self.reference) / len(self.reference)
        right = sum(self.recent) / len(self.recent)
        values = self.reference + self.recent
        center = sum(values) / len(values)
        spread = (sum((value - center) ** 2 for value in values) / len(values)) ** 0.5
        return min(1.0, abs(left - right) / max(spread, 1e-6) / 3.0)
    def dump(self) -> dict:
        return {"limit": self.limit, "temperature": self.temperature, "bias": self.bias, "fitted": self.fitted, "reference": self.reference, "recent": self.recent}
    @classmethod
    def load(cls, data: dict) -> "Calib":
        model = cls(int(data.get("limit", 256)))
        model.temperature = float(data.get("temperature", 1.0))
        model.bias = float(data.get("bias", 0.0))
        model.fitted = bool(data.get("fitted", False))
        model.reference = [float(value) for value in data.get("reference", [])][-model.limit:]
        model.recent = [float(value) for value in data.get("recent", [])][-model.limit:]
        values = [model.temperature, model.bias, *model.reference, *model.recent]
        if model.temperature <= 0.0 or any(not isfinite(value) for value in values):
            raise ValueError("calibration state is invalid")
        return model

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
