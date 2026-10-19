# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Learn task-conditioned semantic and structural fusion."""
from math import ceil, exp, isfinite, log, sqrt
from core.embed import word
from core.model import Trace
class Teacher:
    """Route each task between two aligned retrieval views."""
    views = ("semantic", "structure")
    features = ("sementropy", "strentropy", "divergence", "rankgap", "overlap", "maturity", "coverage", "alignment", "procedure", "symbolic", "context")
    def __init__(self, rate: float = 0.08, temperature: float = 1.0, coldcap: float = 0.15, clip: float = 0.2, kl: float = 0.01):
        if not isfinite(rate) or rate < 0.0:
            raise ValueError("rate must not be negative")
        if not isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not isfinite(coldcap) or not 0.0 <= coldcap <= 1.0:
            raise ValueError("coldcap must be within zero and one")
        if not isfinite(clip) or not 0.0 <= clip < 1.0:
            raise ValueError("clip must be within zero and one")
        if not isfinite(kl) or kl < 0.0:
            raise ValueError("kl must not be negative")
        self.rate = rate
        self.temperature = temperature
        self.coldcap = coldcap
        self.clip = clip
        self.kl = kl
        self.bias = [0.0] * len(self.views)
        self.matrix = [[0.0] * len(self.features) for _ in self.views]
        self.baseline = 0.0
        self.seen = 0
        self.evidence: dict = {}
    def check(self, values: list[float] | None) -> list[float]:
        values = [0.0] * len(self.features) if values is None else list(values)
        if len(values) != len(self.features) or any(not isfinite(value) for value in values):
            raise ValueError("teacher features are invalid")
        return [max(0.0, min(1.0, value)) for value in values]
    def logits(self, features: list[float] | None = None) -> list[float]:
        values = self.check(features)
        return [self.bias[row] + sum(weight * (value - 0.5) for weight, value in zip(self.matrix[row], values)) for row in range(len(self.views))]
    def weights(self, features: list[float] | None = None, cold: bool = False) -> list[float]:
        logits = self.logits(features)
        peak = max(logits)
        values = [exp((value - peak) / self.temperature) for value in logits]
        total = sum(values)
        values = [value / total for value in values]
        if cold and values[1] > self.coldcap:
            values = [1.0 - self.coldcap, self.coldcap]
        return values
    def budgets(self, weights: list[float], limit: int, total: int) -> list[int]:
        if len(weights) != len(self.views) or any(not isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("view weights are invalid")
        if not isinstance(limit, int) or isinstance(limit, bool) or not isinstance(total, int) or isinstance(total, bool) or limit < 0 or total < 0:
            raise ValueError("budgets must not be negative")
        if total == 0 or limit == 0:
            return [0] * len(self.views)
        return [min(total, max(1, ceil(limit * (0.5 + weight)))) for weight in weights]
    def update(self, trace: Trace, chosen: list[str], advantage: float) -> None:
        if not isfinite(advantage) or not trace.hits:
            return
        keys = {word(key) for key in chosen}
        selected = [hit for hit in trace.hits if word(hit.key) in keys]
        if not selected:
            return
        focus = [sum(getattr(hit, view) for hit in selected) / len(selected) for view in self.views]
        weights = trace.weights or self.weights(trace.features, trace.cold)
        expected = sum(weight * value for weight, value in zip(weights, focus))
        features = self.check(trace.features)
        for row in range(len(self.views)):
            direction = focus[row] - expected
            self.bias[row] = max(-8.0, min(8.0, self.bias[row] + self.rate * advantage * direction))
            for col, value in enumerate(features):
                change = self.rate * advantage * direction * (value - 0.5)
                self.matrix[row][col] = max(-8.0, min(8.0, self.matrix[row][col] + change))
        self.evidence = {"task": trace.task, "subtasks": trace.subtasks, "chosen": [hit.key for hit in selected], "advantage": advantage, "features": features, "focus": focus, "weights": weights}
    def action(self, trace: Trace, chosen: list[str]) -> int | None:
        keys = {word(key) for key in chosen}
        selected = [hit for hit in trace.hits if word(hit.key) in keys]
        if not selected:
            return None
        focus = [sum(getattr(hit, view) for hit in selected) / len(selected) for view in self.views]
        return max(range(len(focus)), key=focus.__getitem__)
    def learn(self, trace: Trace, chosen: list[str], utility: float) -> float:
        if not isfinite(utility):
            raise ValueError("utility must be finite")
        advantage = utility - self.baseline
        self.update(trace, chosen, advantage)
        self.seen += 1
        self.baseline += (utility - self.baseline) / self.seen
        return advantage
    def supervise(self, samples: list[tuple[list[float], list[float]]]) -> float:
        if not samples:
            return 0.0
        loss = 0.0
        for features, target in samples:
            values = self.check(features)
            if len(target) != len(self.views) or any(not isfinite(value) or value < 0.0 for value in target) or sum(target) <= 0.0:
                raise ValueError("teacher target is invalid")
            target = [value / sum(target) for value in target]
            current = self.weights(values)
            loss += -sum(goal * log(max(score, 1e-12)) for goal, score in zip(target, current))
            for row in range(len(self.views)):
                direction = target[row] - current[row]
                self.bias[row] = max(-8.0, min(8.0, self.bias[row] + self.rate * direction))
                for col, value in enumerate(values):
                    change = self.rate * direction * (value - 0.5)
                    self.matrix[row][col] = max(-8.0, min(8.0, self.matrix[row][col] + change))
        loss /= len(samples)
        self.evidence = {"mode": "supervised", "samples": len(samples), "loss": loss}
        return loss
    def group(self, samples: list[tuple[Trace, list[str], float]]) -> list[float]:
        if any(not isfinite(item[2]) for item in samples):
            raise ValueError("group utilities must be finite")
        if not samples:
            return []
        values = [item[2] for item in samples]
        center = sum(values) / len(values)
        spread = sqrt(sum((value - center) ** 2 for value in values) / len(values))
        advantages = [(value - center) / spread for value in values] if spread > 1e-12 else [0.0] * len(values)
        scaled: list[float] = []
        for (trace, chosen, _), advantage in zip(samples, advantages):
            action = self.action(trace, chosen)
            current = self.weights(trace.features, trace.cold)
            old = trace.weights or current
            ratio = 1.0 if action is None else current[action] / max(old[action], 1e-12)
            clipped = max(1.0 - self.clip, min(1.0 + self.clip, ratio))
            scaled.append(min(ratio * advantage, clipped * advantage))
        for (trace, chosen, utility), advantage in zip(samples, scaled):
            self.update(trace, chosen, advantage)
            self.seen += 1
            self.baseline += (utility - self.baseline) / self.seen
        shrink = max(0.0, 1.0 - self.rate * self.kl)
        self.bias = [value * shrink for value in self.bias]
        self.matrix = [[value * shrink for value in row] for row in self.matrix]
        return advantages
    def state(self, features: list[float] | None = None, cold: bool = False) -> dict:
        values = self.weights(features, cold)
        lead = self.views[max(range(len(values)), key=values.__getitem__)]
        return {"evidence": self.evidence, "belief": {view: value for view, value in zip(self.views, values)}, "guidance": {"focus": lead, "confidence": abs(values[0] - values[1])}}
    def dump(self) -> dict:
        return {"rate": self.rate, "temperature": self.temperature, "coldcap": self.coldcap, "clip": self.clip, "kl": self.kl, "bias": self.bias, "matrix": self.matrix, "baseline": self.baseline, "seen": self.seen, "evidence": self.evidence}
    @classmethod
    def load(cls, data: dict) -> "Teacher":
        model = cls(float(data.get("rate", 0.08)), float(data.get("temperature", 1.0)), float(data.get("coldcap", 0.15)), float(data.get("clip", 0.2)), float(data.get("kl", 0.01)))
        bias = list(data.get("bias", data.get("logits", model.bias)))
        matrix = [list(row) for row in data.get("matrix", model.matrix)]
        if len(bias) == 3:
            bias = [bias[0], (bias[1] + bias[2]) / 2.0]
        if len(matrix) == 3:
            matrix = [matrix[0], [(left + right) / 2.0 for left, right in zip(matrix[1], matrix[2])]]
        matrix = [row[:len(model.features)] + [0.0] * max(0, len(model.features) - len(row)) for row in matrix]
        if len(bias) != len(model.views) or any(not isfinite(value) for value in bias):
            raise ValueError("teacher bias is invalid")
        if len(matrix) != len(model.views) or any(len(row) != len(model.features) or any(not isfinite(value) for value in row) for row in matrix):
            raise ValueError("teacher matrix is invalid")
        model.bias = bias
        model.matrix = matrix
        model.baseline = float(data.get("baseline", 0.0))
        model.seen = int(data.get("seen", 0))
        if not isfinite(model.baseline) or model.seen < 0:
            raise ValueError("teacher state is invalid")
        model.evidence = dict(data.get("evidence", {}))
        return model

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
