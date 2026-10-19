# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Detect tool identity with a reject option."""
from dataclasses import dataclass, field
from math import isfinite, sqrt
from statistics import fmean
from core.embed import cosine, fold, word
from core.model import Decision, Tool
@dataclass(slots=True)
class Cluster:
    label: str
    limit: int
    items: list[list[float]] = field(default_factory=list)
    def add(self, vector: list[float]) -> None:
        self.items.append(list(vector))
        if len(self.items) > self.limit:
            self.items.pop(0)
    def mean(self) -> list[float]:
        return [fmean(values) for values in zip(*self.items)] if self.items else []
    def variance(self) -> list[float]:
        center = self.mean()
        if len(self.items) < 2:
            return [0.0] * len(center)
        return [sum((item[pos] - center[pos]) ** 2 for item in self.items) / (len(self.items) - 1) for pos in range(len(center))]
    def dump(self) -> dict:
        return {"label": self.label, "limit": self.limit, "items": self.items}
    @classmethod
    def load(cls, data: dict) -> "Cluster":
        return cls(data["label"], int(data["limit"]), list(data.get("items", [])))
class Novelty:
    """Combine adaptive similarity and shrinkage distance."""
    def __init__(self, dim: int, prior: float = 2.0, radius: float = 1.5, threshold: float = 0.72, limit: int = 24, epsilon: float = 1e-4, method: str = "hybrid", neighbors: int = 3):
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 16:
            raise ValueError("dim must be at least 16")
        if not isfinite(prior) or prior <= 0.0:
            raise ValueError("prior must be positive")
        if not isfinite(radius) or radius < 0.0:
            raise ValueError("radius must not be negative")
        if not isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be within negative one and one")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be positive")
        if not isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        method = fold(method)
        if method not in ("shrink", "knn", "energy", "hybrid"):
            raise ValueError("novelty method is invalid")
        if not isinstance(neighbors, int) or isinstance(neighbors, bool) or neighbors < 1:
            raise ValueError("neighbor count is invalid")
        self.dim = dim
        self.prior = prior
        self.radius = radius
        self.threshold = threshold
        self.limit = limit
        self.epsilon = epsilon
        self.method = method
        self.neighbors = neighbors
        self.clusters: dict[str, Cluster] = {}
        self.scores: dict[str, list[float]] = {}
        self.items: list[list[float]] = []
    def gate(self, label: str) -> float:
        label = word(label)
        local = self.scores.get(label, [])
        pooled = [value for values in self.scores.values() for value in values]
        def bound(values: list[float]) -> float:
            if len(values) < 2:
                return self.threshold
            center = fmean(values)
            spread = sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))
            return max(-1.0, min(1.0, center - 0.8 * spread))
        weight = self.prior / (len(local) + self.prior)
        return (1.0 - weight) * bound(local) + weight * bound(pooled)
    def globalvar(self) -> list[float]:
        if len(self.items) < 2:
            return [1.0 / self.dim] * self.dim
        center = [fmean(values) for values in zip(*self.items)]
        return [sum((item[pos] - center[pos]) ** 2 for item in self.items) / (len(self.items) - 1) for pos in range(self.dim)]
    def distance(self, vector: list[float], cluster: Cluster) -> float:
        center = cluster.mean()
        if len(vector) != self.dim or len(center) != self.dim:
            return float("inf")
        local = cluster.variance()
        shared = self.globalvar()
        weight = self.prior / (len(cluster.items) + self.prior)
        variance = [(1.0 - weight) * own + weight * common + self.epsilon for own, common in zip(local, shared)]
        return sum((value - mean) ** 2 / scale for value, mean, scale in zip(vector, center, variance)) / self.dim
    def compatible(self, left: str, right: str) -> bool:
        return not left or not right or fold(left) == fold(right)
    def measures(self, vector: list[float], cluster: Cluster) -> dict[str, float]:
        shrink = self.distance(vector, cluster)
        items = sorted(sqrt(sum((left - right) ** 2 for left, right in zip(vector, item))) for item in cluster.items)
        knn = sum(items[:self.neighbors]) / min(len(items), self.neighbors) if items else float("inf")
        energy = 1.0 - max(((cosine(vector, item) + 1.0) / 2.0 for item in cluster.items), default=0.0)
        return {"shrink": shrink, "knn": knn, "energy": energy}
    def accepted(self, values: dict[str, float]) -> bool:
        tests = {"shrink": values["shrink"] <= self.radius, "knn": values["knn"] <= 0.75, "energy": values["energy"] <= 0.25}
        return sum(tests.values()) >= 2 if self.method == "hybrid" else tests[self.method]
    def measure(self, values: dict[str, float]) -> float:
        scaled = {"shrink": values["shrink"] / max(self.radius, self.epsilon), "knn": values["knn"] / 0.75, "energy": values["energy"] / 0.25}
        return sum(scaled.values()) / len(scaled) if self.method == "hybrid" else scaled[self.method]
    def features(self, vector: list[float], scores: dict[str, float], schema: str = "", tools: dict[str, Tool] | None = None) -> list[float]:
        self.check(vector)
        clean = sorted((value, word(key)) for key, value in scores.items() if isfinite(value))
        if not clean:
            return [0.0, 0.0, (self.threshold + 1.0) / 2.0, 0.0, 0.0, 0.0, min(1.0, len(self.clusters) / 16.0), float(bool(schema))]
        score, label = clean[-1]
        second = clean[-2][0] if len(clean) > 1 else -1.0
        cluster = self.clusters.get(label)
        density = 0.0 if cluster is None else max(0.0, min(1.0, 1.0 - self.distance(vector, cluster) / max(self.radius, self.epsilon)))
        target = (tools or {}).get(label)
        conflict = target is not None and not self.compatible(schema, target.schema)
        evidence = 0.0 if cluster is None else min(1.0, len(cluster.items) / self.limit)
        return [(score + 1.0) / 2.0, max(0.0, min(1.0, (score - second) / 2.0)), (self.gate(label) + 1.0) / 2.0, density, float(conflict), evidence, min(1.0, len(self.clusters) / 16.0), float(bool(schema))]
    def decide(self, vector: list[float], scores: dict[str, float], schema: str = "", tools: dict[str, Tool] | None = None) -> Decision:
        self.check(vector)
        clean = {word(key): value for key, value in scores.items() if isfinite(value)}
        if not clean:
            return Decision("create", None, 1.0, "no known candidate")
        label = max(clean, key=clean.get)
        score = clean[label]
        target = (tools or {}).get(label)
        fits = target is None or self.compatible(schema, target.schema)
        if score >= self.gate(label):
            mode = "reuse" if fits else "split"
            reason = "calibrated semantic match" if fits else "schema conflict"
            return Decision(mode, label, max(0.0, min(1.0, (score + 1.0) / 2.0)), reason)
        distances = {name: self.measures(vector, self.clusters[name]) for name in clean if name in self.clusters and ((tools or {}).get(name) is None or self.compatible(schema, (tools or {})[name].schema))}
        if distances:
            best = min(distances, key=lambda name: self.measure(distances[name]))
            if self.accepted(distances[best]):
                confidence = max(0.0, min(1.0, 1.0 - self.measure(distances[best])))
                return Decision("reuse", best, confidence, "calibrated density match")
        if target is not None and not fits and score >= self.gate(label) - 0.1:
            return Decision("split", label, max(0.0, min(1.0, (score + 1.0) / 2.0)), "near match with schema conflict")
        return Decision("create", None, max(0.0, min(1.0, 1.0 - max(0.0, score))), "known tools rejected")
    def learn(self, label: str, vector: list[float], confidence: float | None = None) -> None:
        self.check(vector)
        label = word(label)
        if confidence is not None and not isfinite(confidence):
            raise ValueError("confidence must be finite")
        cluster = self.clusters.setdefault(label, Cluster(label, self.limit))
        cluster.add(vector)
        self.items.append(list(vector))
        if len(self.items) > self.limit * max(2, len(self.clusters)):
            self.items.pop(0)
        if confidence is not None:
            values = self.scores.setdefault(label, [])
            values.append(confidence)
            if len(values) > self.limit:
                values.pop(0)
    def check(self, vector: list[float]) -> None:
        if len(vector) != self.dim or any(not isfinite(value) for value in vector):
            raise ValueError("vector must be finite and match dim")
    def remove(self, label: str) -> None:
        label = word(label)
        self.clusters.pop(label, None)
        self.scores.pop(label, None)
    def dump(self) -> dict:
        return {"dim": self.dim, "prior": self.prior, "radius": self.radius, "threshold": self.threshold, "limit": self.limit, "epsilon": self.epsilon, "method": self.method, "neighbors": self.neighbors, "clusters": [cluster.dump() for cluster in self.clusters.values()], "scores": self.scores, "items": self.items}
    @classmethod
    def load(cls, data: dict) -> "Novelty":
        model = cls(int(data["dim"]), float(data.get("prior", 2.0)), float(data.get("radius", 1.5)), float(data.get("threshold", 0.72)), int(data.get("limit", 24)), float(data.get("epsilon", 1e-4)), str(data.get("method", "hybrid")), int(data.get("neighbors", 3)))
        for item in data.get("clusters", []):
            cluster = Cluster.load(item)
            cluster.label = word(cluster.label)
            cluster.limit = model.limit
            cluster.items = [vector for vector in cluster.items if isinstance(vector, list) and len(vector) == model.dim and all(isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value) for value in vector)][-model.limit:]
            saved = model.clusters.get(cluster.label)
            if saved is None:
                model.clusters[cluster.label] = cluster
            else:
                saved.items = (saved.items + cluster.items)[-model.limit:]
        for key, values in data.get("scores", {}).items():
            label = word(key)
            clean = [float(value) for value in values if isinstance(values, list) and isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)] if isinstance(values, list) else []
            model.scores[label] = (model.scores.get(label, []) + clean)[-model.limit:]
        items = data.get("items", [])
        model.items = [vector for vector in items if isinstance(vector, list) and len(vector) == model.dim and all(isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value) for value in vector)] if isinstance(items, list) else []
        return model

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
