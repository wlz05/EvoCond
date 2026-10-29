# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Fuse calibrated independent retrieval views."""
from math import exp, isfinite, log, sqrt
from core.calib import Calib
from core.embed import Embed, fold, word
from core.graph import Graph
from core.index import Aligned, Semantic, Structural
from core.model import Hit, Trace
from learn.teach import Teacher
class Moment:
    """Track an online fallback calibration."""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.square = 0.0
    def add(self, value: float) -> None:
        if not isfinite(value):
            raise ValueError("score must be finite")
        self.count += 1
        shift = value - self.mean
        self.mean += shift / self.count
        self.square += shift * (value - self.mean)
    def score(self, value: float) -> float:
        if self.count < 3 or self.square <= 1e-12:
            return max(0.0, min(1.0, value))
        spread = sqrt(self.square / (self.count - 1))
        zed = max(-8.0, min(8.0, (value - self.mean) / spread))
        return 1.0 / (1.0 + exp(-zed))
    def dump(self) -> dict:
        return {"count": self.count, "mean": self.mean, "square": self.square}
    def restore(self, data: dict) -> None:
        self.count = max(0, int(data.get("count", 0)))
        self.mean = float(data.get("mean", 0.0))
        self.square = max(0.0, float(data.get("square", 0.0)))
        if not isfinite(self.mean) or not isfinite(self.square):
            raise ValueError("rank statistics must be finite")
class Ranker:
    """Retrieve from bounded semantic and graph stores."""
    def __init__(self, graph: Graph, embedder: Embed, teacher: Teacher, semantic: Semantic, structural: Structural):
        self.graph = graph
        self.embedder = embedder
        self.teacher = teacher
        self.semindex = semantic
        self.strindex = structural
        self.semcal = Moment()
        self.strcal = Moment()
        self.actcal = Moment()
        self.semsup = Calib()
        self.strsup = Calib()
        self.actsup = Calib()
    def distribution(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        peak = max(scores.values())
        values = {key: exp(6.0 * (value - peak)) for key, value in scores.items()}
        total = sum(values.values())
        return {key: value / total for key, value in values.items()}
    def entropy(self, values: dict[str, float]) -> float:
        if len(values) < 2:
            return 0.0
        return -sum(value * log(max(value, 1e-12)) for value in values.values()) / log(len(values))
    def features(self, query: str, semantic: dict[str, float], structure: dict[str, float], context: list[str], budget: int, alignment: float, subtasks: list[str]) -> list[float]:
        left = self.distribution(semantic)
        right = self.distribution(structure)
        middle = {key: 0.5 * (left.get(key, 0.0) + right.get(key, 0.0)) for key in semantic}
        diverge = 0.5 * sum(value * log(max(value, 1e-12) / max(middle[key], 1e-12)) for key, value in left.items())
        diverge += 0.5 * sum(value * log(max(value, 1e-12) / max(middle[key], 1e-12)) for key, value in right.items())
        semtop = max(semantic, key=semantic.get)
        strtop = max(structure, key=structure.get)
        topsem = set(sorted(semantic, key=semantic.get, reverse=True)[:budget])
        topstr = set(sorted(structure, key=structure.get, reverse=True)[:budget])
        overlap = len(topsem & topstr) / max(1, len(topsem | topstr))
        incident = {key for edge in self.graph.edges.values() for key in (edge.source, edge.target)}
        maturity = min(1.0, sum(edge.count for edge in self.graph.edges.values()) / max(1, len(self.graph.tools) * 2))
        coverage = len(incident) / max(1, len(self.graph.tools))
        procedure = min(1.0, max(0, len(subtasks) - 1) / 3.0)
        symbolic = float(any(value.isdigit() or value in "+-*/=%" for value in query))
        return [self.entropy(left), self.entropy(right), min(1.0, diverge / log(2.0)), float(semtop != strtop), overlap, maturity, coverage, alignment, procedure, symbolic, float(bool(context))]
    def calibrate(self, raw: dict[str, float], moment: Moment, model: Calib) -> dict[str, float]:
        for value in raw.values():
            moment.add(value)
        return {key: model.predict(value) if model.fitted else moment.score(value) for key, value in raw.items()}
    def fit(self, view: str, scores: list[float], labels: list[int]) -> float:
        view = fold(view)
        models = {"semantic": self.semsup, "structure": self.strsup, "spread": self.actsup}
        if view not in models:
            raise ValueError("calibration view is invalid")
        return models[view].fit(scores, labels)
    def drift(self) -> dict[str, float]:
        return {"semantic": self.semsup.drift(), "structure": self.strsup.drift(), "spread": self.actsup.drift()}
    def search(self, query: str, context: list[str] | None = None, limit: int = 5, subtasks: list[str] | None = None, task: str = "", stamp: float = 0.0, fusion: str = "teacher") -> Trace:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError("limit must be an integer")
        if limit < 0:
            raise ValueError("limit must not be negative")
        if fusion not in ("teacher", "semantic", "structure", "equal"):
            raise ValueError("fusion mode is invalid")
        query = fold(query)
        if not query:
            raise ValueError("query must not be empty")
        subtasks = [fold(item) for item in (subtasks or [query]) if fold(item)]
        if not subtasks:
            subtasks = [query]
        context = [word(key) for key in (context or [])]
        context = list(dict.fromkeys(key for key in context if key in self.graph.tools))
        if not self.graph.tools or limit == 0:
            cold = not self.graph.edges
            weights = self.teacher.weights(None, cold) if fusion == "teacher" else {"semantic": [1.0, 0.0], "structure": [0.0, 1.0], "equal": [0.5, 0.5]}[fusion]
            return Trace(query, [], context, [0.0] * len(self.teacher.features), weights, cold, stamp, subtasks, task)
        alignment = Aligned.sync(self.graph, self.semindex, self.strindex)
        rows = [{key: (value + 1.0) / 2.0 for key, value in self.semindex.search(self.embedder.vector(part)).items() if key in self.graph.tools} for part in subtasks]
        rawsemantic = {key: max(row.get(key, 0.0) for row in rows) for key in self.graph.tools}
        anchors = list(context)
        for row in rows:
            if row:
                anchors.append(max(row, key=row.get))
        anchors = list(dict.fromkeys(anchors))[:max(2, len(subtasks))]
        rawstructure = {key: (value + 1.0) / 2.0 for key, value in self.strindex.search(anchors, self.graph).items()}
        seeds = {key: max(0.0, rawsemantic[key]) for key in anchors}
        rawspread = self.graph.spread(seeds)
        peak = max(rawspread.values(), default=0.0)
        rawspread = {key: rawspread.get(key, 0.0) / peak if peak > 0.0 else 0.0 for key in self.graph.tools}
        semantics = self.calibrate(rawsemantic, self.semcal, self.semsup)
        roles = self.calibrate(rawstructure, self.strcal, self.strsup)
        spreads = self.calibrate(rawspread, self.actcal, self.actsup)
        structures = {key: 0.5 * roles[key] + 0.5 * spreads[key] for key in self.graph.tools}
        base = min(max(1, limit), len(self.graph.tools))
        features = self.features(query, semantics, structures, context, base, alignment, subtasks)
        cold = features[5] < 0.25 or features[6] < 0.25 or features[7] < 1.0 or max(self.drift().values()) > 0.5
        weights = self.teacher.weights(features, cold) if fusion == "teacher" else {"semantic": [1.0, 0.0], "structure": [0.0, 1.0], "equal": [0.5, 0.5]}[fusion]
        budgets = self.teacher.budgets(weights, limit, len(self.graph.tools))
        views = (semantics, structures)
        selected = [set(sorted(view, key=view.get, reverse=True)[:budget]) for view, budget in zip(views, budgets)]
        candidates = set().union(*selected)
        hits: list[Hit] = []
        for key in candidates:
            tool = self.graph.tools[key]
            score = sum(weight * view[key] for weight, view in zip(weights, views))
            risk = 0.1 * max(0.0, 1.0 - tool.confidence) + 0.05 * max(0.0, -tool.quality)
            sources = [name for name, keys in zip(self.teacher.views, selected) if key in keys]
            hits.append(Hit(key, tool.name, tool.text, tool.code, score - risk, semantics[key], structures[key], spreads[key], sources))
        hits.sort(key=lambda hit: (-hit.score, hit.key))
        return Trace(query, hits[:min(limit, len(hits))], context, features, weights, cold, stamp, subtasks, task)
    def dump(self) -> dict:
        return {"semantic": self.semcal.dump(), "structure": self.strcal.dump(), "activation": self.actcal.dump(), "semsup": self.semsup.dump(), "strsup": self.strsup.dump(), "actsup": self.actsup.dump()}
    def restore(self, data: dict) -> None:
        self.semcal.restore(data.get("semantic", {}))
        self.strcal.restore(data.get("structure", {}))
        self.actcal.restore(data.get("activation", {}))
        self.semsup = Calib.load(data.get("semsup", {}))
        self.strsup = Calib.load(data.get("strsup", {}))
        self.actsup = Calib.load(data.get("actsup", {}))

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
