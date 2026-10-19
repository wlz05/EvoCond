# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Maintain a typed evolving tool graph."""
from collections import defaultdict
from math import isfinite, sqrt
from core.embed import word
from core.model import Edge, Tool
class Graph:
    """Store tools and outcome-weighted relations."""
    kinds = ("precedes", "supports", "substitutes", "conflicts")
    def __init__(self, rate: float = 0.25, halflife: float = 128.0):
        if not isfinite(rate) or rate <= 0.0:
            raise ValueError("rate must be positive")
        if not isfinite(halflife) or halflife <= 0.0:
            raise ValueError("halflife must be positive")
        self.rate = rate
        self.halflife = halflife
        self.tools: dict[str, Tool] = {}
        self.edges: dict[tuple[str, str, str], Edge] = {}
        self.tasks: list[str] = []
        self.revision = 0
    def put(self, tool: Tool) -> None:
        tool.key = word(tool.key)
        tool.name = word(tool.name)
        tool.aliases = list(dict.fromkeys(word(item) for item in tool.aliases))
        numbers = [tool.quality, tool.created, tool.used, tool.confidence, *tool.vector]
        if any(not isfinite(value) for value in numbers) or not 0.0 <= tool.confidence <= 1.0:
            raise ValueError("tool values are invalid")
        if not isinstance(tool.trials, int) or tool.trials < 0 or not isinstance(tool.recent, int) or tool.recent < 0:
            raise ValueError("tool counts are invalid")
        if tool.state not in ("active", "pending", "retired"):
            raise ValueError("tool state is invalid")
        saved = self.tools.get(tool.key)
        if saved is None:
            self.tools[tool.key] = tool
        else:
            left = max(1, saved.trials)
            right = max(1, tool.trials)
            saved.quality = (saved.quality * left + tool.quality * right) / (left + right)
            saved.confidence = (saved.confidence * left + tool.confidence * right) / (left + right)
            saved.trials += tool.trials
            saved.aliases = list(dict.fromkeys(saved.aliases + tool.aliases + [tool.name]))
            saved.created = min(saved.created, tool.created)
            saved.recent += tool.recent
            if tool.used >= saved.used:
                saved.name = tool.name
                saved.text = tool.text
                saved.code = tool.code or saved.code
                saved.schema = tool.schema or saved.schema
                saved.vector = tool.vector
                saved.state = tool.state
            saved.used = max(saved.used, tool.used)
        self.revision += 1
    def drop(self, key: str) -> None:
        key = word(key)
        if self.tools.pop(key, None) is None:
            return
        for pair in list(self.edges):
            if key in pair[:2]:
                self.edges.pop(pair)
        self.revision += 1
    def fade(self, stamp: float | None = None) -> None:
        now = max((edge.used for edge in self.edges.values()), default=0.0) + 1.0 if stamp is None else stamp
        if not isfinite(now):
            raise ValueError("stamp must be finite")
        changed = False
        for pair, edge in list(self.edges.items()):
            if now <= edge.used:
                continue
            elapsed = max(0.0, now - edge.used)
            edge.fade(2.0 ** (-elapsed / self.halflife), now)
            changed = True
            if edge.positive + edge.negative < 1e-12:
                self.edges.pop(pair)
        self.revision += int(changed)
    def add(self, source: str, target: str, kind: str, value: float, stamp: float, task: str) -> None:
        source = word(source)
        target = word(target)
        if source == target or source not in self.tools or target not in self.tools:
            return
        if kind not in self.kinds:
            raise ValueError("edge kind is invalid")
        if not isfinite(value) or not isfinite(stamp) or not isinstance(task, str) or not task:
            raise ValueError("edge evidence must be finite")
        if task not in self.tasks:
            self.tasks.append(task)
        pair = (source, target, kind)
        edge = self.edges.setdefault(pair, Edge(source, target, kind, used=stamp))
        self.revision += int(edge.add(value, stamp, task))
    def link(self, sequence: list[str], reward: float = 1.0, stamp: float | None = None, task: str = "") -> None:
        if not isfinite(reward):
            raise ValueError("reward must be finite")
        if not isinstance(task, str):
            raise TypeError("task identity must be a string")
        now = max((tool.used for tool in self.tools.values()), default=0.0) + 1.0 if stamp is None else stamp
        if not isfinite(now):
            raise ValueError("stamp must be finite")
        keys = [word(key) for key in sequence]
        keys = list(dict.fromkeys(key for key in keys if key in self.tools))
        task = task or f"task{self.revision + 1}"
        fresh = task not in self.tasks and bool(keys)
        if fresh:
            self.tasks.append(task)
            self.fade(float(len(self.tasks)))
        edgeclock = float(self.tasks.index(task) + 1) if task in self.tasks else 0.0
        signal = max(-1.0, min(1.0, reward))
        for key in keys:
            self.tools[key].used = max(self.tools[key].used, now)
            self.tools[key].recent += 1
        self.revision += int(bool(keys))
        for pos, source in enumerate(keys):
            for end in range(pos + 1, len(keys)):
                target = keys[end]
                scale = 1.0 / (end - pos)
                self.add(source, target, "precedes", (0.25 + 0.75 * abs(signal)) * scale, edgeclock, task)
                if signal:
                    self.add(source, target, "supports", signal * self.rate * scale, edgeclock, task)
                if signal < 0.0:
                    self.add(source, target, "conflicts", -signal * self.rate * scale, edgeclock, task)
    def strength(self, key: str) -> float:
        key = word(key)
        return sum(abs(edge.weight) * edge.confidence for pair, edge in self.edges.items() if key in pair[:2])
    def relation(self, edge: Edge) -> float:
        frequency = sqrt(edge.count / max(1, len(self.tasks)))
        value = edge.weight * edge.confidence * frequency
        return -abs(value) if edge.kind == "conflicts" else value
    def shape(self, key: str, order: int = 3, damp: float = 0.7) -> list[float]:
        key = word(key)
        if not isinstance(order, int) or isinstance(order, bool) or order < 0:
            raise ValueError("order must not be negative")
        if not isfinite(damp) or not 0.0 <= damp <= 1.0:
            raise ValueError("damp must be within zero and one")
        keys = sorted(self.tools)
        if key not in self.tools:
            return [0.0] * (len(self.kinds) * 6 + order)
        size = max(1, len(keys) - 1)
        incoming = [0.0] * len(self.kinds)
        outgoing = [0.0] * len(self.kinds)
        paths = [0.0] * (len(self.kinds) ** 2)
        positions = {kind: pos for pos, kind in enumerate(self.kinds)}
        index = {name: pos for pos, name in enumerate(keys)}
        matrix = [[0.0] * len(keys) for _ in keys]
        for edge in self.edges.values():
            value = self.relation(edge)
            if edge.target == key:
                incoming[positions[edge.kind]] += abs(value)
            if edge.source == key:
                outgoing[positions[edge.kind]] += abs(value)
            if value > 0.0:
                matrix[index[edge.source]][index[edge.target]] += value
        for first in self.edges.values():
            if first.source != key:
                continue
            left = abs(self.relation(first))
            for second in self.edges.values():
                if second.source != first.target:
                    continue
                pos = positions[first.kind] * len(self.kinds) + positions[second.kind]
                paths[pos] += left * abs(self.relation(second)) / size
        state = [0.0] * len(keys)
        state[index[key]] = 1.0
        walks: list[float] = []
        for step in range(1, order + 1):
            state = [sum(state[left] * matrix[left][right] for left in range(len(keys))) for right in range(len(keys))]
            walks.append((damp ** step) * state[index[key]])
        return [value / size for value in incoming + outgoing] + paths + walks
    def spread(self, seeds: dict[str, float], depth: int = 2, damp: float = 0.55) -> dict[str, float]:
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
            raise ValueError("depth must not be negative")
        if not isfinite(damp) or not 0.0 <= damp <= 1.0:
            raise ValueError("damp must be within zero and one")
        if any(not isfinite(value) for value in seeds.values()):
            raise ValueError("seed values must be finite")
        clean = {word(key): max(0.0, value) for key, value in seeds.items()}
        scores = defaultdict(float, {key: value for key, value in clean.items() if key in self.tools})
        frontier = dict(scores)
        for step in range(depth):
            following: dict[str, float] = defaultdict(float)
            totals: dict[str, float] = defaultdict(float)
            degrees: dict[str, int] = defaultdict(int)
            for edge in self.edges.values():
                value = self.relation(edge)
                totals[edge.source] += abs(value)
                degrees[edge.source] += 1
                degrees[edge.target] += 1
            for edge in self.edges.values():
                if edge.source not in frontier or totals[edge.source] <= 0.0:
                    continue
                value = frontier[edge.source] * self.relation(edge) / totals[edge.source]
                value /= sqrt((degrees[edge.source] + 1) * (degrees[edge.target] + 1))
                following[edge.target] += value * (damp ** (step + 1))
            for key, value in following.items():
                scores[key] = max(0.0, scores[key] + value)
            frontier = {key: max(0.0, value) for key, value in following.items() if value > 0.0}
        return dict(scores)
    def forget(self, age: float, floor: float, stamp: float | None = None) -> list[str]:
        if not isfinite(age) or age < 0.0:
            raise ValueError("age must not be negative")
        if not isfinite(floor) or floor < 0.0:
            raise ValueError("floor must not be negative")
        now = max((tool.used for tool in self.tools.values()), default=0.0) + 1.0 if stamp is None else stamp
        if not isfinite(now):
            raise ValueError("stamp must be finite")
        removed: list[str] = []
        changed = False
        for key, tool in list(self.tools.items()):
            if self.strength(key) < floor and max(0.0, now - tool.used) > age and tool.recent == 0:
                removed.append(key)
                self.drop(key)
            else:
                changed = changed or tool.recent != 0
                tool.recent = 0
        self.revision += int(changed)
        return removed
    def dump(self) -> dict:
        return {"rate": self.rate, "halflife": self.halflife, "revision": self.revision, "tasks": self.tasks, "tools": [tool.dump() for tool in self.tools.values()], "edges": [edge.dump() for edge in self.edges.values()]}
    @classmethod
    def load(cls, data: dict) -> "Graph":
        graph = cls(float(data.get("rate", 0.25)), float(data.get("halflife", 128.0)))
        for item in data.get("tools", []):
            graph.put(Tool.load(item))
        for item in data.get("edges", []):
            edge = Edge.load(item)
            edge.source = word(edge.source)
            edge.target = word(edge.target)
            if edge.kind not in graph.kinds or edge.source not in graph.tools or edge.target not in graph.tools or edge.source == edge.target:
                continue
            edge.tasks = list(dict.fromkeys(str(task) for task in edge.tasks if str(task)))
            edge.count = len(edge.tasks)
            if any(not isfinite(value) or value < 0.0 for value in (edge.positive, edge.negative)) or not isfinite(edge.used) or edge.count < 0:
                continue
            pair = (edge.source, edge.target, edge.kind)
            saved = graph.edges.get(pair)
            if saved is None:
                graph.edges[pair] = edge
            else:
                saved.positive += edge.positive
                saved.negative += edge.negative
                saved.count += edge.count
                saved.tasks = list(dict.fromkeys(saved.tasks + edge.tasks))
                saved.count = len(saved.tasks)
                saved.used = max(saved.used, edge.used)
        tasks = data.get("tasks", [])
        graph.tasks = list(dict.fromkeys(str(task) for task in tasks if str(task))) if isinstance(tasks, list) else []
        graph.tasks = list(dict.fromkeys(graph.tasks + [task for edge in graph.edges.values() for task in edge.tasks]))
        graph.revision = max(graph.revision, int(data.get("revision", graph.revision)))
        return graph

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
