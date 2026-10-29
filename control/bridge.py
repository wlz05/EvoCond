# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Offline, read-only EVOLT signals for EvoCond evidence-graph planning.'
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from hashlib import blake2b
from math import isfinite, log, sqrt
import re
from statistics import mean
from typing import Any, Iterable, Mapping
import unicodedata
from control.recourse import EvidenceGraph, Goal, ProspectivePlan, Scenario
def _vector(text: str, dim: int) -> tuple[float, ...]:
    'Use the deterministic hashing formula in EVOLT/core/embed.py.'
    folded = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    terms: list[str] = []
    for chunk in re.findall(r"[^\W_]+", folded, flags=re.UNICODE):
        terms.append(chunk)
        span = 3 if chunk.isascii() else 2
        if len(chunk) > span:
            terms.extend(chunk[pos:pos + span] for pos in range(len(chunk) - span + 1))
    values = [0.0] * dim
    for term, count in Counter(terms).items():
        digest = blake2b(term.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        values[index] += sign * sqrt(count) * (1.0 + log(1.0 + len(term)))
    size = sqrt(sum(value * value for value in values))
    return tuple(value / size for value in values) if size else tuple(values)
def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("Vector dimensions differ")
    left_size = sqrt(sum(value * value for value in left))
    right_size = sqrt(sum(value * value for value in right))
    if left_size == 0.0 or right_size == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_size * right_size)
@dataclass(frozen=True)
class MemoryTool:
    key: str
    name: str
    vector: tuple[float, ...]
@dataclass(frozen=True)
class MemoryEdge:
    source: str
    target: str
    kind: str
    positive: float
    negative: float
    tasks: frozenset[str]
    def relation(self, total_tasks: int) -> float:
        "EVOLT's confidence- and task-frequency-weighted edge relation."
        volume = self.positive + self.negative
        weight = (self.positive - self.negative) / (volume + 2.0)
        confidence = volume / (volume + 2.0)
        frequency = sqrt(len(self.tasks) / max(1, total_tasks))
        value = weight * confidence * frequency
        return -abs(value) if self.kind == "conflicts" else value
@dataclass(frozen=True)
class Signal:
    semantic: float
    structural: float
    fused: float
    supported_pairs: int
@dataclass(frozen=True)
class Choice:
    selected: ProspectivePlan | None
    cost_only: ProspectivePlan | None
    used_memory: bool
    reason: str
    signals: Mapping[tuple[str, ...], Signal]
class EvoltBridge:
    'Align exact tool identities and compare hard-feasible recourse plans.'
    def __init__(
        self,
        tools: Mapping[str, MemoryTool],
        edges: Iterable[MemoryEdge],
        tasks: Iterable[str],
        tool_map: Mapping[str, str] | None = None,
        semantic_weight: float = 0.5,
        min_tasks: int = 3,
    ) -> None:
        if not isfinite(semantic_weight) or not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("Semantic weight must be within [0, 1]")
        if isinstance(min_tasks, bool) or not isinstance(min_tasks, int) or min_tasks < 1:
            raise ValueError("Minimum task support must be positive")
        self.tools = dict(tools)
        self.edges = tuple(edges)
        self.tasks = frozenset(tasks)
        self.tool_map = dict(tool_map or {})
        self.semantic_weight = semantic_weight
        self.min_tasks = min_tasks
        dimensions = {len(tool.vector) for tool in self.tools.values()}
        if len(dimensions) > 1 or any(not isfinite(value) for tool in self.tools.values() for value in tool.vector):
            raise ValueError("EVOLT tool vectors must have one finite dimension")
        self.dim = next(iter(dimensions), 0)
        names: dict[str, str] = {}
        ambiguous: set[str] = set()
        for key, tool in self.tools.items():
            if key != tool.key or not tool.name:
                raise ValueError("EVOLT tool identity is invalid")
            if tool.name in names:
                ambiguous.add(tool.name)
            names[tool.name] = key
        self.by_name = {name: key for name, key in names.items() if name not in ambiguous}
        if any(key not in self.tools for key in self.tool_map.values()):
            raise ValueError("Explicit tool mapping references an unknown EVOLT key")
        for edge in self.edges:
            if edge.source not in self.tools or edge.target not in self.tools or edge.kind not in {"supports", "conflicts"}:
                raise ValueError("Unsupported EVOLT edge")
            if any(not isfinite(value) or value < 0.0 for value in (edge.positive, edge.negative)):
                raise ValueError("EVOLT edge evidence must be finite and nonnegative")
            if not edge.tasks.issubset(self.tasks):
                raise ValueError("EVOLT edge cites tasks absent from the frozen snapshot")
    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        tool_map: Mapping[str, str] | None = None,
        semantic_weight: float = 0.5,
        min_tasks: int = 3,
    ) -> "EvoltBridge":
        'Read EVOLT Agent.dump() or Graph.dump() without importing EVOLT.'
        graph = state.get("graph", state)
        if not isinstance(graph, Mapping):
            raise ValueError("EVOLT graph is missing")
        raw_tools = graph.get("tools", ())
        raw_edges = graph.get("edges", ())
        raw_tasks = graph.get("tasks", ())
        if not isinstance(raw_tools, list) or not isinstance(raw_edges, list) or not isinstance(raw_tasks, list):
            raise ValueError("EVOLT graph lists are malformed")
        tools: dict[str, MemoryTool] = {}
        for raw in raw_tools:
            if not isinstance(raw, Mapping) or raw.get("state", "active") != "active":
                continue
            key, name, vector = raw.get("key"), raw.get("name"), raw.get("vector")
            if not isinstance(key, str) or not isinstance(name, str) or not isinstance(vector, list):
                raise ValueError("EVOLT tool record is malformed")
            if key in tools or not vector or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in vector):
                raise ValueError("EVOLT tool vector or key is invalid")
            tools[key] = MemoryTool(key, name, tuple(float(v) for v in vector))
        edges: list[MemoryEdge] = []
        for raw in raw_edges:
            if not isinstance(raw, Mapping):
                raise ValueError("EVOLT edge record is malformed")
            sources = raw.get("tasks", ())
            if not isinstance(sources, list) or any(not isinstance(item, str) for item in sources):
                raise ValueError("EVOLT edge task identities are malformed")
            if not set(sources).issubset(raw_tasks):
                raise ValueError("EVOLT edge cites tasks absent from the frozen snapshot")
            if raw.get("kind") not in {"supports", "conflicts"}:
                continue  # `precedes` is association, never a hard dependency.
            source, target = raw.get("source"), raw.get("target")
            if source not in tools or target not in tools:
                continue
            edges.append(MemoryEdge(source, target, str(raw["kind"]), float(raw.get("positive", 0.0)), float(raw.get("negative", 0.0)), frozenset(sources)))
        if any(not isinstance(task, str) for task in raw_tasks):
            raise ValueError("EVOLT task identities are malformed")
        return cls(tools, edges, raw_tasks, tool_map, semantic_weight, min_tasks)
    def _key(self, action: str) -> str | None:
        return self.tool_map.get(action, self.by_name.get(action))
    def signal(self, query: str, actions: tuple[str, ...]) -> Signal | None:
        if not self.dim or not query.strip() or not actions or len(self.tasks) < self.min_tasks:
            return None
        keys = [self._key(action) for action in actions]
        if any(key is None for key in keys):
            return None
        mapped = [str(key) for key in keys]
        qvec = _vector(query, self.dim)
        semantic = mean((1.0 + _cosine(qvec, self.tools[key].vector)) / 2.0 for key in mapped)
        pairs = list(zip(mapped, mapped[1:]))
        relations: list[float] = []
        supported = 0
        for source, target in pairs:
            matching = [edge for edge in self.edges if edge.source == source and edge.target == target and len(edge.tasks) >= self.min_tasks]
            supported += int(bool(matching))
            relations.append(max(-1.0, min(1.0, sum(edge.relation(len(self.tasks)) for edge in matching))))
        if not supported:
            return None
        structural = mean((1.0 + value) / 2.0 for value in relations)
        fused = self.semantic_weight * semantic + (1.0 - self.semantic_weight) * structural
        return Signal(semantic, structural, fused, supported)
    def choose(
        self,
        graph: EvidenceGraph,
        known: Iterable[str],
        context: Mapping[str, int],
        goal: Goal,
        scenarios: tuple[Scenario, ...],
        query: str,
        *,
        max_extra_cost: float = 0.0,
        max_steps: int = 6,
        max_states: int = 4096,
    ) -> Choice:
        'Rerank only verified-model-feasible plans inside a cost envelope.'
        if not isfinite(max_extra_cost) or max_extra_cost < 0.0:
            raise ValueError("Additional cost allowance must be nonnegative")
        candidates = list(graph.prospective_candidates(known, context, goal, scenarios, max_steps, max_states))
        if not candidates:
            return Choice(None, None, False, "no_hard_feasible_plan", {})
        baseline = min(candidates, key=lambda item: (item.expected_total, item.plan.cost, item.plan.actions))
        eligible = [item for item in candidates if item.expected_total <= baseline.expected_total + max_extra_cost + 1e-9]
        if len(eligible) < 2:
            return Choice(baseline, baseline, False, "one_cost_eligible_plan", {})
        signals = {item.plan.actions: self.signal(query, item.plan.actions) for item in eligible}
        if any(value is None for value in signals.values()):
            return Choice(baseline, baseline, False, "cold_or_unaligned_memory", {})
        ready = {key: value for key, value in signals.items() if value is not None}
        selected = min(eligible, key=lambda item: (-ready[item.plan.actions].fused, item.expected_total, item.plan.actions))
        return Choice(selected, baseline, selected is not baseline, "memory_rerank" if selected is not baseline else "cost_only_retained", ready)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
