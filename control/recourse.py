# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Small, deterministic research model for versioned evidence recourse.'
from __future__ import annotations
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from math import isclose, isfinite
from typing import Callable, Iterable, Mapping
@dataclass(frozen=True)
class Artifact:
    name: str
    kind: str
    value: str
    parents: frozenset[str] = frozenset()
    versions: tuple[tuple[str, int], ...] = ()
    verified: bool = True
@dataclass(frozen=True)
class ToolStep:
    name: str
    inputs: frozenset[str]
    outputs: frozenset[str]
    cost: float
Goal = Callable[[frozenset[str], Mapping[str, Artifact]], bool]
@dataclass(frozen=True)
class Scenario:
    probability: float
    context: Mapping[str, int]
    observations: frozenset[str]
    goal: Goal
@dataclass(frozen=True)
class Plan:
    actions: tuple[str, ...]
    cost: float
    artifacts: frozenset[str]
@dataclass(frozen=True)
class ProspectivePlan:
    plan: Plan
    expected_repair: float
    @property
    def expected_total(self) -> float:
        return self.plan.cost + self.expected_repair
@dataclass(frozen=True)
class EvidenceGraph:
    artifacts: Mapping[str, Artifact]
    tools: tuple[ToolStep, ...]
    _ordered_tools: tuple[ToolStep, ...] = field(init=False, repr=False)
    def __post_init__(self) -> None:
        for name, artifact in self.artifacts.items():
            if name != artifact.name:
                raise ValueError(f"Artifact key does not match its name: {name}")
            if not artifact.parents.issubset(self.artifacts):
                raise ValueError(f"Unknown parents for artifact {name}")
        checked: set[str] = set()
        visiting: set[str] = set()
        def check_acyclic(name: str) -> None:
            if name in checked:
                return
            if name in visiting:
                raise ValueError("Artifact provenance must be acyclic")
            visiting.add(name)
            for parent in self.artifacts[name].parents:
                check_acyclic(parent)
            visiting.remove(name)
            checked.add(name)
        for name in self.artifacts:
            check_acyclic(name)
        tool_names: set[str] = set()
        for tool in self.tools:
            if tool.name in tool_names:
                raise ValueError(f"Duplicate tool: {tool.name}")
            tool_names.add(tool.name)
            if not tool.inputs.issubset(self.artifacts) or not tool.outputs.issubset(self.artifacts):
                raise ValueError(f"Unknown artifact in tool {tool.name}")
            if not tool.outputs or not isfinite(tool.cost) or tool.cost < 0:
                raise ValueError(f"Invalid outputs or cost for tool {tool.name}")
        object.__setattr__(self, "_ordered_tools", tuple(sorted(self.tools, key=lambda tool: tool.name)))
    def valid(self, known: Iterable[str], context: Mapping[str, int]) -> frozenset[str]:
        'Keep only witnessed nodes whose recorded dependencies still hold.'
        present = frozenset(known)
        if not present.issubset(self.artifacts):
            raise ValueError("Unknown known artifact")
        cache: dict[str, bool] = {}
        visiting: set[str] = set()
        def check(name: str) -> bool:
            if name not in present:
                return False
            if name in cache:
                return cache[name]
            if name in visiting:
                raise ValueError("Artifact provenance must be acyclic")
            visiting.add(name)
            artifact = self.artifacts[name]
            compatible = artifact.verified and all(context.get(key) == version for key, version in artifact.versions)
            result = compatible and all(check(parent) for parent in artifact.parents)
            visiting.remove(name)
            cache[name] = result
            return result
        return frozenset(name for name in present if check(name))
    def _reachable(
        self,
        known: Iterable[str],
        context: Mapping[str, int],
        max_steps: int,
        max_states: int,
    ) -> Iterable[Plan]:
        if max_steps < 0 or max_states < 1:
            raise ValueError("Search limits must be nonnegative and nonzero")
        start = self.valid(known, context)
        serial = count()
        frontier: list[tuple[float, int, tuple[str, ...], int, frozenset[str]]] = []
        heappush(frontier, (0.0, 0, (), next(serial), start))
        best: dict[frozenset[str], tuple[float, int, tuple[str, ...]]] = {start: (0.0, 0, ())}
        while frontier:
            price, steps, actions, _, state = heappop(frontier)
            if best[state] != (price, steps, actions):
                continue
            yield Plan(actions, price, state)
            if steps == max_steps:
                continue
            for tool in self._ordered_tools:
                if not tool.inputs.issubset(state) or tool.outputs.issubset(state):
                    continue
                proposed = state | tool.outputs
                updated = self.valid(proposed, context)
                if not tool.outputs.issubset(updated):
                    continue
                rank = (price + tool.cost, steps + 1, (*actions, tool.name))
                if updated not in best or rank < best[updated]:
                    best[updated] = rank
                    if len(best) > max_states:
                        raise RuntimeError("Explicit search state limit reached")
                    heappush(frontier, (*rank, next(serial), updated))
    def shortest(
        self,
        known: Iterable[str],
        context: Mapping[str, int],
        goal: Goal,
        max_steps: int = 6,
        max_states: int = 4096,
    ) -> Plan | None:
        for plan in self._reachable(known, context, max_steps, max_states):
            if goal(plan.artifacts, self.artifacts):
                return plan
        return None
    def prospective(
        self,
        known: Iterable[str],
        context: Mapping[str, int],
        goal: Goal,
        scenarios: tuple[Scenario, ...],
        max_steps: int = 6,
        max_states: int = 4096,
    ) -> ProspectivePlan | None:
        'Choose the least costly plan in the explicit two-stage model.'
        winner: ProspectivePlan | None = None
        for candidate in self.prospective_candidates(known, context, goal, scenarios, max_steps, max_states):
            plan = candidate.plan
            if winner is None or (candidate.expected_total, plan.cost, plan.actions) < (
                winner.expected_total,
                winner.plan.cost,
                winner.plan.actions,
            ):
                winner = candidate
        return winner
    def prospective_candidates(
        self,
        known: Iterable[str],
        context: Mapping[str, int],
        goal: Goal,
        scenarios: tuple[Scenario, ...],
        max_steps: int = 6,
        max_states: int = 4096,
    ) -> Iterable[ProspectivePlan]:
        'Expose only hard-feasible plans for optional, non-authoritative reranking.'
        if not scenarios or any(not isfinite(s.probability) or s.probability < 0 for s in scenarios):
            raise ValueError("Scenarios need nonnegative finite probabilities")
        if not isclose(sum(s.probability for s in scenarios), 1.0, abs_tol=1e-9):
            raise ValueError("Scenario probabilities must sum to one")
        for plan in self._reachable(known, context, max_steps, max_states):
            if not goal(plan.artifacts, self.artifacts):
                continue
            expected_repair = 0.0
            for scenario in scenarios:
                if scenario.probability == 0:
                    continue
                repaired = self.shortest(
                    plan.artifacts | scenario.observations,
                    scenario.context,
                    scenario.goal,
                    max_steps,
                    max_states,
                )
                if repaired is None:
                    expected_repair = float("inf")
                    break
                expected_repair += scenario.probability * repaired.cost
            if isfinite(expected_repair):
                yield ProspectivePlan(plan, expected_repair)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
