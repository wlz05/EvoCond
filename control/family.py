# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Conservative operator-family backoff for frozen CostBench trace memory.'
from __future__ import annotations
from dataclasses import dataclass
from statistics import mean
from control.bridge import EvoltBridge, MemoryEdge
DOMAINS = frozenset({"location", "transportation", "accommodation", "attraction", "dining", "shopping"})
def operator_family(name: str) -> tuple[str, str | None]:
    'Return the operator template and domain encoded in CostBench names.'
    parts = [part.casefold() for part in name.split("_") if part]
    domain = next((part for part in parts if part in DOMAINS), None)
    family = "_".join(part for part in parts if part not in DOMAINS)
    return family, domain
def _posterior(edges: list[MemoryEdge], unique_tasks: int) -> float:
    positive = sum(edge.positive for edge in edges)
    negative = sum(edge.negative for edge in edges)
    volume = positive + negative
    if volume > unique_tasks > 0:
        factor = unique_tasks / volume
        positive *= factor
        negative *= factor
    return (positive + 1.0) / (positive + negative + 2.0)
@dataclass(frozen=True)
class FamilyRouteScore:
    value: float
    exact_pairs: int
    family_pairs: int
    unknown_pairs: int
def score_route(memory: EvoltBridge, actions: tuple[str, ...]) -> FamilyRouteScore | None:
    'Shrink sparse family outcomes toward 0.5 and penalize missing edges.'
    if len(actions) < 2 or len(memory.tasks) < memory.min_tasks:
        return None
    values: list[float] = []
    exact = family = unknown = 0
    for source, target in zip(actions, actions[1:]):
        source_key, target_key = memory._key(source), memory._key(target)
        direct = [edge for edge in memory.edges if edge.kind == "supports" and edge.source == source_key and edge.target == target_key]
        direct_tasks = set().union(*(edge.tasks for edge in direct)) if direct else set()
        if len(direct_tasks) >= memory.min_tasks:
            values.append(_posterior(direct, len(direct_tasks)))
            exact += 1
            continue
        source_family, _ = operator_family(source)
        target_family, _ = operator_family(target)
        related = []
        for edge in memory.edges:
            if edge.kind != "supports":
                continue
            historical_source_family, historical_domain = operator_family(memory.tools[edge.source].name)
            historical_target_family, historical_target_domain = operator_family(memory.tools[edge.target].name)
            if (historical_source_family, historical_target_family) == (source_family, target_family) and historical_domain is not None and historical_domain == historical_target_domain:
                related.append(edge)
        tasks = set().union(*(edge.tasks for edge in related)) if related else set()
        domains = {operator_family(memory.tools[edge.source].name)[1] for edge in related}
        if len(tasks) >= memory.min_tasks and len(domains) >= 2:
            values.append(0.5 + 0.5 * (_posterior(related, len(tasks)) - 0.5))
            family += 1
        else:
            values.append(0.5)
            unknown += 1
    count = len(values)
    score = mean(values) - (0.12 * unknown + 0.04 * family) / count
    return FamilyRouteScore(score, exact, family, unknown)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
