# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
from math import isfinite
from statistics import median
from typing import Any, Mapping, Protocol
from state.schema import Evidence, Scale
ISSUES = 4
WINDOW = 32
SILENT = 0.75
SHRINK = 0.5
TIMEOUT = 2
class Revising(Protocol):
    def revise(self, suspect: list[str], sharpness: float, shrink: float, step: int, remaining: int, timeout: int) -> Mapping[str, Any]:
        ...
class Challenger:
    def __init__(self, source: str = "rule"):
        value = str(source).strip().lower()
        if not value:
            raise ValueError("Challenger source must not be empty")
        self.source = value
    def revise(self, suspect: list[str], sharpness: float, shrink: float, step: int, remaining: int, timeout: int) -> dict[str, Any]:
        due = min(timeout, max(1, remaining))
        return {"width": sharpness * shrink / 2.0, "forbid": sorted(set(suspect)), "due": due, "source": self.source, "attempt": 1}
@dataclass(frozen=True)
class Health:
    scale: Scale
    score: float = 1.0
    coverage: float = 1.0
    integrity: float = 1.0
    expiry: float = 1.0
    relevance: float = 1.0
    nonvacuity: float = 1.0
    reference: float = 1.0
    silent: bool = False
    critical: bool = False
    issues: tuple[str, ...] = ()
class Audit:
    def __init__(self, window: int = WINDOW, silent: float = SILENT, shrink: float = SHRINK, timeout: int = TIMEOUT, challenger: Revising | None = None):
        if isinstance(window, bool) or not isinstance(window, int) or window < 2:
            raise ValueError("Audit window must exceed one")
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
            raise ValueError("Audit timeout must be positive")
        if not 0.0 <= float(silent) <= 1.0 or not 0.0 <= float(shrink) <= 1.0:
            raise ValueError("Audit rates must lie inside zero and one")
        self.window = window
        self.silentline = float(silent)
        self.shrink = float(shrink)
        self.timeout = timeout
        self.challenger = challenger or Challenger()
        self.trace: dict[Scale, deque[Evidence]] = {scale: deque(maxlen=window) for scale in Scale}
        self.orders: dict[Scale, dict[str, Any]] = {}
        self.attempts: dict[tuple[Scale, int], int] = {}
        self.pending: dict[Scale, tuple[int, int]] = {}
        self.before: dict[Scale, list[float]] = {scale: [] for scale in Scale}
        self.recovered: dict[Scale, int] = {scale: 0 for scale in Scale}
        self.outcomes: dict[Scale, deque[tuple[bool, bool]]] = {scale: deque(maxlen=window) for scale in Scale}
    def assess(self, evidence: Evidence) -> Health:
        coverage = evidence.observable / max(1, evidence.declared)
        integrity = float(evidence.integrity)
        expiry = float(not evidence.expired)
        relevance = float(evidence.relevant > 0)
        nonvacuity = 1.0 if not evidence.due else float(not evidence.flat and (evidence.progressed or evidence.waiting or bool(evidence.invariants)))
        reference = evidence.ready / max(1, evidence.observable)
        values = (coverage, integrity, expiry, relevance, nonvacuity, reference)
        score = sum(values) / len(values)
        issues: list[str] = []
        for name, value in zip(("coverage", "integrity", "expiry", "relevance", "vacuity", "reference"), values):
            if value < 1.0:
                issues.append(name)
        enough = len([item for item in self.trace[evidence.scale] if item.due]) >= ISSUES
        silent = enough and score < self.silentline
        critical = not evidence.integrity or evidence.expired or evidence.observable == 0 and evidence.relevant > 0
        return Health(evidence.scale, score, coverage, integrity, expiry, relevance, nonvacuity, reference, silent, critical, tuple(issues))
    def observe(self, evidence: Evidence) -> Health:
        self.trace[evidence.scale].append(evidence)
        health = self.assess(evidence)
        pending = self.pending.get(evidence.scale)
        if pending is not None and evidence.contract is not None and evidence.contract != pending[0] and evidence.due and not evidence.expired and health.score >= self.silentline:
            self.pending.pop(evidence.scale, None)
            self.recovered[evidence.scale] += 1
        return health
    def health(self, scale: Scale) -> Health:
        return Health(scale) if not self.trace[scale] else self.assess(self.trace[scale][-1])
    def suspect(self, scale: Scale) -> list[str]:
        keys: set[str] = set()
        for evidence in self.trace[scale]:
            keys.update(evidence.flat)
            keys.update(evidence.unverifiable)
            keys.update(evidence.missing)
        return sorted(keys)
    def sharpness(self, scale: Scale) -> float:
        values = [item.sharpness for item in self.trace[scale] if item.due and item.sharpness > 0.0]
        return median(values) if values else 0.0
    def canstart(self, scale: Scale, contract: int | None) -> bool:
        return contract is not None and self.attempts.get((scale, contract), 0) < 1 and scale not in self.pending
    def record(self, scale: Scale, contract: int, step: int, remaining: int) -> dict[str, Any]:
        if not self.canstart(scale, contract):
            raise RuntimeError("Audit attempt is exhausted")
        health = self.health(scale)
        self.before[scale].append(health.score)
        self.attempts[(scale, contract)] = 1
        suspect = self.suspect(scale)
        raw = self.challenger.revise(suspect, self.sharpness(scale), self.shrink, step, remaining, self.timeout)
        if not isinstance(raw, Mapping):
            raise TypeError("Challenger order must be a mapping")
        orders = dict(raw)
        if set(orders) - {"width", "forbid", "due", "source", "attempt"}:
            raise ValueError("Challenger order has forbidden fields")
        due = orders.get("due")
        width = orders.get("width")
        forbid = orders.get("forbid")
        source = str(orders.get("source", "")).strip().lower()
        if isinstance(due, bool) or not isinstance(due, int) or due < 1 or due > min(self.timeout, max(1, remaining)):
            raise ValueError("Challenger deadline is invalid")
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not isfinite(float(width)) or width < 0.0:
            raise ValueError("Challenger width is invalid")
        if not isinstance(forbid, (list, tuple)) or not set(suspect).issubset(set(map(str, forbid))) or not source or orders.get("attempt") != 1:
            raise ValueError("Challenger order is invalid")
        orders.update({"width": float(width), "forbid": sorted(set(map(str, forbid))), "source": source})
        deadline = step + due
        self.pending[scale] = (contract, deadline)
        self.orders[scale] = orders
        return dict(orders)
    def timedout(self, scale: Scale, step: int) -> bool:
        pending = self.pending.get(scale)
        return pending is not None and step >= pending[1]
    def close(self, scale: Scale) -> None:
        self.pending.pop(scale, None)
    def diagnose(self, scale: Scale, success: bool, fired: bool, frozen: bool = False) -> None:
        if not frozen:
            raise ValueError("Diagnostic outcomes require a frozen policy")
        self.outcomes[scale].append((bool(success), bool(fired)))
    def separation(self, scale: Scale) -> float | None:
        history = self.outcomes[scale]
        failures = [fired for success, fired in history if not success]
        successes = [fired for success, fired in history if success]
        if not failures or not successes:
            return None
        hit = (sum(failures) + 1.0) / (len(failures) + 2.0)
        false = (sum(successes) + 1.0) / (len(successes) + 2.0)
        return max(-1.0, min(1.0, hit - false))
    def report(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for scale in Scale:
            health = self.health(scale)
            data = asdict(health)
            data["scale"] = scale.value
            data.update({"audits": sum(key[0] is scale for key in self.attempts), "before": list(self.before[scale]), "orders": self.orders.get(scale), "pending": self.pending.get(scale), "recovered": self.recovered[scale], "separation": self.separation(scale)})
            result[scale.value] = data
        return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
