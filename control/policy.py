# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Mapping, Sequence
from control.audit import Audit, Health
from control.credit import Ledger
from control.router import Router
from state.schema import Decision, Evidence, Move, Scale, Snapshot
SIZE = 12
CAP = 8
LAMBDAS = (0.125, 0.25, 0.5, 1.0)
class EProcess:
    def __init__(self, margin: float, confidence: float):
        if not 0.0 <= margin <= 1.0 or not 0.0 < confidence < 1.0:
            raise ValueError("Sequential test rates are invalid")
        self.margin = margin
        self.confidence = confidence
        self.logwealth = [0.0 for _ in LAMBDAS]
        self.samples = 0
        self.delayed = 0
        self.fired = False
    def update(self, value: float, horizon: int) -> None:
        if horizon != 1:
            self.delayed += 1
            return
        bounded = max(-1.0, min(1.0, float(value)))
        self.logwealth = [wealth + rate * (bounded - self.margin) - rate ** 2 / 2.0 for wealth, rate in zip(self.logwealth, LAMBDAS)]
        self.samples += 1
    def logvalue(self) -> float:
        peak = max(self.logwealth)
        return peak + log(sum(exp(value - peak) for value in self.logwealth) / len(self.logwealth))
    def value(self) -> float:
        return exp(min(log(1e300), self.logvalue()))
    def alarm(self, minimum: int) -> bool:
        self.fired = self.fired or self.samples >= minimum and self.logvalue() >= -log(self.confidence)
        return self.fired
    def report(self, minimum: int) -> dict[str, float | int | bool]:
        return {"evalue": self.value(), "samples": self.samples, "delayed": self.delayed, "alarm": self.alarm(minimum)}
@dataclass
class Arm:
    wins: float = 1.0
    losses: float = 1.0
    cost: float = 0.0
    trials: int = 0
    mass: float = 0.0
    def rate(self) -> float:
        return self.wins / (self.wins + self.losses)
    def update(self, success: bool, cost: float, weight: float = 1.0) -> None:
        self.wins += weight * float(success)
        self.losses += weight * float(not success)
        self.trials += 1
        self.mass += weight
        self.cost += weight * (cost - self.cost) / self.mass
class Ridge:
    def __init__(self, size: int = SIZE, penalty: float = 1.0, forgetting: float = 0.99):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("Ridge size must be positive")
        if not isfinite(float(penalty)) or float(penalty) <= 0.0 or not isfinite(float(forgetting)) or not 0.0 < float(forgetting) <= 1.0:
            raise ValueError("Ridge rates are invalid")
        self.size = size
        self.forgetting = forgetting
        self.weight = [0.0] * size
        self.matrix = [[penalty if row == column else 0.0 for column in range(size)] for row in range(size)]
    def predict(self, features: Sequence[float]) -> float:
        return max(-1.0, min(1.0, sum(weight * value for weight, value in zip(self.weight, features))))
    def update(self, features: Sequence[float], target: float) -> None:
        if len(features) != self.size:
            raise ValueError("Signature size is invalid")
        product = [sum(entry * value for entry, value in zip(row, features)) for row in self.matrix]
        divisor = self.forgetting + sum(value * item for value, item in zip(features, product))
        if abs(divisor) < 1e-9:
            return
        gain = [item / divisor for item in product]
        error = max(-1.0, min(1.0, target)) - self.predict(features)
        self.weight = [weight + item * error for weight, item in zip(self.weight, gain)]
        self.matrix = [[(entry - item * product[column]) / self.forgetting for column, entry in enumerate(row)] for row, item in zip(self.matrix, gain)]
class Scheduler:
    def __init__(self, margins: Mapping[Scale, float] | None = None, samples: int = 2, dwell: int = 1, inertia: float = 0.05, discount: float = 1.0, confidence: float = 0.05, channels: str = "vdh", auditing: bool = True, audit: Audit | None = None, ledger: Ledger | None = None, router: Router | None = None):
        self.margins = {scale: 0.1 for scale in Scale}
        for rawscale, rawvalue in (margins or {}).items():
            scale = rawscale if isinstance(rawscale, Scale) else Scale(str(rawscale).strip().lower())
            value = float(rawvalue)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("Forecast margins must lie inside zero and one")
            self.margins[scale] = value
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError("Sequential samples must be positive")
        if isinstance(dwell, bool) or not isinstance(dwell, int) or dwell < 0:
            raise ValueError("Dwell must be nonnegative")
        if not isfinite(float(inertia)) or float(inertia) < 0.0:
            raise ValueError("Inertia must be finite and nonnegative")
        if not isfinite(float(discount)) or not 0.0 < float(discount) <= 1.0:
            raise ValueError("Discount must lie inside zero and one")
        if not isfinite(float(confidence)) or not 0.0 < float(confidence) < 1.0:
            raise ValueError("Confidence must lie inside zero and one")
        selected = "".join(dict.fromkeys(str(channels).strip().lower()))
        if not selected or set(selected) - set("vdh"):
            raise ValueError("Evidence channels are invalid")
        if not isinstance(auditing, bool):
            raise TypeError("Auditing flag must be boolean")
        self.samples = samples
        self.dwell = dwell
        self.inertia = float(inertia)
        self.discount = float(discount)
        self.channels = selected
        self.auditing = auditing
        self.audit = audit or Audit()
        self.ledger = ledger or Ledger()
        self.router = router
        self.trend: dict[tuple[Scale, int], deque[float]] = {}
        self.tests = {scale: EProcess(self.margins[scale], float(confidence)) for scale in Scale}
        self.seen: set[tuple[Scale, int]] = set()
        self.last = Move.CONTINUE
        self.laststep = -1000000
        self.lastscale: Scale | None = None
        self.prior = {Move.CONTINUE: 0.5, Move.REPAIR: 0.9, Move.BRIDGE: 0.9, Move.GLOBAL: 0.95, Move.AUDIT: 0.9}
        self.arms = {move: Arm(self.prior[move], 1.0 - self.prior[move]) for move in Move}
        self.models = {move: Ridge() for move in Move}
        self.effect = {Move.CONTINUE: {Scale.LOCAL: 0.0, Scale.BRIDGE: 0.0, Scale.GLOBAL: 0.0}, Move.REPAIR: {Scale.LOCAL: 0.9, Scale.BRIDGE: 0.0, Scale.GLOBAL: 0.0}, Move.BRIDGE: {Scale.LOCAL: 0.4, Scale.BRIDGE: 0.9, Scale.GLOBAL: 0.0}, Move.GLOBAL: {Scale.LOCAL: 0.7, Scale.BRIDGE: 0.85, Scale.GLOBAL: 0.95}, Move.AUDIT: {Scale.LOCAL: 0.0, Scale.BRIDGE: 0.0, Scale.GLOBAL: 0.0}}
        self.basecost = {Move.CONTINUE: 0.0, Move.REPAIR: 0.05, Move.BRIDGE: 0.2, Move.GLOBAL: 0.45, Move.AUDIT: 0.15}
        self.locks: dict[Scale, int] = {}
        self.pending: tuple[Move, tuple[float, ...], float] | None = None
        self.initial: int | None = None
    def collect(self, evidence: Mapping[Scale, Evidence]) -> None:
        if "d" not in self.channels:
            return
        for scale, item in evidence.items():
            if item.contract is None or not item.supported() or (scale, item.contract) in self.seen:
                continue
            self.trend.setdefault((scale, item.horizon), deque(maxlen=max(8, self.samples * 2))).append(item.difference)
            self.tests[scale].update(item.difference, item.horizon)
            self.seen.add((scale, item.contract))
    def pressure(self, evidence: Mapping[Scale, Evidence]) -> dict[Scale, float]:
        self.collect(evidence)
        result: dict[Scale, float] = {}
        for scale in Scale:
            item = evidence.get(scale, Evidence(scale))
            hard = min(1.0, max(0.0, item.violation)) if "v" in self.channels and item.due else 0.0
            values = list(self.trend.get((scale, item.horizon), ())) [-self.samples:]
            average = sum(values) / len(values) if len(values) >= self.samples else 0.0
            forecast = max(0.0, average - self.margins[scale]) if "d" in self.channels else 0.0
            if "d" in self.channels and item.horizon == 1 and self.tests[scale].alarm(self.samples):
                forecast = 1.0
            result[scale] = max(hard, forecast * self.discount ** min(item.horizon, CAP))
        return result
    def sequential(self) -> dict[str, dict[str, float | int | bool]]:
        return {scale.value: self.tests[scale].report(self.samples) for scale in Scale}
    def signature(self, evidence: Mapping[Scale, Evidence], health: Mapping[Scale, Health], snapshot: Snapshot) -> tuple[float, ...]:
        violations = [evidence.get(scale, Evidence(scale)).violation if "v" in self.channels else 0.0 for scale in Scale]
        differences = [max(-1.0, min(1.0, evidence.get(scale, Evidence(scale)).difference)) if "d" in self.channels else 0.0 for scale in Scale]
        scores = [health.get(scale, Health(scale)).score if "h" in self.channels else 0.0 for scale in Scale]
        if self.initial is None:
            self.initial = max(1, snapshot.budget.tokens)
        features = (1.0, *violations, *differences, *scores, snapshot.budget.tokens / self.initial, min(1.0, snapshot.step / 64.0))
        if len(features) != SIZE:
            raise ValueError("Signature layout is invalid")
        return features
    def urgent(self, snapshot: Snapshot) -> bool:
        events = {"".join(character for character in str(item).strip().lower() if character.isalnum()) for item in snapshot.history[-1].get("events", ())} if snapshot.history else set()
        safe = snapshot.environment.get("safe")
        unsafe = safe is False or isinstance(safe, str) and safe.strip().lower() in {"false", "no", "unsafe", "0"}
        return bool(events & {"goal", "goalchange", "policy", "policychange", "permission", "permissionchange", "safety", "safetychange"}) or unsafe
    def lock(self, scale: Scale, version: int | None) -> None:
        if version is not None:
            self.locks[scale] = version
    def blocked(self, scale: Scale, evidence: Mapping[Scale, Evidence]) -> bool:
        version = self.locks.get(scale)
        if version is None:
            return False
        item = evidence.get(scale)
        if item is not None and item.contract != version:
            self.locks.pop(scale, None)
            return False
        return True
    def fallback(self, scale: Scale) -> tuple[Move, Scale | None]:
        return (Move.BRIDGE, scale) if scale is Scale.LOCAL else (Move.GLOBAL, scale)
    def candidates(self, pressure: Mapping[Scale, float], evidence: Mapping[Scale, Evidence], health: Mapping[Scale, Health], step: int) -> list[tuple[Move, Scale | None]]:
        items: list[tuple[Move, Scale | None]] = [(Move.CONTINUE, None)]
        failsafe = False
        if pressure[Scale.GLOBAL] > 0.0:
            items.append((Move.GLOBAL, Scale.GLOBAL))
        elif pressure[Scale.BRIDGE] > 0.0:
            items.extend(((Move.BRIDGE, Scale.BRIDGE), (Move.GLOBAL, Scale.BRIDGE)))
        elif pressure[Scale.LOCAL] > 0.0:
            items.extend(((Move.REPAIR, Scale.LOCAL), (Move.BRIDGE, Scale.LOCAL)))
        for scale in Scale:
            item = evidence.get(scale, Evidence(scale))
            state = health.get(scale, Health(scale))
            unhealthy = "h" in self.channels and self.auditing and (state.critical or item.due and state.silent)
            if item.contract is None or pressure[scale] > 0.0 or self.blocked(scale, evidence) or not unhealthy:
                continue
            if self.audit.canstart(scale, item.contract):
                items.append((Move.AUDIT, scale))
            elif self.audit.timedout(scale, step):
                items.append(self.fallback(scale))
                failsafe = True
        unique = list(dict.fromkeys(items))
        return [item for item in unique if item[0] is not Move.CONTINUE] if failsafe else unique
    def reduction(self, move: Move, pressure: Mapping[Scale, float], features: Sequence[float]) -> float:
        prior = sum(pressure[scale] * self.effect[move][scale] for scale in Scale)
        return max(0.0, prior + self.models[move].predict(features))
    def value(self, move: Move, scale: Scale | None, pressure: Mapping[Scale, float], features: Sequence[float], health: Mapping[Scale, Health]) -> float:
        reliability = max(0.2, self.arms[move].rate())
        if move is Move.AUDIT and scale is not None:
            state = health.get(scale, Health(scale))
            gain = max(1.0 - state.score, float(state.critical))
            return gain * reliability - self.basecost[move] - self.arms[move].cost
        return self.reduction(move, pressure, features) * reliability - self.basecost[move] - self.arms[move].cost
    def settle(self, total: float) -> None:
        if self.pending is None:
            return
        move, features, before = self.pending
        self.models[move].update(features, before - total)
        self.pending = None
    def select(self, evidence: Mapping[Scale, Evidence], health: Mapping[Scale, Health], snapshot: Snapshot) -> Decision:
        pressure = self.pressure(evidence)
        total = sum(pressure.values())
        self.settle(total)
        features = self.signature(evidence, health, snapshot)
        candidates = self.candidates(pressure, evidence, health, snapshot.step)
        routed = {} if self.router is None else self.router.scores(features)
        scores: dict[str, float] = {}
        for move, scale in candidates:
            key = move.value if scale is None else f"{move.value}:{scale.value}"
            cost = self.inertia if move is not Move.CONTINUE and move is not self.last else 0.0
            scores[key] = self.value(move, scale, pressure, features, health) + routed.get(key, 0.0) - cost
        move, scale = max(candidates, key=lambda item: scores[item[0].value if item[1] is None else f"{item[0].value}:{item[1].value}"])
        key = move.value if scale is None else f"{move.value}:{scale.value}"
        score = scores[key]
        urgent = self.urgent(snapshot)
        if urgent:
            move, scale, score = Move.GLOBAL, Scale.GLOBAL, 1.0
            scores[f"{Move.GLOBAL.value}:{Scale.GLOBAL.value}"] = max(scores.get(f"{Move.GLOBAL.value}:{Scale.GLOBAL.value}", 0.0), score)
        elif move is not Move.CONTINUE and (Move.CONTINUE, None) in candidates and snapshot.step - self.laststep < self.dwell:
            move, scale, score = Move.CONTINUE, None, scores.get(Move.CONTINUE.value, 0.0)
        if move is not Move.CONTINUE:
            self.laststep = snapshot.step
        self.last = move
        self.lastscale = scale
        self.pending = (move, features, total)
        reasons = tuple(f"v:{item.value}" for item in Scale if "v" in self.channels and evidence.get(item, Evidence(item)).violation > 0.0)
        reasons += tuple(f"d:{item.value}" for item in Scale if "d" in self.channels and pressure[item] > (evidence.get(item, Evidence(item)).violation if "v" in self.channels else 0.0))
        reasons += tuple(f"h:{item.value}" for item in Scale if "h" in self.channels and (health.get(item, Health(item)).silent or health.get(item, Health(item)).critical))
        if urgent:
            reasons += ("event",)
        return Decision(move, scale, score, scores, reasons)
    def learn(self, move: Move, scale: Scale | None, support: float, success: bool, cost: float) -> None:
        if not isinstance(success, bool):
            raise TypeError("Feedback success must be boolean")
        value = float(cost)
        if not isfinite(value) or value < 0.0:
            raise ValueError("Feedback cost must be finite and nonnegative")
        self.arms[move].update(success, value, max(0.05, min(1.0, float(support))))

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
