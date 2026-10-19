# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from enum import Enum
from math import isfinite
from numbers import Real
from typing import Any, Mapping
from state.data import canonical, isfield, lower, normalize, walk
ALPHA = 0.2
class Scale(str, Enum):
    LOCAL = "ll"
    BRIDGE = "gl"
    GLOBAL = "gg"
class Move(str, Enum):
    CONTINUE = "continue"
    REPAIR = "repair"
    BRIDGE = "bridge"
    GLOBAL = "global"
    AUDIT = "audit"
class Kind(str, Enum):
    INTERVAL = "interval"
    CATEGORICAL = "categorical"
class View(str, Enum):
    STATE = "state"
    DERIVED = "derived"
    NONE = "none"
def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    return result if isfinite(result) else None
def label(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value).strip().lower()
@dataclass(frozen=True)
class Budget:
    tokens: int = 100000
    seconds: float = 3600.0
    tools: float = 100.0
    def __post_init__(self) -> None:
        if isinstance(self.tokens, bool) or not isinstance(self.tokens, int) or self.tokens < 0:
            raise ValueError("Token budget must be a nonnegative integer")
        for name, value in (("seconds", self.seconds), ("tools", self.tools)):
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or value < 0:
                raise ValueError(f"{name.capitalize()} budget must be finite and nonnegative")
    def spend(self, tokens: int = 0, seconds: float = 0.0, tools: float = 0.0) -> "Budget":
        Budget(tokens, seconds, tools)
        if tokens > self.tokens or seconds > self.seconds or tools > self.tools:
            raise RuntimeError("Budget would be exceeded")
        return Budget(self.tokens - tokens, self.seconds - seconds, self.tools - tools)
    def ratio(self, initial: "Budget") -> dict[str, float]:
        return {"tokens": self.tokens / max(1, initial.tokens), "seconds": self.seconds / max(1.0, initial.seconds), "tools": self.tools / max(1.0, initial.tools)}
@dataclass(frozen=True)
class Prediction:
    kind: Kind
    lower: float = 0.0
    upper: float = 0.0
    alpha: float = ALPHA
    shares: Mapping[str, float] = field(default_factory=dict)
    verifier: View = View.STATE
    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, Kind) else Kind(str(self.kind).strip().lower())
        verifier = self.verifier if isinstance(self.verifier, View) else View(str(self.verifier).strip().lower())
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real):
            raise TypeError("Prediction alpha must be numeric")
        alpha = float(self.alpha)
        if not isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("Prediction alpha must lie inside zero and one")
        if kind is Kind.INTERVAL:
            lower = numeric(self.lower)
            upper = numeric(self.upper)
            if lower is None or upper is None or upper < lower:
                raise ValueError("Interval predictions need ordered finite bounds")
            if self.shares:
                raise ValueError("Interval predictions cannot carry shares")
            object.__setattr__(self, "lower", lower)
            object.__setattr__(self, "upper", upper)
        else:
            if not isinstance(self.shares, Mapping) or not self.shares:
                raise ValueError("Categorical predictions need shares")
            shares: dict[str, float] = {}
            for key, value in self.shares.items():
                share = numeric(value)
                name = label(key)
                if not name or share is None or share < 0.0:
                    raise ValueError("Categorical shares must be finite and nonnegative")
                shares[name] = shares.get(name, 0.0) + share
            total = sum(shares.values())
            if total <= 0.0:
                raise ValueError("Categorical shares must have positive mass")
            object.__setattr__(self, "shares", {key: value / total for key, value in shares.items()})
            object.__setattr__(self, "lower", 0.0)
            object.__setattr__(self, "upper", 0.0)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "verifier", verifier)
        object.__setattr__(self, "alpha", alpha)
    def width(self) -> float:
        return self.upper - self.lower if self.kind is Kind.INTERVAL else 0.0
    def accepts(self, target: Any) -> bool:
        if self.kind is Kind.INTERVAL:
            value = numeric(target)
            return value is not None and self.lower <= value <= self.upper
        name = label(target)
        best = max(self.shares.values())
        return self.shares.get(name, 0.0) == best and best > 0.0
    @classmethod
    def cast(cls, value: Any, width: float = 0.0, alpha: float = ALPHA, verifier: View = View.STATE) -> "Prediction":
        if isinstance(value, Prediction):
            return value
        view = verifier if isinstance(verifier, View) else View(str(verifier).strip().lower())
        if isinstance(value, Mapping) and "kind" in value:
            kind = Kind(str(value["kind"]).strip().lower())
            inner = View(str(value.get("verifier", view.value)).strip().lower())
            if kind is Kind.INTERVAL:
                return cls(kind, value.get("lower", 0.0), value.get("upper", 0.0), alpha, {}, inner)
            shares = value.get("shares", {})
            if not isinstance(shares, Mapping):
                raise TypeError("Categorical shares must be a mapping")
            return cls(kind, 0.0, 0.0, alpha, shares, inner)
        if isinstance(value, bool):
            return cls(Kind.CATEGORICAL, shares={label(value): 1.0}, alpha=alpha, verifier=view)
        number = numeric(value)
        if number is not None:
            half = numeric(width)
            if half is None or half < 0.0:
                raise ValueError("Prediction width must be nonnegative")
            return cls(Kind.INTERVAL, number - half, number + half, alpha, {}, view)
        if not isinstance(value, str) or not value.strip():
            raise TypeError("Prediction values must be scalar or structured")
        return cls(Kind.CATEGORICAL, shares={label(value): 1.0}, alpha=alpha, verifier=view)
    def export(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind.value, "alpha": self.alpha, "verifier": self.verifier.value}
        if self.kind is Kind.INTERVAL:
            data.update({"lower": self.lower, "upper": self.upper})
        else:
            data["shares"] = dict(self.shares)
        return data
@dataclass(frozen=True)
class Contract:
    scale: Scale
    version: int
    parent: int | None
    created: int
    due: int
    assumptions: tuple[str, ...] = ()
    guarantees: tuple[str, ...] = ()
    predictions: Mapping[str, Prediction] = field(default_factory=dict)
    alpha: float = ALPHA
    progress: str = ""
    invariants: tuple[str, ...] = ()
    wait: bool = False
    def __post_init__(self) -> None:
        scale = self.scale if isinstance(self.scale, Scale) else Scale(str(self.scale).strip().lower())
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("Contract version must be positive")
        if self.parent is not None and (isinstance(self.parent, bool) or not isinstance(self.parent, int) or self.parent < 1):
            raise ValueError("Contract parent must be positive")
        if isinstance(self.created, bool) or isinstance(self.due, bool) or not isinstance(self.created, int) or not isinstance(self.due, int) or self.created < 0 or self.due <= self.created:
            raise ValueError("Contract time bounds are invalid")
        if not isinstance(self.predictions, Mapping) or not self.predictions:
            raise ValueError("Contract predictions must not be empty")
        alpha = float(self.alpha)
        if not isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("Contract alpha must lie inside zero and one")
        predictions: dict[str, Prediction] = {}
        for rawkey, rawvalue in self.predictions.items():
            key = canonical(rawkey)
            item = Prediction.cast(rawvalue, 0.0, alpha)
            if not key or item.alpha != alpha:
                raise ValueError("Contract prediction is invalid")
            predictions[key] = item
        progress = "dataset" if isfield(self.progress) else str(self.progress).strip()
        invariants = tuple("dataset" if isfield(item) else str(item).strip() for item in self.invariants)
        if progress and (progress not in predictions or predictions[progress].verifier is View.NONE):
            raise ValueError("Progress must reference an observable prediction")
        if any(not key or key not in predictions or predictions[key].verifier is View.NONE for key in invariants):
            raise ValueError("Invariants must reference observable predictions")
        if progress and progress in invariants:
            raise ValueError("Progress and invariants must be disjoint")
        if not isinstance(self.wait, bool):
            raise TypeError("Contract wait flag must be boolean")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "assumptions", self.words(self.assumptions))
        object.__setattr__(self, "guarantees", self.words(self.guarantees))
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "progress", progress)
        object.__setattr__(self, "invariants", tuple(dict.fromkeys(invariants)))
    @classmethod
    def build(cls, scale: Scale, version: int, parent: int | None, step: int, data: Mapping[str, Any], alpha: float = ALPHA, cap: int | None = None) -> "Contract":
        if not isinstance(data, Mapping):
            raise TypeError("Contract data must be a mapping")
        rawdelay = data.get("due", 1)
        if isinstance(rawdelay, Mapping):
            fields = [rawdelay[key] for key in ("steps", "step", "horizon", "delay", "value") if key in rawdelay]
            if len(fields) != 1:
                raise ValueError("Contract due mapping must contain one step count")
            rawdelay = fields[0]
        if isinstance(rawdelay, bool):
            raise ValueError("Contract due delay must be positive")
        try:
            delay = float(rawdelay)
        except (TypeError, ValueError) as error:
            raise ValueError("Contract due delay must be positive") from error
        if not isfinite(delay) or not delay.is_integer() or delay < 1.0:
            raise ValueError("Contract due delay must be positive")
        horizon = int(delay)
        if cap is not None:
            if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
                raise ValueError("Contract horizon cap must be positive")
            horizon = min(horizon, cap)
        rawpredictions = data.get("predictions", {})
        rawwidth = data.get("tolerance", {})
        rawverifiers = data.get("verifiers", {})
        if not isinstance(rawpredictions, Mapping) or not rawpredictions:
            raise ValueError("Contract predictions must not be empty")
        if not isinstance(rawwidth, Mapping) or not isinstance(rawverifiers, Mapping):
            raise TypeError("Contract metadata must be mappings")
        widths: dict[str, Any] = {}
        verifiers: dict[str, Any] = {}
        for source, target, title in ((rawwidth, widths, "tolerance"), (rawverifiers, verifiers, "verifier")):
            for rawkey, value in source.items():
                key = canonical(rawkey)
                if not key:
                    raise ValueError("Contract metadata keys must not be empty")
                if key in target and target[key] != value:
                    raise ValueError(f"Conflicting contract {title}: {key}")
                target[key] = value
        predictions: dict[str, Prediction] = {}
        for rawkey, rawvalue in rawpredictions.items():
            key = canonical(rawkey)
            if not key:
                raise ValueError("Prediction keys must not be empty")
            if key == "dataset" and not isinstance(rawvalue, (str, Mapping)):
                raise TypeError("Dataset predictions must be categorical")
            value = lower(rawvalue) if key == "dataset" and isinstance(rawvalue, str) else rawvalue
            if key in predictions:
                raise ValueError(f"Conflicting prediction field: {key}")
            verifier = View(str(verifiers.get(key, "state")).strip().lower())
            width = numeric(widths.get(key, 0.0))
            if width is None or width < 0.0:
                raise ValueError("Prediction width must be nonnegative")
            predictions[key] = Prediction.cast(value, width, alpha, verifier)
            if key == "dataset" and predictions[key].kind is not Kind.CATEGORICAL:
                raise TypeError("Dataset predictions must be categorical")
        rawprogress = data.get("progress", "")
        progress = "" if rawprogress is None or not str(rawprogress).strip() else canonical(rawprogress)
        rawinvariants = data.get("invariants", ())
        if rawinvariants is None:
            invariants: Any = ()
        elif isinstance(rawinvariants, str):
            invariants = (rawinvariants,)
        elif isinstance(rawinvariants, Mapping):
            invariants = tuple(rawinvariants)
        else:
            invariants = rawinvariants
        if not isinstance(invariants, (list, tuple)):
            raise TypeError("Contract invariants must be a sequence")
        wait = data.get("wait", False)
        if not isinstance(wait, bool):
            raise TypeError("Contract wait flag must be boolean")
        return cls(scale, version, parent, step, step + horizon, cls.words(data.get("assumptions", ())), cls.words(data.get("guarantees", ())), predictions, alpha, progress, tuple(canonical(item) for item in invariants), wait)
    @staticmethod
    def words(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            return (value,)
        if not isinstance(value, (list, tuple)):
            raise TypeError("Contract text must be a string or sequence")
        return tuple(str(item) for item in value)
    def horizon(self) -> int:
        return self.due - self.created
    def export(self) -> dict[str, Any]:
        data = asdict(self)
        data["scale"] = self.scale.value
        data["predictions"] = {key: value.export() for key, value in self.predictions.items()}
        return data
@dataclass(frozen=True)
class Observation:
    step: int
    values: Mapping[str, Any]
    events: tuple[str, ...] | list[str] | set[str] | str = ()
    verified: bool = True
    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("Observation step must be nonnegative")
        if not isinstance(self.verified, bool):
            raise TypeError("Observation verified flag must be boolean")
        events = (self.events,) if isinstance(self.events, str) else self.events
        if not isinstance(events, (list, tuple, set)):
            raise TypeError("Observation events must be a sequence")
        object.__setattr__(self, "values", normalize(self.values))
        object.__setattr__(self, "events", tuple(label(item) for item in events if label(item)))
@dataclass(frozen=True)
class Action:
    name: str
    arguments: Mapping[str, Any]
    expected: Mapping[str, Any]
    tolerance: Mapping[str, float]
    contract: int
    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name or not isinstance(self.arguments, Mapping) or not isinstance(self.expected, Mapping) or not isinstance(self.tolerance, Mapping):
            raise ValueError("Action fields are invalid")
        if isinstance(self.contract, bool) or not isinstance(self.contract, int) or self.contract < 1:
            raise ValueError("Action contract must be positive")
        expected: dict[str, Any] = {}
        for rawkey, rawvalue in self.expected.items():
            key = canonical(rawkey)
            if not key:
                raise ValueError("Action expectation keys must not be empty")
            value = lower(rawvalue) if key == "dataset" and isinstance(rawvalue, str) else walk(rawvalue)
            if key in expected and expected[key] != value:
                raise ValueError(f"Conflicting action expectation: {key}")
            expected[key] = value
        tolerance: dict[str, float] = {}
        for rawkey, rawvalue in self.tolerance.items():
            value = numeric(rawvalue)
            key = canonical(rawkey)
            if value is None or value < 0.0:
                raise ValueError("Action tolerance must be nonnegative")
            tolerance[key] = value
        if not expected or not set(tolerance).issubset(expected):
            raise ValueError("Action expectations are invalid")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "arguments", normalize(self.arguments))
        object.__setattr__(self, "expected", expected)
        object.__setattr__(self, "tolerance", tolerance)
@dataclass(frozen=True)
class Evidence:
    scale: Scale
    violation: float = 0.0
    difference: float = 0.0
    display: float = 0.0
    sharpness: float = 0.0
    own: float = 0.0
    reference: float = 0.0
    violations: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    flat: tuple[str, ...] = ()
    unverifiable: tuple[str, ...] = ()
    observable: int = 0
    declared: int = 0
    relevant: int = 0
    ready: int = 0
    horizon: int = 1
    due: bool = False
    expired: bool = False
    integrity: bool = True
    contract: int | None = None
    progressed: bool = False
    waiting: bool = False
    invariants: tuple[str, ...] = ()
    def __post_init__(self) -> None:
        scale = self.scale if isinstance(self.scale, Scale) else Scale(str(self.scale).strip().lower())
        for name in ("violation", "difference", "display", "sharpness", "own", "reference"):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError("Evidence scores must be finite")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.violation <= 1.0 or not -1.0 <= self.difference <= 1.0 or not -1.0 <= self.display <= 1.0:
            raise ValueError("Evidence normalized scores are invalid")
        if min(self.sharpness, self.own, self.reference) < 0.0:
            raise ValueError("Evidence losses must be nonnegative")
        for name in ("observable", "declared", "relevant", "ready", "horizon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < int(name == "horizon"):
                raise ValueError("Evidence counts are invalid")
        if self.observable > self.declared or self.relevant > self.declared or self.ready > self.observable:
            raise ValueError("Evidence counts are inconsistent")
        if self.contract is not None and (isinstance(self.contract, bool) or not isinstance(self.contract, int) or self.contract < 1):
            raise ValueError("Evidence contract is invalid")
        for name in ("due", "expired", "integrity", "progressed", "waiting"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError("Evidence flags must be boolean")
        if self.expired and not self.due:
            raise ValueError("Expired evidence must be due")
        object.__setattr__(self, "scale", scale)
        for name in ("violations", "missing", "flat", "unverifiable", "invariants"):
            object.__setattr__(self, name, tuple(str(item) for item in getattr(self, name)))
    def supported(self) -> bool:
        return self.due and self.observable > 0 and self.ready > 0
    def keys(self) -> int:
        return self.declared
@dataclass(frozen=True)
class Decision:
    move: Move
    scale: Scale | None
    score: float
    scores: Mapping[str, float]
    reasons: tuple[str, ...]
    def __post_init__(self) -> None:
        if self.scale is not None and not isinstance(self.scale, Scale):
            object.__setattr__(self, "scale", Scale(str(self.scale).strip().lower()))
@dataclass(frozen=True)
class Snapshot:
    goal: str
    constraints: tuple[str, ...]
    facts: Mapping[str, Any]
    plan: tuple[str, ...]
    cursor: int
    budget: Budget
    environment: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...] = ()
    step: int = 0
    version: int = 0
    status: str = "active"
    def __post_init__(self) -> None:
        goal = str(self.goal).strip()
        if not goal:
            raise ValueError("Goal must not be empty")
        if isinstance(self.cursor, bool) or not isinstance(self.cursor, int):
            raise TypeError("Plan cursor must be an integer")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("Snapshot step must be nonnegative")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 0:
            raise ValueError("Snapshot version must be nonnegative")
        if not isinstance(self.budget, Budget):
            raise TypeError("Snapshot budget must be a Budget")
        constraints = (self.constraints,) if isinstance(self.constraints, str) else self.constraints
        plan = (self.plan,) if isinstance(self.plan, str) else self.plan
        if not isinstance(constraints, (list, tuple)) or not isinstance(plan, (list, tuple)) or not isinstance(self.history, (list, tuple)):
            raise TypeError("Snapshot sequences are invalid")
        history: list[Mapping[str, Any]] = []
        for entry in self.history:
            if not isinstance(entry, Mapping):
                raise TypeError("Snapshot history entries must be mappings")
            record = normalize(entry)
            events = record.get("events", ())
            events = (events,) if isinstance(events, str) else events
            if not isinstance(events, (list, tuple, set)):
                raise TypeError("History events must be a sequence")
            record["events"] = [label(item) for item in events if label(item)]
            history.append(record)
        items = tuple(str(item).strip() for item in plan if str(item).strip())
        status = label(self.status)
        if not status:
            raise ValueError("Status must not be empty")
        object.__setattr__(self, "goal", goal)
        object.__setattr__(self, "constraints", tuple(str(item).strip() for item in constraints if str(item).strip()))
        object.__setattr__(self, "facts", normalize(self.facts))
        object.__setattr__(self, "plan", items)
        object.__setattr__(self, "cursor", min(max(0, self.cursor), len(items)))
        object.__setattr__(self, "environment", normalize(self.environment))
        object.__setattr__(self, "history", tuple(history))
        object.__setattr__(self, "status", status)
    def terminal(self) -> bool:
        return self.status != "active"

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
