# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
from statistics import median
from typing import Any, Iterable, Mapping
from state.data import canonical, normalize
from state.schema import Kind, Prediction, Scale, label, numeric
TRACE = 1e-9
UNIT = 1.0
WARMUP = 4
class State(str, Enum):
    SCORED = "scored"
    BROKEN = "broken"
@dataclass(frozen=True)
class Score:
    state: State
    sharpness: float = 0.0
    own: float = 0.0
    reference: float = 0.0
    difference: float = 0.0
    normalized: float = 0.0
    display: float = 0.0
    ready: bool = False
def interval(lower: float, upper: float, alpha: float, target: float) -> float:
    if not all(isfinite(float(value)) for value in (lower, upper, alpha, target)) or upper < lower or not 0.0 < alpha < 1.0:
        raise ValueError("Interval score inputs are invalid")
    penalty = 0.0
    if target < lower:
        penalty = 2.0 * (lower - target) / alpha
    elif target > upper:
        penalty = 2.0 * (target - upper) / alpha
    return upper - lower + penalty
def brier(shares: Mapping[str, float], target: str) -> float:
    values = [float(value) for value in shares.values()]
    if not shares or not all(isfinite(value) and value >= 0.0 for value in values) or abs(sum(values) - 1.0) > TRACE:
        raise ValueError("Brier shares must form a distribution")
    keys = set(shares) | {target}
    return sum((float(shares.get(key, 0.0)) - float(key == target)) ** 2 for key in keys)
def relative(own: float, reference: float) -> float:
    if not isfinite(float(own)) or not isfinite(float(reference)):
        raise ValueError("Losses must be finite")
    return own - reference
def display(own: float, reference: float) -> float:
    if not isfinite(float(own)) or not isfinite(float(reference)) or own < 0.0 or reference < 0.0:
        raise ValueError("Display losses must be finite and nonnegative")
    total = own + reference
    return 0.0 if total <= TRACE else max(-1.0, min(1.0, (own - reference) / (total + TRACE)))
def quantile(values: list[float], probability: float) -> float:
    if not values or not isfinite(float(probability)) or not 0.0 <= probability <= 1.0 or not all(isfinite(float(value)) for value in values):
        raise ValueError("Quantile inputs are invalid")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight
class Reference:
    def __init__(self, window: int = 64, alpha: float = 0.2, unit: float = UNIT, warmup: int = WARMUP, frozen: bool = False):
        if isinstance(window, bool) or not isinstance(window, int) or window < 2:
            raise ValueError("Reference window must exceed one")
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1 or warmup > window:
            raise ValueError("Reference warmup is invalid")
        if not isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0 or not isfinite(float(unit)) or float(unit) <= 0.0:
            raise ValueError("Reference score rates are invalid")
        if not isinstance(frozen, bool):
            raise TypeError("Reference frozen flag must be boolean")
        self.window = window
        self.alpha = float(alpha)
        self.unit = float(unit)
        self.warmup = warmup
        self.frozen = bool(frozen)
        self.series: dict[str, list[Any]] = {}
        self.conditioned: dict[tuple[Scale, str, str, int], list[Any]] = {}
        self.calibration: dict[tuple[Scale, str, str, int], float] = {}
        self.training: tuple[str, ...] = ()
        self.holdout: tuple[str, ...] = ()
    @classmethod
    def crossfit(cls, rows: Iterable[Mapping[str, Any]], holdout: str | Iterable[str], window: int = 64, alpha: float = 0.2, unit: float = UNIT, warmup: int = WARMUP) -> "Reference":
        rawholdout = (holdout,) if isinstance(holdout, str) else tuple(holdout)
        held = {label(item) for item in rawholdout if isinstance(item, str) and item.strip()}
        if not held:
            raise ValueError("Crossfit holdout must not be empty")
        reference = cls(window, alpha, unit, warmup)
        training: set[str] = set()
        observed: set[str] = set()
        differentials: dict[tuple[Scale, str, str, int], list[float]] = {}
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("template"), str) or not str(row["template"]).strip():
                raise ValueError("Crossfit rows need a template")
            template = label(row["template"])
            observed.add(template)
            if template in held:
                continue
            values = row.get("values", {})
            contexts = row.get("contexts", {})
            horizons = row.get("horizons", {})
            differences = row.get("differences", {})
            if not isinstance(values, Mapping) or not isinstance(contexts, Mapping) or not isinstance(horizons, Mapping) or not isinstance(differences, Mapping):
                raise TypeError("Crossfit row fields must be mappings")
            parsedcontexts = {(rawscale if isinstance(rawscale, Scale) else Scale(str(rawscale).strip().lower())): str(context) for rawscale, context in contexts.items()}
            parsedhorizons: dict[Scale, int] = {}
            for rawscale, rawhorizon in horizons.items():
                scale = rawscale if isinstance(rawscale, Scale) else Scale(str(rawscale).strip().lower())
                if isinstance(rawhorizon, bool) or not isinstance(rawhorizon, int) or rawhorizon < 1:
                    raise ValueError("Crossfit horizons must be positive")
                parsedhorizons[scale] = rawhorizon
            if set(parsedhorizons) - set(parsedcontexts) or differences and not parsedcontexts:
                raise ValueError("Crossfit conditions are incomplete")
            clean = normalize(values)
            reference.update(clean, parsedcontexts, parsedhorizons)
            for scale, context in parsedcontexts.items():
                horizon = parsedhorizons.get(scale, 1)
                for rawkey, rawvalue in differences.items():
                    value = numeric(rawvalue)
                    key = canonical(rawkey)
                    if not key or value is None or key not in clean:
                        raise ValueError("Crossfit differences must be finite")
                    differentials.setdefault((scale, key, reference.token(context), horizon), []).append(value)
            training.add(template)
        if not training or not held.issubset(observed):
            raise ValueError("Crossfit split is empty or unknown")
        for key, values in differentials.items():
            reference.calibration[key] = max(TRACE, median(abs(value) for value in values))
        reference.training = tuple(sorted(training))
        reference.holdout = tuple(sorted(held))
        reference.freeze()
        return reference
    def freeze(self) -> None:
        self.frozen = True
    def thaw(self) -> None:
        self.frozen = False
    def token(self, value: Any) -> str:
        return label(value)
    def update(self, values: Mapping[str, Any], contexts: Mapping[Scale, str] | None = None, horizons: Mapping[Scale, int] | None = None) -> None:
        if self.frozen:
            return
        contexts = contexts or {}
        horizons = horizons or {}
        self.observe(values)
        for scale, context in contexts.items():
            self.resolve(values, scale, context, max(1, int(horizons.get(scale, 1))))
    def observe(self, values: Mapping[str, Any]) -> None:
        if self.frozen:
            return
        for rawkey, value in values.items():
            key = str(rawkey)
            self._append(self.series.setdefault(key, []), value)
    def resolve(self, values: Mapping[str, Any], scale: Scale, context: str, horizon: int) -> None:
        if self.frozen:
            return
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("Reference horizon must be positive")
        span = horizon
        for rawkey, value in values.items():
            key = str(rawkey)
            bucket = self.conditioned.setdefault((scale, key, self.token(context), span), [])
            self._append(bucket, value)
    def _append(self, target: list[Any], value: Any) -> None:
        target.append(value)
        if len(target) > self.window:
            del target[0]
    def history(self, key: str, scale: Scale, context: str, horizon: int) -> list[Any]:
        exact = self.conditioned.get((scale, key, self.token(context), max(1, horizon)), [])
        return exact if len(exact) >= self.warmup else self.series.get(key, [])
    def samples(self, key: str, scale: Scale, context: str, horizon: int) -> int:
        return len(self.conditioned.get((scale, key, self.token(context), max(1, horizon)), []))
    def ready(self, key: str, scale: Scale, context: str, horizon: int) -> bool:
        return self.samples(key, scale, context, horizon) >= self.warmup
    def changes(self, values: list[Any]) -> list[float]:
        numbers = [value for value in (numeric(item) for item in values) if value is not None]
        return [abs(right - left) for left, right in zip(numbers, numbers[1:])]
    def scale(self, key: str, scale: Scale, context: str, horizon: int) -> float:
        calibrated = self.calibration.get((scale, key, self.token(context), max(1, horizon)))
        if calibrated is not None:
            return calibrated
        changes = self.changes(self.history(key, scale, context, horizon))
        return max(TRACE, median(changes) if changes else self.unit)
    def flat(self, key: str, scale: Scale, context: str, horizon: int) -> bool:
        values = self.history(key, scale, context, horizon)
        if len(values) < self.warmup:
            return False
        return len({self.token(value) for value in values}) <= 1
    def deviates(self, before: Any, after: Any, tolerance: float = TRACE) -> bool:
        left = numeric(before)
        right = numeric(after)
        if left is not None and right is not None:
            return abs(right - left) > tolerance
        return label(before) != label(after)
    def baseline(self, key: str, prediction: Prediction, anchor: Any, scale: Scale, context: str, horizon: int) -> Prediction:
        values = self.history(key, scale, context, horizon)
        numbers = [value for value in (numeric(item) for item in values) if value is not None]
        if prediction.kind is Kind.INTERVAL:
            center = numeric(anchor)
            center = median(numbers) if center is None and numbers else center
            center = 0.0 if center is None else center
            if self.ready(key, scale, context, horizon) and numbers:
                lower = quantile(numbers, prediction.alpha / 2.0)
                upper = quantile(numbers, 1.0 - prediction.alpha / 2.0)
                return Prediction(Kind.INTERVAL, lower, upper, prediction.alpha)
            width = max(self.unit, self.scale(key, scale, context, horizon) * sqrt(max(1, horizon)))
            return Prediction(Kind.INTERVAL, center - width, center + width, prediction.alpha)
        labels = [label(item) for item in values if numeric(item) is None]
        counts = Counter(labels)
        if not counts:
            return Prediction(Kind.CATEGORICAL, shares={"unknown": 1.0}, alpha=prediction.alpha)
        total = float(sum(counts.values()))
        return Prediction(Kind.CATEGORICAL, shares={key: count / total for key, count in counts.items()}, alpha=prediction.alpha)
    def measure(self, key: str, prediction: Prediction, target: Any, anchor: Any, scale: Scale, context: str, horizon: int) -> Score:
        baseline = self.baseline(key, prediction, anchor, scale, context, horizon)
        ready = self.ready(key, scale, context, horizon)
        if prediction.kind is Kind.INTERVAL:
            value = numeric(target)
            if value is None or baseline.kind is not Kind.INTERVAL:
                return Score(State.BROKEN, prediction.width(), ready=ready)
            own = interval(prediction.lower, prediction.upper, prediction.alpha, value)
            reference = interval(baseline.lower, baseline.upper, baseline.alpha, value)
        else:
            if numeric(target) is not None or baseline.kind is not Kind.CATEGORICAL:
                return Score(State.BROKEN, ready=ready)
            targetlabel = label(target)
            own = brier(prediction.shares, targetlabel)
            reference = brier(baseline.shares, targetlabel)
        difference = relative(own, reference)
        normalized = max(-1.0, min(1.0, difference / self.scale(key, scale, context, horizon)))
        return Score(State.SCORED, prediction.width(), own, reference, difference, normalized, display(own, reference), ready)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
