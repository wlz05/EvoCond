# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
from math import isfinite
from typing import Any, Mapping
from control.scoring import Reference, State
from state.data import canonical
from state.schema import Contract, Evidence, Scale, Snapshot, View
BADINTEGRITY = {"corrupt", "hashmismatch", "integrityfail"}
def event(value: Any) -> str:
    return "".join(character for character in str(value).strip().lower() if character.isalnum())
class Verifier:
    def __init__(self, reference: Reference | None = None, weights: Mapping[Scale | str, Mapping[str, float]] | None = None):
        self.reference = reference or Reference()
        self.weights: dict[Scale, dict[str, float]] = {scale: {} for scale in Scale}
        for rawscale, rawweights in (weights or {}).items():
            scale = rawscale if isinstance(rawscale, Scale) else Scale(str(rawscale).strip().lower())
            if not isinstance(rawweights, Mapping):
                raise TypeError("Verifier weights must be mappings")
            for rawkey, rawvalue in rawweights.items():
                value = float(rawvalue)
                key = canonical(rawkey)
                if not key or not isfinite(value) or value <= 0.0:
                    raise ValueError("Verifier weights must be finite and positive")
                self.weights[scale][key] = value
    def weight(self, scale: Scale, key: str) -> float:
        return self.weights[scale].get(key, 1.0)
    def values(self, snapshot: Snapshot) -> dict[str, Any]:
        values = dict(snapshot.facts)
        values.update({"goal": snapshot.goal, "step": snapshot.step, "cursor": snapshot.cursor, "status": snapshot.status, "budget.tokens": snapshot.budget.tokens, "budget.seconds": snapshot.budget.seconds, "budget.tools": snapshot.budget.tools})
        return values
    def check(self, contract: Contract, before: Snapshot, after: Snapshot, context: str = "") -> Evidence:
        horizon = contract.horizon()
        old = self.values(before)
        values = self.values(after)
        declared = len(contract.predictions)
        relevant = int(bool(contract.progress)) + len(contract.invariants)
        available = [key for key, prediction in contract.predictions.items() if prediction.verifier is not View.NONE and key in values]
        unverifiable = tuple(key for key, prediction in contract.predictions.items() if prediction.verifier is View.NONE)
        missing = tuple(key for key, prediction in contract.predictions.items() if prediction.verifier is not View.NONE and key not in values)
        readykeys = sum(self.reference.ready(key, contract.scale, context, horizon) for key in available)
        record = after.history[-1] if after.history else {}
        events = {event(item) for item in record.get("events", ())}
        integrity = bool(record.get("verified", True)) and not bool(events & BADINTEGRITY)
        if after.step < contract.due:
            return Evidence(contract.scale, missing=missing, unverifiable=unverifiable, observable=len(available), declared=declared, relevant=relevant, ready=readykeys, horizon=horizon, integrity=integrity, contract=contract.version, waiting=contract.wait, invariants=contract.invariants)
        if after.step > contract.due:
            return Evidence(contract.scale, missing=missing, unverifiable=unverifiable, observable=len(available), declared=declared, relevant=relevant, ready=readykeys, horizon=horizon, due=True, expired=True, integrity=integrity, contract=contract.version, waiting=contract.wait, invariants=contract.invariants)
        violations: list[str] = []
        missingitems = list(missing)
        flat: list[str] = []
        unverifiableitems = list(unverifiable)
        normalized: list[float] = []
        displays: list[float] = []
        own: list[float] = []
        references: list[float] = []
        widths: list[float] = []
        losses: list[float] = []
        comparisons: list[float] = []
        ready = 0
        observable = 0
        strict = set(contract.invariants)
        if contract.progress:
            strict.add(contract.progress)
        progressed = False
        for key, prediction in contract.predictions.items():
            if prediction.verifier is View.NONE:
                continue
            if key not in values:
                if key in strict:
                    violations.append(key)
                continue
            observable += 1
            score = self.reference.measure(key, prediction, values[key], old.get(key), contract.scale, context, horizon)
            if score.state is State.BROKEN:
                missingitems.append(key)
                if key in strict:
                    violations.append(key)
                continue
            if score.ready:
                normalized.append(score.normalized)
                displays.append(score.display)
                comparisons.append(self.weight(contract.scale, key))
            own.append(score.own)
            references.append(score.reference)
            widths.append(score.sharpness)
            losses.append(self.weight(contract.scale, key))
            ready += int(score.ready)
            if key in strict and not prediction.accepts(values[key]):
                violations.append(key)
            if key == contract.progress:
                if key not in old:
                    missingitems.append(key)
                    violations.append(key)
                    progressed = False
                    continue
                progressed = self.reference.deviates(old[key], values[key])
                if not progressed and not contract.wait:
                    violations.append(key)
                if self.reference.flat(key, contract.scale, context, horizon) and not progressed:
                    flat.append(key)
        if not integrity:
            violations.append("integrity")
        violation = 1.0 if violations else 0.0
        difference = sum(value * weight for value, weight in zip(normalized, comparisons)) / sum(comparisons) if comparisons else 0.0
        shown = sum(value * weight for value, weight in zip(displays, comparisons)) / sum(comparisons) if comparisons else 0.0
        sharpness = sum(value * weight for value, weight in zip(widths, losses)) / sum(losses) if losses else 0.0
        ownloss = sum(value * weight for value, weight in zip(own, losses)) / sum(losses) if losses else 0.0
        referenceloss = sum(value * weight for value, weight in zip(references, losses)) / sum(losses) if losses else 0.0
        return Evidence(contract.scale, violation, difference, shown, sharpness, ownloss, referenceloss, tuple(dict.fromkeys(violations)), tuple(dict.fromkeys(missingitems)), tuple(dict.fromkeys(flat)), tuple(dict.fromkeys(unverifiableitems)), observable, declared, relevant, ready, horizon, True, False, integrity, contract.version, progressed, contract.wait, contract.invariants)
    def scan(self, contracts: Mapping[Scale, Contract], before: Snapshot, after: Snapshot, contexts: Mapping[Scale, str] | None = None, anchors: Mapping[Scale, Snapshot] | None = None) -> dict[Scale, Evidence]:
        contexts = contexts or {}
        anchors = anchors or {}
        return {scale: Evidence(scale) if scale not in contracts else self.check(contracts[scale], anchors.get(scale, before), after, contexts.get(scale, "")) for scale in Scale}

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
