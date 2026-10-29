# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
from control.audit import Audit
from control.policy import Scheduler
from control.residual import Verifier
from state.schema import Budget, Contract, Move, Scale, Snapshot
@dataclass(frozen=True)
class Probe:
    name: str
    expected: str
    facts: Mapping[str, Any]
    early: bool = False
class TriPerturb:
    def __init__(self):
        self.before = Snapshot("Reach target", (), {"local": 0, "bridge": 0, "global": "stable"}, ("advance",), 0, Budget(), {"local": 0, "bridge": 0, "global": "stable"})
        self.probes = (Probe("localstall", "repair:ll", {"local": 0, "bridge": 1, "global": "stable"}), Probe("bridgeblock", "bridge:gl", {"local": 1, "bridge": 0, "global": "stable"}), Probe("globalchange", "global:gg", {"local": 1, "bridge": 1, "global": "changed"}), Probe("monitormissing", "audit:gl", {"local": 0, "global": "stable"}, True), Probe("none", "continue", {"local": 1, "bridge": 1, "global": "stable"}))
    def contracts(self, early: bool) -> dict[Scale, Contract]:
        due = 5 if early else 1
        bridgekey = "missing" if early else "bridge"
        return {Scale.LOCAL: Contract.build(Scale.LOCAL, 1, 1, 0, {"predictions": {"local": 0 if early else 1}, "progress": "local", "wait": early, "due": due}), Scale.BRIDGE: Contract.build(Scale.BRIDGE, 1, 1, 0, {"predictions": {bridgekey: 1}, "invariants": [bridgekey] if early else (), "progress": "" if early else bridgekey, "due": due}), Scale.GLOBAL: Contract.build(Scale.GLOBAL, 1, None, 0, {"predictions": {"global": "stable"}, "invariants": ["global"], "due": due})}
    def after(self, facts: Mapping[str, Any]) -> Snapshot:
        record = {"step": 1, "values": dict(facts), "events": [], "verified": True}
        return Snapshot("Reach target", (), dict(facts), ("advance",), 0, Budget(), dict(facts), (record,), 1)
    def label(self, move: Move, scale: Scale | None) -> str:
        return move.value if scale is None else f"{move.value}:{scale.value}"
    def run(self) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for probe in self.probes:
            verifier = Verifier()
            audit = Audit()
            evidence = verifier.scan(self.contracts(probe.early), self.before, self.after(probe.facts), {Scale.LOCAL: "advance", Scale.BRIDGE: "advance", Scale.GLOBAL: "target"})
            health = {scale: audit.observe(item) for scale, item in evidence.items()}
            decision = Scheduler(dwell=0, audit=audit).select(evidence, health, self.after(probe.facts))
            result.append({"probe": probe.name, "expected": probe.expected, "predicted": self.label(decision.move, decision.scale)})
        return result
    def report(self) -> dict[str, Any]:
        rows = self.run()
        labels = sorted({row["expected"] for row in rows} | {row["predicted"] for row in rows})
        scores: list[float] = []
        for label in labels:
            true = sum(row["expected"] == label and row["predicted"] == label for row in rows)
            false = sum(row["expected"] != label and row["predicted"] == label for row in rows)
            missed = sum(row["expected"] == label and row["predicted"] != label for row in rows)
            precision = true / max(1, true + false)
            recall = true / max(1, true + missed)
            scores.append(0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall))
        return {"cases": len(rows), "accuracy": sum(row["expected"] == row["predicted"] for row in rows) / len(rows), "macrof1": sum(scores) / len(scores), "rows": rows}

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
