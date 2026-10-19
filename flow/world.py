# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
from math import isfinite
from typing import Any, Mapping
from model.client import Model, contractual, validated
from model.prompt import GLOBAL
from state.data import canonical
from state.schema import ALPHA, Contract, Evidence, Scale
from state.store import Store
class Governor:
    def __init__(self, model: Model, alpha: float = ALPHA):
        self.model = model
        self.alpha = alpha
    # Refresh the global contract from current evidence.
    def create(self, store: Store, violation: Mapping[str, Any] | Evidence | None = None, audit: Mapping[str, Any] | None = None) -> Contract:
        payload = store.view(Scale.GLOBAL)
        payload["kind"] = "gg"
        payload["violation"] = self._violation(violation)
        payload["audit"] = dict(audit) if audit else {}
        standing = store.contract(Scale.GLOBAL)
        def accept(raw: dict[str, Any]) -> tuple[dict[str, Any], Contract]:
            data = contractual(raw)
            if audit and standing is not None:
                data["assumptions"] = list(standing.assumptions)
                data["guarantees"] = list(standing.guarantees)
            contract = Contract.build(Scale.GLOBAL, store.nextversion(Scale.GLOBAL), None, store.read().step, data, self.alpha, (audit or {}).get("due"))
            self._audit(contract, audit)
            return data, contract
        data, contract = validated(self.model, GLOBAL, payload, accept)
        goal = None
        if "goal" in data and not audit:
            rawconstraints = data.get("constraints", payload["constraints"])
            constraints = [rawconstraints] if isinstance(rawconstraints, str) else rawconstraints
            if not isinstance(constraints, (list, tuple)):
                raise TypeError("Constraints must be a string or sequence")
            goal = (str(data["goal"]), list(constraints))
        store.commit(contract, goal=goal)
        return contract
    def _violation(self, violation: Mapping[str, Any] | Evidence | None) -> Any:
        if isinstance(violation, Evidence):
            return {"scale": violation.scale.value, "violation": violation.violation, "difference": violation.difference, "violations": list(violation.violations)}
        return violation
    def _audit(self, contract: Contract, audit: Mapping[str, Any] | None) -> None:
        if not audit:
            return
        forbid = {canonical(item) for item in audit.get("forbid", ())}
        if forbid & set(contract.predictions):
            raise ValueError("Audit reused a forbidden prediction")
        if "width" in audit:
            limit = float(audit["width"])
            if not isfinite(limit) or limit < 0.0 or any(prediction.width() / 2.0 > limit + 1e-9 for prediction in contract.predictions.values()):
                raise ValueError("Audit exceeded the sharpness limit")

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
