# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
from math import isfinite
from typing import Any, Mapping
from model.client import Model, contractual, validated
from model.prompt import LOCAL
from state.data import canonical
from state.schema import ALPHA, Action, Contract, Evidence, Scale
from state.store import Store
class Executor:
    def __init__(self, model: Model, alpha: float = ALPHA):
        self.model = model
        self.alpha = alpha
    # Bind each action to one local predictive contract.
    def propose(self, store: Store, violation: Mapping[str, Any] | Evidence | None = None, audit: Mapping[str, Any] | None = None) -> Action:
        parent = store.contract(Scale.BRIDGE)
        if parent is None:
            raise RuntimeError("Bridge contract is required")
        payload = store.view(Scale.LOCAL)
        payload["kind"] = "ll"
        payload["violation"] = self._violation(violation)
        payload["audit"] = dict(audit) if audit else {}
        standing = store.contract(Scale.LOCAL)
        def accept(raw: dict[str, Any]) -> tuple[dict[str, Any], str, Mapping[str, Any], Contract]:
            data = contractual(raw)
            if audit and standing is not None:
                data["assumptions"] = list(standing.assumptions)
                data["guarantees"] = list(standing.guarantees)
            name = str(data.get("action", "wait")).strip()
            arguments = data.get("arguments", {})
            if not name or not isinstance(arguments, Mapping):
                raise ValueError("Action name and arguments are invalid")
            contract = Contract.build(Scale.LOCAL, store.nextversion(Scale.LOCAL), parent.version, store.read().step, data, self.alpha, (audit or {}).get("due"))
            self._audit(contract, audit)
            return data, name, arguments, contract
        data, name, arguments, contract = validated(self.model, LOCAL, payload, accept)
        store.commit(contract)
        expected = {key: prediction.export() for key, prediction in contract.predictions.items()}
        tolerance = {key: prediction.width() / 2.0 for key, prediction in contract.predictions.items()}
        return Action(name, dict(arguments), expected, tolerance, contract.version)
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
