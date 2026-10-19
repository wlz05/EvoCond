# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, replace
from threading import RLock
from typing import Any, Mapping
from state.schema import Budget, Contract, Observation, Scale, Snapshot
class Store:
    def __init__(self, snapshot: Snapshot):
        self._state = snapshot
        self._initial = snapshot.budget
        self._contracts: dict[Scale, Contract] = {}
        self._anchors: dict[Scale, tuple[int, Snapshot]] = {}
        self._versions: dict[Scale, int] = {scale: 0 for scale in Scale}
        self._lock = RLock()
    def read(self) -> Snapshot:
        with self._lock:
            return deepcopy(self._state)
    def nextversion(self, scale: Scale) -> int:
        with self._lock:
            return self._versions[scale] + 1
    def contract(self, scale: Scale) -> Contract | None:
        with self._lock:
            return deepcopy(self._contracts.get(scale))
    def contracts(self) -> dict[Scale, Contract]:
        with self._lock:
            return deepcopy(self._contracts)
    def anchors(self) -> dict[Scale, Snapshot]:
        with self._lock:
            return {scale: deepcopy(item[1]) for scale, item in self._anchors.items() if self._contracts.get(scale) is not None and self._contracts[scale].version == item[0]}
    # Commit state and contracts atomically.
    def commit(self, contract: Contract, goal: tuple[str, str | list[str] | tuple[str, ...]] | None = None, plan: str | list[str] | tuple[str, ...] | None = None) -> None:
        with self._lock:
            version = self._versions[contract.scale] + 1
            if contract.version != version:
                raise ValueError("Contract version must increase by one")
            if contract.created != self._state.step or contract.due <= contract.created:
                raise ValueError("Contract time bounds are invalid")
            if contract.scale is Scale.GLOBAL and contract.parent is not None:
                raise ValueError("Global contract must not have a parent")
            if contract.scale is Scale.BRIDGE:
                parent = self._contracts.get(Scale.GLOBAL)
                if parent is None or contract.parent != parent.version:
                    raise ValueError("Bridge contract has a stale global parent")
            if contract.scale is Scale.LOCAL:
                parent = self._contracts.get(Scale.BRIDGE)
                if parent is None or contract.parent != parent.version:
                    raise ValueError("Local contract has a stale bridge parent")
            state = self._state
            if goal is not None:
                value = str(goal[0]).strip()
                if not value:
                    raise ValueError("Goal must not be empty")
                rawconstraints = (goal[1],) if isinstance(goal[1], str) else goal[1]
                if not isinstance(rawconstraints, (list, tuple)):
                    raise TypeError("Constraints must be a string or sequence")
                constraints = tuple(str(item).strip() for item in rawconstraints if str(item).strip())
                state = replace(state, goal=value, constraints=constraints, version=state.version + 1)
            if plan is not None:
                rawplan = (plan,) if isinstance(plan, str) else plan
                if not isinstance(rawplan, (list, tuple)):
                    raise TypeError("Plan must be a string or sequence")
                items = tuple(str(item).strip() for item in rawplan if str(item).strip())
                state = replace(state, plan=items, cursor=0, version=state.version + 1)
            self._contracts[contract.scale] = deepcopy(contract)
            self._versions[contract.scale] = contract.version
            if contract.scale is Scale.GLOBAL:
                self._contracts.pop(Scale.BRIDGE, None)
                self._contracts.pop(Scale.LOCAL, None)
                self._anchors.pop(Scale.BRIDGE, None)
                self._anchors.pop(Scale.LOCAL, None)
            elif contract.scale is Scale.BRIDGE:
                self._contracts.pop(Scale.LOCAL, None)
                self._anchors.pop(Scale.LOCAL, None)
            self._state = state
            self._anchors[contract.scale] = (contract.version, deepcopy(state))
    def apply(self, observation: Observation, tokens: int = 0, seconds: float = 0.0, tools: float = 0.0) -> Snapshot:
        with self._lock:
            if observation.step <= self._state.step:
                raise ValueError("Observation step must increase")
            values = dict(observation.values)
            rawstatus = values.pop("status", None)
            status = self._state.status
            if observation.verified and rawstatus is not None:
                status = str(rawstatus).strip().lower()
                if not status:
                    raise ValueError("Status must not be empty")
            environment = dict(self._state.environment)
            environment.update(values)
            facts = dict(self._state.facts)
            if observation.verified:
                facts.update(values)
            entry = {"step": observation.step, "values": dict(observation.values), "events": list(observation.events), "verified": observation.verified}
            history = (self._state.history + (entry,))[-64:]
            self._state = replace(self._state, facts=facts, environment=environment, budget=self._state.budget.spend(tokens, seconds, tools), history=history, step=observation.step, version=self._state.version + 1, status=status)
            return deepcopy(self._state)
    def setplan(self, plan: str | list[str] | tuple[str, ...], cursor: int = 0) -> Snapshot:
        with self._lock:
            if isinstance(cursor, bool) or not isinstance(cursor, int):
                raise TypeError("Plan cursor must be an integer")
            rawplan = (plan,) if isinstance(plan, str) else plan
            if not isinstance(rawplan, (list, tuple)):
                raise TypeError("Plan must be a string or sequence")
            items = tuple(str(item).strip() for item in rawplan if str(item).strip())
            self._state = replace(self._state, plan=items, cursor=min(max(0, cursor), len(items)), version=self._state.version + 1)
            return deepcopy(self._state)
    def advance(self) -> Snapshot:
        with self._lock:
            cursor = min(self._state.cursor + 1, len(self._state.plan))
            if cursor == self._state.cursor:
                return deepcopy(self._state)
            self._state = replace(self._state, cursor=cursor, version=self._state.version + 1)
            return deepcopy(self._state)
    def setgoal(self, goal: str, constraints: str | list[str] | tuple[str, ...]) -> Snapshot:
        with self._lock:
            value = str(goal).strip()
            if not value:
                raise ValueError("Goal must not be empty")
            rawconstraints = (constraints,) if isinstance(constraints, str) else constraints
            if not isinstance(rawconstraints, (list, tuple)):
                raise TypeError("Constraints must be a string or sequence")
            items = tuple(str(item).strip() for item in rawconstraints if str(item).strip())
            self._state = replace(self._state, goal=value, constraints=items, version=self._state.version + 1)
            return deepcopy(self._state)
    def finish(self, status: str) -> Snapshot:
        with self._lock:
            value = str(status).strip().lower()
            if not value:
                raise ValueError("Status must not be empty")
            self._state = replace(self._state, status=value, version=self._state.version + 1)
            return deepcopy(self._state)
    def view(self, scale: Scale) -> dict[str, Any]:
        with self._lock:
            state = self._state
            base: dict[str, Any] = {"goal": state.goal, "constraints": list(state.constraints), "facts": dict(state.facts), "environment": dict(state.environment), "budget": asdict(state.budget), "budgetratio": state.budget.ratio(self._initial), "step": state.step, "version": state.version, "status": state.status}
            if scale is Scale.LOCAL:
                base["subgoal"] = state.plan[state.cursor] if state.cursor < len(state.plan) else state.goal
                base["recent"] = list(state.history[-3:])
                parent = self._contracts.get(Scale.BRIDGE)
                base["parent"] = None if parent is None else parent.export()
            elif scale is Scale.BRIDGE:
                base["plan"] = list(state.plan)
                base["cursor"] = state.cursor
                base["recent"] = list(state.history[-12:])
                parent = self._contracts.get(Scale.GLOBAL)
                base["parent"] = None if parent is None else parent.export()
            else:
                base["plan"] = list(state.plan)
                base["cursor"] = state.cursor
                base["history"] = list(state.history)
                base["contracts"] = {key.value: value.export() for key, value in self._contracts.items()}
            return base

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
