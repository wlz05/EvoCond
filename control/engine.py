# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
from copy import deepcopy
from threading import RLock
from typing import Any, Callable, Mapping
from control.audit import Audit, Health
from control.credit import Ledger
from control.policy import Scheduler
from control.residual import Verifier
from control.scoring import Reference
from flow.action import Executor
from flow.plan import Compiler
from flow.world import Governor
from state.schema import Action, Decision, Evidence, Move, Observation, Scale, Snapshot
from state.store import Store
class Engine:
    def __init__(self, store: Store, governor: Governor, compiler: Compiler, executor: Executor, verifier: Verifier | None = None, scheduler: Scheduler | None = None, audit: Audit | None = None, ledger: Ledger | None = None, reference: Reference | None = None, terminal: Callable[[Snapshot], bool] | None = None):
        self.store = store
        self.governor = governor
        self.compiler = compiler
        self.executor = executor
        if verifier is not None and reference is not None and verifier.reference is not reference:
            raise ValueError("Verifier and reference must agree")
        self.reference = verifier.reference if verifier is not None else reference or Reference()
        self.verifier = verifier or Verifier(self.reference)
        if scheduler is not None and audit is not None and scheduler.audit is not audit:
            raise ValueError("Scheduler and audit must agree")
        if scheduler is not None and ledger is not None and scheduler.ledger is not ledger:
            raise ValueError("Scheduler and ledger must agree")
        self.scheduler = scheduler or Scheduler(audit=audit or Audit(), ledger=ledger or Ledger())
        self.audit = self.scheduler.audit
        self.ledger = self.scheduler.ledger
        self.terminal = terminal or (lambda snapshot: snapshot.terminal())
        self.reference.observe(self.verifier.values(store.read()))
        self.lastmove = Move.CONTINUE
        self.lastscale: Scale | None = None
        self.evidence: Mapping[Scale, Evidence] = {}
        self.health: Mapping[Scale, Health] = {}
        self.advice: dict[Scale, dict[str, Any]] = {}
        self.started = False
        self.pending: Action | None = None
        self.repair: Evidence | None = None
        self.awaiting = False
        self.resolved: set[tuple[Scale, int]] = set()
        self.lock = RLock()
    def start(self) -> None:
        with self.lock:
            if self.started:
                return
            if self.terminal(self.store.read()):
                raise RuntimeError("Engine is not active")
            self.governor.create(self.store)
            self.compiler.create(self.store)
            self.started = True
    def propose(self) -> Action:
        with self.lock:
            if not self.started:
                raise RuntimeError("Engine must be started")
            if self.terminal(self.store.read()):
                raise RuntimeError("Engine is not active")
            if self.pending is None:
                self.pending = self.executor.propose(self.store, self.repair, self.advice.get(Scale.LOCAL))
                self.repair = None
            return deepcopy(self.pending)
    def contexts(self, snapshot: Snapshot, action: Action | None) -> dict[Scale, str]:
        subgoal = snapshot.plan[snapshot.cursor] if snapshot.cursor < len(snapshot.plan) else snapshot.goal
        return {Scale.LOCAL: "" if action is None else action.name.lower(), Scale.BRIDGE: subgoal.lower(), Scale.GLOBAL: snapshot.goal.lower()}
    def observe(self, observation: Observation, tokens: int = 0, seconds: float = 0.0, tools: float = 0.0) -> dict[str, Any]:
        with self.lock:
            if not self.started:
                raise RuntimeError("Engine must be started")
            if self.terminal(self.store.read()):
                raise RuntimeError("Engine is not active")
            before = self.store.read()
            action = self.pending
            if action is None:
                raise RuntimeError("Observation requires a pending action")
            contexts = self.contexts(before, action)
            after = self.store.apply(observation, tokens, seconds, tools)
            if self.terminal(after) and not after.terminal():
                after = self.store.finish("done")
            self.pending = None
            contracts = self.store.contracts()
            evidence = self.verifier.scan(contracts, before, after, contexts, self.store.anchors())
            pending = set(self.audit.pending)
            health = {scale: self.audit.observe(item) for scale, item in evidence.items()}
            for scale in pending - set(self.audit.pending):
                self.advice.pop(scale, None)
            if self.terminal(after):
                decision = Decision(Move.CONTINUE, None, 0.0, {Move.CONTINUE.value: 0.0}, ("status",))
                self.scheduler.settle(0.0)
            else:
                decision = self.scheduler.select(evidence, health, after)
            if observation.verified:
                values = self.verifier.values(after)
                self.reference.observe(values)
                for scale, contract in contracts.items():
                    key = (scale, contract.version)
                    item = evidence[scale]
                    if key in self.resolved or not item.due or item.expired:
                        continue
                    targets = {name: values[name] for name in contract.predictions if name in values}
                    self.reference.resolve(targets, scale, contexts[scale], contract.horizon())
                    self.resolved.add(key)
            expired = [scale for scale in Scale if self.audit.timedout(scale, after.step)]
            for scale in expired:
                self.audit.close(scale)
                self.advice.pop(scale, None)
            self.lastmove = decision.move
            self.lastscale = decision.scale
            self.evidence = evidence
            self.health = health
            self.awaiting = not self.terminal(after)
            if not self.terminal(after):
                self.act(decision, evidence)
            return self.report(decision, evidence, health, after)
    def progress(self, evidence: Mapping[Scale, Evidence], scale: Scale) -> bool:
        item = evidence.get(scale)
        return bool(item and item.due and item.progressed and item.violation == 0.0)
    def act(self, decision: Decision, evidence: Mapping[Scale, Evidence]) -> None:
        move, scale = decision.move, decision.scale
        if move is Move.AUDIT and scale is not None:
            standing = self.store.contract(scale)
            if standing is None:
                raise RuntimeError("Audit requires a contract")
            remaining = max(1, standing.due - self.store.read().step)
            self.scheduler.lock(scale, standing.version)
            self.advice[scale] = self.audit.record(scale, standing.version, self.store.read().step, remaining)
        if self.progress(evidence, Scale.LOCAL):
            self.store.advance()
        if move is Move.GLOBAL:
            source = evidence.get(scale or Scale.GLOBAL)
            self.governor.create(self.store, source, self.advice.get(Scale.GLOBAL))
            self.compiler.create(self.store, source, self.advice.get(Scale.BRIDGE))
        elif move is Move.BRIDGE:
            self.compiler.create(self.store, evidence.get(scale or Scale.BRIDGE), self.advice.get(Scale.BRIDGE))
        elif move is Move.REPAIR:
            self.repair = evidence.get(scale or Scale.LOCAL)
        elif move is Move.AUDIT and scale is Scale.GLOBAL:
            self.governor.create(self.store, evidence.get(scale), self.advice.get(scale))
            self.compiler.create(self.store, evidence.get(scale), self.advice.get(Scale.BRIDGE))
        elif move is Move.AUDIT and scale is Scale.BRIDGE:
            self.compiler.create(self.store, evidence.get(scale), self.advice.get(scale))
        elif move is Move.CONTINUE and self.progress(evidence, Scale.BRIDGE):
            self.compiler.create(self.store, evidence.get(Scale.BRIDGE), self.advice.get(Scale.BRIDGE))
    def report(self, decision: Decision, evidence: Mapping[Scale, Evidence], health: Mapping[Scale, Health], snapshot: Snapshot) -> dict[str, Any]:
        rows = {scale.value: {"violation": item.violation, "difference": item.difference, "display": item.display, "sharpness": item.sharpness, "own": item.own, "reference": item.reference, "violations": list(item.violations), "missing": list(item.missing), "flat": list(item.flat), "unverifiable": list(item.unverifiable), "observable": item.observable, "declared": item.declared, "relevant": item.relevant, "ready": item.ready, "horizon": item.horizon, "due": item.due, "expired": item.expired, "integrity": item.integrity, "contract": item.contract, "progressed": item.progressed, "waiting": item.waiting} for scale, item in evidence.items()}
        healthrows = {scale.value: {"score": item.score, "coverage": item.coverage, "integrity": item.integrity, "expiry": item.expiry, "relevance": item.relevance, "nonvacuity": item.nonvacuity, "reference": item.reference, "silent": item.silent, "critical": item.critical, "issues": list(item.issues)} for scale, item in health.items()}
        return {"step": snapshot.step, "move": decision.move.value, "target": None if decision.scale is None else decision.scale.value, "score": decision.score, "scores": dict(decision.scores), "reasons": list(decision.reasons), "evidence": rows, "health": healthrows, "sequence": self.scheduler.sequential(), "audit": self.audit.report()}
    def feedback(self, success: bool, cost: float = 0.0) -> None:
        with self.lock:
            if not self.awaiting:
                raise RuntimeError("Feedback requires a new decision")
            support = self.ledger.support(self.lastmove, self.lastscale, self.evidence, self.health, self.scheduler.channels)
            self.scheduler.learn(self.lastmove, self.lastscale, support, success, cost)
            self.awaiting = False

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
