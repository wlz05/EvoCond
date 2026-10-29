# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import json
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping, cast
from control.audit import Audit, Challenger, Health
from control.credit import Ledger
from control.engine import Engine
from control.guard import intervene
from control.metric import auroc, macrof1, paired, recall, regret
from control.policy import Scheduler
from control.probe import TriPerturb
from control.residual import Verifier
from control.router import ACTIONS, Router
from control.scoring import Reference, brier, display, interval, relative
from flow.action import Executor
from flow.plan import Compiler
from flow.world import Governor
from model.client import Http, Rule, Script, contractual, parse, validated
from model.audit import Monitor
from serve import toolcalls
from state.data import identify, normalize
from state.record import Journal, document, local, rows
from state.schema import ALPHA, Budget, Contract, Evidence, Kind, Move, Observation, Prediction, Scale, Snapshot, View
from state.store import Store
from state.task import Terminal
def fails(error: type[BaseException], call: Callable[[], object]) -> None:
    try:
        call()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")
def repairs() -> dict[str, str]:
    calls: list[Mapping[str, object]] = []
    def handler(prompt: str, payload: Mapping[str, object]) -> dict[str, object]:
        calls.append(payload)
        return {"predictions": {}} if len(calls) == 1 else {"predictions": {"x": 1}, "due": 1}
    contract = validated(Script(handler), "schema", {"task": "test"}, lambda data: Contract.build(Scale.GLOBAL, 1, None, 0, data))
    assert contract.predictions and len(calls) == 2 and "repair" in calls[1]
    cleaned = contractual({"predictions": {"progress": 1}, "progress": "progress", "invariants": ["progress", "dataset"]})
    assert cleaned["invariants"] == []
    return {"invalid": "rejected", "retry": "bounded", "invariants": "conservative"}
def scores() -> dict[str, str]:
    assert interval(0.0, 1.0, ALPHA, 0.5) == 1.0
    assert interval(0.0, 1.0, ALPHA, 2.0) == 1.0 + 2.0 / ALPHA
    assert brier({"a": 1.0}, "a") == 0.0
    assert brier({"a": 1.0}, "b") == 2.0
    fails(ValueError, lambda: brier({"a": 2.0}, "a"))
    fails(ValueError, lambda: interval(1.0, 0.0, ALPHA, 0.5))
    assert relative(2.0, 3.0) == -1.0
    assert display(0.0, 0.0) == 0.0
    probability = 0.7
    baseline = 0.5
    def ratio(report: float) -> float:
        hit = (1.0 - report) ** 2
        miss = report ** 2
        refhit = (1.0 - baseline) ** 2
        refmiss = baseline ** 2
        return probability * hit / (hit + refhit) + (1.0 - probability) * miss / (miss + refmiss)
    assert ratio(0.982) < ratio(probability)
    reference = Reference(warmup=2)
    contexts = {Scale.LOCAL: "left"}
    horizons = {Scale.LOCAL: 1}
    reference.update({"x": 1}, contexts, horizons)
    reference.update({"x": 1}, contexts, horizons)
    reference.update({"x": 9}, {Scale.LOCAL: "right"}, horizons)
    reference.update({"x": 9}, {Scale.LOCAL: "right"}, horizons)
    prediction = Prediction(Kind.INTERVAL, 0.0, 2.0, ALPHA)
    left = reference.baseline("x", prediction, 0, Scale.LOCAL, "left", 1)
    right = reference.baseline("x", prediction, 0, Scale.LOCAL, "right", 1)
    assert left.lower == left.upper == 1.0
    assert right.lower == right.upper == 9.0
    assert reference.samples("x", Scale.LOCAL, "unseen", 1) == 0
    before = reference.samples("x", Scale.LOCAL, "left", 1)
    reference.freeze()
    reference.update({"x": 7}, contexts, horizons)
    assert reference.samples("x", Scale.LOCAL, "left", 1) == before
    rows = [{"template": "Train", "values": {"DataSet": "ALFWorld", "x": 1}, "contexts": {"ll": "advance"}, "horizons": {"ll": 1}, "differences": {"x": 0.2}}, {"template": "TRAIN", "values": {"dataset": "alfworld", "x": 1}, "contexts": {"ll": "advance"}, "horizons": {"ll": 1}, "differences": {"x": -0.4}}, {"template": "Holdout", "values": {"dataset": "WebArena", "x": 9}, "contexts": {"ll": "advance"}, "horizons": {"ll": 1}, "differences": {"x": 9.0}}]
    folded = Reference.crossfit(rows, "HOLDOUT", warmup=2)
    assert folded.frozen and folded.training == ("train",) and folded.holdout == ("holdout",)
    assert folded.samples("x", Scale.LOCAL, "advance", 1) == 2
    assert abs(folded.scale("x", Scale.LOCAL, "advance", 1) - 0.3) < 1e-9
    assert folded.series["dataset"] == ["alfworld", "alfworld"]
    fails(ValueError, lambda: Reference.crossfit(rows, "unknown", warmup=2))
    return {"proper": "ok", "difference": "ok", "ratio": "rejected", "conditioned": "ok", "frozen": "ok", "crossfit": "disjoint"}
def contracts() -> dict[str, str]:
    reference = Reference(warmup=1)
    reference.update({"progress": 0, "policy": "stable"})
    verifier = Verifier(reference)
    before = Snapshot("Test", (), {"progress": 0, "policy": "stable"}, (), 0, Budget(), {"progress": 0, "policy": "stable"})
    changed = Snapshot("Test", (), {"progress": 1, "policy": "stable"}, (), 0, Budget(), {"progress": 1, "policy": "stable"}, step=1)
    progress = Contract.build(Scale.LOCAL, 1, None, 0, {"predictions": {"progress": 1}, "progress": "progress", "due": 1})
    evidence = verifier.check(progress, before, changed, "advance")
    assert evidence.progressed and evidence.violation == 0.0
    stalled = verifier.check(progress, before, replace(changed, facts={"progress": 0, "policy": "stable"}, environment={"progress": 0, "policy": "stable"}), "advance")
    assert not stalled.progressed and stalled.violation == 1.0
    wait = Contract.build(Scale.LOCAL, 1, None, 0, {"predictions": {"progress": 0}, "progress": "progress", "wait": True, "due": 1})
    waiting = verifier.check(wait, before, replace(changed, facts={"progress": 0, "policy": "stable"}, environment={"progress": 0, "policy": "stable"}), "wait")
    assert waiting.waiting and waiting.violation == 0.0 and not waiting.progressed
    invariant = Contract.build(Scale.GLOBAL, 1, None, 0, {"predictions": {"policy": "stable"}, "invariants": ["policy"], "due": 1})
    steady = verifier.check(invariant, before, changed, "test")
    assert steady.violation == 0.0 and not steady.progressed and steady.invariants == ("policy",)
    broken = verifier.check(invariant, before, replace(changed, facts={"progress": 1, "policy": "changed"}, environment={"progress": 1, "policy": "changed"}), "test")
    assert broken.violation == 1.0 and broken.violations == ("policy",)
    fails(ValueError, lambda: Contract.build(Scale.LOCAL, 1, None, 0, {"predictions": {"x": 1}, "progress": "hidden", "due": 1}))
    fails(ValueError, lambda: Contract.build(Scale.LOCAL, 1, None, 0, {"predictions": {"x": 1}, "invariants": ["hidden"], "due": 1}))
    fails(TypeError, lambda: Contract.build(Scale.LOCAL, 1, None, 0, {"predictions": {"x": 1}, "wait": "false", "due": 1}))
    fails(ValueError, lambda: Contract.build(Scale.LOCAL, 1, None, 0, {"predictions": {"x": 1}, "progress": "x", "invariants": ["x"], "due": 1}))
    dataset = Contract.build(Scale.GLOBAL, 1, None, 0, {"predictions": {"Benchmark Name": {"kind": "categorical", "shares": {"ALFWorld": 1.0}}}, "invariants": ["DATASET"], "due": 1})
    assert dataset.predictions["dataset"].shares == {"alfworld": 1.0}
    fails(TypeError, lambda: Contract.build(Scale.GLOBAL, 1, None, 0, {"predictions": {"dataset": 1}, "due": 1}))
    unseen = Snapshot("Test", (), {"policy": "stable"}, (), 0, Budget(), {"policy": "stable"})
    unknown = verifier.check(progress, unseen, changed, "advance")
    assert unknown.violation == 1.0 and not unknown.progressed and "progress" in unknown.missing
    untrusted = Store(before).apply(Observation(1, {"progress": 1}, ("Hash_Mismatch",), False))
    corrupt = verifier.check(progress, before, untrusted, "advance")
    assert not corrupt.integrity and corrupt.violation == 1.0
    early = Contract.build(Scale.BRIDGE, 1, None, 0, {"predictions": {"missing": "ready"}, "invariants": ["missing"], "due": 5})
    structural = verifier.check(early, before, changed, "test")
    monitor = Audit()
    structuralhealth = monitor.observe(structural)
    assert not structural.due and structuralhealth.critical
    mapping = {scale: Evidence(scale) for scale in Scale}
    mapping[Scale.BRIDGE] = structural
    healthmap = {scale: Health(scale) for scale in Scale}
    healthmap[Scale.BRIDGE] = structuralhealth
    earlydecision = Scheduler(dwell=0, audit=monitor).select(mapping, healthmap, changed)
    assert earlydecision.move is Move.AUDIT and earlydecision.scale is Scale.BRIDGE
    weightedref = Reference(warmup=1)
    weightedref.update({"x": 0, "y": 0}, {Scale.LOCAL: "test"}, {Scale.LOCAL: 1})
    auxiliary = Contract.build(Scale.LOCAL, 1, None, 0, {"predictions": {"x": 10, "y": 0}, "due": 1})
    pairbefore = Snapshot("Test", (), {"x": 0, "y": 0}, (), 0, Budget(), {"x": 0, "y": 0})
    pairafter = replace(pairbefore, history=({"step": 1, "values": {"x": 0, "y": 0}, "events": [], "verified": True},), step=1)
    plain = Verifier(weightedref).check(auxiliary, pairbefore, pairafter, "test")
    weighted = Verifier(weightedref, {Scale.LOCAL: {"x": 9.0, "y": 1.0}}).check(auxiliary, pairbefore, pairafter, "test")
    assert abs(plain.difference - 0.5) < 1e-9 and abs(weighted.difference - 0.9) < 1e-9 and plain.violation == weighted.violation == 0.0
    fails(ValueError, lambda: Verifier(weights={Scale.LOCAL: {"x": 0.0}}))
    assert Contract.build(Scale.LOCAL, 1, None, 0, {"predictions": {"x": 1}, "due": 9}, cap=2).horizon() == 2
    assert Contract.build(Scale.LOCAL, 1, None, 0, {"predictions": {"x": 1}, "due": {"steps": 2}}).horizon() == 2
    assert Contract.build(Scale.GLOBAL, 1, None, 0, {"predictions": {"x": 1}, "invariants": {"x": "stable"}, "due": 1}).invariants == ("x",)
    assert Contract.build(Scale.GLOBAL, 1, None, 0, {"predictions": {"x": 1}, "invariants": None, "due": 1}).invariants == ()
    fails(ValueError, lambda: Contract.build(Scale.LOCAL, 1, None, 0, {"predictions": {"x": 1}, "due": {"when": "soon"}}))
    return {"progress": "ok", "invariant": "ok", "wait": "ok", "terminal": "external", "cap": "ok"}
def audits() -> dict[str, str]:
    audit = Audit()
    assert isinstance(audit.challenger, Challenger)
    bad = Evidence(Scale.BRIDGE, due=True, contract=1, declared=1, relevant=1, unverifiable=("intent",))
    health = Health(Scale.BRIDGE)
    for _ in range(4):
        health = audit.observe(bad)
    assert health.silent and health.critical and health.coverage == 0.0
    orders = audit.record(Scale.BRIDGE, 1, 4, 4)
    assert orders["source"] == "rule" and orders["forbid"] == ["intent"] and orders["due"] == 2
    assert not audit.canstart(Scale.BRIDGE, 1)
    assert not audit.timedout(Scale.BRIDGE, 5) and audit.timedout(Scale.BRIDGE, 6)
    fails(RuntimeError, lambda: audit.record(Scale.BRIDGE, 1, 5, 1))
    good = Evidence(Scale.BRIDGE, due=True, contract=2, declared=1, relevant=1, observable=1, ready=1, progressed=True)
    recovered = audit.observe(good)
    assert recovered.score == 1.0 and audit.report()["gl"]["recovered"] == 1
    fails(ValueError, lambda: audit.diagnose(Scale.BRIDGE, True, False))
    audit.diagnose(Scale.BRIDGE, True, False, True)
    audit.diagnose(Scale.BRIDGE, False, True, True)
    assert audit.separation(Scale.BRIDGE) > 0.0
    monitor = Monitor(Script(lambda prompt, payload: {"width": payload["limits"]["width"], "forbid": payload["suspect"], "due": payload["limits"]["due"]}))
    modeled = Audit(challenger=monitor)
    modeled.observe(bad)
    assert modeled.record(Scale.BRIDGE, 1, 1, 3)["source"] == "model"
    extra = Audit(challenger=Monitor(Script(lambda prompt, payload: {"width": 0.0, "forbid": payload["suspect"], "due": 1, "goal": "replace"})))
    extra.observe(bad)
    fails(ValueError, lambda: extra.record(Scale.BRIDGE, 1, 1, 3))
    omitted = Audit(challenger=Monitor(Script(lambda prompt, payload: {"width": 0.0, "forbid": [], "due": 1})))
    omitted.observe(bad)
    fails(ValueError, lambda: omitted.record(Scale.BRIDGE, 1, 1, 3))
    return {"health": "ok", "bounded": "ok", "recovery": "due", "schema": "strict", "frozen": "ok"}
def schedules() -> dict[str, str]:
    hard = {scale: Evidence(scale) for scale in Scale}
    hard[Scale.LOCAL] = Evidence(Scale.LOCAL, violation=1.0, due=True, contract=1, declared=1, relevant=1, observable=1, ready=1)
    low = {scale: Health(scale) for scale in Scale}
    low[Scale.LOCAL] = Health(Scale.LOCAL, score=0.0, silent=True, critical=True)
    scheduler = Scheduler(samples=2, dwell=0)
    pressure = scheduler.pressure(hard)
    assert pressure[Scale.LOCAL] == 1.0
    decision = scheduler.select(hard, low, Snapshot("Target", (), {}, (), 0, Budget(), {}, step=1))
    assert decision.move is Move.REPAIR and all(not key.startswith("audit") for key in decision.scores)
    audit = Audit()
    scheduler = Scheduler(samples=2, dwell=100, audit=audit)
    quiet = {scale: Evidence(scale) for scale in Scale}
    quiet[Scale.BRIDGE] = Evidence(Scale.BRIDGE, due=True, contract=1, declared=1, relevant=1)
    unhealthy = {scale: Health(scale) for scale in Scale}
    unhealthy[Scale.BRIDGE] = Health(Scale.BRIDGE, score=0.0, silent=True, critical=True)
    decision = scheduler.select(quiet, unhealthy, Snapshot("Target", (), {}, (), 0, Budget(), {}, step=1))
    assert decision.move is Move.AUDIT and decision.scale is Scale.BRIDGE
    audit.record(Scale.BRIDGE, 1, 1, 4)
    waiting = scheduler.select(quiet, unhealthy, Snapshot("Target", (), {}, (), 0, Budget(), {}, step=2))
    assert waiting.move is Move.CONTINUE
    fallback = scheduler.select(quiet, unhealthy, Snapshot("Target", (), {}, (), 0, Budget(), {}, step=3))
    assert fallback.move is Move.GLOBAL
    discounted = Scheduler(samples=2, dwell=0, discount=0.5)
    distant = dict(hard)
    distant[Scale.LOCAL] = replace(hard[Scale.LOCAL], horizon=8)
    assert discounted.pressure(distant)[Scale.LOCAL] == 1.0
    trend = Scheduler(samples=2, dwell=0)
    health = {scale: Health(scale) for scale in Scale}
    first = {scale: Evidence(scale) for scale in Scale}
    first[Scale.LOCAL] = Evidence(Scale.LOCAL, difference=0.8, due=True, contract=1, declared=1, relevant=1, observable=1, ready=1)
    assert trend.pressure(first)[Scale.LOCAL] == 0.0
    second = dict(first)
    second[Scale.LOCAL] = replace(first[Scale.LOCAL], contract=2)
    assert trend.pressure(second)[Scale.LOCAL] > 0.0
    assert trend.select(second, health, Snapshot("Target", (), {}, (), 0, Budget(), {}, step=2)).move is Move.REPAIR
    statistical = Scheduler(samples=2, dwell=0, confidence=0.2)
    for version in range(1, 13):
        sample = {scale: Evidence(scale) for scale in Scale}
        sample[Scale.LOCAL] = Evidence(Scale.LOCAL, difference=1.0, due=True, contract=version, declared=1, relevant=1, observable=1, ready=1)
        statistical.pressure(sample)
    assert statistical.sequential()["ll"]["alarm"]
    delayed = Scheduler(samples=2, dwell=0, confidence=0.2)
    for version in range(1, 13):
        sample = {scale: Evidence(scale) for scale in Scale}
        sample[Scale.LOCAL] = Evidence(Scale.LOCAL, difference=1.0, due=True, contract=version, declared=1, relevant=1, observable=1, ready=1, horizon=2)
        delayed.pressure(sample)
    assert delayed.sequential()["ll"]["samples"] == 0 and delayed.sequential()["ll"]["delayed"] == 12 and not delayed.sequential()["ll"]["alarm"]
    vonly = Scheduler(samples=1, dwell=0, channels="v")
    forecastonly = {scale: Evidence(scale) for scale in Scale}
    forecastonly[Scale.LOCAL] = Evidence(Scale.LOCAL, difference=1.0, due=True, contract=1, declared=1, relevant=1, observable=1, ready=1)
    assert vonly.pressure(forecastonly)[Scale.LOCAL] == 0.0 and vonly.sequential()["ll"]["samples"] == 0
    donly = Scheduler(samples=1, dwell=0, channels="d")
    assert donly.pressure(hard)[Scale.LOCAL] == 0.0
    noaudit = Scheduler(samples=2, dwell=0, channels="vdh", auditing=False)
    assert noaudit.select(quiet, unhealthy, Snapshot("Target", (), {}, (), 0, Budget(), {}, step=1)).move is Move.CONTINUE
    route = Router.build({"labels": list(ACTIONS), "mean": [0.0] * 12, "scale": [1.0] * 12, "weights": [[0.0] * 12 for _ in ACTIONS], "bias": [10.0 if label == "bridge:ll" else 0.0 for label in ACTIONS]})
    routed = Scheduler(samples=2, dwell=0, router=route)
    routeddecision = routed.select(hard, {scale: Health(scale) for scale in Scale}, Snapshot("Target", (), {}, (), 0, Budget(), {}, step=1))
    assert routeddecision.move is Move.BRIDGE and routeddecision.scale is Scale.LOCAL
    fails(ValueError, lambda: Scheduler(channels=""))
    return {"hard": "ok", "health": "separate", "timeout": "ok", "sequential": "ok", "eprocess": "h1", "multihorizon": "delayed", "ablation": "isolated", "router": "loaded"}
def credits() -> dict[str, str]:
    ledger = Ledger()
    evidence = {scale: Evidence(scale) for scale in Scale}
    evidence[Scale.GLOBAL] = Evidence(Scale.GLOBAL, violation=0.9, due=True, contract=1)
    health = {scale: Health(scale) for scale in Scale}
    assert ledger.support(Move.CONTINUE, None, evidence, health) < 0.2
    assert ledger.support(Move.REPAIR, None, evidence, health) == 0.0
    assert ledger.support(Move.GLOBAL, None, evidence, health) == 0.9
    health[Scale.LOCAL] = Health(Scale.LOCAL, score=0.2, silent=True)
    assert abs(ledger.support(Move.AUDIT, Scale.LOCAL, evidence, health) - 0.8) < 1e-9
    return {"aligned": "ok", "audit": "health"}
def engine() -> dict[str, str]:
    rule = Rule()
    seen: list[dict[str, object]] = []
    def handler(prompt: str, payload: Mapping[str, object]) -> dict[str, object]:
        seen.append(dict(payload))
        return rule.generate(prompt, payload)
    model = Script(handler)
    reference = Reference(warmup=2)
    snapshot = Snapshot("Reach the target", ("Keep policy",), {"progress": 0, "policy": "stable"}, (), 0, Budget(), {"progress": 0, "policy": "stable"})
    store = Store(snapshot)
    agent = Engine(store, Governor(model), Compiler(model), Executor(model), reference=reference)
    agent.start()
    fails(RuntimeError, lambda: agent.observe(Observation(1, {"progress": 1})))
    action = agent.propose()
    first = agent.observe(Observation(1, {"progress": 0, "policy": "stable"}))
    assert first["move"] == "repair" and first["evidence"]["ll"]["violation"] == 1.0
    assert first["evidence"]["ll"]["ready"] == 0
    assert reference.samples("progress", Scale.LOCAL, action.name, 1) == 1
    assert len(reference.series["progress"]) == 2
    agent.feedback(False, 0.1)
    agent.propose()
    assert cast(dict[str, object], seen[-1]["violation"])["violation"] == 1.0
    second = agent.observe(Observation(2, {"progress": 1, "policy": "stable"}))
    assert second["evidence"]["ll"]["progressed"]
    assert second["evidence"]["ll"]["ready"] == 0
    assert reference.samples("progress", Scale.LOCAL, action.name, 1) == 2
    agent.feedback(True)
    third = agent.propose()
    cursor = store.read().cursor
    final = agent.observe(Observation(3, {"progress": 2, "policy": "stable", "status": "DONE"}))
    assert final["move"] == "continue" and final["evidence"]["ll"]["ready"] == 1 and store.read().terminal()
    assert third.name == action.name and store.read().cursor == cursor
    fails(RuntimeError, agent.propose)
    predicate = Terminal("Solved", ("YES",))
    gatedstore = Store(Snapshot("Reach", (), {"progress": 0, "policy": "stable", "solved": "no"}, (), 0, Budget(), {"progress": 0, "policy": "stable", "solved": "no"}))
    gated = Engine(gatedstore, Governor(rule), Compiler(rule), Executor(rule), terminal=predicate)
    gated.start()
    gated.propose()
    gated.observe(Observation(1, {"progress": 1, "policy": "stable", "solved": "yes"}))
    assert gatedstore.read().terminal()
    untrustedstore = Store(Snapshot("Reach", (), {"progress": 0, "policy": "stable", "solved": "no"}, (), 0, Budget(), {"progress": 0, "policy": "stable", "solved": "no"}))
    untrusted = Engine(untrustedstore, Governor(rule), Compiler(rule), Executor(rule), terminal=predicate)
    untrusted.start()
    untrusted.propose()
    untrusted.observe(Observation(1, {"progress": 1, "solved": "yes"}, verified=False))
    assert not untrustedstore.read().terminal()
    return {"leakage": "blocked", "progress": "ok", "terminal": "external"}
def boundaries() -> dict[str, str]:
    values = normalize({"DataSet": " ALFWorld ", "meta": {"benchmark_name": " SWE-Bench "}})
    assert values == {"dataset": "alfworld", "meta": {"dataset": "swe-bench"}}
    assert identify(" ALFWORLD ", {"alfworld": "matched"}) == "matched"
    fails(ValueError, lambda: normalize({"dataset": "gaia", "Benchmark": "alfworld"}))
    fails(RuntimeError, lambda: Budget(tokens=1).spend(2))
    fails(ValueError, lambda: Prediction(Kind.INTERVAL, 2.0, 1.0))
    fails(ValueError, lambda: Prediction(Kind.CATEGORICAL, shares={"a": 1.0, "b": -1.0}))
    assert Prediction.cast(True).shares == {"true": 1.0}
    assert Prediction.cast("READY").shares == {"ready": 1.0}
    assert Prediction.cast(1, verifier=View.NONE).verifier is View.NONE
    assert parse('prefix {"first":1} suffix {"second":2}') == {"first": 1}
    content, calls, rejected = toolcalls('before<tool_call>{"name":"Search","arguments":{"q":1}}</tool_call>after')
    assert content == "beforeafter" and calls[0]["function"]["name"] == "Search" and calls[0]["function"]["arguments"] == '{"q":1}' and rejected == []
    invalid, absent, rejected = toolcalls('<tool_call>{"name":</tool_call>')
    assert invalid == '<tool_call>{"name":</tool_call>' and absent == [] and rejected == []
    tools = [{"type": "function", "function": {"name": "blocked", "parameters": {"required": ["Preference", "TimeInfo"]}}}, {"type": "function", "function": {"name": "advance", "description": "This tool has a cost of 2 units. The output type of this tool is Preference.", "parameters": {"required": ["Candidate", "TimeInfo"]}}}]
    history = [{"role": "assistant", "tool_calls": [{"function": {"name": "blocked", "arguments": "{}"}}]}, {"role": "tool", "name": "blocked", "content": "[ERROR] Missing required input types. Required: ['Preference', 'TimeInfo'], Current: ['Candidate', 'TimeInfo']"}]
    revised, visible, event = intervene(history, tools)
    assert event is not None and event["scale"] == "gl" and event["action"] == "bridge" and event["path"] == ["advance"] and [item["function"]["name"] for item in visible] == ["advance"] and revised[-1]["role"] == "system"
    repeated = history * 2
    assert intervene(repeated, tools)[2]["scale"] == "gl"
    assert intervene(history * 4, tools)[2]["scale"] == "gl"
    assert intervene(history * 4, tools, False, "ll")[2]["scale"] == "ll"
    assert intervene(history * 4, tools, False, "llgl")[2]["scale"] == "gl"
    filtered, filteredtools, filterevent = intervene(history, tools, False, "filter")
    assert filterevent is not None and filterevent["scale"] == "filter" and filteredtools[0]["function"]["name"] == "advance" and filtered[-1]["role"] == "tool"
    planner = [{"type": "function", "function": {"name": name, "description": f"This tool has a cost of {price} units. The output type of this tool is {produced}.", "parameters": {"required": needs}}} for name, needs, produced, price in (("Direct", ["TimeInfo"], "FinalSet", 12), ("Seed", ["TimeInfo"], "RawSet", 2), ("Refine", ["RawSet"], "FinalSet", 3), ("Select_Final_Item", ["FinalSet"], "Choice", 1))]
    first = [{"role": "user", "content": "Plan my trip during <TimeInfo03119>."}]
    planned, available, proposal = intervene(first, planner, False, "full")
    assert proposal is not None and proposal["action"] == "plan" and proposal["path"] == ["Seed", "Refine", "Select_Final_Item"] and proposal["cost"] == 6 and available == planner and planned[-1]["role"] == "system"
    typed = [{"type": "function", "function": {"name": name, "description": f"This tool has a cost of {price} units. The output type of this tool is {produced}.", "parameters": {"required": needs}}} for name, needs, produced, price in (("Location_Preference_and_Search", ["LocationCategory", "LocationTier", "LocationStyle", "LocationFeaturePackage", "TimeInfo"], "LocationCandidate_Raw", 7), ("Location_Refinement_Step1", ["LocationCandidate_Raw"], "LocationCandidate_L1_Available", 4), ("Select_Final_Location", ["LocationCandidate_L1_Available"], "TravelLocation", 3))]
    context = [{"role": "system", "content": "You should infer LocationCategory, LocationTier, LocationStyle, LocationFeaturePackage, TimeInfo from the user query."}, {"role": "user", "content": "Visit a city during <TimeInfo03119>."}]
    _, _, typedplan = intervene(context, typed, False, "full")
    assert typedplan is not None and typedplan["path"] == ["Location_Preference_and_Search", "Location_Refinement_Step1", "Select_Final_Location"]
    assert intervene(first, planner, False, "full", False)[2] is None
    assert intervene(first, planner, False, "filter")[2] is None
    changed = [*history, {"role": "user", "content": "I changed my preference: I now want another destination."}]
    assert intervene(changed, tools, False, "full")[2]["scale"] == "gg"
    updated = [*context, {"role": "assistant", "tool_calls": [{"function": {"name": "Location_Preference_and_Search", "arguments": "{}"}}]}, {"role": "tool", "content": "Get raw candidate set (Type: LocationCandidate_Raw)"}, {"role": "user", "content": "I think I'll go with different requirements: explore a mountain instead."}]
    assert intervene(updated, typed, False, "full")[2]["action"] == "replan"
    dynamictools = [{"type": "function", "function": {"name": name, "description": f"This tool has a cost of {price} units. The output type of this tool is {produced}.", "parameters": {"required": needs}}} for name, needs, produced, price in (("Decide_Item_Preference", ["TimeInfo", "LocationPreference"], "ItemPreference", 10), ("Select_Final_Item", ["ItemPreference"], "Choice", 2))]
    dynamic = [{"role": "user", "content": "Plan during <TimeInfo02502> at <LocationPreference00157>."}, {"role": "assistant", "tool_calls": [{"function": {"name": "Decide_Item_Preference", "arguments": "{}"}}]}, {"role": "tool", "content": "Preference decided."}, {"role": "user", "content": "I changed my requirements: This time go to <LocationPreference00082>."}]
    _, _, recourse = intervene(dynamic, dynamictools, False, "full")
    assert recourse is not None and recourse["path"] == ["Decide_Item_Preference", "Select_Final_Item"] and recourse["cost"] == 12 and recourse["stable_time"] == "<TimeInfo02502>"
    shifted = [*dynamic[:-1], {"role": "user", "content": "I changed my requirements and travel date to tomorrow at <LocationPreference00082>."}]
    assert intervene(shifted, dynamictools, False, "full")[2]["path"] == []
    newtime = [*dynamic[:-1], {"role": "user", "content": "I changed my requirements to <TimeInfo03000> at <LocationPreference00082>."}]
    assert intervene(newtime, dynamictools, False, "full")[2]["stable_time"] == ""
    tooltext = [*context, {"role": "assistant", "tool_calls": [{"function": {"name": "Location_Preference_and_Search", "arguments": "{}"}}]}, {"role": "tool", "content": "Try a different requirement instead."}]
    assert intervene(tooltext, typed, False, "full")[2] is None
    malformed = [{"role": "assistant", "tool_calls": [{"function": {"name": "blocked", "arguments": "{}"}}]}, {"role": "tool", "name": "blocked", "content": "[ERROR] Invalid ID length in the input."}]
    assert intervene(malformed, tools, False, "full")[2]["action"] == "repair"
    absent = [{"role": "assistant", "tool_calls": [{"function": {"name": "blocked", "arguments": "{}"}}]}, {"role": "tool", "name": "blocked", "content": "[ERROR] Unfortunately, this tool is currently malfunctioning."}]
    assert intervene(absent, tools, False, "full")[2]["scale"] == "filter"
    successful = [{"role": "assistant", "tool_calls": [{"function": {"name": "advance", "arguments": "{}"}}]}, {"role": "tool", "name": "advance", "content": "Get a candidate (Type: Candidate)."}]
    assert intervene(successful * 2, tools, False, "full")[2] is None
    fails(ValueError, lambda: intervene(history, tools, False, "invalid"))
    audittools = [*tools, {"type": "function", "function": {"name": "Select_Final_Item", "parameters": {"required": ["Candidate"]}}}]
    audithistory = [{"role": "assistant", "tool_calls": [{"function": {"name": "Item_Refinement_Step2", "arguments": "{}"}}]}, {"role": "tool", "name": "Item_Refinement_Step2", "content": "Candidates are ready."}]
    reviewed, available, precommit = intervene(audithistory, audittools)
    assert precommit is not None and precommit["scale"] == "gl" and precommit["action"] == "audit" and available == audittools and reviewed[-1]["role"] == "system"
    assert intervene(audithistory, audittools, False)[2] is None
    _, selected, rejected = toolcalls('<tool_call>{"name":"Select_Final_Item","arguments":{"candidate":"Candidate00763"}}</tool_call>', audittools)
    assert json.loads(selected[0]["function"]["arguments"])["candidate"] == "00763"
    assert rejected == []
    missing, accepted, rejected = toolcalls('<tool_call>{"name":"blocked","arguments":{}}</tool_call>', filteredtools)
    assert missing is None and accepted == [] and rejected == ["blocked"]
    missing, accepted, rejected = toolcalls('<tool_call>{"name":"advance","arguments":{}}</tool_call>', filteredtools)
    assert missing is None and accepted[0]["function"]["name"] == "advance" and rejected == []
    fails(ValueError, lambda: parse('{"value":NaN}'))
    fails(ValueError, lambda: parse('{"value":1,"value":2}'))
    assert Http("https://example.com/v1/chat/completions", "model", "").url == "https://example.com/v1/chat/completions"
    fails(ValueError, lambda: Http("ftp://model", "model", ""))
    store = Store(Snapshot("Copy", (), {"value": 1}, (), 0, Budget(), {"value": 1}))
    copied = store.read()
    cast(dict[str, object], copied.environment)["value"] = 9
    assert store.read().environment["value"] == 1
    versions = Store(Snapshot("Versions", (), {}, (), 0, Budget(), {}))
    versions.commit(Contract.build(Scale.GLOBAL, 1, None, 0, {"predictions": {"x": 1}, "due": 1}))
    versions.commit(Contract.build(Scale.BRIDGE, 1, 1, 0, {"predictions": {"x": 1}, "due": 1}))
    versions.commit(Contract.build(Scale.LOCAL, 1, 1, 0, {"predictions": {"x": 1}, "due": 1}))
    versions.commit(Contract.build(Scale.GLOBAL, 2, None, 0, {"predictions": {"x": 1}, "due": 1}))
    versions.commit(Contract.build(Scale.BRIDGE, 2, 2, 0, {"predictions": {"x": 1}, "due": 1}))
    assert versions.nextversion(Scale.LOCAL) == 2
    timeline = Store(Snapshot("Timeline", (), {"progress": 0}, (), 0, Budget(), {"progress": 0}))
    timeline.commit(Contract.build(Scale.GLOBAL, 1, None, 0, {"predictions": {"progress": 2}, "due": 2}))
    timeline.commit(Contract.build(Scale.BRIDGE, 1, 1, 0, {"predictions": {"progress": 2}, "due": 2}))
    timeline.commit(Contract.build(Scale.LOCAL, 1, 1, 0, {"predictions": {"progress": 2}, "progress": "progress", "due": 2}))
    issued = timeline.anchors()
    middle = timeline.apply(Observation(1, {"progress": 2}))
    ending = timeline.apply(Observation(2, {"progress": 2}))
    horizon = Verifier(Reference(warmup=1)).scan(timeline.contracts(), middle, ending, anchors=issued)[Scale.LOCAL]
    assert horizon.progressed and not horizon.expired
    late = Verifier().check(Contract.build(Scale.LOCAL, 1, None, 0, {"predictions": {"progress": 1}, "progress": "progress", "due": 1}), issued[Scale.LOCAL], replace(ending, step=3))
    assert late.expired and late.due and late.violation == late.difference == 0.0
    def adversary(prompt: str, payload: Mapping[str, object]) -> dict[str, object]:
        if payload["kind"] == "gg":
            return {"goal": "Replaced", "assumptions": ["Changed"], "guarantees": ["Changed"], "predictions": {"safe": "yes"}, "due": 9, "invariants": ["safe"]}
        return {"plan": ["replaced"], "assumptions": ["Changed"], "guarantees": ["Changed"], "predictions": {"safe": "yes"}, "due": 9, "invariants": ["safe"]}
    protected = Store(Snapshot("Protected", (), {"safe": "yes"}, ("keep",), 0, Budget(), {"safe": "yes"}))
    Governor(Script(adversary)).create(protected, audit={"due": 1, "width": 0.0})
    assert protected.read().goal == "Protected"
    Compiler(Script(adversary)).create(protected, audit={"due": 1, "width": 0.0})
    assert protected.read().plan == ("keep",)
    assert not local("runs/reference.jsonl").is_absolute()
    fails(ValueError, lambda: local(str(Path.cwd())))
    record = local("recordtest.jsonl")
    try:
        Journal(str(record)).write({"template": "one", "values": {"dataset": "ALFWorld"}})
        assert rows(str(record))[0]["template"] == "one"
        assert document(str(record))["template"] == "one"
    finally:
        record.unlink(missing_ok=True)
    return {"dataset": "lower", "json": "ok", "copy": "safe", "paths": "relative", "horizon": "anchored", "expiry": "isolated"}
def probes() -> dict[str, object]:
    report = TriPerturb().report()
    assert report["cases"] == 5 and report["accuracy"] == 1.0 and report["macrof1"] == 1.0
    return report
def metrics() -> dict[str, object]:
    interval = paired([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], resamples=200, seed=7)
    assert interval["estimate"] == interval["lower"] == interval["upper"] == 1.0
    assert regret([2.0, 4.0], [1.0, 3.0]) == 1.0
    assert macrof1(["ll", "gl", "gg"], ["ll", "gl", "gg"]) == 1.0
    labels = [True, True, False, False]
    scores = [0.9, 0.8, 0.2, 0.1]
    assert auroc(labels, scores) == 1.0
    fixed = recall(labels, scores, 0.0)
    assert fixed["recall"] == 1.0 and fixed["fpr"] == 0.0
    fails(ValueError, lambda: paired([1.0], [1.0, 2.0]))
    fails(TypeError, lambda: paired([True], [0.0], resamples=100))
    fails(ValueError, lambda: paired([1.0], [0.0], resamples=100, confidence=float("nan")))
    fails(TypeError, lambda: paired([1.0], [0.0], resamples=100, seed=True))
    fails(ValueError, lambda: recall(labels, scores, float("nan")))
    return {"bootstrap": "task", "regret": "paired", "auroc": 1.0, "fixedfpr": fixed["recall"], "finite": "strict"}
if __name__ == "__main__":
    print(json.dumps({"repair": repairs(), "scores": scores(), "contracts": contracts(), "audit": audits(), "schedule": schedules(), "credit": credits(), "engine": engine(), "boundaries": boundaries(), "probes": probes(), "metrics": metrics()}, ensure_ascii=True, indent=2))

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
