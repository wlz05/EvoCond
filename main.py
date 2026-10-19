# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import argparse
import json
import os
import sys
from control.engine import Engine
from control.audit import Audit
from control.policy import Scheduler
from control.residual import Verifier
from control.router import Router
from flow.action import Executor
from flow.plan import Compiler
from flow.world import Governor
from model.client import Http, Rule, load
from model.audit import Monitor
from control.scoring import Reference
from state.data import lower, normalize
from state.record import Journal, document, rows
from state.schema import Budget, Observation, Scale, Snapshot
from state.store import Store
from state.task import Terminal
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="evocond")
    parser.add_argument("task")
    parser.add_argument("--constraint", action="append", default=[])
    parser.add_argument("--state", default="{}")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--reference", default="")
    parser.add_argument("--holdout", action="append", default=[])
    parser.add_argument("--log", default="")
    parser.add_argument("--weights", default="")
    parser.add_argument("--router", default="")
    parser.add_argument("--terminalkey", default="")
    parser.add_argument("--terminalvalue", action="append", default=[])
    parser.add_argument("--url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--key", default="")
    parser.add_argument("--auditurl", default="")
    parser.add_argument("--auditmodel", default="")
    parser.add_argument("--auditkey", default="")
    parser.add_argument("--channels", default="vdh")
    parser.add_argument("--noaudit", action="store_true")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--health", type=float, default=0.75)
    parser.add_argument("--llmargin", type=float, default=0.1)
    parser.add_argument("--glmargin", type=float, default=0.1)
    parser.add_argument("--ggmargin", type=float, default=0.1)
    parser.add_argument("--tokens", type=int, default=100000)
    parser.add_argument("--seconds", type=float, default=3600.0)
    parser.add_argument("--tools", type=float, default=100.0)
    return parser.parse_args()
def build(args: argparse.Namespace) -> Engine:
    environment = load(args.state)
    if not isinstance(environment, dict):
        raise ValueError("State must be a JSON object")
    # Normalize the dataset before model dispatch.
    environment = normalize(environment)
    status = str(environment.pop("status", "active")).strip().lower()
    dataset = getattr(args, "dataset", "")
    if dataset:
        environment["dataset"] = lower(dataset)
    key = args.key or os.getenv("OPENAI_API_KEY", "")
    if bool(args.url) != bool(args.model):
        raise ValueError("Model URL and name must be provided together")
    if bool(args.auditurl) != bool(args.auditmodel):
        raise ValueError("Audit URL and model must be provided together")
    if args.auditurl and args.auditurl == args.url and args.auditmodel == args.model:
        raise ValueError("Audit model must use an independent endpoint or checkpoint")
    model = Http(args.url, args.model, key) if args.url else Rule()
    snapshot = Snapshot(args.task, tuple(args.constraint), dict(environment), (), 0, Budget(args.tokens, args.seconds, args.tools), dict(environment), status=status)
    store = Store(snapshot)
    if bool(args.reference) != bool(args.holdout):
        raise ValueError("Reference data and holdout must be provided together")
    if bool(args.terminalkey) != bool(args.terminalvalue):
        raise ValueError("Terminal key and values must be provided together")
    reference = Reference.crossfit(rows(args.reference), args.holdout) if args.reference else None
    terminal = Terminal(args.terminalkey, tuple(args.terminalvalue)) if args.terminalkey else None
    auditkey = args.auditkey or os.getenv("OPENAI_API_KEY", "")
    challenger = Monitor(Http(args.auditurl, args.auditmodel, auditkey)) if args.auditurl else None
    audit = Audit(silent=args.health, challenger=challenger)
    margins = {Scale.LOCAL: args.llmargin, Scale.BRIDGE: args.glmargin, Scale.GLOBAL: args.ggmargin}
    router = Router.build(document(args.router)) if args.router else None
    scheduler = Scheduler(margins=margins, samples=args.samples, confidence=args.confidence, channels=args.channels, auditing=not args.noaudit, audit=audit, router=router)
    weights = document(args.weights) if args.weights else None
    verifier = Verifier(reference, weights) if weights is not None else None
    return Engine(store, Governor(model), Compiler(model), Executor(model), verifier=verifier, scheduler=scheduler, audit=audit, reference=reference, terminal=terminal)
def run() -> int:
    args = arguments()
    engine = build(args)
    journal = Journal(args.log) if args.log else None
    engine.start()
    while engine.store.read().status == "active":
        action = engine.propose()
        request = {"action": action.name, "arguments": dict(action.arguments), "expected": dict(action.expected), "contract": action.contract}
        print(json.dumps(request, ensure_ascii=True))
        if journal is not None:
            journal.write({"kind": "action", **request})
        line = sys.stdin.readline()
        if not line:
            break
        data = load(line)
        if not isinstance(data, dict):
            raise ValueError("Observation must be a JSON object")
        events = data.pop("events", ())
        verified = data.pop("verified", True)
        report = engine.observe(Observation(engine.store.read().step + 1, data, events, verified))
        print(json.dumps(report, ensure_ascii=True))
        if journal is not None:
            journal.write({"kind": "report", **report})
    return 0
if __name__ == "__main__":
    raise SystemExit(run())

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
