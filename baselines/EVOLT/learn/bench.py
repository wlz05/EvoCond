# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Run paired plans from one world snapshot."""
from dataclasses import dataclass, field
from core.env import Spec, World
@dataclass(slots=True)
class Plan:
    calls: list[tuple[str, dict]] = field(default_factory=list)
@dataclass(slots=True)
class Result:
    success: bool
    calls: int
    errors: int
    state: str
    trace: list[dict]
    @property
    def utility(self) -> float:
        return float(self.success) - 0.05 * self.calls - 0.2 * self.errors
@dataclass(slots=True)
class Case:
    name: str
    world: World
    plans: list[Plan]
    goal: dict
    action: str
def run(world: World, plan: Plan, goal: dict) -> Result:
    branch = world.clone()
    start = len(branch.history)
    errors = 0
    for name, args in plan.calls:
        errors += int(not branch.execute(name, args)["ok"])
    trace = branch.history[start:]
    return Result(branch.verify(goal), len(plan.calls), errors, branch.fingerprint(), trace)
def compare(world: World, plans: list[Plan], goal: dict) -> list[Result]:
    return [run(world, plan, goal) for plan in plans]
def suite() -> list[Case]:
    alias = World({}, [Spec("search", {"query": "string"}, {}, {"found": "$arg:query"})])
    alias.shift("search", "finder")
    schema = World({}, [Spec("search", {"query": "string"}, {}, {"found": "$arg:query"})])
    schema.shift("search", inputs={"query": "integer"})
    novel = World({}, [Spec("solve", {"value": "integer"}, {}, {"solved": "$arg:value"})])
    retired = World({}, [Spec("oldsearch", {"query": "string"}, {}, {"found": "$arg:query"})])
    retired.shift("oldsearch", active=False)
    retired.put(Spec("newsearch", {"query": "string"}, {}, {"found": "$arg:query"}))
    order = World({"ready": False}, [Spec("login", {"token": "string"}, {}, {"ready": True}), Spec("search", {"query": "string"}, {"ready": True}, {"found": "$arg:query"})])
    return [Case("alias", alias, [Plan([("search", {"query": "facts"})]), Plan([("missing", {})])], {"found": "facts"}, "reuse"), Case("schema", schema, [Plan([("search", {"query": 7})]), Plan([("search", {"query": "7"})])], {"found": 7}, "split"), Case("novel", novel, [Plan([("solve", {"value": 4})]), Plan([("missing", {})])], {"solved": 4}, "create"), Case("retired", retired, [Plan([("newsearch", {"query": "facts"})]), Plan([("oldsearch", {"query": "facts"})])], {"found": "facts"}, "abstain"), Case("order", order, [Plan([("login", {"token": "x"}), ("search", {"query": "facts"})]), Plan([("search", {"query": "facts"}), ("login", {"token": "x"})])], {"found": "facts"}, "reuse")]
def evaluate(cases: list[Case] | None = None) -> dict:
    cases = suite() if cases is None else cases
    rows = []
    for case in cases:
        if not isinstance(case, Case) or not case.plans:
            raise ValueError("benchmark case is invalid")
        results = compare(case.world, case.plans, case.goal)
        best = max(range(len(results)), key=lambda pos: results[pos].utility)
        rows.append({"name": case.name, "action": case.action, "best": best, "utility": [item.utility for item in results], "success": [item.success for item in results]})
    return {"cases": rows, "oracle": sum(item["best"] == 0 for item in rows) / max(1, len(rows))}

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
