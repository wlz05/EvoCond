# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Evidence-gated subtask DAGs for the dynamic-tool SEARL environment.'
from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
import json
FORMAT = "evocond_searl_subtasks_v1"
def _hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()
def make_plan(question: str, proposals: list[dict], *, epoch: int = 0) -> dict:
    if not isinstance(question, str) or not question.strip() or type(epoch) is not int or epoch < 0:
        raise ValueError("A plan requires a visible user goal and nonnegative epoch")
    if not isinstance(proposals, list) or not 1 <= len(proposals) <= 16:
        raise ValueError("Plan must contain 1 to 16 subtasks")
    tasks = {}
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError("Each subtask must be an object")
        identity, goal, dependencies, evidence = (proposal.get("id"), proposal.get("goal"),
                                                   proposal.get("depends_on"), proposal.get("evidence_required"))
        if (not isinstance(identity, str) or not identity or identity in tasks
                or not isinstance(goal, str) or not goal.strip()
                or not isinstance(dependencies, list) or len(dependencies) != len(set(dependencies))
                or not isinstance(evidence, str) or not evidence.strip()):
            raise ValueError("Malformed or duplicate subtask")
        tasks[identity] = {"id": identity, "goal": goal.strip(), "depends_on": dependencies,
                           "evidence_required": evidence.strip(), "status": "pending",
                           "evidence": None}
        for field in ('inputs', 'outputs'):
            value = proposal.get(field)
            if value is not None and not isinstance(value, dict):
                raise ValueError('Subtask input and output specifications must be objects or missing')
            json.dumps(value, allow_nan=False)
            if field in proposal:
                tasks[identity][field] = deepcopy(value)
    if any(not isinstance(dep, str) or dep not in tasks or dep == identity
           for identity, task in tasks.items() for dep in task["depends_on"]):
        raise ValueError("Subtask prerequisite is missing or self-referential")
    visiting, done = set(), set()
    def visit(identity: str) -> None:
        if identity in visiting:
            raise ValueError("Subtask dependency graph contains a cycle")
        if identity in done:
            return
        visiting.add(identity)
        for dep in tasks[identity]["depends_on"]:
            visit(dep)
        visiting.remove(identity)
        done.add(identity)
    for identity in tasks:
        visit(identity)
    return {"format": FORMAT, "epoch": epoch, "question_sha256": _hash(question.strip()),
            "tasks": tasks, "revision_of": None}
def ready(plan: dict) -> list[dict]:
    if plan.get("format") != FORMAT or not isinstance(plan.get("tasks"), dict):
        raise ValueError("Invalid subtask plan")
    tasks = plan["tasks"]
    return [deepcopy(task) for task in tasks.values()
            if task["status"] == "pending" and all(tasks[dep]["status"] == "completed"
                                                   for dep in task["depends_on"])]
def complete(plan: dict, identity: str, *, observed_tool_version: str,
             evidence: str, epoch: int) -> dict:
    if epoch != plan.get("epoch") or identity not in plan.get("tasks", {}):
        raise ValueError("Stale goal epoch or unknown subtask")
    if not isinstance(observed_tool_version, str) or not observed_tool_version:
        raise ValueError("Completion needs an observed tool version")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("Completion needs an observed result, not a planned result")
    available = {task["id"] for task in ready(plan)}
    if identity not in available:
        raise ValueError("Subtask is not ready or is already completed")
    updated = deepcopy(plan)
    updated["tasks"][identity]["status"] = "completed"
    updated["tasks"][identity]["evidence"] = {
        "tool_version": observed_tool_version, "result_sha256": _hash(evidence), "epoch": epoch}
    return updated
def revise(plan: dict, new_question: str, proposals: list[dict]) -> dict:
    if plan.get("format") != FORMAT:
        raise ValueError("Unknown previous plan")
    updated = make_plan(new_question, proposals, epoch=plan["epoch"] + 1)
    updated["revision_of"] = plan["question_sha256"]
    return updated
def parse_proposal(text: str, question: str, *, epoch: int = 0) -> dict:
    'Accept only a JSON array, never executable or free-form model output.'
    proposals = json.loads(text)
    return make_plan(question, proposals, epoch=epoch)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
