# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Executed LL/GL/GG recovery for the adapted dynamic-tool environment.'
from __future__ import annotations
import ast
from copy import deepcopy
from hashlib import sha256
import json
import re
from control import subtasks
from control.contracts import strict_json
FORMAT = "evocond_searl_recovery_v1"
CONTROL_TOOL = "submit_revised_plan"
MAX_FAILURES = 1024
def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()
def _goal_hash(question):
    if not isinstance(question, str) or not question.strip() or len(question) > 1_000_000:
        raise ValueError("Recovery requires a bounded visible user question")
    return sha256(question.strip().encode()).hexdigest()
def _proposals(plan):
    return [{key: item[key] for key in ("id", "goal", "depends_on", "evidence_required", 'inputs', 'outputs') if key in item}
            for item in plan["tasks"].values()]
def _make_plan(question, proposals, epoch):
    if not isinstance(proposals, list) or any(not isinstance(item, dict)
            or not isinstance(item.get("depends_on"), list)
            or any(not isinstance(dep, str) for dep in item["depends_on"]) for item in proposals):
        raise ValueError("Subtask dependencies must be string identifiers")
    return subtasks.make_plan(question, proposals, epoch=epoch)
def _checked_plan(plan, question, epoch):
    if plan is None:
        return None
    if (not isinstance(plan, dict) or plan.get("format") != subtasks.FORMAT
            or plan.get("epoch") != epoch or plan.get("question_sha256") != _goal_hash(question)
            or not isinstance(plan.get("tasks"), dict)):
        raise ValueError("Plan is not bound to this current visible goal epoch")
    task_fields = {"id", "goal", "depends_on", "evidence_required", "status", "evidence"}
    if any(not isinstance(identity, str) or not isinstance(task, dict) or not task_fields <= set(task) or set(task)-task_fields-{'inputs', 'outputs'}
           or task["id"] != identity for identity, task in plan["tasks"].items()):
        raise ValueError("Subtask ledger keys and identifiers must match")
    rebuilt = _make_plan(question, _proposals(plan), epoch)
    for identity, task in plan["tasks"].items():
        status, evidence = task.get("status"), task.get("evidence")
        if status not in ("pending", "completed") or status == "pending" and evidence is not None:
            raise ValueError("Invalid subtask execution ledger")
        if status == "completed":
            if (not isinstance(evidence, dict) or set(evidence) != {"tool_version", "result_sha256", "epoch"}
                    or evidence["epoch"] != epoch or not isinstance(evidence["tool_version"], str)
                    or not evidence["tool_version"] or not isinstance(evidence["result_sha256"], str)
                    or not re.fullmatch(r"[a-f0-9]{64}", evidence["result_sha256"])):
                raise ValueError("Completed subtask lacks current-version runtime evidence")
            rebuilt["tasks"][identity].update(status=status, evidence=deepcopy(evidence))
    if any(task["status"] == "completed" and any(rebuilt["tasks"][dep]["status"] != "completed"
            for dep in task["depends_on"]) for task in rebuilt["tasks"].values()):
        raise ValueError("Completed subtask depends on uncompleted evidence")
    rebuilt["revision_of"] = plan.get("revision_of")
    return rebuilt
def validate(state):
    fields = {"format", "question", "question_sha256", "goal_epoch", "revision", "plan",
              "failed_calls", "failure_streak", "pending", "last_scale", "last_reason"}
    if (not isinstance(state, dict) or set(state) != fields or state.get("format") != FORMAT
            or type(state.get("goal_epoch")) is not int or state["goal_epoch"] < 0
            or type(state.get("revision")) is not int or state["revision"] < 0
            or state.get("question_sha256") != _goal_hash(state.get("question"))
            or not isinstance(state.get("failed_calls"), dict) or len(state["failed_calls"]) > MAX_FAILURES
            or not isinstance(state.get("failure_streak"), dict) or state.get("last_scale") not in ("ll", "gl", "gg")
            or not isinstance(state.get("last_reason"), str)):
        raise ValueError("Malformed recovery state")
    _checked_plan(state["plan"], state["question"], state["goal_epoch"])
    for key, record in state["failed_calls"].items():
        if (not isinstance(record, dict) or set(record) != {"tool", "tool_version", "arguments_hash", "subtask", "goal_epoch"}
                or not isinstance(record["tool"], str) or not record["tool"]
                or record["tool_version"] is not None and (not isinstance(record["tool_version"], str) or not record["tool_version"])
                or record["subtask"] is not None and not isinstance(record["subtask"], str)
                or not isinstance(record["arguments_hash"], str) or not re.fullmatch(r"[a-f0-9]{64}", record["arguments_hash"])
                or key != _hash({k: record[k] for k in ("tool", "tool_version", "arguments_hash", "goal_epoch")})
                or record["goal_epoch"] != state["goal_epoch"]):
            raise ValueError("Failure record provenance differs from this recovery epoch")
    if any(not isinstance(key, str) or type(count) is not int or count < 1
           for key, count in state["failure_streak"].items()):
        raise ValueError("Invalid failure streak")
    pending = state["pending"]
    if pending is not None:
        if (not isinstance(pending, dict) or set(pending) != {"scale", "reason", "tasks"}
                or pending["scale"] not in ("gl", "gg") or not isinstance(pending["reason"], str)
                or not isinstance(pending["tasks"], list) or any(not isinstance(item, str) for item in pending["tasks"])
                or len(pending["tasks"]) != len(set(pending["tasks"]))):
            raise ValueError("Invalid pending recovery request")
        if state["plan"] is not None and not set(pending["tasks"]) <= set(state["plan"]["tasks"]):
            raise ValueError("Pending recovery names an unknown subtask")
    return state
def new_state(question, plan=None):
    epoch = plan.get("epoch", 0) if isinstance(plan, dict) else 0
    checked = _checked_plan(plan, question, epoch)
    state = {"format": FORMAT, "question": question, "question_sha256": _goal_hash(question),
             "goal_epoch": epoch, "revision": 0, "plan": checked, "failed_calls": {}, "failure_streak": {},
             "pending": None, "last_scale": "ll", "last_reason": "no_observed_failure"}
    if checked is None:
        state.update(pending={"scale": "gl", "reason": "missing_valid_initial_plan", "tasks": []},
                     last_scale="gl", last_reason="missing_valid_initial_plan")
    validate(state)
    return state
def sync_plan(state, plan):
    'Accept only runner-owned completion updates, never a model revision.'
    validate(state)
    checked = _checked_plan(plan, state["question"], state["goal_epoch"])
    original = state["plan"]
    if (original is None) != (checked is None) or original is not None and _proposals(original) != _proposals(checked):
        raise ValueError("Plan structure may change only through explicit submit_revised_plan")
    if state["pending"] and checked is not None:
        for identity in state["pending"]["tasks"]:
            if checked["tasks"][identity]["status"] != "pending":
                raise ValueError("Pending replanning cannot be bypassed by completion evidence")
    updated = deepcopy(state)
    updated["plan"] = checked
    return updated
def _call(state, tool, arguments, subtask, tool_version):
    if not isinstance(tool, str) or not tool.strip() or len(tool) > 256:
        raise ValueError("Call needs a bounded tool name")
    if isinstance(arguments, str):
        arguments = strict_json(arguments)
    if not isinstance(arguments, dict) or tool_version is not None and (not isinstance(tool_version, str) or not tool_version):
        raise ValueError("Call arguments/version are invalid")
    arguments = strict_json(json.dumps(arguments, allow_nan=False))
    if subtask is not None and (not isinstance(subtask, str) or state["plan"] is None or subtask not in state["plan"]["tasks"]):
        raise ValueError("Unknown call subtask; do not infer scope from free-form output")
    identity = {"tool": tool, "tool_version": tool_version, "arguments_hash": _hash(arguments),
                "goal_epoch": state["goal_epoch"]}
    return _hash(identity), {**identity, "subtask": subtask}
def check_call(state, tool, arguments, *, subtask=None, tool_version=None):
    validate(state)
    if tool == CONTROL_TOOL:
        return {"blocked": False, "reason": "control_plan_submission_requires_separate_validation", "scale": state["last_scale"]}
    key, _ = _call(state, tool, arguments, subtask, tool_version)
    pending = state["pending"]
    if pending and (pending["scale"] == "gg" or not pending["tasks"] or subtask is None or subtask in pending["tasks"]):
        return {"blocked": True, "reason": "explicit_revised_plan_required", "scale": pending["scale"]}
    if key in state["failed_calls"]:
        return {"blocked": True, "reason": "identical_failed_call_requires_repair", "scale": "ll"}
    return {"blocked": False, "reason": "no_recorded_recovery_block", "scale": "ll"}
def _affected(plan, subtask):
    if plan is None:
        return []
    if subtask is None:
        return sorted(plan["tasks"])
    result = {subtask}
    while True:
        added = {identity for identity, task in plan["tasks"].items() if set(task["depends_on"]) & result}
        if added <= result:
            return sorted(result)
        result.update(added)
def record_call(state, tool, arguments, *, success, subtask=None, tool_version=None, plan=None, gl_after=2):
    'Use a reliable execution receipt, not final reward or returned prose.'
    validate(state)
    if type(success) not in (bool, type(None)) or type(gl_after) is not int or gl_after < 2:
        raise ValueError("Need an explicit execution outcome and conservative GL threshold")
    updated = sync_plan(state, plan) if plan is not None else deepcopy(state)
    key, record = _call(updated, tool, arguments, subtask, tool_version)
    streak_key = subtask or "__unscoped__"
    if success is None:
        return updated
    if success:
        updated["failure_streak"].pop(streak_key, None)
        if not updated["pending"]:
            updated.update(last_scale="ll", last_reason="observed_execution_success_not_answer_truth")
        return updated
    if key not in updated["failed_calls"] and len(updated["failed_calls"]) >= MAX_FAILURES:
        raise ValueError("Recovery failure ledger budget exhausted")
    updated["failed_calls"][key] = record
    streak = updated["failure_streak"].get(streak_key, 0)+1
    updated["failure_streak"][streak_key] = streak
    if not updated["pending"]:
        updated.update(last_scale="ll", last_reason="observed_tool_execution_failure")
    if streak >= gl_after and (not updated["pending"] or updated["pending"]["scale"] == "gl"):
        affected = _affected(updated["plan"], subtask)
        if updated["pending"]:
            affected = sorted(set(affected) | set(updated["pending"]["tasks"]))
        if updated["plan"]:
            for identity in affected:
                updated["plan"]["tasks"][identity].update(status="pending", evidence=None)
        updated.update(pending={"scale": "gl", "reason": "repeated_execution_failure", "tasks": affected},
                       last_scale="gl", last_reason="explicit_subtask_replan_required")
    validate(updated)
    return updated
def revise_goal(state, question, *, source_role="user"):
    'Trusted host event only: never expose this as a model/tool operation.'
    validate(state)
    if source_role != "user":
        raise ValueError("GG requires an actual new visible user-goal event")
    updated = deepcopy(state)
    updated.update(question=question, question_sha256=_goal_hash(question), goal_epoch=state["goal_epoch"]+1,
                   revision=state["revision"]+1, plan=None, failed_calls={}, failure_streak={},
                   pending={"scale": "gg", "reason": "trusted_user_goal_version_changed", "tasks": []},
                   last_scale="gg", last_reason="old_goal_evidence_invalidated")
    validate(updated)
    return updated
def parse_searl_plan(text, question, *, epoch):
    if not isinstance(text, str) or len(text) > 1_000_000:
        raise ValueError("SEARL plan text is missing or too long")
    tasks = {}
    for match in re.finditer(r"^\s*##\s*(ST\d+):?([^\n]*)\n([\s\S]*?)(?=\n\s*##|\Z)", text, re.M):
        if match[1] in tasks:
            raise ValueError("Duplicate SEARL subtask")
        tasks[match[1]] = {"id": match[1], "goal": (match[2]+"\n"+match[3]).strip(), "depends_on": [],
                           "evidence_required": "Observed current-goal tool execution, not final-answer truth"}
    matches = list(re.finditer(r"##DAG_LIST[^\n]*\n([\s\S]*?)(?=\n##|\Z)", text))
    if not tasks or len(matches) != 1:
        raise ValueError("A revised SEARL plan needs subtasks and one explicit DAG_LIST")
    raw = re.sub(r"^```[^\n]*\n|\n```$", "", matches[0][1].strip())
    raw = re.sub(r'(?<![\w\"\x27])\b(ST\d+)\b(?![\w\"\x27])', r'"\1"', raw)
    if len(raw) > 8192:
        raise ValueError("Plan graph is too large")
    try:
        edges = ast.literal_eval(raw)
    except (ValueError, SyntaxError, RecursionError) as exc:
        raise ValueError("SEARL DAG_LIST must be a literal list, not executable code") from exc
    if not isinstance(edges, list):
        raise ValueError("SEARL DAG_LIST must be a list")
    for edge in edges:
        nodes = [edge] if isinstance(edge, str) else edge
        if not isinstance(nodes, (list, tuple)) or len(nodes) not in (1, 2) or any(not isinstance(n, str) or n not in tasks for n in nodes):
            raise ValueError("SEARL DAG refers to an unknown subtask")
        if len(nodes) == 2 and nodes[0] not in tasks[nodes[1]]["depends_on"]:
            tasks[nodes[1]]["depends_on"].append(nodes[0])
    return _make_plan(question, list(tasks.values()), epoch)
def submit_plan(state, payload):
    'Only this explicit validated control action clears GL/GG pending state.'
    validate(state)
    if isinstance(payload, str):
        payload = strict_json(payload)
    fields = {"goal_epoch", "base_revision", "question_sha256"}
    if not isinstance(payload, dict) or set(payload) not in (fields | {"tasks"}, fields | {"searl_plan"}):
        raise ValueError("Plan submission needs current goal_epoch/base_revision/question_sha256 and exactly one plan")
    if (type(payload["goal_epoch"]) is not int or type(payload["base_revision"]) is not int
            or payload["goal_epoch"] != state["goal_epoch"] or payload["base_revision"] != state["revision"]
            or payload["question_sha256"] != state["question_sha256"]):
        raise ValueError("Stale or unrelated plan submission")
    if "tasks" in payload:
        proposals = payload["tasks"]
        if not isinstance(proposals, list) or any(not isinstance(task, dict) or set(task) !=
                {"id", "goal", "depends_on", "evidence_required"} for task in proposals):
            raise ValueError("Policy may propose tasks, not completed statuses or invented evidence")
        fresh = _make_plan(state["question"], proposals, state["goal_epoch"])
    else:
        fresh = parse_searl_plan(payload["searl_plan"], state["question"], epoch=state["goal_epoch"])
    previous, pending = state["plan"], state["pending"]
    if previous is not None and _proposals(previous) == _proposals(fresh):
        raise ValueError("Revised plan must change the proposed goals or dependency structure")
    if previous is not None and pending and pending["scale"] == "gl":
        affected = set(pending["tasks"])
        roots = [identity for identity in affected if not set(previous["tasks"][identity]["depends_on"]) & affected]
        for identity in roots:
            old, replacement = previous["tasks"][identity], fresh["tasks"].get(identity)
            if replacement and all(old[key] == replacement[key] for key in ("goal", "depends_on", "evidence_required")):
                raise ValueError("A revised plan must address every pending recovery root, not only unrelated tasks")
    # Preserve only unchanged, unaffected runtime-backed completions.
    if previous is not None and pending and pending["scale"] == "gl":
        affected = set(pending["tasks"])
        for identity, task in fresh["tasks"].items():
            old = previous["tasks"].get(identity)
            if old and identity not in affected and old["status"] == "completed" and all(old[key] == task[key]
                    for key in ("goal", "depends_on", "evidence_required")):
                task.update(status="completed", evidence=deepcopy(old["evidence"]))
        # A changed predecessor invalidates all downstream completion claims.
        while True:
            invalid = [task for task in fresh["tasks"].values() if task["status"] == "completed"
                       and any(fresh["tasks"][dep]["status"] != "completed" for dep in task["depends_on"])]
            if not invalid:
                break
            for task in invalid:
                task.update(status="pending", evidence=None)
    fresh["revision_of"] = state["question_sha256"] if previous else None
    updated = deepcopy(state)
    updated.update(plan=fresh, revision=state["revision"]+1, pending=None, failure_streak={},
                   last_scale="ll", last_reason="explicit_revised_plan_validated")
    validate(updated)
    return updated
def control_schema(state):
    validate(state)
    return {"name": CONTROL_TOOL, "description": "Explicitly submit a revised subtask plan after GL/GG recovery. "
            "Use only the current user goal; this validates provenance and DAG shape, not answer truth.",
            "parameters": {"type": "object", "properties": {
                "goal_epoch": {"type": "integer", "const": state["goal_epoch"]},
                "base_revision": {"type": "integer", "const": state["revision"]},
                "question_sha256": {"type": "string", "const": state["question_sha256"]},
                "tasks": {"type": "array", "minItems": 1, "maxItems": 16, "items": {"type": "object",
                    "properties": {"id": {"type": "string"}, "goal": {"type": "string"},
                                   "depends_on": {"type": "array", "items": {"type": "string"}},
                                   "evidence_required": {"type": "string"}},
                    "required": ["id", "goal", "depends_on", "evidence_required"], "additionalProperties": False}},
                "searl_plan": {"type": "string"}},
                "required": ["goal_epoch", "base_revision", "question_sha256"], "additionalProperties": False}}

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
