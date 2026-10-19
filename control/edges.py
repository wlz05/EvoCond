# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Context-conditioned directed edge evidence for dynamic tool retrieval.'
from __future__ import annotations
from copy import deepcopy
import math
from control import dynamic, boundary
FORMAT = "evocond_searl_edge_memory_v2"
DEFAULT_MIN_COSINE = 0.6
def _number(value, *, low: float, high: float, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Invalid {name}")
    return float(value)
def _context(value, encoder_id) -> list[float] | None:
    if value is None:
        if encoder_id is not None and (not isinstance(encoder_id, str) or not encoder_id.strip()):
            raise ValueError("Invalid context encoder identity")
        return None
    if not isinstance(encoder_id, str) or not encoder_id.strip():
        raise ValueError("Visible context requires a versioned encoder identity")
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Missing visible-context embedding")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError("Nonfinite visible-context embedding")
    norm = math.hypot(*value)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Visible-context embedding must have a finite nonzero norm")
    return [float(x) / norm for x in value]
def _hash(memory: dict) -> str:
    return dynamic.digest({key: value for key, value in memory.items() if key != "sha256"})
def _empty(retention: float, learning_rate: float) -> dict:
    memory = {"format": FORMAT, "config": {"retention": retention, "learning_rate": learning_rate},
              "clock": 0, "tasks": {}, "edges": {}, "legacy_provenance": None}
    memory["sha256"] = _hash(memory)
    return memory
def validate_edge_memory(registry: dict) -> None:
    'Validate v2 provenance and evidence; legacy graphs remain readable.'
    dynamic.validate(registry)
    if "edge_memory" not in registry:
        return
    memory = registry["edge_memory"]
    if not isinstance(memory, dict) or memory.get("format") != FORMAT:
        raise ValueError("Unknown edge memory version")
    if memory.get("sha256") != _hash(memory):
        raise ValueError("Edge memory content hash does not match")
    config = memory.get("config")
    if not isinstance(config, dict) or set(config) != {"retention", "learning_rate"}:
        raise ValueError("Malformed edge memory configuration")
    _number(config["retention"], low=0, high=1, name="edge retention")
    rate = _number(config["learning_rate"], low=0, high=1, name="edge learning rate")
    if rate <= 0:
        raise ValueError("Edge learning rate must be positive")
    tasks, edges = memory.get("tasks"), memory.get("edges")
    if (type(memory.get("clock")) is not int or memory["clock"] < 0
            or not isinstance(tasks, dict) or not isinstance(edges, dict)):
        raise ValueError("Malformed edge memory clock or records")
    if (any(not isinstance(key, str) or not key.strip() or type(step) is not int
            for key, step in tasks.items())
            or sorted(tasks.values()) != list(range(1, memory["clock"] + 1))):
        raise ValueError("Edge task clock must count each independent training task once")
    nodes = registry["structure"]["nodes"]
    dimensions = {}
    for key, edge in edges.items():
        if (not isinstance(edge, dict) or set(edge) != {"source", "target", "observations"}
                or edge.get("source") not in nodes or edge.get("target") not in nodes
                or key != f"{edge['source']}->{edge['target']}" or not isinstance(edge.get("observations"), dict)):
            raise ValueError("Malformed directed transition")
        for task_id, event in edge["observations"].items():
            fields = {"step", "split", "context_vector", "encoder_id", "direct_feedback", "task_success", "coactivated"}
            if (task_id not in tasks or not isinstance(event, dict)
                    or not fields <= set(event) or set(event) - fields - {"policy_evidence", "reuse_boundary"}
                    or event.get("step") != tasks[task_id] or type(event.get("step")) is not int
                    or event.get("split") != "train" or type(event.get("coactivated")) is not bool
                    or event.get("direct_feedback") not in {"verified", "failed", "unknown"}
                    or type(event.get("task_success")) not in (bool, type(None))):
                raise ValueError("Malformed or nontraining transition evidence")
            boundary.validate(event.get("reuse_boundary"), registry=registry,
                                          source_version=edge["source"], target_version=edge["target"])
            if "policy_evidence" in event:
                _validate_policy_evidence(event["policy_evidence"])
                summary = _policy_summary(event["policy_evidence"])
                if (event["direct_feedback"], event["task_success"]) != summary:
                    raise ValueError("Policy evidence summary differs from observed rollouts")
            vector = _context(event.get("context_vector"), event.get("encoder_id"))
            if vector is not None:
                if not math.isclose(math.hypot(*event["context_vector"]), 1.0, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError("Stored visible context must be normalized")
                encoder = event["encoder_id"]
                if encoder in dimensions and dimensions[encoder] != len(vector):
                    raise ValueError("One encoder identity cannot have multiple dimensions")
                dimensions[encoder] = len(vector)
    legacy = memory.get("legacy_provenance")
    if legacy is not None and (not isinstance(legacy, dict) or set(legacy) != {"sha256", "edge_count"}
                              or not isinstance(legacy["sha256"], str) or len(legacy["sha256"]) != 64
                              or any(char not in "0123456789abcdef" for char in legacy["sha256"])
                              or type(legacy["edge_count"]) is not int or legacy["edge_count"] < 0):
        raise ValueError("Malformed legacy edge provenance")
def migrate_legacy_edges(registry: dict, *, retention: float = 0.995,
                         learning_rate: float = 0.02) -> dict:
    'Preserve old edges, record their origin, and start empty contextual evidence.'
    validate_edge_memory(registry)
    if "edge_memory" in registry:
        raise ValueError("Edge memory has already been initialized")
    retention = _number(retention, low=0, high=1, name="edge retention")
    learning_rate = _number(learning_rate, low=0, high=1, name="edge learning rate")
    updated = deepcopy(registry)
    memory = updated["edge_memory"] = _empty(retention, learning_rate)
    legacy = registry["structure"]["edges"]
    memory["legacy_provenance"] = {"sha256": dynamic.digest(legacy), "edge_count": len(legacy)}
    memory["sha256"] = _hash(memory)
    validate_edge_memory(updated)
    return updated
def advance_training_task(registry: dict, *, task_id: str, split: str,
                          retention: float = 0.995, learning_rate: float = 0.02) -> dict:
    'Advance the logical clock once, including tasks with no tool transitions.'
    validate_edge_memory(registry)
    if split != "train" or not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("Only identified independent training tasks may advance edge memory")
    retention = _number(retention, low=0, high=1, name="edge retention")
    learning_rate = _number(learning_rate, low=0, high=1, name="edge learning rate")
    updated = deepcopy(registry)
    memory = updated.setdefault("edge_memory", _empty(retention, learning_rate))
    if memory["config"] != {"retention": retention, "learning_rate": learning_rate}:
        raise ValueError("Cannot change frozen edge configuration while accumulating evidence")
    if task_id not in memory["tasks"]:
        memory["clock"] += 1
        memory["tasks"][task_id] = memory["clock"]
    memory["sha256"] = _hash(memory)
    validate_edge_memory(updated)
    return updated
def observe_transition(registry: dict, *, source_name: str, target_name: str,
                       task_id: str, split: str, context_vector, encoder_id: str | None,
                       direct_feedback: str, task_success: bool | None,
                       coactivated: bool = True, retention: float = 0.995,
                       learning_rate: float = 0.02, source_version: str | None = None,
                       target_version: str | None = None, reuse_boundary: dict | None = None) -> dict:
    'Record one finalized observation per directed edge and independent task.'
    validate_edge_memory(registry)
    if split != "train" or not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("Only identified independent training tasks may update edge memory")
    if (direct_feedback not in {"verified", "failed", "unknown"}
            or type(task_success) not in (bool, type(None)) or type(coactivated) is not bool):
        raise ValueError("Invalid directly attributed transition feedback")
    retention = _number(retention, low=0, high=1, name="edge retention")
    learning_rate = _number(learning_rate, low=0, high=1, name="edge learning rate")
    vector = _context(context_vector, encoder_id)
    source = dynamic.resolve_tool_version(registry, source_name, source_version)
    target = dynamic.resolve_tool_version(registry, target_name, target_version)
    boundary.validate(reuse_boundary, registry=registry, source_version=source, target_version=target)
    updated = deepcopy(registry)
    memory = updated.setdefault("edge_memory", _empty(retention, learning_rate))
    if memory["config"] != {"retention": retention, "learning_rate": learning_rate}:
        raise ValueError("Cannot change frozen edge configuration while accumulating evidence")
    if task_id not in memory["tasks"]:
        memory["clock"] += 1
        memory["tasks"][task_id] = memory["clock"]
    key = f"{source}->{target}"
    edge = memory["edges"].setdefault(key, {"source": source, "target": target, "observations": {}})
    event = {"step": memory["tasks"][task_id], "split": "train", "context_vector": vector,
             "encoder_id": encoder_id, "direct_feedback": direct_feedback,
             "task_success": task_success, "coactivated": coactivated}
    if reuse_boundary is not None:
        event["reuse_boundary"] = deepcopy(reuse_boundary)
    if task_id in edge["observations"]:
        if edge["observations"][task_id] != event:
            raise ValueError("Contradictory observations for one independent task and edge")
        return updated
    edge["observations"][task_id] = event
    memory["sha256"] = _hash(memory)
    validate_edge_memory(updated)
    return updated
def _validate_policy_evidence(evidence):
    if (not isinstance(evidence, dict) or set(evidence) != {"policy_step", "rollouts"}
            or type(evidence.get("policy_step")) is not int or evidence["policy_step"] < 0
            or not isinstance(evidence.get("rollouts"), dict) or not evidence["rollouts"]):
        raise ValueError("Malformed policy-indexed edge evidence")
    for identity, row in evidence["rollouts"].items():
        if (not isinstance(identity, str) or not identity.strip() or len(identity) > 128
                or not isinstance(row, dict) or set(row) != {"feedback", "task_success"}
                or type(row.get("task_success")) not in (bool, type(None))
                or not isinstance(row.get("feedback"), list) or not 1 <= len(row["feedback"]) <= 1000
                or any(value not in ("verified", "failed", "unknown") for value in row["feedback"])):
            raise ValueError("Malformed finalized rollout transition feedback")
def _raw_signal(feedback, task_success):
    return (-1.0 if feedback == "failed" else
            1.0 if feedback == "verified" and task_success is True else None)
def _policy_summary(evidence):
    labels = {value for row in evidence["rollouts"].values() for value in row["feedback"] if value != "unknown"}
    outcomes = {row["task_success"] for row in evidence["rollouts"].values()}
    return (next(iter(labels)) if len(labels) == 1 else "unknown",
            next(iter(outcomes)) if len(outcomes) == 1 else None)
def _event_signal(event):
    evidence = event.get("policy_evidence")
    if evidence is None:
        return _raw_signal(event["direct_feedback"], event["task_success"])
    rollout_means = []
    for row in evidence["rollouts"].values():
        signals = [_raw_signal(value, row["task_success"]) for value in row["feedback"]]
        known = [value for value in signals if value is not None]
        if known:
            rollout_means.append(sum(known)/len(known))
    return sum(rollout_means)/len(rollout_means) if rollout_means else None
def observe_policy_transition(registry: dict, *, source_name: str, target_name: str,
                              task_id: str, split: str, context_vector, encoder_id: str | None,
                              policy_step: int, rollout_id: str, feedback: list[str], task_success: bool | None,
                              coactivated: bool = True, retention: float = 0.995,
                              learning_rate: float = 0.02, source_version: str | None = None,
                              target_version: str | None = None, reuse_boundary: dict | None = None) -> dict:
    'Refresh finalized feedback while counting one independent question once.'
    if not isinstance(rollout_id, str) or not rollout_id.strip() or len(rollout_id) > 128:
        raise ValueError("Invalid finalized rollout identity")
    sample = {"feedback": deepcopy(feedback), "task_success": task_success}
    incoming = {"policy_step": policy_step, "rollouts": {rollout_id: sample}}
    _validate_policy_evidence(incoming)
    validate_edge_memory(registry)
    if split != "train" or type(coactivated) is not bool:
        raise ValueError("Only finalized training transitions may refresh edge evidence")
    source = dynamic.resolve_tool_version(registry, source_name, source_version)
    target = dynamic.resolve_tool_version(registry, target_name, target_version)
    boundary.validate(reuse_boundary, registry=registry, source_version=source, target_version=target)
    vector = _context(context_vector, encoder_id)
    updated = advance_training_task(registry, task_id=task_id, split=split,
                                    retention=retention, learning_rate=learning_rate)
    memory = updated["edge_memory"]
    edge = memory["edges"].setdefault(f"{source}->{target}", {"source": source, "target": target, "observations": {}})
    previous = edge["observations"].get(task_id)
    if previous is not None:
        if "policy_evidence" not in previous:
            raise ValueError("Legacy finalized feedback lacks policy provenance; use a new experiment")
        prior = previous["policy_evidence"]
        if policy_step < prior["policy_step"]:
            raise ValueError("Stale policy feedback cannot overwrite a newer observation")
        if policy_step == prior["policy_step"]:
            if previous.get("reuse_boundary") != reuse_boundary:
                raise ValueError("One question/policy cannot mix different reuse boundaries")
            if previous["context_vector"] != vector or previous["encoder_id"] != encoder_id:
                raise ValueError("One question/policy cannot mix different context embeddings")
            if previous["coactivated"] != coactivated:
                raise ValueError("One question/policy cannot mix coactivation semantics")
            old_sample = prior["rollouts"].get(rollout_id)
            if old_sample is not None:
                if old_sample != sample or previous["coactivated"] != coactivated:
                    raise ValueError("Conflicting feedback for one finalized rollout")
                return updated
            incoming = deepcopy(prior)
            incoming["rollouts"][rollout_id] = sample
    direct, outcome = _policy_summary(incoming)
    edge["observations"][task_id] = {"step": memory["tasks"][task_id], "split": "train",
        "context_vector": vector, "encoder_id": encoder_id, "direct_feedback": direct,
        "task_success": outcome, "coactivated": coactivated, "policy_evidence": incoming}
    if reuse_boundary is not None:
        edge["observations"][task_id]["reuse_boundary"] = deepcopy(reuse_boundary)
    memory["sha256"] = _hash(memory)
    validate_edge_memory(updated)
    return updated
def edge_view(registry: dict, source_version: str | None, target_version: str,
              *, context_vector, encoder_id: str | None,
              min_cosine: float = DEFAULT_MIN_COSINE, prior_support: float = 3.0,
              reuse_boundary_mode: str = "off", current_user_text: str | None = None) -> dict:
    'Read a frozen edge; unrelated/missing contexts supply zero historical utility.'
    validate_edge_memory(registry)
    boundary.visible_text(current_user_text, mode=reuse_boundary_mode)
    vector, min_cosine, prior_support = _view_arguments(context_vector, encoder_id, min_cosine, prior_support)
    memory = registry.get("edge_memory")
    edge = memory["edges"].get(f"{source_version}->{target_version}") if memory else None
    return _read_edge(memory, edge, vector, encoder_id, min_cosine, prior_support, reuse_boundary_mode, current_user_text)
def batch_edge_views(registry: dict, *, context_vector, encoder_id: str | None,
                     min_cosine: float = DEFAULT_MIN_COSINE, prior_support: float = 3.0,
                     reuse_boundary_mode: str = "off", current_user_text: str | None = None) -> dict:
    'Read all contextual edges with one registry/provenance validation.'
    validate_edge_memory(registry)
    boundary.visible_text(current_user_text, mode=reuse_boundary_mode)
    vector, min_cosine, prior_support = _view_arguments(context_vector, encoder_id, min_cosine, prior_support)
    memory = registry.get("edge_memory")
    if memory is None:
        return {}
    return {key: _read_edge(memory, edge, vector, encoder_id, min_cosine, prior_support, reuse_boundary_mode, current_user_text)
            for key, edge in memory["edges"].items()}
def _view_arguments(context_vector, encoder_id, min_cosine, prior_support):
    min_cosine = _number(min_cosine, low=0, high=1, name="context threshold")
    prior_support = _number(prior_support, low=0, high=1e9, name="prior support")
    if min_cosine >= 1 or prior_support <= 0:
        raise ValueError("Context threshold must be below one and prior support positive")
    vector = _context(context_vector, encoder_id)
    return vector, min_cosine, prior_support
def _read_edge(memory, edge, vector, encoder_id, min_cosine, prior_support, boundary_mode, current_user_text):
    result = {"association": 0.0, "utility": 0.0, "reliability": 0.0,
              "support": 0.0, "score": 0.0, "context_match": 0.0,
              "independent_tasks": 0, "matching_tasks": 0, "format": FORMAT}
    if boundary_mode == "explicit":
        result["reuse_boundary"] = {"matched": 0, "conflict": 0, "unknown": 0, "reasons": {}}
    if edge is None:
        return result
    retention, rate = memory["config"]["retention"], memory["config"]["learning_rate"]
    association = utility_mass = support = 0.0
    previous_step = 0
    for event in sorted(edge["observations"].values(), key=lambda item: item["step"]):
        decay = retention ** (memory["clock"] - event["step"])
        association *= retention ** (event["step"] - previous_step)
        previous_step = event["step"]
        if boundary_mode == "explicit":
            check = boundary.compare(event.get("reuse_boundary"), current_user_text, checked=True)
            audit = result["reuse_boundary"]
            audit[check["state"]] += 1
            audit["reasons"][check["reason"]] = audit["reasons"].get(check["reason"], 0)+1
            if not check["historical_signal_allowed"]:
                continue
        if event["coactivated"]:
            association = min(1.0, association + rate)
        known = event["context_vector"]
        if vector is None or known is None or event["encoder_id"] != encoder_id:
            continue
        if len(vector) != len(known):
            raise ValueError("Query context dimension differs from its encoder identity")
        cosine = max(-1.0, min(1.0, sum(left * right for left, right in zip(vector, known))))
        result["context_match"] = max(result["context_match"], cosine)
        relevance = max(0.0, (cosine - min_cosine) / (1.0 - min_cosine))
        signal = _event_signal(event)
        if relevance > 0 and signal is not None and decay > 0:
            weight = relevance * decay
            support += weight
            utility_mass += weight * signal
            result["matching_tasks"] += 1
    association *= retention ** (memory["clock"] - previous_step)
    result.update(association=association, support=support,
                  utility=utility_mass / support if support else 0.0,
                  reliability=support / (support + prior_support),
                  independent_tasks=len(edge["observations"]))
    result["score"] = result["association"] * result["utility"] * result["reliability"]
    return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
