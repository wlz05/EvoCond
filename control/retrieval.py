# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Aligned semantic/structural retrieval with a train-only abstaining teacher.'
from __future__ import annotations
from hashlib import sha256
import json
import math
from typing import Protocol
from control import dynamic
from control.fusion import decide as teacher_decide
FORMAT = "evocond_searl_alignment_v1"
class Encoder(Protocol):
    def encode(self, text: str) -> list[float]: ...
def _vector(value: list[float], size: int | None = None) -> list[float]:
    if not isinstance(value, (list, tuple)) or not value or (size is not None and len(value) != size):
        raise ValueError("Missing or mismatched retrieval embedding")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError("Nonfinite retrieval embedding")
    return [float(x) for x in value]
def _cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    return sum(x*y for x, y in zip(left, right)) / denominator if denominator else 0.0
def structure_text(node: dict, *, predecessor_names: list[str]) -> str:
    return ("Input contract: " + json.dumps(node["arguments"], sort_keys=True)
            + ". Output contract: " + node["returns"]
            + ". Verified predecessors: " + ", ".join(sorted(predecessor_names)))
def fit_alignment(pairs: list[dict], *, train_ids: set[str],
                  validation_ids: set[str], test_ids: set[str], ridge: float = 0.1) -> dict:
    'Learn a structural-to-semantic map from paired training tool identities.'
    if not pairs or train_ids & (validation_ids | test_ids) or validation_ids & test_ids:
        raise ValueError("Alignment needs disjoint training data")
    if not 0 < ridge <= 100:
        raise ValueError("Invalid alignment regularizer")
    seen = set()
    semantic, structural = [], []
    for pair in pairs:
        identity = (pair.get("query_id"), pair.get("tool_version"))
        if pair.get("query_id") not in train_ids or identity in seen:
            raise ValueError("Duplicate or nontraining alignment pair")
        seen.add(identity)
        semantic.append(_vector(pair.get("semantic")))
        structural.append(_vector(pair.get("structural"), len(semantic[-1])))
    dim = len(semantic[0])
    if any(len(row) != dim for row in semantic + structural):
        raise ValueError("Alignment dimensions differ")
    import numpy as np
    x, y = np.asarray(structural, dtype=np.float64), np.asarray(semantic, dtype=np.float64)
    matrix = np.linalg.solve(x.T @ x + ridge * np.eye(dim), x.T @ y)
    return {"format": FORMAT, "dimension": dim, "matrix": matrix.tolist(),
            "train_ids": sorted(train_ids), "validation_ids": sorted(validation_ids),
            "test_ids": sorted(test_ids),
            "training_sha256": sha256(json.dumps(pairs, sort_keys=True).encode("utf-8")).hexdigest()}
def _project(alignment: dict, values: list[float]) -> list[float]:
    dim = alignment.get("dimension")
    matrix = alignment.get("matrix")
    if (alignment.get("format") != FORMAT or type(dim) is not int or dim < 1
            or not isinstance(matrix, list) or len(matrix) != dim
            or any(not isinstance(row, list) or len(row) != dim
                   or any(type(x) not in (int, float) or not math.isfinite(x) for x in row)
                   for row in matrix)):
        raise ValueError("Malformed alignment snapshot")
    source = _vector(values, dim)
    return [sum(source[i] * matrix[i][j] for i in range(dim)) for j in range(dim)]
def teacher_features(query_embedding: list[float], *, turn: int,
                     goal_changed: bool, prior_failures: int, completed_subtasks: int,
                     semantic_score: float, structural_score: float,
                     reliability: float, novelty: float) -> list[float]:
    '44 features compatible with the paired-outcome retrieval teacher.'
    query = _vector(query_embedding, 32)
    state = [1.0, min(turn, 20)/20, float(goal_changed), min(prior_failures, 5)/5,
             min(completed_subtasks, 16)/16, 0.0, 0.0, 0.0]
    tail = [semantic_score, structural_score, reliability, novelty]
    if any(not math.isfinite(x) for x in tail):
        raise ValueError("Nonfinite retrieval view")
    return [*state, *query, *tail]
def rank(registry: dict, encoder: Encoder, alignment: dict, teacher: dict | None,
         *, query: str, previous_tool: str | None = None, turn: int = 0,
         goal_changed: bool = False, prior_failures: int = 0,
         completed_subtasks: int = 0, top_k: int = 5, policy_hash: str | None = None,
         previous_version: str | None = None) -> dict:
    'Rank only admitted tools; abstention ignores the old edge graph.'
    if teacher is None or teacher.get("format") == "evocond_teacher_policy_v2":
        from control.ranking import rank as rank_v2
        return rank_v2(registry, encoder, alignment, teacher, query=query,
                       previous_tool=previous_tool, previous_version=previous_version,
                       turn=turn, goal_changed=goal_changed,
                       prior_failures=prior_failures, completed_subtasks=completed_subtasks,
                       top_k=top_k, policy_hash=policy_hash)
    dynamic.validate(registry)
    if previous_version is not None:
        raise ValueError("Explicit execution versions require the v2 retrieval teacher")
    if not isinstance(query, str) or not query.strip() or not 1 <= top_k <= 100:
        raise ValueError("Invalid retrieval request")
    qvec = _vector(encoder.encode(query), alignment.get("dimension"))
    if len(qvec) != 32:
        raise ValueError("Current teacher requires a 32-dimensional encoder")
    previous_version = registry["aliases"].get(previous_tool) if previous_tool else None
    results = []
    for version, item in registry["semantic"].items():
        if registry["aliases"].get(item["name"]) != version:
            continue
        node = registry["structure"]["nodes"][version]
        predecessors = []
        for edge in registry["structure"]["edges"].values():
            if edge["target"] == version and edge["weight"] >= 0.5:
                predecessors.append(registry["structure"]["nodes"][edge["source"]]["name"])
        semantic = _cosine(qvec, _vector(encoder.encode(item["description"]), 32))
        aligned = _cosine(qvec, _project(alignment, _vector(encoder.encode(
            structure_text(node, predecessor_names=predecessors)), 32)))
        edge = registry["structure"]["edges"].get(f"{previous_version}->{version}")
        support = len(edge["evidence"]) if edge else 0
        reliability = support/(support + 3) if support else 0.0
        edge_score = 2*edge["weight"] - 1 if edge else 0.0
        structure_score = 0.5*aligned + 0.5*edge_score*reliability
        values = teacher_features(qvec, turn=turn, goal_changed=goal_changed,
                                  prior_failures=prior_failures, completed_subtasks=completed_subtasks,
                                  semantic_score=semantic, structural_score=structure_score,
                                  reliability=reliability, novelty=0.0)
        decision = teacher_decide(teacher, values, minimum_support=0.0)
        alpha = decision["structure_weight"]
        score = (1-alpha)*semantic + alpha*structure_score
        results.append({"name": item["name"], "version": version, "score": score,
                        "semantic_score": semantic, "structural_score": structure_score,
                        "support": support, "teacher": decision})
    results.sort(key=lambda row: (-row["score"], row["name"], row["version"]))
    return {"query": query, "previous_tool": previous_tool,
            "tools": results[:top_k], "candidate_count": len(results),
            "snapshot_hash": dynamic.digest(registry),
            "warning": "Legacy v1 per-candidate router; use v2 for new experiments"}

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
