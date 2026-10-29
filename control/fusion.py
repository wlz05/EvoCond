# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Train-only outcome teacher for legacy paired retrieval fusion.'
from __future__ import annotations
from hashlib import sha256
import json
import math
from typing import Any
from control.controller import context
FORMAT = "evocond_retrieval_teacher_v1"
EXTRA = 4
def features(state: dict, semantic_score: float, structural_score: float,
             edge_reliability: float, novelty: float) -> list[float]:
    tail = [semantic_score, structural_score, edge_reliability, novelty]
    if any(not math.isfinite(float(x)) for x in tail):
        raise ValueError("Nonfinite teacher feature")
    return [*context(state), *[max(-1.0, min(1.0, float(x))) for x in tail]]
def _hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")).hexdigest()
def fit(pairs: list[dict], *, train_indices: list[int],
        validation_indices: list[int], test_indices: list[int],
        ridge: float = 0.1) -> tuple[dict, dict]:
    """Fit a small CPU ridge teacher on matched semantic/structural rewards."""
    if not pairs or not 0 < ridge < 100 or not train_indices:
        raise ValueError("Missing paired training data")
    if set(train_indices) & (set(validation_indices) | set(test_indices)) or set(validation_indices) & set(test_indices):
        raise ValueError("Teacher splits overlap")
    if any(type(i) is not int or i < 0 for i in train_indices + validation_indices + test_indices):
        raise ValueError("Invalid split index")
    seen = set()
    xs, ys = [], []
    for item in pairs:
        index, mode, seed = item.get("index"), item.get("mode"), item.get("seed")
        key = (index, mode, seed)
        if type(index) is not int or index not in train_indices or key in seen:
            raise ValueError("Unpaired, duplicate, or nontraining teacher example")
        seen.add(key)
        semantic_reward, structural_reward = item.get("semantic_reward"), item.get("structural_reward")
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
               for v in (semantic_reward, structural_reward)):
            raise ValueError("Teacher needs verified paired rewards in [0,1]")
        x = item.get("features")
        if not isinstance(x, list) or len(x) != 44 or any(type(v) not in (int, float) or not math.isfinite(v) for v in x):
            raise ValueError("Malformed teacher feature vector")
        xs.append([1.0, *x])
        ys.append(float(structural_reward - semantic_reward))
    informative = sum(abs(y) > 1e-9 for y in ys)
    if informative < 3:
        raise ValueError("Fewer than three paired reward differences; teacher training is unsupported")
    import numpy as np
    matrix = np.asarray(xs, dtype=np.float64)
    targets = np.asarray(ys, dtype=np.float64)
    penalty = np.eye(matrix.shape[1]) * ridge
    penalty[0, 0] = 0.0
    weight = np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ targets)
    predicted = matrix @ weight
    residual = float(np.sqrt(np.mean((predicted - targets) ** 2)))
    model = {"format": FORMAT, "weights": weight.tolist(),
             "margin": max(0.05, min(0.5, residual)),
             "train_indices": sorted(train_indices),
             "validation_indices": sorted(validation_indices),
             "test_indices": sorted(test_indices),
             "paired_training_hash": _hash(pairs),
             "objective": "paired_downstream_reward_difference_structure_minus_semantics",
             "update_policy": "frozen_on_validation_and_test"}
    validate(model)
    return model, {"paired_examples": len(pairs), "informative_pairs": informative,
                   "train_rmse": residual, "teacher_parameters": len(weight),
                   "warning": "Training fit is not held-out benefit or causal tool-edge credit"}
def validate(model: dict) -> None:
    weights = model.get("weights")
    if model.get("format") != FORMAT or not isinstance(weights, list) or len(weights) != 45:
        raise ValueError("Invalid retrieval teacher")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in weights):
        raise ValueError("Nonfinite retrieval teacher")
    if type(model.get("margin")) not in (int, float) or not 0 <= model["margin"] <= 1:
        raise ValueError("Invalid teacher abstention margin")
    if not isinstance(model.get("paired_training_hash"), str) or len(model["paired_training_hash"]) != 64:
        raise ValueError("Missing teacher data provenance")
    parts = [model.get(key) for key in ("train_indices", "validation_indices", "test_indices")]
    if any(not isinstance(part, list) or any(type(i) is not int or i < 0 for i in part) for part in parts):
        raise ValueError("Invalid teacher split provenance")
    if set(parts[0]) & (set(parts[1]) | set(parts[2])) or set(parts[1]) & set(parts[2]):
        raise ValueError("Teacher split overlap")
def decide(model: dict, values: list[float], *, minimum_support: float = 0.0) -> dict:
    validate(model)
    if len(values) != 44 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("Invalid live teacher features")
    delta = model["weights"][0] + sum(a*b for a, b in zip(model["weights"][1:], values))
    evidence = float(values[-2])
    if evidence <= minimum_support or delta <= model["margin"]:
        return {"decision": "abstain", "structure_weight": 0.0, "predicted_delta": delta,
                "reason": "insufficient_support_or_margin"}
    alpha = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, 6*delta))))
    return {"decision": "fuse", "structure_weight": alpha,
            "predicted_delta": delta, "reason": "paired_train_reward_teacher"}

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
