# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Calibrated action-level retrieval teacher fitted on executed training pairs.'
from __future__ import annotations
from hashlib import sha256
import json
import math
from typing import Any
FORMAT = "evocond_teacher_policy_v2"
PAIR_FORMAT = "evocond_teacher_pairs_v2"
ACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
FEATURE_NAMES = ("semantic_top_score", "structural_top_score", "semantic_margin",
                 "structural_margin", "rank_agreement", "edge_reliability",
                 "query_novelty", "goal_changed", "prior_failures", "turn",
                 "completed_subtasks")
PROVENANCE_FIELDS = ("context_hash", "candidate_hash", "registry_hash",
                     "policy_hash", "alignment_hash", "encoder_id")
def _number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False
def _hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")).hexdigest()
def _digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))
def _vector(values: Any, size: int | None = None) -> list[float]:
    if (not isinstance(values, (list, tuple)) or not values
            or (size is not None and len(values) != size)
            or any(not _number(value) for value in values)):
        raise ValueError("Invalid teacher query embedding")
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Teacher query embedding must be nonzero")
    return [float(value)/norm for value in values]
def _features(values: Any) -> dict[str, float]:
    if not isinstance(values, dict) or set(values) != set(FEATURE_NAMES):
        raise ValueError("Teacher features must contain exactly the public feature names")
    result = {}
    for name in FEATURE_NAMES:
        value = values[name]
        low = -1.0 if name in ("semantic_top_score", "structural_top_score") else 0.0
        if not _number(value) or not low <= value <= 1.0:
            raise ValueError(f"Teacher feature {name} is outside its normalized range")
        result[name] = float(value)
    return result
def _novelty(vector: list[float], prototypes: list[dict], excluded_id: str | None = None) -> float:
    similarities = [sum(a*b for a, b in zip(vector, item["embedding"]))
                    for item in prototypes if item["query_id"] != excluded_id]
    return max(0.0, min(1.0, 1.0-max(similarities))) if similarities else 1.0
def query_novelty(model: dict, embedding: list[float], encoder_id: str) -> float:
    'Distance from the nearest fit-only query, without using held-out labels.'
    validate(model)
    if encoder_id != model["encoder_id"]:
        raise ValueError("Teacher query encoder differs from fitted encoder")
    return _novelty(_vector(embedding, model["embedding_dimension"]), model["query_prototypes"])
def _identities(values: Any, name: str) -> set[str]:
    if not isinstance(values, (set, list, tuple)) or any(not isinstance(x, str) or not x for x in values):
        raise ValueError(f"Invalid {name} identities")
    result = set(values)
    if len(result) != len(values):
        raise ValueError(f"Duplicate {name} identities")
    return result
def _record(row: dict, allowed: set[str]) -> dict:
    if not isinstance(row, dict) or row.get("schema") != PAIR_FORMAT:
        raise ValueError("Invalid teacher pair schema")
    if row.get("query_id") not in allowed or not isinstance(row.get("decision_id"), str) or not row["decision_id"]:
        raise ValueError("Teacher pair needs an allowed question and decision identity")
    if type(row.get("seed")) is not int or row["seed"] < 0:
        raise ValueError("Teacher pair needs an explicit nonnegative seed")
    for name in PROVENANCE_FIELDS:
        if name == "encoder_id":
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError("Teacher pair needs an encoder identity")
        elif not _digest(row.get(name)):
            raise ValueError(f"Teacher pair needs a SHA256 {name}")
    values = _features(row.get("features"))
    vector = _vector(row.get("query_embedding"))
    outcomes = row.get("actions")
    if not isinstance(outcomes, list) or len(outcomes) < 2:
        raise ValueError("Teacher pair needs executed semantic and alternative actions")
    rewards = {}
    for action in outcomes:
        if not isinstance(action, dict):
            raise ValueError("Malformed teacher action receipt")
        alpha = action.get("alpha")
        if not _number(alpha) or alpha not in ACTIONS or alpha in rewards:
            raise ValueError("Invalid or duplicate teacher action")
        if (any(action.get(name) != row[name] for name in PROVENANCE_FIELDS)
                or type(action.get("seed")) is not int or action["seed"] != row["seed"]):
            raise ValueError("Teacher actions were not run from matched state, candidates, policy and seed")
        if (action.get("executed") is not True or action.get("action_scope") != "next_retrieval_only"
                or type(action.get("success")) is not bool):
            raise ValueError("Teacher labels require independently executed next-retrieval continuations")
        if not _digest(action.get("trace_hash")):
            raise ValueError("Teacher actions require trace provenance")
        reward = action.get("reward")
        if not _number(reward) or not 0 <= reward <= 1:
            raise ValueError("Teacher action rewards must be finite and in [0,1]")
        rewards[float(alpha)] = float(reward)
    if 0.0 not in rewards:
        raise ValueError("Teacher pair lacks an executed semantic baseline")
    return {"raw": row, "features": values, "embedding": vector, "rewards": rewards}
def fit(records: list[dict], *, train_ids: set[str], calibration_ids: set[str],
        validation_ids: set[str], test_ids: set[str], ridge: float = 0.1,
        minimum_benefit: float = 0.02, max_novelty: float = 0.4,
        ood_tolerance: float = 0.15) -> tuple[dict, dict]:
    'Fit action uplift; reserve independent training questions for calibration.'
    parts = [_identities(value, name) for value, name in
             ((train_ids, "fit"), (calibration_ids, "calibration"),
              (validation_ids, "validation"), (test_ids, "test"))]
    if any(parts[i] & parts[j] for i in range(4) for j in range(i+1, 4)):
        raise ValueError("Teacher fit, calibration, validation and test questions overlap")
    if not records or any(not _number(x) for x in (ridge, minimum_benefit, max_novelty, ood_tolerance)):
        raise ValueError("Missing teacher training data or invalid settings")
    if not 0 < ridge <= 100 or not 0 < minimum_benefit <= 1 or not 0 <= max_novelty <= 1 or not 0 <= ood_tolerance <= 1:
        raise ValueError("Invalid teacher calibration configuration")
    fit_ids, cal_ids = parts[:2]
    rows = [_record(row, fit_ids | cal_ids) for row in records]
    execution_contexts = {row.get('execution_context_hash') for row in records}
    if execution_contexts != {None}:
        if len(execution_contexts) != 1 or not _digest(next(iter(execution_contexts))):
            raise ValueError('Teacher pairs mix incompatible execution contexts')
        if any(action.get('execution_context_hash') != row['execution_context_hash'] for row in records for action in row['actions']):
            raise ValueError('Teacher action execution context differs from its paired record')
    keys = [(row["raw"]["query_id"], row["raw"]["decision_id"], row["raw"]["seed"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate matched teacher decision")
    for name in ("encoder_id", "policy_hash", "alignment_hash"):
        if len({row["raw"][name] for row in rows}) != 1:
            raise ValueError(f"Teacher dataset mixes incompatible {name} versions")
    dimensions = {len(row["embedding"]) for row in rows}
    if len(dimensions) != 1:
        raise ValueError("Teacher query dimensions differ")
    fitting = [row for row in rows if row["raw"]["query_id"] in fit_ids]
    calibration = [row for row in rows if row["raw"]["query_id"] in cal_ids]
    if (len({row["raw"]["query_id"] for row in fitting}) < 3
            or len({row["raw"]["query_id"] for row in calibration}) < 3):
        raise ValueError("Teacher needs at least three independent fit and three calibration training questions")
    prototype_map = {(row["raw"]["query_id"], tuple(row["embedding"])): row for row in fitting}
    prototypes = [{"query_id": key[0], "embedding": list(key[1])} for key in sorted(prototype_map)]
    for row in rows:
        own = row["raw"]["query_id"] if row["raw"]["query_id"] in fit_ids else None
        row["features"]["query_novelty"] = _novelty(row["embedding"], prototypes, own)
    bounds = {name: [min(row["features"][name] for row in fitting),
                     max(row["features"][name] for row in fitting)] for name in FEATURE_NAMES}
    import numpy as np
    models = {}
    action_report = {}
    for alpha in ACTIONS[1:]:
        observed = [row for row in fitting if alpha in row["rewards"]]
        held = [row for row in calibration if alpha in row["rewards"]]
        observed_ids = {row["raw"]["query_id"] for row in observed}
        held_ids = {row["raw"]["query_id"] for row in held}
        key = str(alpha)
        if len(observed_ids) < 3 or len(held_ids) < 3:
            action_report[key] = {"eligible": False, "reason": "insufficient_independent_executed_pairs"}
            continue
        counts = {identity: sum(row["raw"]["query_id"] == identity for row in observed) for identity in observed_ids}
        matrix = np.asarray([[1.0, *[row["features"][name] for name in FEATURE_NAMES]] for row in observed])
        target = np.asarray([row["rewards"][alpha]-row["rewards"][0.0] for row in observed])
        weights = np.asarray([1.0/counts[row["raw"]["query_id"]] for row in observed])
        penalty = ridge*np.eye(len(FEATURE_NAMES)+1)
        penalty[0, 0] = 0.0
        coef = np.linalg.solve(matrix.T @ (weights[:, None]*matrix)+penalty,
                               matrix.T @ (weights*target))
        errors = {}
        for row in held:
            predicted = float(coef @ np.asarray([1.0, *[row["features"][name] for name in FEATURE_NAMES]]))
            observed_gain = row["rewards"][alpha]-row["rewards"][0.0]
            identity = row["raw"]["query_id"]
            errors[identity] = max(errors.get(identity, -math.inf), predicted-observed_gain)
        radius = max(0.0, max(errors.values()))
        models[key] = {"alpha": alpha, "weights": coef.tolist(), "calibration_radius": radius,
                       "fit_query_ids": sorted(observed_ids), "calibration_query_ids": sorted(held_ids),
                       "group_overprediction_errors": errors}
        action_report[key] = {"eligible": True, "fit_questions": len(observed_ids),
                              "calibration_questions": len(held_ids), "calibration_radius": radius}
    model = {"format": FORMAT, "feature_names": list(FEATURE_NAMES), "action_models": models,
             "train_ids": sorted(fit_ids), "calibration_ids": sorted(cal_ids),
             "validation_ids": sorted(parts[2]), "test_ids": sorted(parts[3]),
             "encoder_id": rows[0]["raw"]["encoder_id"], "embedding_dimension": dimensions.pop(),
             "policy_hash": rows[0]["raw"]["policy_hash"], "alignment_hash": rows[0]["raw"]["alignment_hash"],
             "query_prototypes": prototypes, "feature_bounds": bounds,
             "minimum_benefit": float(minimum_benefit), "max_novelty": float(max_novelty),
             "ood_tolerance": float(ood_tolerance), "ridge": float(ridge),
             "training_sha256": _hash(records), "update_policy": "frozen_on_validation_and_test",
             "calibration_kind": "max_observed_question_group_overprediction_not_coverage_guarantee",
             "action_scope": "next_retrieval_only"}
    if execution_contexts != {None}:
        model['execution_context_hash'] = next(iter(execution_contexts))
    validate(model)
    report = {"fit_questions": len({row["raw"]["query_id"] for row in fitting}),
              "calibration_questions": len({row["raw"]["query_id"] for row in calibration}),
              "matched_decisions": len(rows), "actions": action_report,
              "warning": "Empirical train-only calibration; no statistical coverage or held-out benefit claim"}
    return model, report
def validate(model: dict) -> None:
    if (not isinstance(model, dict) or model.get("format") != FORMAT
            or model.get("feature_names") != list(FEATURE_NAMES)
            or model.get("action_scope") != "next_retrieval_only"):
        raise ValueError("Invalid action-level teacher format")
    parts = [_identities(model.get(name), name) for name in
             ("train_ids", "calibration_ids", "validation_ids", "test_ids")]
    if any(parts[i] & parts[j] for i in range(4) for j in range(i+1, 4)) or len(parts[0]) < 3 or len(parts[1]) < 3:
        raise ValueError("Invalid teacher question split provenance")
    if not isinstance(model.get("encoder_id"), str) or not model["encoder_id"]:
        raise ValueError("Missing teacher encoder identity")
    if any(not _digest(model.get(name)) for name in ("policy_hash", "alignment_hash", "training_sha256")):
        raise ValueError("Missing teacher snapshot provenance")
    dimension = model.get("embedding_dimension")
    if type(dimension) is not int or dimension < 1:
        raise ValueError("Invalid teacher embedding dimension")
    prototypes = model.get("query_prototypes")
    if not isinstance(prototypes, list) or not prototypes:
        raise ValueError("Missing fit-only teacher query prototypes")
    for item in prototypes:
        if not isinstance(item, dict) or item.get("query_id") not in parts[0]:
            raise ValueError("Nontraining teacher novelty prototype")
        normed = _vector(item.get("embedding"), dimension)
        if any(abs(a-b) > 1e-8 for a, b in zip(normed, item["embedding"])):
            raise ValueError("Teacher novelty prototypes must be normalized")
    if len({item["query_id"] for item in prototypes}) < 3:
        raise ValueError("Teacher novelty support needs three independent training questions")
    for name in ("minimum_benefit", "max_novelty", "ood_tolerance"):
        if not _number(model.get(name)) or not 0 <= model[name] <= 1:
            raise ValueError("Invalid teacher decision setting")
    if model["minimum_benefit"] <= 0:
        raise ValueError("Teacher must require positive action benefit")
    bounds = model.get("feature_bounds")
    if not isinstance(bounds, dict) or set(bounds) != set(FEATURE_NAMES):
        raise ValueError("Invalid teacher feature support")
    for name, value in bounds.items():
        if not isinstance(value, list) or len(value) != 2 or any(not _number(x) for x in value) or value[0] > value[1]:
            raise ValueError("Invalid teacher feature bounds")
    _features({name: values[0] for name, values in bounds.items()})
    _features({name: values[1] for name, values in bounds.items()})
    actions = model.get("action_models")
    if not isinstance(actions, dict):
        raise ValueError("Invalid teacher action models")
    for key, action in actions.items():
        if (not isinstance(action, dict) or not _number(action.get("alpha")) or action["alpha"] not in ACTIONS[1:]
                or key != str(action["alpha"])):
            raise ValueError("Invalid teacher action identity")
        weights = action.get("weights")
        if not isinstance(weights, list) or len(weights) != len(FEATURE_NAMES)+1 or any(not _number(x) for x in weights):
            raise ValueError("Invalid teacher action weights")
        if not _number(action.get("calibration_radius")) or action["calibration_radius"] < 0:
            raise ValueError("Invalid teacher empirical calibration radius")
        for name, allowed in (("fit_query_ids", parts[0]), ("calibration_query_ids", parts[1])):
            ids = _identities(action.get(name), name)
            if len(ids) < 3 or not ids <= allowed:
                raise ValueError("Teacher action lacks independent question support")
        errors = action.get("group_overprediction_errors")
        if (not isinstance(errors, dict) or set(errors) != set(action["calibration_query_ids"])
                or any(not _number(error) for error in errors.values())
                or abs(action["calibration_radius"]-max(0.0, max(errors.values()))) > 1e-8):
            raise ValueError("Teacher action calibration evidence disagrees with its radius")
def decide(model: dict, features: dict, *, query_embedding: list[float] | None = None,
           encoder_id: str | None = None, policy_hash: str | None = None,
           alignment_hash: str | None = None) -> dict:
    'Choose an action only for the pinned policy, alignment and live query.'
    validate(model)
    values = _features(features)
    if query_embedding is None or any(value is None for value in (encoder_id, policy_hash, alignment_hash)):
        raise ValueError("Live teacher decision requires query embedding, encoder, policy and alignment identities")
    for name, expected in (("encoder_id", encoder_id), ("policy_hash", policy_hash),
                           ("alignment_hash", alignment_hash)):
        if expected != model[name]:
            raise ValueError(f"Live {name} differs from fitted teacher")
    values["query_novelty"] = query_novelty(model, query_embedding, encoder_id)
    fallback = {"decision": "abstain", "action": 0.0, "structure_weight": 0.0,
                "predicted_delta": 0.0, "lower_bound": 0.0}
    if values["query_novelty"] > model["max_novelty"]:
        return {**fallback, "reason": "novel_query_outside_training_support"}
    outside = [name for name in FEATURE_NAMES if name != "query_novelty"
               and not model["feature_bounds"][name][0]-model["ood_tolerance"] <= values[name]
               <= model["feature_bounds"][name][1]+model["ood_tolerance"]]
    if outside:
        return {**fallback, "reason": "features_outside_training_support", "outside_features": outside}
    candidates = []
    for action in model["action_models"].values():
        try:
            prediction = math.fsum([action["weights"][0], *[
                weight*values[name] for weight, name in zip(action["weights"][1:], FEATURE_NAMES)]])
        except (OverflowError, ValueError):
            return {**fallback, "reason": "nonfinite_action_prediction"}
        lower = prediction-action["calibration_radius"]
        if not math.isfinite(prediction) or not math.isfinite(lower):
            return {**fallback, "reason": "nonfinite_action_prediction"}
        if lower > model["minimum_benefit"]:
            candidates.append((lower, -action["alpha"], prediction, action["alpha"]))
    if not candidates:
        return {**fallback, "reason": "no_calibrated_positive_action_benefit"}
    lower, _, prediction, alpha = max(candidates)
    return {"decision": "fuse", "action": alpha, "structure_weight": alpha,
            "predicted_delta": prediction, "lower_bound": lower,
            "reason": "executed_action_train_calibrated_benefit"}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
