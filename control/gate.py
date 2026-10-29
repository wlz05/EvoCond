# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'A small, supervised *episode-level* intervention gate, not an edge gate.'
from __future__ import annotations
from collections import Counter
from hashlib import blake2b, sha256
import math
import random
import re
from typing import Any, Mapping, Sequence
ARMS = ("filter", "evocond", "joint_exact", "joint_family")
FORMAT = "evocond_initial_task_gate_v1"
DIMENSION = 256
COST_WEIGHT = 0.05
STEP_WEIGHT = 0.01
def initial_user(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            return content.strip() if isinstance(content, str) else ""
    return ""
def normalize(text: str) -> str:
    text = text.casefold()
    # Typed identifiers retain their type, never their instance/answer number.
    text = re.sub(r"<([a-z_]+)\d+>", r"<\1>", text)
    text = re.sub(r"\b((?:[a-z]+)?(?:preference|candidateset|category|featurepackage|tier|style)|timeinfo|query)\d+\b", r"\1", text)
    # Real numerical constraints (prices, distances, counts) are useful inputs.
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE)[:512])
def features(text: str, dimension: int = DIMENSION) -> list[float]:
    if dimension < 8:
        raise ValueError("Feature dimension is too small")
    words = normalize(text).split()
    terms = words + [a + ":" + b for a, b in zip(words, words[1:])]
    vector = [0.0] * dimension
    for term, count in Counter(terms).items():
        code = blake2b(term.encode("utf-8"), digest_size=8).digest()
        index = 1 + int.from_bytes(code[:4], "big") % (dimension - 1)
        vector[index] += (1.0 if code[4] & 1 else -1.0) * math.log1p(count)
    norm = math.sqrt(sum(x * x for x in vector))
    if norm:
        vector = [x / norm for x in vector]
    vector[0] = 1.0
    return vector
def utility(outcome: Mapping[str, Any], base: Mapping[str, Any]) -> float:
    cost_delta = (outcome["cost"] - base["cost"]) / max(1.0, base["cost"])
    step_delta = (outcome["steps"] - base["steps"]) / 20.0
    return (int(outcome["success"]) - int(base["success"])
            - COST_WEIGHT * max(-2.0, min(2.0, cost_delta))
            - STEP_WEIGHT * max(-1.0, min(1.0, step_delta)))
def fit(examples: list[dict[str, Any]], seed: int = 20260927) -> dict[str, Any]:
    if not examples:
        raise ValueError("No training examples")
    weights = {arm: [0.0] * DIMENSION for arm in ARMS[1:]}
    per_query = Counter(row["index"] for row in examples)
    vectors = [features(row["text"]) for row in examples]
    targets = [{arm: utility(row["outcomes"][arm], row["outcomes"]["filter"])
                for arm in ARMS[1:]} for row in examples]
    rng = random.Random(seed)
    order = list(range(len(examples)))
    # Pure-stdlib regression; no torch install or GPU needed to fit the gate.
    for epoch in range(240):
        rng.shuffle(order)
        rate = 0.08 / math.sqrt(1.0 + epoch / 30.0)
        for i in order:
            x = vectors[i]
            scale = 1.0 / per_query[examples[i]["index"]]
            for arm in ARMS[1:]:
                w = weights[arm]
                error = sum(a * b for a, b in zip(w, x)) - targets[i][arm]
                for j in range(DIMENSION):
                    penalty = 0.002 * w[j] if j else 0.0
                    w[j] -= rate * scale * (error * x[j] + penalty)
    return {"format": FORMAT, "dimension": DIMENSION, "weights": weights,
            "training_vectors": [list(v) for v in dict.fromkeys(tuple(v) for v in vectors)],
            "threshold": None, "min_similarity": 0.15,
            "seed": seed, "epochs": 240, "cost_weight": COST_WEIGHT,
            "step_weight": STEP_WEIGHT, "scope": "initial_user_episode_policy",
            "fallback": "filter", "calibrated_probability": False}
def validate_model(model: Mapping[str, Any]) -> None:
    if model.get("format") != FORMAT or model.get("dimension") != DIMENSION:
        raise ValueError("Unknown task-gate format")
    if model.get("fallback") != "filter" or model.get("scope") != "initial_user_episode_policy":
        raise ValueError("Gate has an incompatible inference scope")
    threshold = model.get("threshold")
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or threshold < 0):
        raise ValueError("Invalid gate threshold")
    similarity = model.get("min_similarity")
    if isinstance(similarity, bool) or not isinstance(similarity, (float, int)) or not math.isfinite(similarity) or not 0 <= similarity <= 1:
        raise ValueError("Invalid similarity threshold")
    weights = model.get("weights")
    if not isinstance(weights, dict) or set(weights) != set(ARMS[1:]):
        raise ValueError("Gate action heads are invalid")
    vectors = model.get("training_vectors")
    if not isinstance(vectors, list) or not vectors:
        raise ValueError("Missing train-only feature reference vectors")
    for row in [*weights.values(), *vectors]:
        if not isinstance(row, list) or len(row) != DIMENSION or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in row):
            raise ValueError("Malformed gate parameters")
def decide(model: Mapping[str, Any], text: str) -> dict[str, Any]:
    x = features(text, model["dimension"])
    scores = {"filter": 0.0, **{arm: sum(a * b for a, b in zip(w, x))
                                for arm, w in model["weights"].items()}}
    nearest = max((sum(a * b for a, b in zip(x[1:], v[1:]))
                   for v in model["training_vectors"]), default=0.0)
    proposal = max(ARMS, key=lambda arm: (scores[arm], -ARMS.index(arm)))
    threshold = model.get("threshold")
    reason = "learned_positive_utility"
    arm = proposal
    if not normalize(text):
        arm, reason = "filter", "empty_initial_user"
    elif threshold is None:
        arm, reason = "filter", "validation_abstain_all"
    elif nearest < model["min_similarity"]:
        arm, reason = "filter", "low_training_similarity"
    elif proposal == "filter" or scores[proposal] <= threshold:
        arm, reason = "filter", "insufficient_predicted_utility"
    return {"arm": arm, "proposal": proposal, "reason": reason,
            "predicted_utilities": {a: round(v, 6) for a, v in scores.items()},
            "training_similarity": round(nearest, 6),
            "initial_user_sha256": sha256(text.encode("utf-8")).hexdigest()}
def evaluate(model: Mapping[str, Any], examples: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = []
    selected_queries: set[int] = set()
    total = Counter()
    for row in examples:
        decision = decide(model, row["text"])
        arm = decision["arm"]
        chosen, base = row["outcomes"][arm], row["outcomes"]["filter"]
        total["correct"] += int(chosen["success"])
        total["filter_correct"] += int(base["success"])
        total["gain"] += int(chosen["success"] and not base["success"])
        total["loss"] += int(base["success"] and not chosen["success"])
        total["cost"] += chosen["cost"]
        total["filter_cost"] += base["cost"]
        total["steps"] += chosen["steps"]
        total["utility"] += utility(chosen, base)
        total["selected"] += int(arm != "filter")
        if arm != "filter":
            selected_queries.add(row["index"])
        decisions.append({"index": row["index"], "mode": row["mode"], **decision})
    n = len(examples)
    return {"evaluation_kind": "paired_static_policy_replay_not_live_gate",
            "n": n, "unique_queries": len({r["index"] for r in examples}),
            "correct": total["correct"], "filter_correct": total["filter_correct"],
            "gain": total["gain"], "loss": total["loss"],
            "mean_cost": total["cost"] / n if n else None,
            "mean_filter_cost": total["filter_cost"] / n if n else None,
            "mean_steps": total["steps"] / n if n else None,
            "utility_sum": total["utility"], "selected": total["selected"],
            "selected_unique_queries": len(selected_queries), "decisions": decisions}
def calibrate(model: dict[str, Any], validation: list[dict[str, Any]]) -> dict[str, Any]:
    if not validation:
        raise ValueError("No separate validation data")
    trials = []
    # Always-defer is an explicit candidate, not a hidden failure mode.
    for threshold in (None, 0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5):
        report = evaluate({**model, "threshold": threshold}, validation)
        eligible = threshold is None or (
            report["loss"] == 0 and report["selected_unique_queries"] >= 3
            and report["utility_sum"] > 0
            and (report["correct"] > report["filter_correct"]
                 or report["mean_cost"] <= report["mean_filter_cost"]))
        trials.append({"threshold": threshold, "eligible": eligible,
                       **{k: v for k, v in report.items() if k != "decisions"}})
    best = max((t for t in trials if t["eligible"]),
               key=lambda t: (t["utility_sum"], -t["selected"]))
    model["threshold"] = best["threshold"]
    return {"rule": "validation-only zero observed harm, positive utility, >=3 selected queries; otherwise Filter",
            "warning": "Empirical model selection, not a statistical safety guarantee",
            "selected_threshold": best["threshold"], "trials": trials}
def settings(arm: str, memory: Any) -> tuple[str, Any, str]:
    if arm not in ARMS:
        raise ValueError("Invalid gate action")
    return ("filter" if arm == "filter" else "full",
            memory if arm.startswith("joint_") else None,
            "family" if arm == "joint_family" else "exact")

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
