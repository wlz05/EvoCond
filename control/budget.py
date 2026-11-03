# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'A bounded, read-only historical view, never a hard prerequisite graph.'
from __future__ import annotations
from copy import deepcopy
import math
from control.dynamic import digest
FORMAT = "evocond_structure_budget_v1"
def finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False

def policy(value=None):
    if value is None:
        return {"mode": "dense"}
    if not isinstance(value, dict) or value.get("mode") not in ("dense", "sparse", "none"):
        raise ValueError("Unknown historical structure policy")
    if value["mode"] != "sparse":
        if set(value) != {"mode"}:
            raise ValueError("Dense/none structure policy takes no sparse parameters")
        return deepcopy(value)
    fields = {"mode", "max_incident_edges", "min_matching_tasks", "min_abs_score", "selection", "seed"}
    if set(value) != fields:
        raise ValueError("Sparse structure policy must explicitly freeze all budget parameters")
    for key, low, high in (("max_incident_edges", 0, 100), ("min_matching_tasks", 1, 1000000), ("seed", 0, 2**31-1)):
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError(f"Invalid structure budget: {key}")
    score = value["min_abs_score"]
    if not finite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid structure score threshold")
    if value["selection"] not in ("score", "hash"):
        raise ValueError("Structure selection must be score or a seeded hash control")
    return deepcopy(value)
def select(views, value=None, *, allowed_versions, previous_version=None):
    'Greedy bounded-degree view of signed contextual execution associations.'
    spec = policy(value)
    if (not isinstance(views, dict) or not isinstance(allowed_versions, (list, tuple, set, frozenset))
            or any(not isinstance(item, str) or not item.strip() or item != item.strip()
                   or '->' in item for item in allowed_versions)):
        raise ValueError("Structure views require explicit current tool versions")
    if previous_version is not None and (not isinstance(previous_version, str)
            or not previous_version.strip() or previous_version != previous_version.strip()
            or '->' in previous_version):
        raise ValueError("Previous tool version must identify the observed invocation")
    allowed = set(allowed_versions)
    available = {}
    for key, view in views.items():
        if not isinstance(key, str):
            raise ValueError("Malformed versioned edge identity")
        parts = key.split("->")
        if len(parts) != 2 or any(not part or part != part.strip() for part in parts):
            raise ValueError("Malformed versioned edge identity")
        if parts[1] in allowed and (parts[0] in allowed or parts[0] == previous_version) and parts[0] != parts[1]:
            if (not isinstance(view, dict) or type(view.get('matching_tasks')) is not int
                    or view['matching_tasks'] < 0 or not finite(view.get('score'))
                    or not -1 <= view['score'] <= 1 or not finite(view.get('support'))
                    or view['support'] < 0):
                raise ValueError("Structure views require finite scores and observed support")
            available[key] = view
    if spec["mode"] == "dense":
        chosen = available
    elif spec["mode"] == "none":
        chosen = {}
    else:
        eligible = {key: row for key, row in available.items()
                    if row["matching_tasks"] >= spec["min_matching_tasks"]
                    and abs(row["score"]) > 0 and abs(row["score"]) >= spec["min_abs_score"]}
        def order(key):
            row = eligible[key]
            if spec["selection"] == "hash":
                return (digest({"seed": spec["seed"], "edge": key}), key)
            return (-abs(row["score"]), -row["support"], key)
        chosen, degrees = {}, {}
        for key in sorted(eligible, key=order):
            source, target = key.split("->")
            if max(degrees.get(source, 0), degrees.get(target, 0)) >= spec["max_incident_edges"]:
                continue
            chosen[key] = eligible[key]
            degrees[source] = degrees.get(source, 0)+1
            degrees[target] = degrees.get(target, 0)+1
    positive = sum(row["score"] > 0 for row in chosen.values())
    negative = sum(row["score"] < 0 for row in chosen.values())
    degree = {}
    for key in chosen:
        for version in key.split("->"):
            degree[version] = degree.get(version, 0)+1
    receipt = {"format": FORMAT, "policy": spec, "policy_hash": digest(spec),
               "available_edges": len(available), "selected_edges": len(chosen),
               "positive_edges": positive, "negative_edges": negative,
               "max_incident_edges_observed": max(degree.values(), default=0),
               "selected_edge_keys": sorted(chosen), "selected_views_hash": digest(chosen),
               "hard_prerequisites_changed": False,
               "claim": "contextual associations only; neither causal necessity nor generality certified"}
    return deepcopy(chosen), receipt

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
