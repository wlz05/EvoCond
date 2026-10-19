# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Train-only dual tool memory primitives for the proposed teacher method.'
from __future__ import annotations
from copy import deepcopy
import re
from math import isfinite
from typing import Any
from control.controller import catalog, cosine, semantic
FORMAT = "evocond_teacher_memory_v1"
def empty_store() -> dict[str, Any]:
    return {"format": FORMAT, "semantic_store": {}, "structural_store": {"nodes": {}, "edges": {}}, "aliases": {}}
def _same_contract(left: dict, right: dict) -> bool:
    return (left["requires"] == right["requires"] and left["produces"] == right["produces"]
            and left["schema"] == right["schema"])
def validate(store: dict[str, Any]) -> None:
    if store.get("format") != FORMAT:
        raise ValueError("Unknown dual tool-memory format")
    semantic_store = store.get("semantic_store")
    structure = store.get("structural_store")
    aliases = store.get("aliases")
    if not isinstance(semantic_store, dict) or not isinstance(structure, dict) or not isinstance(aliases, dict):
        raise ValueError("Missing dual stores")
    nodes, edges = structure.get("nodes"), structure.get("edges")
    if not isinstance(nodes, dict) or not isinstance(edges, dict) or set(nodes) != set(semantic_store):
        raise ValueError("Semantic and structural stores are not version-aligned")
    for version, entry in semantic_store.items():
        if entry.get("version") != version or not isinstance(entry.get("description"), str):
            raise ValueError("Invalid semantic node")
        if nodes[version].get("version") != version or not isinstance(nodes[version].get("requires"), list):
            raise ValueError("Invalid structural node")
    for name, version in aliases.items():
        if not isinstance(name, str) or version not in nodes:
            raise ValueError("Alias references missing node")
    for key, edge in edges.items():
        if not isinstance(key, str) or edge.get("source") not in nodes or edge.get("target") not in nodes:
            raise ValueError("Edge references missing node")
        if not isinstance(edge.get("evidence"), dict) or not isinstance(edge.get("weight"), (int, float)) or not isfinite(edge["weight"]) or not 0 <= edge["weight"] <= 1:
            raise ValueError("Invalid edge evidence or weight")
def admit_visible_tool(store: dict[str, Any], tool: dict, *,
                       verified_equivalent_to: str | None = None,
                       verification: dict | None = None,
                       threshold: float = 0.92) -> tuple[dict[str, Any], dict[str, Any]]:
    'Admit one currently callable tool; merging requires external evidence.'
    validate(store)
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid novelty threshold")
    visible = catalog([tool])
    if len(visible) != 1:
        raise ValueError("Tool needs a parseable callable interface")
    node = next(iter(visible.values()))
    version, name = node["version"], node["name"]
    result = deepcopy(store)
    if version in result["semantic_store"]:
        result["aliases"][name] = version
        return result, {"decision": "existing", "version": version, "tool": name}
    candidates = []
    for old_version, old in result["semantic_store"].items():
        old_node = result["structural_store"]["nodes"][old_version]
        similarity = cosine(semantic(node), semantic(old))
        if _same_contract(node, old_node) and similarity >= threshold:
            candidates.append((similarity, old_version))
    candidates.sort(reverse=True)
    if verified_equivalent_to is not None:
        if (not isinstance(verification, dict) or verification.get("candidate_version") != verified_equivalent_to
            or verification.get("input_output_checked") is not True
            or verification.get("side_effects_checked") is not True
            or not isinstance(verification.get("evidence_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", verification["evidence_sha256"]) is None):
            raise ValueError("Behavioral equivalence requires a hashed verification record")
        if verified_equivalent_to not in {v for _, v in candidates}:
            raise ValueError("Claimed equivalence lacks semantic and executable-contract support")
        result["aliases"][name] = verified_equivalent_to
        return result, {"decision": "merge_verified", "version": verified_equivalent_to,
                        "tool": name, "semantic_similarity": candidates[0][0],
                        "verification_sha256": verification["evidence_sha256"]}
    result["semantic_store"][version] = {"version": version, "name": name,
                                         "description": node["description"]}
    result["structural_store"]["nodes"][version] = node
    result["aliases"][name] = version
    return result, {"decision": "new_versioned_node", "version": version, "tool": name,
                    "similar_candidate": candidates[0][1] if candidates else None,
                    "merge_requires_behavioral_verification": bool(candidates)}
def record_transition(store: dict[str, Any], *, source: str, target: str,
                      query_id: str, feedback: str, task_success: bool | None,
                      learning_rate: float = 0.1, decay: float = 0.01) -> dict[str, Any]:
    "Update a directed edge from one independent training task's observed step."
    validate(store)
    if not isinstance(query_id, str) or not query_id or feedback not in {"verified", "failed", "unknown"}:
        raise ValueError("Invalid transition evidence")
    if type(task_success) not in (bool, type(None)) or not 0 < learning_rate <= 1 or not 0 <= decay < 1:
        raise ValueError("Invalid feedback update")
    nodes = store["structural_store"]["nodes"]
    if source not in nodes or target not in nodes:
        raise ValueError("Transition refers to unknown tool version")
    if feedback == "unknown" or feedback == "verified" and task_success is not True:
        return deepcopy(store)
    result = deepcopy(store)
    edges = result["structural_store"]["edges"]
    key = f"{source}->{target}"
    edge = edges.setdefault(key, {"source": source, "target": target, "weight": 0.5, "evidence": {}})
    signal = 1.0 if feedback == "verified" else 0.0
    old = edge["evidence"].get(query_id)
    if old is not None:
        if old != signal:
            raise ValueError("Contradictory evidence for one independent task")
        return result
    edge["evidence"][query_id] = signal
    edge["weight"] = min(1.0, max(0.0, (1 - decay) * edge["weight"] + learning_rate * (signal - edge["weight"])))
    return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
