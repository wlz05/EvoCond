# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Contextual dual-view retrieval; one teacher action per candidate ranking.'
from __future__ import annotations
import math
import json
from copy import deepcopy
from control import dynamic, edges, boundary
from control import teacher as teacher_policy
from control.retrieval import _vector, _project, fit_alignment
from control.budget import policy as structure_policy_spec, select as select_structure
FORMAT = "evocond_searl_retrieval_v2"
VIEW_SCHEMA = "evocond_tool_contract_neighbors_v2"
class SentenceTransformerEncoder:
    'Load a local, pinned encoder only when explicitly instantiated.'
    def __init__(self, path: str, *, encoder_id: str, device: str = "cpu",
                 query_prefix: str = "", passage_prefix: str = ""):
        if not isinstance(encoder_id, str) or not encoder_id.strip():
            raise ValueError("An immutable encoder identity is required")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(path, device=device, local_files_only=True)
        self.encoder_id = encoder_id
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
    def encode(self, text: str) -> list[float]:
        return self.model.encode(self.passage_prefix + text, normalize_embeddings=True).tolist()
    def encode_query(self, text: str) -> list[float]:
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()
def active_tools(registry: dict) -> list[dict]:
    dynamic.validate(registry)
    names = {}
    for name, version in registry["aliases"].items():
        names.setdefault(version, []).append(name)
    tools = []
    for version, aliases in sorted(names.items()):
        item = registry["semantic"][version]
        name = item["name"] if item["name"] in aliases else min(aliases)
        tools.append({"name": name, "version": version,
                      "description": item["description"], "arguments": item["arguments"],
                      "returns": item["returns"], "side_effects": item["side_effects"]})
    return tools
def _checked_embedding(encoder, text: str, dimension: int | None = None, *, query: bool = False) -> list[float]:
    encode = getattr(encoder, "encode_query", encoder.encode) if query else encoder.encode
    vector = _vector(encode(text), dimension)
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Retrieval requires finite nonzero embedding norms")
    return [x/norm for x in vector]
def _bounded_cosine(left: list[float], right: list[float]) -> float:
    left, right = _vector(left), _vector(right, len(left))
    left_norm, right_norm = math.hypot(*left), math.hypot(*right)
    if not math.isfinite(left_norm) or not math.isfinite(right_norm):
        raise ValueError("Retrieval views have nonfinite norms")
    if left_norm == 0 or right_norm == 0:
        return 0.0
    cosine = sum((a/left_norm)*(b/right_norm) for a, b in zip(left, right))
    return max(-1.0, min(1.0, cosine))
def validate_alignment(alignment: dict, *, encoder_id: str | None = None,
                       structure_policy: dict | None = None, reuse_boundary_mode: str | None = None) -> None:
    'Validate a frozen map and its exact training/serving view definition.'
    if not isinstance(alignment, dict) or alignment.get("view_schema") != VIEW_SCHEMA:
        raise ValueError("Alignment uses a missing or incompatible structural view schema")
    stored_encoder = alignment.get("encoder_id")
    if not isinstance(stored_encoder, str) or not stored_encoder.strip():
        raise ValueError("Alignment needs an immutable encoder identity")
    if encoder_id is not None and encoder_id != stored_encoder:
        raise ValueError("Encoder differs from the frozen alignment identity")
    stored_policy = structure_policy_spec(alignment.get("structure_policy"))
    if structure_policy is not None and stored_policy != structure_policy_spec(structure_policy):
        raise ValueError("Historical structure policy differs from the frozen alignment view")
    stored_boundary = boundary.validate_mode(alignment.get("reuse_boundary_mode", "off"))
    if reuse_boundary_mode is not None and stored_boundary != boundary.validate_mode(reuse_boundary_mode):
        raise ValueError("Reuse boundary differs from the frozen alignment view")
    dimension = alignment.get("dimension")
    if type(dimension) is not int or dimension < 1:
        raise ValueError("Alignment dimension must be a positive integer")
    _project(alignment, [0.0]*dimension)
def _eligible_tools(registry: dict, eligible_versions: list[str] | None) -> list[dict]:
    active = active_tools(registry)
    if eligible_versions is not None:
        if (not isinstance(eligible_versions, list)
                or any(not isinstance(value, str) or not value for value in eligible_versions)
                or len(set(eligible_versions)) != len(eligible_versions)):
            raise ValueError("Malformed eligible tool versions")
        if not set(eligible_versions) <= {row["version"] for row in active}:
            raise ValueError("Eligibility includes an inactive tool")
        active = [row for row in active if row["version"] in eligible_versions]
    return active
def _structure(tool: dict, active: list[dict], views: dict) -> tuple[str, list[dict]]:
    incoming, outgoing, observations = [], [], []
    for other in active:
        if other["version"] == tool["version"]:
            continue
        for source, target, names in ((other, tool, incoming), (tool, other, outgoing)):
            view = views.get(f"{source['version']}->{target['version']}")
            if view and view["score"] > 0:
                names.append(other["name"])
                observations.append(view)
    text = ("Input contract: " + json.dumps(tool["arguments"], sort_keys=True) + ". Output contract: "
            + tool["returns"] + ". Supported predecessors: " + ", ".join(sorted(incoming))
            + ". Supported successors: " + ", ".join(sorted(outgoing)))
    return text, observations
def build_alignment_pairs(registry: dict, encoder, *, query_id: str, query: str,
                          eligible_versions: list[str] | None = None,
                          structure_policy: dict | None = None, reuse_boundary_mode: str = "off",
                          current_user_text: str | None = None) -> list[dict]:
    'Encode train views using the identical builder used by live retrieval.'
    if not isinstance(query_id, str) or not query_id.strip() or not isinstance(query, str) or not query.strip():
        raise ValueError("Alignment collection needs a question identity and visible subtask")
    encoder_id = getattr(encoder, "encoder_id", None)
    if not isinstance(encoder_id, str) or not encoder_id.strip():
        raise ValueError("Alignment collection needs an immutable encoder identity")
    active = _eligible_tools(registry, eligible_versions)
    query = query.strip()
    qvec = _checked_embedding(encoder, query, query=True)
    views = edges.batch_edge_views(registry, context_vector=qvec, encoder_id=encoder_id,
        reuse_boundary_mode=reuse_boundary_mode, current_user_text=current_user_text)
    boundary_receipt = boundary.receipt(views, reuse_boundary_mode, current_user_text)
    views, structure_receipt = select_structure(views, structure_policy,
                                               allowed_versions=[row["version"] for row in active])
    structure_receipt["reuse_boundary"] = boundary_receipt
    provenance = {"view_schema": VIEW_SCHEMA, "encoder_id": encoder_id,
                  "reuse_boundary_mode": reuse_boundary_mode,
                  "structure_policy": structure_receipt["policy"],
                  "structure_receipt": structure_receipt,
                  "registry_hash": dynamic.digest(registry),
                  "query_hash": dynamic.digest(query),
                  "candidate_versions_hash": dynamic.digest([row["version"] for row in active])}
    pairs = []
    for tool in active:
        text, _ = _structure(tool, active, views)
        pairs.append({"query_id": query_id, "tool_version": tool["version"],
                      "semantic": _checked_embedding(encoder, tool["description"], len(qvec)),
                      "structural": _checked_embedding(encoder, text, len(qvec)),
                      "semantic_text": tool["description"], "structural_text": text,
                      **provenance})
    return pairs
def _summary(rows: list[dict], *, turn: int, goal_changed: bool, prior_failures: int,
             completed_subtasks: int, novelty: float) -> dict:
    semantic = sorted(rows, key=lambda r: (-r["semantic_score"], r["version"]))
    structural = sorted(rows, key=lambda r: (-r["structural_score"], r["version"]))
    def margin(items, name):
        return min(1.0, max(0.0, (items[0][name]-items[1][name])/2)) if len(items)>1 else 0.0
    return {"semantic_top_score": semantic[0]["semantic_score"] if rows else 0.0,
            "structural_top_score": structural[0]["structural_score"] if rows else 0.0,
            "semantic_margin": margin(semantic, "semantic_score"),
            "structural_margin": margin(structural, "structural_score"),
            "rank_agreement": float(bool(rows) and semantic[0]["version"] == structural[0]["version"]),
            "edge_reliability": max((r["edge"]["reliability"] for r in rows), default=0.0),
            "query_novelty": novelty, "goal_changed": float(goal_changed),
            "prior_failures": min(prior_failures, 5)/5, "turn": min(turn, 20)/20,
            "completed_subtasks": min(completed_subtasks, 16)/16}
def describe(registry: dict, encoder, alignment: dict, *, query: str,
             previous_tool: str | None = None, previous_version: str | None = None,
             turn: int = 0, goal_changed: bool = False,
             prior_failures: int = 0, completed_subtasks: int = 0,
             eligible_versions: list[str] | None = None,
             structure_policy: dict | None = None, reuse_boundary_mode: str = "off",
             current_user_text: str | None = None) -> dict:
    'Build the visible predecision views used for both collection and serving.'
    if not isinstance(query, str) or not query.strip() or type(goal_changed) is not bool:
        raise ValueError("Invalid visible retrieval state")
    if any(type(x) is not int or x < 0 for x in (turn, prior_failures, completed_subtasks)):
        raise ValueError("Invalid retrieval counters")
    encoder_id = getattr(encoder, "encoder_id", None)
    if not isinstance(encoder_id, str) or not encoder_id.strip():
        raise ValueError("Retrieval needs an immutable encoder identity")
    # Missing config preserves the dense legacy view, not whatever happens to be inside an.
    structure_policy = structure_policy_spec(structure_policy)
    validate_alignment(alignment, encoder_id=encoder_id, structure_policy=structure_policy,
                       reuse_boundary_mode=reuse_boundary_mode)
    dimension = alignment.get("dimension")
    query = query.strip()
    qvec = _checked_embedding(encoder, query, dimension, query=True)
    active = _eligible_tools(registry, eligible_versions)
    if previous_tool is not None and (not isinstance(previous_tool, str) or not previous_tool):
        raise ValueError("Malformed previous tool name")
    if previous_version is not None:
        if not isinstance(previous_version, str) or previous_version not in registry["semantic"]:
            raise ValueError("Unknown previously invoked tool version")
        observed = registry["semantic"][previous_version]
        previous = dynamic.resolve_tool_version(
            registry, previous_tool or observed["name"], previous_version)
    else:
        if previous_tool is not None and previous_tool not in registry["aliases"]:
            raise ValueError("Unknown previous tool")
        previous = registry["aliases"].get(previous_tool)
    views = edges.batch_edge_views(registry, context_vector=qvec, encoder_id=encoder_id,
        reuse_boundary_mode=reuse_boundary_mode, current_user_text=current_user_text)
    boundary_receipt = boundary.receipt(views, reuse_boundary_mode, current_user_text)
    # Sparse structure only uses current callable endpoints, so fit and serve have the same text.
    allowed = [row["version"] for row in active]
    views, structure_receipt = select_structure(views, structure_policy, allowed_versions=allowed)
    structure_receipt["reuse_boundary"] = boundary_receipt
    neutral = {"association": 0.0, "utility": 0.0, "reliability": 0.0,
               "support": 0.0, "score": 0.0, "context_match": 0.0,
               "independent_tasks": 0, "matching_tasks": 0, "format": edges.FORMAT}
    rows = []
    for tool in active:
        sem = _bounded_cosine(qvec, _checked_embedding(encoder, tool["description"], dimension))
        text, _ = _structure(tool, active, views)
        projected = _project(alignment, _checked_embedding(encoder, text, dimension))
        aligned = _bounded_cosine(qvec, projected)
        edge = deepcopy(views.get(f"{previous}->{tool['version']}", neutral))
        structure_score = (aligned + edge["score"])/2
        rows.append({**tool, "semantic_score": sem, "structural_score": structure_score,
                     "aligned_contract_score": aligned, "edge": edge})
    visible = {"query": query, "previous_tool": previous_tool, "previous_version": previous, "turn": turn,
               "goal_changed": goal_changed, "prior_failures": prior_failures,
               "completed_subtasks": completed_subtasks}
    visible.update(reuse_boundary_mode=reuse_boundary_mode, current_user_hash=boundary_receipt["current_user_hash"])
    return {"format": FORMAT, "visible_state": visible, "query_embedding": qvec,
            "structure_receipt": structure_receipt,
            "encoder_id": encoder_id, "candidates": rows,
            "features": _summary(rows, turn=turn, goal_changed=goal_changed,
                                 prior_failures=prior_failures, completed_subtasks=completed_subtasks,
                                 novelty=1.0),
            "registry_hash": dynamic.digest(registry),
            "alignment_hash": dynamic.digest(alignment),
            "visible_state_hash": dynamic.digest(visible),
            "visible_state_kind": "retrieval_summary_not_full_continuation_snapshot",
            "candidate_hash": dynamic.digest(rows)}
def apply_action(view: dict, alpha: float, *, top_k: int = 5) -> list[dict]:
    if view.get("format") != FORMAT or type(alpha) not in (int, float) or alpha not in teacher_policy.ACTIONS:
        raise ValueError("Unsupported retrieval action")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("Invalid retrieval top_k")
    result = [{**deepcopy(row), "score": (1-alpha)*row["semantic_score"]+alpha*row["structural_score"]}
              for row in view["candidates"]]
    result.sort(key=lambda row: (-row["score"], row["name"], row["version"]))
    return result[:top_k]
def rank(registry: dict, encoder, alignment: dict, teacher: dict | None, *, top_k: int = 5,
         fixed_alpha: float | None = None, policy_hash: str | None = None, **state) -> dict:
    view = describe(registry, encoder, alignment, **state)
    if fixed_alpha is not None:
        decision = {"structure_weight": fixed_alpha, "decision": "fixed",
                    "reason": "declared_action_ablation"}
    elif teacher is None:
        decision = {"structure_weight": 0.0, "decision": "abstain", "reason": "untrained_teacher"}
    else:
        if teacher.get("alignment_hash") != view["alignment_hash"]:
            raise ValueError("Teacher was fitted with a different alignment snapshot")
        if policy_hash is None or teacher.get("policy_hash") != policy_hash:
            raise ValueError("Teacher needs paired data from the current frozen Qwen policy")
        view["features"]["query_novelty"] = teacher_policy.query_novelty(
            teacher, view["query_embedding"], view["encoder_id"])
        decision = teacher_policy.decide(teacher, view["features"],
                                         query_embedding=view["query_embedding"],
                                         encoder_id=view["encoder_id"], policy_hash=policy_hash,
                                         alignment_hash=view["alignment_hash"])
    return {**view, "teacher": decision, "tools": apply_action(view, decision["structure_weight"], top_k=top_k),
            "candidate_count": len(view["candidates"]), "snapshot_hash": view["registry_hash"]}
def prepare_alignment(pairs: list[dict], *, encoder_id: str, train_ids: set[str],
                      calibration_ids: set[str], validation_ids: set[str], test_ids: set[str],
                      ridge: float = 0.1) -> dict:
    if not encoder_id or train_ids & calibration_ids or calibration_ids & (validation_ids | test_ids):
        raise ValueError("Alignment calibration groups overlap")
    if any(row.get("encoder_id") != encoder_id for row in pairs):
        raise ValueError("Alignment rows use a different encoder")
    if any(row.get("view_schema") != VIEW_SCHEMA for row in pairs):
        raise ValueError("Alignment rows must use the current shared structural view builder")
    policies = [structure_policy_spec(row.get("structure_policy")) for row in pairs]
    if policies and any(item != policies[0] for item in policies):
        raise ValueError("Alignment cannot mix historical structure policies")
    boundaries = [boundary.validate_mode(row.get("reuse_boundary_mode", "off")) for row in pairs]
    if boundaries and any(item != boundaries[0] for item in boundaries):
        raise ValueError("Alignment cannot mix reuse boundary policies")
    aligned = fit_alignment(pairs, train_ids=train_ids, validation_ids=validation_ids | calibration_ids,
                            test_ids=test_ids, ridge=ridge)
    aligned.update({"encoder_id": encoder_id, "view_schema": VIEW_SCHEMA,
                    "structure_policy": policies[0] if policies else structure_policy_spec(),
                    "reuse_boundary_mode": boundaries[0] if boundaries else "off",
                    "calibration_ids": sorted(calibration_ids), "validation_ids": sorted(validation_ids),
                    "method": "ridge_contract_and_directional_neighbor_view",
                    "warning": "Adapted tool views; not RAG-GFM WSE/InfoNCE reproduction"})
    validate_alignment(aligned, encoder_id=encoder_id)
    return aligned

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
