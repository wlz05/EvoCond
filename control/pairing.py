# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Collect observed teacher actions from matched pre-decision snapshots.'
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import Protocol
from control.teacher import ACTIONS, FEATURE_NAMES, PROVENANCE_FIELDS
FORMAT = "evocond_teacher_pairs_v2"
ACTION_SCOPE = "next_retrieval_only"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
def _number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False
def snapshot_hash(value: object) -> str:
    'Hash JSON snapshots, rejecting NaN and nonportable Python objects.'
    try:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Continuation snapshots must contain finite JSON values") from error
    return sha256(payload.encode("utf-8")).hexdigest()
@dataclass(frozen=True)
class ContinuationRequest:
    query_id: str
    decision_id: str
    seed: int
    alpha: float
    state: dict
    registry: dict
    candidates: list[dict]
    alignment: dict
    features: dict[str, float]
    query_embedding: list[float]
    context_hash: str
    candidate_hash: str
    registry_hash: str
    policy_hash: str
    alignment_hash: str
    encoder_id: str
    action_scope: str = ACTION_SCOPE
    execution_context_hash: str | None = None
class ContinuationEvaluator(Protocol):
    def __call__(self, request: ContinuationRequest) -> dict:
        'Execute one branch and return a verified outcome and start receipt.'
        ...
class PairCollectionError(RuntimeError):
    'No complete matched record may be emitted for a failed collection.'
def _check_inputs(*, query_id: str, decision_id: str, split: str,
                  state: dict, registry: dict, candidates: list[dict],
                  alignment: dict, features: dict, query_embedding: list[float],
                  encoder_id: str, policy_hash: str, seeds: list[int]) -> None:
    if split != "train":
        raise ValueError("Teacher action collection is training-only")
    if any(not isinstance(value, str) or not value.strip()
           for value in (query_id, decision_id, encoder_id)):
        raise ValueError("Question, decision and encoder identities are required")
    if not isinstance(policy_hash, str) or _SHA256.fullmatch(policy_hash) is None:
        raise ValueError("A frozen Qwen policy SHA-256 is required")
    if any(not isinstance(value, dict) or not value for value in (state, registry, alignment)):
        raise ValueError("Nonempty state, registry and alignment snapshots are required")
    if (not isinstance(candidates, list) or not candidates
            or any(not isinstance(value, dict) or not value for value in candidates)):
        raise ValueError("A frozen nonempty candidate list is required")
    if (not isinstance(features, dict) or set(features) != set(FEATURE_NAMES)
            or any(not _number(value) for value in features.values())):
        raise ValueError("Teacher collection needs the declared named feature schema")
    for name, value in features.items():
        low = -1.0 if name in ("semantic_top_score", "structural_top_score") else 0.0
        if not low <= value <= 1.0:
            raise ValueError(f"Teacher feature {name} is outside its normalized range")
    if (not isinstance(query_embedding, (list, tuple)) or not query_embedding
            or any(not _number(value) for value in query_embedding)
            or not any(value != 0 for value in query_embedding)):
        raise ValueError("A nonzero finite query embedding is required")
    norm = math.hypot(*query_embedding)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Query embedding has an invalid norm")
    if (not isinstance(seeds, (list, tuple)) or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError("Continuation seeds must be distinct nonnegative integers")
def _checked_outcome(result: dict, request: ContinuationRequest) -> dict:
    if not isinstance(result, dict) or result.get("executed") is not True:
        raise ValueError("An unexecuted or estimated action cannot supply a teacher reward")
    if result.get("action_scope") != ACTION_SCOPE:
        raise ValueError("The action must affect only the next retrieval decision")
    alpha = result.get("alpha")
    if not _number(alpha) or alpha != request.alpha:
        raise ValueError("Executed alpha differs from the requested action")
    if type(result.get("seed")) is not int or result["seed"] != request.seed:
        raise ValueError("Continuation seed was not reset to the matched seed")
    for name in PROVENANCE_FIELDS:
        if result.get(name) != getattr(request, name):
            raise ValueError(f"Continuation changed its starting {name}")
    if request.execution_context_hash is not None and result.get('execution_context_hash') != request.execution_context_hash:
        raise ValueError('Continuation changed its execution context')
    reward = result.get("reward")
    if not _number(reward) or not 0 <= reward <= 1:
        raise ValueError("An observed reward in [0,1] is required")
    if type(result.get("success")) is not bool:
        raise ValueError("An observed Boolean success outcome is required")
    trace_hash = result.get("trace_hash")
    if not isinstance(trace_hash, str) or _SHA256.fullmatch(trace_hash) is None:
        raise ValueError("An execution trace SHA-256 is required")
    return {"alpha": float(alpha), "reward": float(reward), "success": result["success"],
            "executed": True, "action_scope": ACTION_SCOPE,
            "seed": request.seed, "trace_hash": trace_hash,
            **{name: getattr(request, name) for name in PROVENANCE_FIELDS},
            **({'execution_context_hash': request.execution_context_hash} if request.execution_context_hash is not None else {})}
def collect_teacher_policy_pairs(*, evaluator: ContinuationEvaluator,
                                 query_id: str, decision_id: str, split: str,
                                 state: dict, registry: dict, candidates: list[dict],
                                 alignment: dict, features: dict[str, float],
                                 query_embedding: list[float], encoder_id: str,
                                 policy_hash: str, seeds: list[int], execution_context_hash: str | None = None) -> list[dict]:
    'Collect all five actually executed actions for each common seed.'
    _check_inputs(query_id=query_id, decision_id=decision_id, split=split,
                  state=state, registry=registry, candidates=candidates,
                  alignment=alignment, features=features, query_embedding=query_embedding,
                  encoder_id=encoder_id, policy_hash=policy_hash, seeds=seeds)
    if not callable(evaluator):
        raise ValueError("A continuation evaluator is required")
    if execution_context_hash is not None and (not isinstance(execution_context_hash, str) or not _SHA256.fullmatch(execution_context_hash)):
        raise ValueError('Execution context needs a SHA-256 identity')
    inputs = {"state": state, "registry": registry, "candidates": candidates,
              "alignment": alignment, "features": features,
              "query_embedding": list(query_embedding)}
    before = snapshot_hash(inputs)
    frozen = deepcopy(inputs)
    provenance = {"context_hash": snapshot_hash(frozen["state"]),
                  "candidate_hash": snapshot_hash(frozen["candidates"]),
                  "registry_hash": snapshot_hash(frozen["registry"]),
                  "policy_hash": policy_hash,
                  "alignment_hash": snapshot_hash(frozen["alignment"]),
                  "encoder_id": encoder_id}
    records = []
    for seed in seeds:
        outcomes = []
        for alpha in ACTIONS:
            request = ContinuationRequest(query_id=query_id, decision_id=decision_id,
                                          seed=seed, alpha=float(alpha), **deepcopy(frozen),
                                          **provenance, execution_context_hash=execution_context_hash)
            try:
                result = evaluator(request)
                if snapshot_hash(inputs) != before:
                    raise ValueError("Evaluator mutated the caller's shared pre-decision artifacts")
                outcomes.append(_checked_outcome(result, request))
            except Exception as error:
                raise PairCollectionError(
                    f"No paired record: {query_id}/{decision_id}, seed={seed}, "
                    f"alpha={alpha}: {type(error).__name__}: {error}") from error
        records.append({"schema": FORMAT, "query_id": query_id,
                        "decision_id": decision_id, "seed": seed,
                        "features": deepcopy(frozen["features"]),
                        "query_embedding": deepcopy(frozen["query_embedding"]),
                        **provenance, "actions": outcomes,
                        **({'execution_context_hash': execution_context_hash} if execution_context_hash is not None else {})})
    return records

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
