# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Fail-closed, versioned admission for tools executed by an isolated backend.'
from __future__ import annotations
import ast
from copy import deepcopy
from hashlib import sha256
import json
import math
import re
from typing import Any, Protocol
from control.isolation import VerifiedSandboxFusion, validate_arguments
FORMAT = "evocond_searl_dynamic_registry_v1"
NAME = re.compile(r"^[A-Za-z_][A-Za-z_0-9]{0,127}$")
SPEC_FIELDS = ("name", "description", "arguments", "returns", "side_effects", "code")
class IsolatedBackend(Protocol):
    isolation: str
    isolation_verified: bool
    def run_function(self, source: str, name: str, inputs: dict,
                     timeout: float) -> dict: ...
class SemanticEncoder(Protocol):
    def encode(self, text: str) -> list[float]: ...
class SandboxFusionBackend:
    'Execute virtual generated tools through SandboxFusion, never locally.'
    def __init__(self, endpoint: str, *, isolation: str,
                 isolation_verified: bool = False, config: dict | None = None,
                 receipt: dict | None = None):
        if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")):
            raise ValueError("SandboxFusion endpoint must be an HTTP URL")
        self.endpoint = endpoint
        self.isolation = isolation
        self.isolation_verified = False
        self._verified = None
        if config is not None or receipt is not None:
            if not isinstance(config, dict) or config.get("endpoint") != endpoint or config.get("isolation") != isolation:
                raise ValueError("Sandbox endpoint disagrees with inspected configuration")
            self._verified = VerifiedSandboxFusion(config, receipt)
            self.isolation_verified = True
    def run_function(self, source: str, name: str, inputs: dict,
                     timeout: float) -> dict:
        _check_backend(self)
        return self._verified.run_function(source, name, inputs, timeout)
def digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")).hexdigest()
def empty_registry() -> dict:
    return {"format": FORMAT, "semantic": {}, "structure": {"nodes": {}, "edges": {}},
            "aliases": {}, "quarantine": []}
def _bind_alias(registry: dict, name: str, version: str) -> None:
    history = registry.setdefault("alias_history", {key: [value] for key, value in registry["aliases"].items()})
    versions = history.setdefault(name, [])
    for item in (registry["aliases"].get(name), version):
        if item is not None and item not in versions:
            versions.append(item)
    registry["aliases"][name] = version
def resolve_tool_version(registry: dict, name: str, version: str | None = None) -> str:
    'Resolve current or historically bound identity for execution receipts.'
    validate(registry)
    if not isinstance(name, str) or NAME.fullmatch(name) is None:
        raise ValueError("Invalid tool alias")
    if version is None:
        version = registry["aliases"].get(name)
    if not isinstance(version, str) or version not in registry["semantic"]:
        raise ValueError("Tool version is not admitted")
    if (registry["semantic"][version]["name"] != name and registry["aliases"].get(name) != version
            and version not in registry.get("alias_history", {}).get(name, [])):
        raise ValueError("Tool name is incompatible with the observed version")
    return version
def _check_backend(backend: IsolatedBackend) -> None:
    if (getattr(backend, "isolation", None) not in {"container", "microvm"}
            or getattr(backend, "isolation_verified", None) is not True):
        raise RuntimeError("Generated tools require an independently verified isolated backend")
def _check_source(name: str, source: str) -> None:
    if not isinstance(name, str) or NAME.fullmatch(name) is None:
        raise ValueError("Invalid generated-tool name")
    if not isinstance(source, str) or not 0 < len(source) <= 32768:
        raise ValueError("Invalid generated-tool source length")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("Generated tool source is not valid Python") from error
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
               for node in tree.body):
        raise ValueError("Generated source does not define its declared function")
def _check_spec(spec: dict) -> None:
    if not isinstance(spec, dict):
        raise ValueError("Tool spec must be an object")
    _check_source(spec.get("name"), spec.get("code"))
    if not isinstance(spec.get("description"), str) or not spec["description"].strip():
        raise ValueError("Tool capability description is missing")
    if not isinstance(spec.get("arguments"), (str, dict)) or not isinstance(spec.get("returns"), str):
        raise ValueError("Tool input/output contract is missing")
    if not isinstance(spec.get("side_effects"), str) or not spec["side_effects"].strip():
        raise ValueError("Tool side-effect declaration is missing")
    if any(len(spec[key].encode("utf-8")) > 16384 for key in ("description", "returns", "side_effects")):
        raise ValueError("Tool metadata exceeds 16 KiB per field")
    try:
        encoded_arguments = json.dumps(spec["arguments"], sort_keys=True, allow_nan=False)
        if len(encoded_arguments.encode("utf-8")) > 65536:
            raise ValueError("Tool argument contract exceeds 64 KiB")
    except (TypeError, ValueError) as error:
        raise ValueError("Tool input contract must be finite JSON") from error
def _version(spec: dict) -> str:
    return digest({key: spec[key] for key in SPEC_FIELDS})
def _distinct_probes(inputs: list[dict] | None) -> list[tuple[str, dict]]:
    if not isinstance(inputs, list):
        raise ValueError("Merging needs at least two distinct behavioral probes")
    probes = {}
    for item in inputs:
        if not isinstance(item, dict):
            raise ValueError("Behavioral probe inputs must be JSON objects")
        try:
            encoded = json.dumps(item, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                 separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise ValueError("Behavioral probe inputs must be finite JSON") from error
        probes.setdefault(sha256(encoded.encode("utf-8")).hexdigest(), item)
    if len(probes) < 2:
        raise ValueError("Merging needs at least two distinct behavioral probes")
    return list(probes.items())
def validate(registry: dict) -> None:
    if not isinstance(registry, dict) or registry.get("format") != FORMAT:
        raise ValueError("Unknown dynamic-tool registry")
    semantic, structure, aliases = registry.get("semantic"), registry.get("structure"), registry.get("aliases")
    if not isinstance(semantic, dict) or not isinstance(structure, dict) or not isinstance(aliases, dict):
        raise ValueError("Malformed dual stores")
    nodes, edges = structure.get("nodes"), structure.get("edges")
    if not isinstance(nodes, dict) or not isinstance(edges, dict) or set(semantic) != set(nodes):
        raise ValueError("Semantic and structural nodes are not version-aligned")
    for version, item in semantic.items():
        if not isinstance(item, dict) or not isinstance(nodes[version], dict):
            raise ValueError("Malformed versioned tool")
        _check_spec(item)
        if item.get("version") != version or nodes[version].get("version") != version:
            raise ValueError("Misaligned tool version")
        if _version(item) != version:
            raise ValueError("Tool description, contract, or source changed after admission")
        if item.get("code_sha256") != sha256(item.get("code", "").encode("utf-8")).hexdigest():
            raise ValueError("Tool source changed after admission")
        if any(nodes[version].get(key) != item[key] for key in ("name", "arguments", "returns")):
            raise ValueError("Semantic and structural tool contracts disagree")
    if any(not isinstance(name, str) or NAME.fullmatch(name) is None or version not in semantic
           for name, version in aliases.items()):
        raise ValueError("Alias points at a missing tool")
    history = registry.get("alias_history")
    if history is not None:
        if not isinstance(history, dict):
            raise ValueError("Malformed tool alias history")
        for name, versions in history.items():
            if (not isinstance(name, str) or NAME.fullmatch(name) is None
                    or not isinstance(versions, list) or not versions
                    or any(not isinstance(version, str) or version not in semantic for version in versions)
                    or len(set(versions)) != len(versions)):
                raise ValueError("Invalid historical tool binding")
        if any(version not in history.get(name, []) for name, version in aliases.items()):
            raise ValueError("Current tool alias is absent from its recorded history")
    for key, edge in edges.items():
        if not isinstance(edge, dict):
            raise ValueError("Malformed legacy edge")
        if edge.get("source") not in nodes or edge.get("target") not in nodes:
            raise ValueError("Edge points at a missing tool")
        if key != f"{edge['source']}->{edge['target']}":
            raise ValueError("Legacy edge identity disagrees with its endpoints")
        if (type(edge.get("weight")) not in (int, float) or not math.isfinite(edge["weight"])
                or not 0 <= edge["weight"] <= 1 or not isinstance(edge.get("evidence"), dict)):
            raise ValueError("Invalid edge")
        if any(not isinstance(task_id, str) or not task_id.strip() or type(signal) not in (int, float)
               or signal not in (0, 1) for task_id, signal in edge["evidence"].items()):
            raise ValueError("Invalid legacy edge feedback provenance")
def _run(backend: IsolatedBackend, spec: dict, inputs: dict, timeout: float) -> dict:
    _check_backend(backend)
    if not isinstance(inputs, dict) or type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError("Invalid isolated execution request")
    validate_arguments(spec["code"], spec["name"], inputs)
    result = backend.run_function(spec["code"], spec["name"], inputs, float(timeout))
    if not isinstance(result, dict) or type(result.get("run_success")) is not bool:
        raise RuntimeError("Isolated backend returned an unverifiable execution result")
    return result
def _similarity(left: list[float], right: list[float]) -> float:
    if (not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple))
            or not left or len(left) != len(right)
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in [*left, *right])):
        raise ValueError("Invalid semantic embeddings")
    denominator = math.sqrt(sum(value*value for value in left) * sum(value*value for value in right))
    return sum(a*b for a, b in zip(left, right)) / denominator if denominator else 0.0
def _candidate(registry: dict, spec: dict, encoder: SemanticEncoder | None) -> tuple[str | None, float | None]:
    if encoder is None or not registry["semantic"]:
        return None, None
    current = encoder.encode(spec["description"])
    scores = [(_similarity(current, encoder.encode(item["description"])), version)
              for version, item in registry["semantic"].items()]
    score, version = max(scores)
    return version, score
def admit(registry: dict, spec: dict, backend: IsolatedBackend, *, probe_inputs: dict,
          task_id: str, equivalence_inputs: list[dict] | None = None,
          merge_with: str | None = None, encoder: SemanticEncoder | None = None,
          similarity_threshold: float = 0.88,
          timeout: float = 15.0) -> tuple[dict, dict]:
    'Admit only after successful sandbox execution; merge only after paired tests.'
    validate(registry)
    _check_spec(spec)
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("Admission needs independent task provenance")
    if (type(similarity_threshold) not in (int, float) or not math.isfinite(similarity_threshold)
            or not 0 <= similarity_threshold <= 1):
        raise ValueError("Invalid semantic novelty threshold")
    spec = deepcopy(spec)
    version = _version(spec)
    if version in registry["semantic"]:
        return deepcopy(registry), {"decision": "already_admitted", "version": version}
    result = _run(backend, spec, probe_inputs, timeout)
    if result["run_success"] is not True:
        updated = deepcopy(registry)
        updated["quarantine"].append({"version": version, "task_id": task_id,
                                       "reason": "sandbox_probe_failed"})
        return updated, {"decision": "quarantine", "version": version}
    updated = deepcopy(registry)
    candidate, similarity = _candidate(registry, spec, encoder)
    same_contract = candidate is not None and (
        spec["arguments"], spec["returns"], spec["side_effects"]) == (
        registry["semantic"][candidate]["arguments"],
        registry["semantic"][candidate]["returns"],
        registry["semantic"][candidate]["side_effects"])
    if merge_with is None and same_contract and similarity >= similarity_threshold:
        updated["quarantine"].append({"version": version, "task_id": task_id,
                                       "reason": "semantic_duplicate_requires_behavioral_verification",
                                       "candidate": candidate, "similarity": similarity})
        return updated, {"decision": "quarantine", "version": version,
                         "candidate": candidate, "similarity": similarity}
    if merge_with is not None:
        if encoder is None or candidate != merge_with or similarity < similarity_threshold:
            raise ValueError("Semantic evidence does not support the requested merge")
        existing = updated["semantic"].get(merge_with)
        if existing is None:
            raise ValueError("Merge target is not an admitted tool")
        if (spec["arguments"], spec["returns"], spec["side_effects"]) != (
                existing["arguments"], existing["returns"], existing["side_effects"]):
            raise ValueError("Cannot merge tools with different contracts or side effects")
        probes = _distinct_probes(equivalence_inputs)
        old_spec = {"name": existing["name"], "code": existing["code"]}
        for _, inputs in probes:
            old, new = _run(backend, old_spec, inputs, timeout), _run(backend, spec, inputs, timeout)
            if not old["run_success"] or not new["run_success"] or old.get("result") != new.get("result"):
                raise ValueError("Behavioral equivalence failed")
        _bind_alias(updated, spec["name"], merge_with)
        return updated, {"decision": "merge_verified", "version": merge_with,
                         "behavioral_probes": len(probes), "similarity": similarity,
                         "behavioral_probe_sha256": [key for key, _ in probes]}
    semantic = {"version": version, "name": spec["name"],
                "description": spec["description"], "arguments": deepcopy(spec["arguments"]),
                "returns": spec["returns"], "side_effects": spec["side_effects"],
                "code": spec["code"], "code_sha256": sha256(spec["code"].encode("utf-8")).hexdigest(),
                "admitted_from_task": task_id}
    from control.primitive import profile
    semantic["abstraction_profile"] = profile(spec)
    updated["semantic"][version] = semantic
    updated["structure"]["nodes"][version] = {"version": version, "name": spec["name"],
                                                 "arguments": deepcopy(spec["arguments"]), "returns": spec["returns"]}
    _bind_alias(updated, spec["name"], version)
    validate(updated)
    return updated, {"decision": "new_versioned_node", "version": version,
                     "nearest_version": candidate, "nearest_similarity": similarity,
                     "semantic_screening": "encoder" if encoder is not None else "unavailable_conservative_new_node"}
def invoke(registry: dict, name: str, inputs: dict, backend: IsolatedBackend,
           *, timeout: float = 15.0, expected_version: str | None = None) -> dict:
    'Invoke the active alias and return the actual immutable tool identity.'
    version = resolve_tool_version(registry, name)
    if expected_version is not None and expected_version != version:
        raise ValueError("Tool alias changed after ranking; retrieve again before invocation")
    result = _run(backend, registry["semantic"][version], inputs, timeout)
    return {**result, "tool_version": version, "invoked_name": name}
def update_edge(registry: dict, *, source_name: str, target_name: str, task_id: str,
                split: str, direct_feedback: str, task_success: bool | None,
                learning_rate: float = 0.1) -> dict:
    'Legacy EMA updater; use edges.observe_transition for contextual v2.'
    validate(registry)
    if "edge_memory" in registry:
        raise ValueError('Contextual edge memory is active; use edges.observe_transition')
    if split != "train" or not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("Validation/test and anonymous tasks cannot update edge memory")
    if (type(learning_rate) not in (int, float) or not math.isfinite(learning_rate)
            or not 0 < learning_rate <= 1):
        raise ValueError("Invalid legacy edge learning rate")
    if direct_feedback not in {"verified", "failed", "unknown"} or type(task_success) not in (bool, type(None)):
        raise ValueError("Invalid edge feedback")
    source, target = registry["aliases"].get(source_name), registry["aliases"].get(target_name)
    if source is None or target is None:
        raise ValueError("Edge uses an unavailable tool")
    if direct_feedback == "unknown" or direct_feedback == "verified" and task_success is not True:
        return deepcopy(registry)
    updated = deepcopy(registry)
    key = f"{source}->{target}"
    edge = updated["structure"]["edges"].setdefault(
        key, {"source": source, "target": target, "weight": 0.5, "evidence": {}})
    signal = 1.0 if direct_feedback == "verified" else 0.0
    prior = edge["evidence"].get(task_id)
    if prior is not None:
        if prior != signal:
            raise ValueError("Contradictory evidence for the same independent task")
        return updated
    edge["evidence"][task_id] = signal
    edge["weight"] += learning_rate * (signal - edge["weight"])
    validate(updated)
    return updated

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
