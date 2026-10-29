# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Schema-grounded subgoals and a trainable, state-conditioned dual-view ranker.'
from __future__ import annotations
from collections import defaultdict
from hashlib import sha256
import heapq
import json
import math
import re
from typing import Any
from control import guard
from control.bridge import _vector
from control.family import operator_family
from control.gate import normalize
FORMAT = "evocond_state_dual_v1"
EXPERIMENTS = ("legacy", "baseline", "prompt_only", "selective")
DIM = 32
CONTEXT = 40
EDGE_DIM = CONTEXT + 6
TYPE = re.compile(r"\bType:\s*([A-Za-z][A-Za-z0-9_]*)", re.I)
def stable_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode()).hexdigest()
def catalog(tools: list[dict]) -> dict[str, dict]:
    result = {}
    for tool in tools:
        name, produced = guard.toolname(tool), guard.output(tool)
        if not name or not produced:
            continue
        if name in result:
            raise ValueError("Duplicate tool name in current catalog")
        # Prices may change without changing the functional interface.
        desc = guard.COST.sub("cost of VARIABLE units", guard.description(tool))
        schema = tool.get("function", {}).get("parameters", {})
        version = stable_hash({"name": name, "description": desc, "schema": schema})
        result[name] = {"name": name, "version": version, "description": desc,
                        "requires": sorted(guard.requirements(tool)), "produces": produced,
                        "cost": guard.cost(tool), "schema": schema}
    return result
def observed_state(messages: list[dict], nodes: dict[str, dict]) -> dict:
    users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    epoch = users[-1] if users else 0
    query = str(messages[epoch].get("content") or "") if users else ""
    # Any later user message is a conservative new epoch, even if a change keyword is not.
    changed = len(users) > 1
    systems = [m for m in messages[:1] if m.get("role") == "system"]
    facts = guard.initial([*systems, *([messages[epoch]] if users else [])])
    stable_time = ""
    if changed and not guard.TIME_ID.search(query) and not guard.TIME_CHANGE.search(query):
        times = [v for m in messages[:epoch] if m.get("role") == "user"
                 for v in guard.TIME_ID.findall(str(m.get("content") or ""))]
        if times:
            stable_time = times[-1]
            facts.add("timeinfo")
    provenance = {value: "current_user_or_schema_inference" for value in facts}
    last_valid, pending, errors, calls, spent = "", "", 0, 0, 0.0
    previous = ""
    for m in messages[epoch + 1:]:
        if m.get("role") == "assistant":
            pending = guard.callname(m)
            if pending:
                calls += 1
                previous = pending
        elif m.get("role") == "tool":
            if pending in nodes:
                spent += nodes[pending]["cost"]
            if guard.failed(m):
                errors += 1
                authoritative = guard.current(m)
                if authoritative:
                    facts = authoritative
                    provenance = {v: "current_environment_error_state" for v in facts}
            else:
                errors = 0
                observed = {guard.canonical(v) for v in TYPE.findall(str(m.get("content") or ""))}
                if pending in nodes and nodes[pending]["produces"] in observed:
                    produced = nodes[pending]["produces"]
                    facts.add(produced)
                    provenance[produced] = pending
                    last_valid = pending
            pending = ""
    return {"query": query, "facts": sorted(facts), "provenance": provenance,
            "changed": changed, "epoch": epoch, "stable_time": stable_time,
            "last_valid": last_valid, "last_call": previous,
            "errors": errors, "calls": calls, "spent": spent}
def semantic(node: dict) -> list[float]:
    # Lightweight deterministic text encoder; not a pretrained semantic model.
    return list(_vector(normalize(node["name"].replace("_", " ") + " " + node["description"]), DIM))
def structural(node: dict) -> list[float]:
    family, _ = operator_family(node["name"])
    text = " ".join(["operator " + family, *["needs " + x for x in node["requires"]],
                     "produces " + node["produces"]])
    return list(_vector(text, DIM))
def context(state: dict) -> list[float]:
    numeric = [1.0, float(state["changed"]), min(state["errors"], 3) / 3,
               min(state["calls"], 20) / 20, min(len(state["facts"]), 20) / 20,
               min(state["spent"], 500) / 500, float(bool(state["last_valid"])),
               float(bool(state["stable_time"]))]
    return [*numeric, *_vector(normalize(state["query"]), 16),
            *_vector(" ".join(state["facts"]), 16)]
def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))
def cosine(left: list[float], right: list[float]) -> float:
    den = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    return sum(x*y for x, y in zip(left, right)) / den if den else 0.0
def edge_evidence(model: dict, source: str, target: str, nodes: dict[str, dict]) -> dict:
    if not source or source not in nodes or target not in nodes:
        return {"kind": "unknown", "signal": 0.0, "support": 0, "reliability": 0.0}
    direct, family = [], []
    sf, sd = operator_family(source)
    tf, td = operator_family(target)
    for edge in model["edges"]:
        a, b = model["nodes"][edge["source"]], model["nodes"][edge["target"]]
        if (a["name"], b["name"], a["version"], b["version"]) == (
                source, target, nodes[source]["version"], nodes[target]["version"]):
            direct.append(edge)
        af, ad = operator_family(a["name"])
        bf, bd = operator_family(b["name"])
        if (af, bf) == (sf, tf) and ad is not None and ad == bd and sd == td:
            family.append(edge)
    for kind, edges in (("exact", direct), ("family", family)):
        by_query = defaultdict(list)
        domains = set()
        for edge in edges:
            domains.add(operator_family(model["nodes"][edge["source"]]["name"])[1])
            for key, value in edge["outcomes"].items():
                by_query[key].append(value)
        n = len(by_query)
        if n < 3 or kind == "family" and len(domains) < 2:
            continue
        # Each independent training query has total weight ONE; retries, modes and teacher policies do.
        positive = sum(sum(values) / len(values) for values in by_query.values())
        posterior = (positive + 1) / (n + 2)
        reliability = n / (n + 3) * (1.0 if kind == "exact" else 0.5)
        return {"kind": kind, "signal": 2 * posterior - 1,
                "support": n, "reliability": reliability}
    return {"kind": "unknown", "signal": 0.0, "support": 0, "reliability": 0.0}
def view_features(model: dict, state: dict, names: list[str], nodes: dict[str, dict]) -> dict:
    query = list(_vector(normalize(state["query"]), DIM))
    ctx = context(state)
    sem, aligned, edges, gates, details = [], [], [], [], []
    for name in names:
        node = nodes[name]
        sv = structural(node)
        projected = [sum(sv[i] * model["alignment"][i][j] for i in range(DIM)) for j in range(DIM)]
        se, st = cosine(query, semantic(node)), cosine(query, projected)
        evidence = edge_evidence(model, state["last_valid"], name, nodes)
        compatible = bool(state["last_valid"] in nodes and
                          nodes[state["last_valid"]]["produces"] in node["requires"])
        known_version = node["version"] in model["nodes"]
        xedge = [*ctx, float(compatible), min(evidence["support"], 10) / 10,
                 evidence["reliability"], float(evidence["kind"] == "family"),
                 se, float(not known_version)]
        sem.append(se)
        aligned.append(st)
        edges.append(evidence["signal"] * evidence["reliability"])
        gates.append(xedge)
        details.append(evidence)
    return {"context": ctx, "semantic": sem, "aligned": aligned,
            "edge": edges, "edge_context": gates, "evidence": details}
def scores(model: dict, state: dict, names: list[str], nodes: dict[str, dict], policy: str) -> tuple[list[float], dict]:
    views = view_features(model, state, names, nodes)
    alpha = 0.5 if policy == "fixed" else sigmoid(sum(a*b for a, b in zip(model["fusion"], views["context"])))
    values, details = [], []
    for i, name in enumerate(names):
        gate = 1.0 if policy == "fixed" else sigmoid(sum(a*b for a, b in zip(model["edge_gate"], views["edge_context"][i])))
        value = (1-alpha) * views["semantic"][i] + alpha * (
            0.5 * views["aligned"][i] + 0.5 * gate * views["edge"][i])
        values.append(value)
        details.append({"tool": name, "semantic": round(views["semantic"][i], 5),
                        "aligned_structure": round(views["aligned"][i], 5),
                        "edge_gate": round(gate, 5), **views["evidence"][i]})
    return values, {"structure_weight": round(alpha, 5), "tools": details}
def routes(state: dict, tools: list[dict], extra_cost: float = 0.5) -> tuple[list[list[str]], dict]:
    nodes = catalog(tools)
    selectors = [name for name in nodes if guard.canonical(name).startswith("selectfinal")]
    if len(selectors) != 1 or not state["facts"] or not all(guard.COST.search(guard.description(t)) for t in tools):
        return [], {"reason": "unknown_schema_or_goal"}
    selector = selectors[0]
    facts = frozenset(state["facts"])
    if nodes[selector]["produces"] in facts:
        # The final tool already returned a goal witness. Leave the answer generation turn alone.
        return [], {"reason": "goal_type_already_observed"}
    blocked = state["last_call"] if state["errors"] else ""
    base = guard.route(set(facts), set(nodes[selector]["requires"]), tools, blocked or selector, 6)
    if base is None or selector == blocked:
        return [], {"reason": "no_type_feasible_route"}
    baseline = [*base, selector]
    budget = sum(nodes[n]["cost"] for n in baseline) + extra_cost
    frontier = [(0.0, (), facts)]
    found, expanded = [], 0
    while frontier and expanded < 4096 and len(found) < 128:
        price, path, available = heapq.heappop(frontier)
        expanded += 1
        if price > budget + 1e-9:
            break
        if path and path[-1] == selector:
            found.append(list(path))
            continue
        if len(path) >= 7:
            continue
        for name, node in sorted(nodes.items()):
            if name == blocked or name in path or not set(node["requires"]).issubset(available):
                continue
            if name != selector and node["produces"] in available:
                continue
            added = price + node["cost"]
            if added <= budget + 1e-9:
                heapq.heappush(frontier, (added, (*path, name), available | {node["produces"]}))
        if len(frontier) > 4096:
            break
    truncated = bool(frontier) and (expanded >= 4096 or len(found) >= 128 or len(frontier) > 4096)
    if truncated:
        return [baseline], {"reason": "search_limit_cost_fallback", "expanded": expanded}
    if baseline not in found:
        found.append(baseline)
    found.sort(key=lambda p: (sum(nodes[n]["cost"] for n in p), len(p), p))
    return found, {"reason": "bounded_type_routes", "expanded": expanded}
def subgoals(path: list[str], state: dict, nodes: dict[str, dict]) -> list[dict]:
    producers = {value: "observed" for value in state["facts"]}
    result = []
    for i, name in enumerate(path):
        node = nodes[name]
        if not set(node["requires"]).issubset(producers):
            raise ValueError("Subgoal has unsatisfied AND prerequisites")
        identity = f"s{i}"
        result.append({"id": identity, "goal_type": node["produces"], "tool": name,
                       "required_inputs": node["requires"],
                       "depends_on": sorted({producers[x] for x in node["requires"] if producers[x] != "observed"}),
                       "completion": "matching observed output type; semantic correctness still requires checking"})
        producers[node["produces"]] = identity
    return result
def planning_note(path: list[str], price: float, state: dict, nodes: dict[str, dict]) -> tuple[str, list[dict]]:
    'One prompt formatter shared by the cost-only and ranked-route arms.'
    graph = subgoals(path, state, nodes)
    scope = "GG changed goal; old derived choices are invalid" if state["changed"] else "current task"
    continuity = f" Reuse unchanged time {state['stable_time']} only." if state["stable_time"] else ""
    note = (f"EvoCond schema-grounded subgoals for {scope}. "
            f"Current observed/inferable types: {', '.join(state['facts'])}. "
            f"Suggested remaining type-feasible route: {' -> '.join(path)}; tool cost {price:g}. "
            + continuity + " Check latest user constraints and actual object IDs at every step; "
            "a type-feasible route is NOT proof of a correct answer. Never reuse stale candidates after a goal update. "
            "Each subgoal requires ALL its inputs; perform at most one valid tool call now. "
            "Subgoals: " + json.dumps(graph, ensure_ascii=False, separators=(",", ":")))
    return note, graph
def _rank_paths(paths: list[list[str]], state: dict, nodes: dict[str, dict], model: dict, policy: str) -> list[dict]:
    candidates = []
    for path in paths:
        hypothetical = {**state, "facts": list(state["facts"])}
        values, evidence = [], []
        for name in path:
            ranked, detail = scores(model, hypothetical, [name], nodes, policy)
            values.extend(ranked)
            evidence.append(detail)
            hypothetical = {**hypothetical, "facts": sorted(set(hypothetical["facts"]) | {nodes[name]["produces"]}),
                            "last_valid": name, "calls": hypothetical["calls"] + 1,
                            "spent": hypothetical["spent"] + nodes[name]["cost"]}
        price = sum(nodes[n]["cost"] for n in path)
        candidates.append({"path": path, "score": sum(values)/len(values), "cost": price, "views": evidence})
    return candidates
def _experimental_intervene(messages: list[dict], tools: list[dict], model: dict, policy: str, experiment: str) -> tuple[list[dict], list[dict], dict]:
    # Match serve.py's original EvoCond full/noaudit, planning enabled, no memory. A cost-only.
    base_messages, base_tools, base_event = guard.intervene(
        messages, tools, False, "full", True, None, 0.0, "exact")
    event = {"action": "state_experiment", "experiment": experiment, "policy": policy,
             "reference": "original_evocond_full_noaudit_planning_no_memory",
             "base_event": base_event, "route_changed": False,
             "route_reference": "cost_only_candidate_not_original_evocond",
             "prompt_changed": False, "tools_changed": False, "actual_intervention": False,
             "fallback": experiment != "baseline", "reason": "baseline_requested"}
    if experiment == "baseline":
        return base_messages, base_tools, event
    try:
        issue, _, _ = guard.pressure(messages)
        if issue:
            event["reason"] = "baseline_recovery"
            return base_messages, base_tools, event
        nodes = catalog(tools)
        state = observed_state(messages, nodes)
        paths, search = routes(state, tools)
        event.update(state=state, search=search, candidate_count=len(paths))
        if not paths:
            event["reason"] = "no_type_feasible_candidates"
            return base_messages, base_tools, event
        cost_path = paths[0]
        event["baseline_path"] = cost_path
        if experiment == "prompt_only":
            chosen = {"path": cost_path, "cost": sum(nodes[n]["cost"] for n in cost_path), "views": []}
        else:
            candidates = _rank_paths(paths, state, nodes, model, policy)
            if any(not math.isfinite(c["score"]) for c in candidates):
                raise ValueError("Nonfinite candidate score")
            base = candidates[0]
            best = max(candidates, key=lambda c: (c["score"], -c["cost"], -len(c["path"])))
            event.update(baseline_score=base["score"], best_score=best["score"], margin=0.02)
            if best["path"] == cost_path or not best["score"] > base["score"] + 0.02:
                event["reason"] = "no_distinct_route_clears_margin"
                return base_messages, base_tools, event
            chosen = best
        note, graph = planning_note(chosen["path"], chosen["cost"], state, nodes)
        revised = [*messages, {"role": "system", "content": note}]
        available = list(tools)
        prompt_changed = revised != base_messages
        tools_changed = available != base_tools
        event.update(path=chosen["path"], cost=chosen["cost"], views=chosen["views"], subgoals=graph,
                     route_changed=chosen["path"] != cost_path, prompt_changed=prompt_changed,
                     tools_changed=tools_changed, actual_intervention=prompt_changed or tools_changed,
                     fallback=False, reason="cost_only_prompt" if experiment == "prompt_only" else "ranked_route_prompt")
        return revised, available, event
    except Exception as error:
        # A failed experimental controller must not silently change its reference. Keep failures.
        event.update(reason="controller_error", controller_error={"type": type(error).__name__, "message": str(error)[:160]})
        return base_messages, base_tools, event
def intervene(messages: list[dict], tools: list[dict], model: dict, policy: str = "learned", *, experiment: str = "legacy") -> tuple[list[dict], list[dict], dict]:
    'Legacy by default; explicit ablations compare payloads with original EvoCond.'
    if policy not in {"fixed", "learned"}:
        raise ValueError("Unknown state-controller policy")
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown state-controller experiment")
    if experiment != "legacy":
        return _experimental_intervene(messages, tools, model, policy, experiment)
    nodes = catalog(tools)
    state = observed_state(messages, nodes)
    # Preserve the established LL/GL repair; never replace verified input repair with an empirical.
    issue, _, _ = guard.pressure(messages)
    if issue:
        revised, available, recovery = guard.intervene(messages, tools, False, "full", False)
        return revised, available, {"action": "state_recovery", "state": state,
                                    "policy": policy, "recovery": recovery}
    paths, search = routes(state, tools)
    if not paths:
        revised, available, recovery = guard.intervene(messages, tools, False, "full", False)
        return revised, available, {"action": "state_abstain", "policy": policy,
                                    "state": state, "search": search, "recovery": recovery}
    candidates = _rank_paths(paths, state, nodes, model, policy)
    # Default abstention retains the cost-only route unless a bounded-cost alternative clears a.
    base = candidates[0]
    best = max(candidates, key=lambda c: (c["score"], -c["cost"], -len(c["path"])))
    chosen = best if best["score"] > base["score"] + 0.02 else base
    note, graph = planning_note(chosen["path"], chosen["cost"], state, nodes)
    return [*messages, {"role": "system", "content": note}], tools, {
        "action": "state_plan", "policy": policy, "state": state, "subgoals": graph,
        "path": chosen["path"], "cost": chosen["cost"], "baseline_path": base["path"],
        "reranked": chosen is not base, "candidate_count": len(candidates), "search": search,
        "views": chosen["views"], "scope": "typed_subgoals_not_general_language_decomposition"}
def validate(model: dict) -> None:
    if model.get("format") != FORMAT:
        raise ValueError("Unknown state controller format")
    vectors = [*model.get("alignment", []), model.get("fusion", []), model.get("edge_gate", [])]
    if len(model.get("alignment", [])) != DIM or any(len(x) != DIM for x in model["alignment"]) or len(model.get("fusion", [])) != CONTEXT or len(model.get("edge_gate", [])) != EDGE_DIM:
        raise ValueError("State controller parameter shape mismatch")
    if any(type(x) not in (int, float) or not math.isfinite(x) for row in vectors for x in row):
        raise ValueError("Nonfinite state controller parameter")
    if not isinstance(model.get("nodes"), dict) or not model["nodes"] or not isinstance(model.get("edges"), list):
        raise ValueError("State controller dual stores missing")
    for key, node in model["nodes"].items():
        if not isinstance(node, dict) or node.get("version") != key or not isinstance(node.get("requires"), list) or not node.get("produces"):
            raise ValueError("Misaligned tool identity")
        if any(not isinstance(node.get(k), str) or not node[k] for k in ("name", "description", "produces")) or not isinstance(node.get("schema"), dict):
            raise ValueError("Invalid tool interface")
        if key != stable_hash({k: node[k] for k in ("name", "description", "schema")}):
            raise ValueError("Tool version does not match its interface")
        if node["requires"] != sorted({guard.canonical(x) for x in node["schema"].get("required", [])}):
            raise ValueError("Tool prerequisites do not match its interface")
        if type(node.get("cost")) not in (int, float) or not math.isfinite(node["cost"]) or node["cost"] < 0:
            raise ValueError("Invalid tool cost")
    for edge in model["edges"]:
        if edge.get("source") not in model["nodes"] or edge.get("target") not in model["nodes"]:
            raise ValueError("Edge references an unknown node version")
        if not edge.get("outcomes") or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in edge["outcomes"].values()):
            raise ValueError("Invalid per-query edge evidence")

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
