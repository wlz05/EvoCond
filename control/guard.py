# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import ast
import heapq
import re
from typing import Any, Mapping, Sequence
from control.bridge import EvoltBridge
from control.family import score_route
CURRENT = re.compile(r"Current:\s*(\[[^\]]*\])", re.IGNORECASE)
REQUIRED = re.compile(r"Required:\s*(\[[^\]]*\])", re.IGNORECASE)
OUTPUT = re.compile(r"output type of this tool is\s+([A-Za-z][A-Za-z0-9_]*)", re.IGNORECASE)
IDENTIFIER = re.compile(r"<([A-Za-z][A-Za-z_]*?)(?:\d{4,})>")
TIME_ID = re.compile(r"<TimeInfo\d{4,}>", re.IGNORECASE)
TIME_CHANGE = re.compile('\\b(?:change|changed|changing|update|updated|different|new|move|moved|reschedule)\\s+(?:my\\s+|the\\s+|our\\s+)?(?:travel\\s+|trip\\s+)?(?:date|time|schedule|day)\\b|\\b(?:tomorrow|next\\s+(?:week|month|year)|this\\s+weekend)\\b|(?:\u6539|\u6362|\u8c03\u6574)(?:\u65f6\u95f4|\u65e5\u671f|\u884c\u7a0b\u65e5\u671f)', re.IGNORECASE)
INFER = re.compile(r"You should infer\s+(.*?)\s+from the user query", re.IGNORECASE | re.DOTALL)
PRIMITIVE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*(?:Category|Tier|Style|FeaturePackage|TimeInfo|LocationPreference)\b")
COST = re.compile(r"\bcost of\s+([0-9]+(?:\.[0-9]+)?)\s+units", re.IGNORECASE)
CHANGE = re.compile('\\b(?:changed?|changing|updated?|revised?|tweak)\\s+(?:my\\s+|the\\s+|user\\s+)?(?:preference|requirement|goal|constraint)s?\\b|\\b(?:different|new)\\s+(?:preference|requirement|goal|constraint)s?\\b|\\bswitch up my requirements\\b|\\bno longer\\b|\\binstead\\b|(?:\u66f4\u6539|\u6539\u53d8|\u66f4\u65b0|\u65b0)(?:\u504f\u597d|\u8981\u6c42|\u76ee\u6807|\u7ea6\u675f)', re.IGNORECASE)
ERRORS = ("[error]", "tool execution failed", "error:")
def canonical(value: Any) -> str:
    return "".join(character for character in str(value).strip().lower() if character.isalnum())
def callname(message: Mapping[str, Any]) -> str:
    calls = message.get("tool_calls")
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)) or not calls:
        return ""
    call = calls[0]
    if not isinstance(call, Mapping):
        return ""
    function = call.get("function", call)
    return str(function.get("name", "")).strip() if isinstance(function, Mapping) else ""
def toolname(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    return str(function.get("name", "")).strip() if isinstance(function, Mapping) else ""
def description(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    return str(function.get("description") or "") if isinstance(function, Mapping) else ""
def requirements(tool: Mapping[str, Any]) -> set[str]:
    function = tool.get("function")
    parameters = function.get("parameters") if isinstance(function, Mapping) else None
    raw = parameters.get("required", ()) if isinstance(parameters, Mapping) else ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return set()
    return {canonical(item) for item in raw if canonical(item)}
def failed(message: Mapping[str, Any]) -> bool:
    content = str(message.get("content") or "").strip().lower()
    return any(token in content for token in ERRORS)
def listed(message: Mapping[str, Any], pattern: re.Pattern[str]) -> set[str]:
    match = pattern.search(str(message.get("content") or ""))
    if match is None:
        return set()
    try:
        values = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return set()
    return {canonical(item) for item in values if canonical(item)} if isinstance(values, list) else set()
def current(message: Mapping[str, Any]) -> set[str]:
    return listed(message, CURRENT)
def output(tool: Mapping[str, Any]) -> str:
    match = OUTPUT.search(description(tool))
    return canonical(match.group(1)) if match else ""
def cost(tool: Mapping[str, Any]) -> float:
    match = COST.search(description(tool))
    return float(match.group(1)) if match else 1000.0
def pathcost(path: Sequence[str], tools: Sequence[Mapping[str, Any]]) -> float:
    prices = {toolname(tool): cost(tool) for tool in tools}
    return sum(prices[name] for name in path)
def initial(messages: Sequence[Mapping[str, Any]]) -> set[str]:
    state: set[str] = set()
    for message in messages:
        if message.get("role") == "user":
            state.update(canonical(match) for match in IDENTIFIER.findall(str(message.get("content") or "")))
        if message.get("role") == "system":
            match = INFER.search(str(message.get("content") or ""))
            if match:
                state.update(canonical(item) for item in PRIMITIVE.findall(match.group(1)))
    return state
def route(state: set[str], target: set[str], tools: Sequence[Mapping[str, Any]], blocked: str = "", limit: int = 5) -> list[str] | None:
    if not state or not target:
        return None
    catalog = sorted((toolname(tool), requirements(tool), output(tool), cost(tool)) for tool in tools if toolname(tool) and toolname(tool) != blocked and output(tool))
    start = frozenset(state)
    frontier: list[tuple[float, int, tuple[str, ...], frozenset[str]]] = [(0.0, 0, (), start)]
    seen: dict[frozenset[str], tuple[float, int]] = {start: (0.0, 0)}
    while frontier and len(seen) <= 1024:
        price, steps, path, available = heapq.heappop(frontier)
        if target.issubset(available):
            return list(path)
        if steps >= limit:
            continue
        for name, needs, produced, added in catalog:
            if produced in available or not needs.issubset(available):
                continue
            updated = available | {produced}
            rank = (price + added, steps + 1)
            if updated not in seen or rank < seen[updated]:
                seen[updated] = rank
                heapq.heappush(frontier, (rank[0], rank[1], (*path, name), updated))
    return None
def memory_route(
    state: set[str],
    target: set[str],
    tools: Sequence[Mapping[str, Any]],
    memory: EvoltBridge | None,
    query: str,
    blocked: str = "",
    limit: int = 5,
    max_extra_cost: float = 0.0,
    backoff: str = "exact",
) -> tuple[list[str] | None, dict[str, Any]]:
    'Softly rerank bounded type-feasible routes; never invent an action.'
    if backoff not in {"exact", "family"}:
        raise ValueError("Unknown memory backoff mode")
    baseline = route(state, target, tools, blocked, limit)
    if baseline is None or memory is None:
        return baseline, {"used": False, "reason": "no_route" if baseline is None else "memory_disabled"}
    if max_extra_cost < 0:
        raise ValueError("Memory cost allowance must be nonnegative")
    budget = pathcost(baseline, tools) + max_extra_cost
    catalog = sorted((toolname(tool), requirements(tool), output(tool), cost(tool)) for tool in tools if toolname(tool) and toolname(tool) != blocked and output(tool))
    frontier: list[tuple[float, int, tuple[str, ...], frozenset[str]]] = [(0.0, 0, (), frozenset(state))]
    candidates: dict[tuple[str, ...], float] = {}
    expanded = 0
    truncated = False
    while frontier and expanded < 4096 and len(candidates) < 128:
        price, steps, actions, available = heapq.heappop(frontier)
        expanded += 1
        if price > budget + 1e-9:
            break
        if target.issubset(available):
            candidates[actions] = price
            continue
        if steps >= limit:
            continue
        for name, needs, produced, added in catalog:
            if name in actions or produced in available or not needs.issubset(available):
                continue
            if price + added <= budget + 1e-9:
                heapq.heappush(frontier, (price + added, steps + 1, (*actions, name), available | {produced}))
                if len(frontier) > 4096:
                    truncated = True
                    break
        if truncated:
            break
    candidates.setdefault(tuple(baseline), pathcost(baseline, tools))
    truncated = truncated or expanded >= 4096 or len(candidates) >= 128
    if truncated or len(candidates) < 2:
        return baseline, {"used": False, "reason": "search_limit" if truncated else "one_eligible_route", "eligible": len(candidates)}
    terminal = blocked if canonical(blocked).startswith("selectfinal") else ""
    if backoff == "family":
        scores = {actions: score_route(memory, (*actions, terminal) if terminal else actions) for actions in candidates}
        prior = scores[tuple(baseline)]
        audit = []
        for actions, price in sorted(candidates.items(), key=lambda item: (item[1], item[0])):
            score = scores[actions]
            status = (
                "insufficient_training_tasks" if score is None else
                "unknown_edges" if score.unknown_pairs else
                "no_exact_anchor" if score.exact_pairs == 0 else
                "eligible"
            )
            audit.append({
                "path": [*actions, terminal] if terminal else list(actions),
                "cost_excluding_terminal": round(price, 4),
                "extra_cost": round(price - candidates[tuple(baseline)], 4),
                "score": round(score.value, 4) if score is not None else None,
                "exact_pairs": score.exact_pairs if score is not None else 0,
                "family_pairs": score.family_pairs if score is not None else 0,
                "unknown_pairs": score.unknown_pairs if score is not None else 0,
                "status": status,
                "baseline": actions == tuple(baseline),
            })
        detail = {"eligible": len(candidates), "candidate_audit_version": 1, "candidates": audit}
        if prior is None:
            return baseline, {"used": False, "reason": "insufficient_training_tasks", **detail}
        supported = {
            actions: score for actions, score in scores.items()
            if score is not None and score.unknown_pairs == 0 and score.exact_pairs >= 1
        }
        if not supported:
            return baseline, {"used": False, "reason": "no_evidence_complete_alternative", "baseline_score": round(prior.value, 4), **detail}
        chosen = min(supported, key=lambda actions: (-supported[actions].value, candidates[actions], actions))
        improvement = supported[chosen].value - prior.value
        if chosen == tuple(baseline) or improvement < 0.12:
            return baseline, {"used": False, "reason": "insufficient_family_margin", "supported": len(supported), "baseline_score": round(prior.value, 4), "best_score": round(supported[chosen].value, 4), **detail}
        evidence = "family_transfer" if supported[chosen].family_pairs else "exact_alternative"
        return list(chosen), {"used": True, "reason": f"{evidence}_rerank", "evidence": evidence, "supported": len(supported), "baseline": baseline, "chosen": list(chosen), "baseline_score": round(prior.value, 4), "chosen_score": round(supported[chosen].value, 4), "exact_pairs": supported[chosen].exact_pairs, "family_pairs": supported[chosen].family_pairs, "extra_cost": candidates[chosen] - candidates[tuple(baseline)], **detail}
    signals = {actions: memory.signal(query, (*actions, terminal) if terminal else actions) for actions in candidates}
    if signals[tuple(baseline)] is None:
        return baseline, {"used": False, "reason": "cold_or_unaligned_baseline", "eligible": len(candidates)}
    aligned = {actions: signal for actions, signal in signals.items() if signal is not None}
    if len(aligned) < 2:
        return baseline, {"used": False, "reason": "one_memory_aligned_route", "eligible": len(candidates), "aligned": len(aligned)}
    chosen = min(aligned, key=lambda actions: (-aligned[actions].fused, candidates[actions], actions))
    return list(chosen), {
        "used": chosen != tuple(baseline),
        "reason": "memory_rerank" if chosen != tuple(baseline) else "cost_route_retained",
        "eligible": len(candidates),
        "aligned": len(aligned),
        "baseline": baseline,
        "chosen": list(chosen),
        "extra_cost": candidates[chosen] - candidates[tuple(baseline)],
    }
def changed(messages: Sequence[Mapping[str, Any]]) -> bool:
    if not messages or messages[-1].get("role") != "user":
        return False
    if not any(message.get("role") == "assistant" and callname(message) for message in messages[:-1]):
        return False
    return bool(CHANGE.search(str(messages[-1].get("content") or ""))) and not failed(messages[-1])
def pressure(messages: Sequence[Mapping[str, Any]]) -> tuple[int, str, set[str]]:
    tools = [message for message in messages if message.get("role") == "tool"]
    errors = 0
    for message in reversed(tools):
        if not failed(message):
            break
        errors += 1
    names = [name for message in messages if message.get("role") == "assistant" and (name := callname(message))]
    streak = 0
    if names:
        for name in reversed(names):
            if name != names[-1]:
                break
            streak += 1
    issue = max(errors, max(0, streak - 1)) if tools and failed(tools[-1]) else 0
    last = names[-1] if names else ""
    state = current(tools[-1]) if tools and failed(tools[-1]) else set()
    return issue, last, state
def precommit(messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, Any] | None]:
    names = [name for message in messages if message.get("role") == "assistant" and (name := callname(message))]
    source = names[-1] if names else ""
    selectors = [tool for tool in tools if canonical(toolname(tool)).startswith("selectfinal")]
    if not canonical(source).endswith("step2") or not selectors:
        return list(messages), list(tools), None
    note = "EvoCond GL precommit audit. Re-read the global user requirements and compare every surviving candidate attribute by attribute using only observed tool evidence. Do not infer missing evidence. If no candidate satisfies every explicit constraint, make one reachable refinement or search call. Otherwise call the final selector with its exact argument schema. Make at most one tool call."
    revised = [dict(message) for message in messages]
    revised.append({"role": "system", "content": note})
    event = {"scale": "gl", "pressure": 0, "action": "audit", "source": source, "visible": len(tools)}
    return revised, list(tools), event
def intervene(messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]], auditing: bool = True, mode: str = "full", planning: bool = True, memory: EvoltBridge | None = None, memory_extra_cost: float = 0.0, memory_backoff: str = "exact") -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, Any] | None]:
    if mode not in {"filter", "ll", "llgl", "full"}:
        raise ValueError("Recovery mode is invalid")
    issue, last, state = pressure(messages)
    if mode == "full" and changed(messages):
        selectors = [tool for tool in tools if canonical(toolname(tool)).startswith("selectfinal")]
        source = [*([message for message in messages if message.get("role") == "system"][:1]), messages[-1]]
        fresh = initial(source)
        update = str(messages[-1].get("content") or "")
        earlier_times = [match for message in messages[:-1] if message.get("role") == "user" for match in TIME_ID.findall(str(message.get("content") or ""))]
        stable_time = earlier_times[-1] if earlier_times and not TIME_ID.search(update) and not TIME_CHANGE.search(update) else ""
        if stable_time:
            fresh.add("timeinfo")
        path, memory_event = memory_route(fresh, requirements(selectors[0]), tools, memory, update, toolname(selectors[0]), max_extra_cost=memory_extra_cost, backoff=memory_backoff) if len(selectors) == 1 else (None, {"used": False, "reason": "ambiguous_selector"})
        plan = [*path, toolname(selectors[0])] if path is not None else []
        alternative_path = route(fresh, requirements(selectors[0]), tools, last) if len(selectors) == 1 else None
        alternative = [*alternative_path, toolname(selectors[0])] if alternative_path is not None else []
        if alternative == plan:
            alternative = []
        suffix = f" A type-feasible route for the updated request is {' -> '.join(plan)} at remaining tool cost {pathcost(plan, tools):g}; use it only if its outputs satisfy the new request." if plan else ""
        contrast = f" An alternative avoiding a repeat of {last} is {' -> '.join(alternative)} at remaining tool cost {pathcost(alternative, tools):g}." if alternative else ""
        continuity = f" The previous time input {stable_time} can be reused only because the update does not explicitly change the travel time; use the newest location and preference identifiers." if stable_time else ""
        note = "EvoCond GG: an explicit goal or preference update was observed. Re-read the latest user requirements and discard stale preference-derived choices. Compare remaining tool cost, not just step count; avoid repeating expensive tools without need." + continuity + suffix + contrast + " Make at most one tool call."
        revised = [*map(dict, messages), {"role": "system", "content": note}]
        return revised, list(tools), {"scale": "gg", "pressure": 0, "action": "replan", "cause": "explicit_change", "path": plan, "cost": pathcost(plan, tools) if plan else None, "alternative": alternative, "alternative_cost": pathcost(alternative, tools) if alternative else None, "stable_time": stable_time, "visible": len(tools), "memory": memory_event if memory is not None else None}
    if issue < 1 or not last:
        if mode == "full" and planning and not last:
            selectors = [tool for tool in tools if canonical(toolname(tool)).startswith("selectfinal")]
            state = initial(messages)
            if len(selectors) == 1 and state and all(COST.search(description(tool)) for tool in tools):
                selector = selectors[0]
                query = next((str(message.get("content") or "") for message in reversed(messages) if message.get("role") == "user"), "")
                path, memory_event = memory_route(state, requirements(selector), tools, memory, query, toolname(selector), max_extra_cost=memory_extra_cost, backoff=memory_backoff)
                if path is not None:
                    plan = [*path, toolname(selector)]
                    price = pathcost(plan, tools)
                    note = f"EvoCond prospective plan: the visible input/output types permit {' -> '.join(plan)} at estimated tool cost {price:g}. This is a type-feasible route, not proof of candidate validity. Prefer it only when it respects the user's constraints and observed evidence; otherwise adapt. Make at most one tool call."
                    revised = [*map(dict, messages), {"role": "system", "content": note}]
                    return revised, list(tools), {"scale": "prior", "pressure": 0, "action": "plan", "path": plan, "cost": price, "current": sorted(state), "visible": len(tools), "memory": memory_event if memory is not None else None}
        return precommit(messages, tools) if auditing else (list(messages), list(tools), None)
    latest = next((message for message in reversed(messages) if message.get("role") == "tool"), {})
    content = str(latest.get("content") or "").lower()
    reachable = [tool for tool in tools if toolname(tool) != last and (not state or requirements(tool).issubset(state))]
    if not reachable:
        reachable = [tool for tool in tools if toolname(tool) != last]
    if not reachable:
        reachable = list(tools)
    if mode == "filter":
        return list(messages), reachable, {"scale": "filter", "pressure": issue, "action": "filter", "failed": last, "current": sorted(state), "visible": len(reachable)}
    if mode == "full":
        if "missing required input types" in content:
            target = listed(latest, REQUIRED)
            path = route(state, target, tools, last, 4) if target else None
            if path and path[0] in {toolname(tool) for tool in reachable}:
                note = f"EvoCond GL: {last} lacks input types {', '.join(sorted(target - state))}. Visible type dependencies permit the upstream route {' -> '.join(path)} before retrying {last}. Use observed object IDs only; call one available upstream tool now."
                revised = [*map(dict, messages), {"role": "system", "content": note}]
                return revised, reachable, {"scale": "gl", "pressure": issue, "action": "bridge", "failed": last, "missing": sorted(target - state), "path": path, "current": sorted(state), "visible": len(reachable)}
        if issue <= 2 and any(token in content for token in ("cannot be parsed as json", "invalid id length", "invalid tool input", "invalid argument")):
            schema = next((sorted(requirements(tool)) for tool in tools if toolname(tool) == last), [])
            note = f"EvoCond LL: {last} failed because of input formatting or an invalid identifier. The required argument types are {', '.join(schema)}. Reuse this tool only with identifiers actually observed in the conversation; otherwise choose another feasible tool. Make at most one tool call."
            revised = [*map(dict, messages), {"role": "system", "content": note}]
            return revised, list(tools), {"scale": "ll", "pressure": issue, "action": "repair", "failed": last, "required": schema, "visible": len(tools)}
        return list(messages), reachable, {"scale": "filter", "pressure": issue, "action": "filter", "failed": last, "cause": "no_verified_bridge", "current": sorted(state), "visible": len(reachable)}
    scale = "ll" if mode == "ll" or issue == 1 else "gl"
    note = f"EvoCond {scale.upper()} intervention. Repair the failed action from observed inputs using a different reachable tool. Do not call {last}. Make at most one tool call."
    revised = [*map(dict, messages), {"role": "system", "content": note}]
    return revised, reachable, {"scale": scale, "pressure": issue, "action": "recover", "failed": last, "current": sorted(state), "visible": len(reachable)}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
