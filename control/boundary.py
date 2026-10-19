# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Explicit historical-context checks, not an oracle for transferable tools.'
from __future__ import annotations
from copy import deepcopy
import json
from control import dynamic, contracts
FORMAT = "evocond_reuse_boundary_v1"
def validate_mode(value="off"):
    if value not in ("off", "explicit"):
        raise ValueError("Reuse boundary must be off or explicit")
    return value
def visible_text(value, *, mode="explicit"):
    validate_mode(mode)
    if mode == "explicit" and (not isinstance(value, str) or not value.strip()
                                or len(value) > contracts.MAX_TEXT):
        raise ValueError("Explicit reuse boundary needs bounded current trusted user text")
    return value
def endpoint_tools(registry, source_version, target_version):
    tools = []
    for version in sorted({source_version, target_version}):
        item = registry["semantic"][version]
        schema = item["arguments"]
        if not isinstance(schema, dict) or schema.get("type") != "object":
            return None
        tools.append({"name": item["name"], "description": item["description"],
                      "parameters": deepcopy(schema), "tool_version": version})
    if len({tool["name"] for tool in tools}) != len(tools):
        return None
    return tools
def _contract(text, tools, aliases):
    return contracts.compile_contract([{"role": "user", "content": text}], tools,
                                           trusted_alias_tools=tuple(aliases))
def capture(registry, source_version, target_version, text, *, trusted_alias_tools=()):
    visible_text(text)
    tools = endpoint_tools(registry, source_version, target_version)
    if tools is None:
        return None
    aliases = sorted(set(trusted_alias_tools) & {tool["name"] for tool in tools})
    report = contracts.contract_report(_contract(text, tools, aliases))
    value = {"format": FORMAT, "source_text": text, "tools": tools,
             "trusted_alias_tools": aliases, "source_contract": report}
    value["sha256"] = dynamic.digest(value)
    return value
def validate(value, *, registry=None, source_version=None, target_version=None):
    if value is None:
        return
    fields = {"format", "source_text", "tools", "trusted_alias_tools", "source_contract", "sha256"}
    if not isinstance(value, dict) or set(value) != fields or value.get("format") != FORMAT:
        raise ValueError("Malformed historical reuse boundary")
    if value["sha256"] != dynamic.digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("Historical reuse boundary hash differs")
    visible_text(value["source_text"])
    aliases = value["trusted_alias_tools"]
    if (not isinstance(aliases, list) or any(not isinstance(name, str) for name in aliases)
            or aliases != sorted(set(aliases))):
        raise ValueError("Malformed boundary alias provenance")
    if registry is not None and value["tools"] != endpoint_tools(registry, source_version, target_version):
        raise ValueError("Reuse boundary describes different endpoint versions")
    report = contracts.contract_report(_contract(value["source_text"], value["tools"], aliases))
    if value["source_contract"] != report:
        raise ValueError("Reuse boundary does not follow from its trusted source text")
def compare(value, current_user_text, *, checked=False):
    visible_text(current_user_text)
    if value is None:
        result = {"decision": "unknown", "reason": "missing_endpoint_context_footprint"}
    else:
        if not checked:
            validate(value)
        contract = contracts.Contract(json.dumps(value["source_contract"], ensure_ascii=False))
        result = contracts.memory_applicability(contract,
            [{"role": "user", "content": current_user_text}], value["tools"],
            trusted_alias_tools=tuple(value["trusted_alias_tools"]))
    state = {"compatible_explicit_assumptions": "matched", "inapplicable": "conflict"}.get(result["decision"], "unknown")
    return {**result, "state": state, "historical_signal_allowed": state == "matched",
            "capability_blocked": False, "generality_certified": False}
def receipt(views, mode, current_user_text):
    validate_mode(mode)
    totals = {key: sum(row.get("reuse_boundary", {}).get(key, 0) for row in views.values())
              for key in ("matched", "unknown", "conflict")}
    return {"mode": mode, "current_user_hash": dynamic.digest(current_user_text) if mode == "explicit" else None,
            "edge_task_observations": totals, "count_kind": "edge_question_pairs_not_unique_questions",
            "capabilities_removed": 0, "unknown_labeled_as_failure": False,
            "claim": "explicit historical-context compatibility only; not tool preconditions or causal benefit"}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
