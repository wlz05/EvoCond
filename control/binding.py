# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Rebind only evidence-backed current-task fields; never infer intent.'
from copy import deepcopy
from control.contracts import (compile_contract, contract_report, check_call,
                                  strict_json, _equal, _finite)
STRATEGIES = ("reject", "rebind")
CREDIT_RULES = ("native_only", "composite")
FORMAT = "evocond_current_binding_v1"
def validate_strategy(strategy, *, enabled=True):
    if strategy not in STRATEGIES:
        raise ValueError("binding_strategy must be reject or rebind")
    if strategy == "rebind" and not enabled:
        raise ValueError("Rebinding requires value_contracts=true")
def validate_credit(rule):
    if rule not in CREDIT_RULES:
        raise ValueError('binding_credit must be native_only or composite')
def prepare(messages, tools, name, arguments, *, strategy="reject", trusted_alias_tools=()):
    'Return a new argument object and auditable before/after checks.'
    validate_strategy(strategy)
    values = strict_json(arguments) if isinstance(arguments, str) else deepcopy(arguments)
    _finite(values)
    if not isinstance(values, dict):
        raise ValueError("Arguments must be a JSON object")
    proposed = deepcopy(values)
    active = contract_report(compile_contract(messages, tools, trusted_alias_tools=trusted_alias_tools))
    before = check_call(messages, tools, name, values, trusted_alias_tools=trusted_alias_tools)
    target = [row for row in active["constraints"] if row["tool"] == name]
    ambiguous = [row for row in active["abstentions"]
                 if row.get("reason") == "contradictory_current_requirements" and row.get("tool") == name]
    rewrites = []
    if strategy == "rebind" and not ambiguous:
        for row in target:
            field = row["field"]
            if row["op"] != "eq" or (field in values and _equal(values[field], row["value"])):
                continue
            rewrites.append({"field": field, "was_present": field in values,
                             "proposed": deepcopy(values.get(field)), "bound": deepcopy(row["value"]),
                             "evidence": {key: row[key] for key in ("start", "end", "quote", "scope_evidence")}})
            values[field] = deepcopy(row["value"])
    after = check_call(messages, tools, name, values, trusted_alias_tools=trusted_alias_tools)
    if strategy == "rebind" and ambiguous:
        after.update(decision="conflict", blocked=True, reason="ambiguous_current_binding",
                     conflicts=["contradictory_current_requirements"])
    constrained = {row["field"] for row in target}
    audit = {"format": FORMAT, "strategy": strategy, "tool": name,
             "proposed_arguments": proposed, "effective_arguments": deepcopy(values),
             "tool_version": next((tool.get('tool_version') for tool in tools if tool.get('name') == name), None),
             "user_version": active["user_version"], "catalog_sha256": active["catalog_sha256"],
             "before": before, "after": after, "rewrites": rewrites, "changed": bool(rewrites),
             "covered_fields": sorted(constrained), "unverified_fields": sorted(set(values)-constrained),
             "abstentions": active["abstentions"], "semantic_correctness_certified": False,
             "policy_credit_rule": "mask_entire_corrected_assistant_turn"}
    return values, audit
def apply_rewrites(raw_arguments, audit):
    'Preserve explicit result references on fields that were not corrected.'
    values = deepcopy(raw_arguments)
    for row in audit["rewrites"]:
        values[row["field"]] = deepcopy(row["bound"])
    return values
def response_loss_mask(responses, corrected_turns):
    'Plan=0, observation=1, first action=2; zero only corrected actions.'
    if (not isinstance(corrected_turns, list) or len(set(corrected_turns)) != len(corrected_turns)
            or any(type(i) is not int or i <= 0 or i % 2 or i >= len(responses) for i in corrected_turns)):
        raise ValueError("Corrected turns must identify actual even-indexed action segments")
    suppressed = set(corrected_turns)
    return [0 if i in suppressed else (i+1) % 2
            for i, segment in enumerate(responses) for _ in segment]

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
