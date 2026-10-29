# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Read-only, evidence-checked review of preference arguments before execution.'
from __future__ import annotations
import json
import re
from typing import Any
from control.guard import toolname
PREFERENCE = re.compile(r"(?:Category|Tier|Style|FeaturePackage)$")
def proposal(call: dict, tools: list[dict]) -> dict | None:
    function = call.get("function", call)
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    raw = function.get("arguments")
    try:
        arguments = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return None
    if not isinstance(arguments, dict):
        return None
    tool = next((item for item in tools if toolname(item) == name), None)
    if tool is None:
        return None
    schema = tool.get("function", {}).get("parameters", {})
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    if not isinstance(properties, dict):
        properties = {}
    fields = {}
    for key, value in arguments.items():
        if not PREFERENCE.search(key) or not isinstance(value, str):
            continue
        spec = properties.get(key, {})
        fields[key] = {"proposed": value, "schema": spec if isinstance(spec, dict) else {}}
    return {"tool": name, "fields": fields} if fields else None
def latest_user(messages: list[dict]) -> str:
    return next((str(m.get("content") or "") for m in reversed(messages)
                 if m.get("role") == "user"), "")
def review_prompt(user: str, item: dict) -> list[dict]:
    instruction = (
        "You audit preference values before a travel-planning tool call. "
        "Use ONLY the latest user request and visible field schemas below. "
        "Compare each proposed Category, Tier, Style, and FeaturePackage value "
        "against the user's actual priority, including negations and paraphrases. "
        "Do not infer one field's value merely from another field; weigh the "
        "user's explicit priorities and negations independently for each field. "
        "Reject only when a specific proposed value contradicts clear user evidence. "
        "For each rejected field, quote an exact contiguous phrase from the latest "
        "user request that supports your objection. If evidence is ambiguous, "
        "return uncertain, not reject. Do not invent facts, use past preferences, "
        "or infer hidden answers. Respond ONLY as JSON: "
        '{"decision":"accept|reject|uncertain","mismatches":'
        '[{"field":"field name","evidence":"exact user phrase","reason":"brief"}]}. '
        "An empty mismatches list is required unless decision is reject."
    )
    payload = {"latest_user_request": user, "proposed_tool": item["tool"],
               "preference_fields": item["fields"]}
    return [{"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
def interpret(text: str, user: str, item: dict) -> dict[str, Any]:
    'Reject only with a schema-matching field and a verbatim user witness.'
    try:
        start = text.index("{")
        raw, _ = json.JSONDecoder().raw_decode(text[start:])
    except (ValueError, TypeError):
        return {"decision": "uncertain", "mismatches": [], "reason": "unparsed_review"}
    if not isinstance(raw, dict) or raw.get("decision") not in {"accept", "reject", "uncertain"}:
        return {"decision": "uncertain", "mismatches": [], "reason": "invalid_review"}
    if raw["decision"] != "reject":
        return {"decision": raw["decision"], "mismatches": []}
    mismatches = []
    for mismatch in raw.get("mismatches", []):
        if not isinstance(mismatch, dict):
            continue
        field = mismatch.get("field")
        evidence = mismatch.get("evidence")
        if field not in item["fields"] or not isinstance(evidence, str):
            continue
        evidence = evidence.strip()
        if len(evidence) < 6 or evidence.casefold() not in user.casefold():
            continue
        mismatches.append({"field": field, "proposed": item["fields"][field]["proposed"],
                           "evidence": evidence, "reason": str(mismatch.get("reason") or "")[:160]})
    if not mismatches:
        return {"decision": "uncertain", "mismatches": [], "reason": "unsupported_rejection"}
    return {"decision": "reject", "mismatches": mismatches}

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
