# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Evidence-bound checks, not a natural-language correctness oracle.'
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import tempfile
from typing import Any
FORMAT = "evocond_task_contract_v1"
MAX_TEXT = 1_000_000
MAX_PROPOSALS = 128
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
PREFIX = re.compile(r"(?<![\w.])(?:(?P<tool>[A-Za-z_][A-Za-z0-9_-]*)\.)?"
                    r"(?P<field>[A-Za-z_]\w*)\s*(?P<op>!=|=(?!=))\s*")
def _finite(value: Any, depth: int = 0) -> None:
    if depth > 32:
        raise ValueError("Payload nesting exceeds 32 levels")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite values are forbidden")
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Object keys must be strings")
        for item in value.values():
            _finite(item, depth + 1)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _finite(item, depth + 1)
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError("Only JSON values are allowed")
def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result
def strict_json(text: str, *, max_text: int = MAX_TEXT) -> Any:
    if type(max_text) is not int or max_text < 1 or not isinstance(text, str) or len(text) > max_text:
        raise ValueError("Invalid JSON text size")
    value = json.loads(text, object_pairs_hook=_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(
                           ValueError("Non-finite JSON constant")))
    _finite(value)
    return value
def _hash(value: Any) -> str:
    _finite(value)
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")).hexdigest()
def _equal(a: Any, b: Any) -> bool:
    # JSON boolean is not the numeric value 1 or 0.
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[key], b[key]) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b
def current_frame(messages: list[dict], tools: list[dict]) -> dict:
    'Derive identities from visible messages and the current public schema.'
    if not isinstance(messages, list) or not isinstance(tools, list) or len(tools) > 2048:
        raise ValueError("Expected bounded message and catalog lists")
    latest = [(i, m.get("content")) for i, m in enumerate(messages)
              if isinstance(m, dict) and m.get("role") == "user"]
    index, content = latest[-1] if latest else (-1, "")
    # Multimodal / hidden-message representations need an explicit adapter.
    if not isinstance(content, str) or len(content) > MAX_TEXT:
        raise ValueError("Latest user content must be bounded visible text")
    catalog, identities = {}, {}
    for entry in tools:
        if not isinstance(entry, dict):
            raise ValueError("Tool catalog entries must be objects")
        function = entry.get("function", entry)
        if not isinstance(function, dict):
            raise ValueError("Malformed function schema")
        name = function.get("name")
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise ValueError("Unsupported tool name")
        if name in catalog:
            raise ValueError("Duplicate tool name")
        schema = function.get("parameters", function.get("inputSchema"))
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError("Tool parameters must declare an object schema")
        if not isinstance(schema.get("properties", {}), dict):
            raise ValueError("Tool properties must be an object")
        _finite(schema)
        catalog[name] = json.loads(json.dumps(schema))
        version = function.get("tool_version", entry.get("tool_version"))
        description = function.get("description", "")
        if (version is not None and not isinstance(version, str)) or not isinstance(description, str):
            raise ValueError("Tool version and description must be text")
        identities[name] = {"schema": catalog[name], "tool_version": version, "description": description}
    return {"user_index": index, "user_text": content,
            "user_version": _hash({"index": index, "text": content}),
            "catalog": catalog, "catalog_sha256": _hash(identities)}
def _schema_check(value: Any, schema: dict, path: str = "$", depth: int = 0) -> tuple[list, list]:
    'Check a conservative JSON Schema subset; unsupported constraints abstain.'
    if depth > 24 or not isinstance(schema, dict):
        return [], [f"{path}: unsupported schema"]
    errors, unknown = [], []
    supported = {"type", "properties", "required", "additionalProperties", "items",
                 "enum", "const", "minimum", "maximum", "exclusiveMinimum",
                 "exclusiveMaximum", "minLength", "maxLength", "minItems", "maxItems",
                 "description", "title", "default", "examples", "$schema", "$id",
                 "x-task-contract-aliases"}
    unknown.extend(f"{path}: unsupported keyword {key}" for key in schema
                   if key not in supported)
    kind = schema.get("type")
    kinds = {"null": value is None, "boolean": isinstance(value, bool),
             "integer": isinstance(value, int) and not isinstance(value, bool),
             "number": isinstance(value, (int, float)) and not isinstance(value, bool),
             "string": isinstance(value, str), "array": isinstance(value, list),
             "object": isinstance(value, dict)}
    if kind is not None and (not isinstance(kind, str) or kind not in kinds):
        unknown.append(f"{path}: unsupported type declaration")
    elif kind is not None and not kinds[kind]:
        return [f"{path}: expected {kind}"], unknown
    for key in ("enum", "const"):
        if key in schema:
            allowed = schema[key] if key == "enum" else [schema[key]]
            if not isinstance(allowed, list):
                unknown.append(f"{path}: malformed {key}")
            elif not any(_equal(value, item) for item in allowed):
                errors.append(f"{path}: violates {key}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            return errors, unknown + [f"{path}: malformed object schema"]
        for field in required:
            if not isinstance(field, str):
                unknown.append(f"{path}: malformed required field")
            elif field not in value:
                errors.append(f"{path}.{field}: required")
        for field, item in value.items():
            rule = properties.get(field)
            if rule is None:
                extra = schema.get("additionalProperties", True)
                if extra is False:
                    errors.append(f"{path}.{field}: extra property")
                elif isinstance(extra, dict):
                    rule = extra
                elif extra is not True:
                    unknown.append(f"{path}: malformed additionalProperties")
            if rule is not None:
                bad, uncertain = _schema_check(item, rule, f"{path}.{field}", depth + 1)
                errors += bad
                unknown += uncertain
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            bad, uncertain = _schema_check(item, schema["items"], f"{path}[{i}]", depth + 1)
            errors += bad
            unknown += uncertain
    if isinstance(value, (str, list)):
        for key, is_min in (("minLength" if isinstance(value, str) else "minItems", True),
                            ("maxLength" if isinstance(value, str) else "maxItems", False)):
            if key in schema:
                limit = schema[key]
                if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
                    unknown.append(f"{path}: malformed {key}")
                elif (len(value) < limit if is_min else len(value) > limit):
                    errors.append(f"{path}: violates {key}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        for key, test in (("minimum", lambda a, b: a < b), ("maximum", lambda a, b: a > b),
                          ("exclusiveMinimum", lambda a, b: a <= b),
                          ("exclusiveMaximum", lambda a, b: a >= b)):
            if key in schema:
                limit = schema[key]
                if not isinstance(limit, (int, float)) or isinstance(limit, bool):
                    unknown.append(f"{path}: malformed {key}")
                elif test(value, limit):
                    errors.append(f"{path}: violates {key}")
    return errors, unknown
def _bindings(frame: dict) -> list[dict]:
    'Extract only an exact assignment token, never execute its contents.'
    text = frame["user_text"]
    decoder = json.JSONDecoder(object_pairs_hook=_pairs)
    result = []
    for match in PREFIX.finditer(text):
        try:
            value, length = decoder.raw_decode(text[match.end():])
            _finite(value)
        except (ValueError, TypeError, RecursionError):
            continue
        end = match.end() + length
        # Do not turn the prefix of a value, e.g. 1e999 or 100kg, into a claim.
        if end < len(text) and (text[end].isalnum() or text[end] == "_"
                               or (text[end] == "." and end + 1 < len(text) and text[end + 1].isdigit())):
            continue
        name, field = match.group("tool"), match.group("field")
        candidates = [key for key, schema in frame["catalog"].items()
                      if field in schema.get("properties", {})]
        scope = "explicit_tool" if name else "unique_current_schema_field"
        context_start = max(text.rfind(mark, 0, match.start()) for mark in ("\n", ";", '\u3002', '\uff1b', "!", "?")) + 1
        prefix = text[context_start:match.start()]
        if (re.search("\\b(?:not|never|avoid|don't|example|hypothetical|previously|formerly|was|if|used\\s+to)\\b|\\u4e0d\\u8981|\\u4e0d\\u60f3|\\u4e0d\\u662f|\\u4f8b\\u5982|\\u4ee5\\u524d|\\u4e4b\\u524d|\\u5047\\u8bbe", prefix, re.I)
                or any(prefix.count(mark) % 2 for mark in ('"', '`'))):
            scope = "unsupported_context"
        if name is None and len(candidates) == 1:
            name = candidates[0]
        result.append({"tool": name, "field": field, "op": "ne" if match.group("op") == "!=" else "eq",
                       "value": value, "start": match.start(), "end": end,
                       "quote": text[match.start():end], "scope_evidence": scope})
        if len(result) > MAX_PROPOSALS:
            raise ValueError("Too many explicit requirements")
    return result
def _alias_bindings(frame: dict, trusted_alias_tools: tuple[str, ...]) -> tuple[list, list]:
    'Resolve only operator-trusted schema aliases, never policy-made aliases.'
    text = frame["user_text"]
    matches, abstentions = [], []
    for tool in trusted_alias_tools:
        schema = frame["catalog"].get(tool)
        if schema is None:
            raise ValueError("Trusted alias tool is absent from the current catalog")
        for field, rule in schema.get("properties", {}).items():
            aliases = rule.get("x-task-contract-aliases", []) if isinstance(rule, dict) else []
            if not isinstance(aliases, list) or len(aliases) > MAX_PROPOSALS:
                raise ValueError("Malformed trusted alias table")
            for alias in aliases:
                if (not isinstance(alias, dict) or not {"phrase", "value"} <= set(alias)
                        or set(alias) - {"phrase", "value", "scope_phrase"}
                        or not isinstance(alias["phrase"], str) or not alias["phrase"].strip()):
                    raise ValueError("Malformed trusted alias")
                phrase = alias["phrase"]
                scope = alias.get("scope_phrase")
                if scope is not None and (not isinstance(scope, str) or not scope.strip()):
                    raise ValueError("Malformed alias scope phrase")
                for match in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.I):
                    start, end = match.span()
                    left = max(text.rfind(mark, 0, start) for mark in (".", "!", "?", "\n", '\u3002', '\uff1b', ";")) + 1
                    ends = [text.find(mark, end) for mark in (".", "!", "?", "\n", '\u3002', '\uff1b', ";")]
                    right = min((pos for pos in ends if pos >= 0), default=len(text))
                    clause = text[left:right]
                    if scope and not re.search(r"(?<!\w)" + re.escape(scope) + r"(?!\w)", clause, re.I):
                        continue
                    negation = re.search("\\b(?:not|no|never|without|avoid|except|don't|isn't|was|previously|formerly|if|unless|maybe|example|hypothetical|used\\s+to)\\b|\\u4e0d\\u8981|\\u4e0d\\u60f3|\\u4e0d\\u662f|\\u6392\\u9664|\\u4ee5\\u524d|\\u4e4b\\u524d|\\u5047\\u8bbe|\\u4f8b\\u5982|\\u4e5f\\u8bb8", clause, re.I)
                    quoted = any(clause[:start-left].count(mark) % 2 for mark in ('"', '`'))
                    if negation or quoted:
                        abstentions.append({"reason": "negated_historical_or_quoted_alias", "start": start})
                        continue
                    matches.append({"tool": tool, "field": field, "op": "eq", "value": alias["value"],
                                    "start": start, "end": end, "quote": text[start:end],
                                    "scope_evidence": "trusted_schema_alias"})
                    if len(matches) > MAX_PROPOSALS:
                        raise ValueError("Too many alias requirements")
    # One span matching multiple tool/field bindings cannot establish scope.
    ambiguous = {(a["start"], a["end"]) for a in matches for b in matches
                 if (a["start"], a["end"]) == (b["start"], b["end"])
                 and (a["tool"], a["field"]) != (b["tool"], b["field"])}
    for start, _ in sorted(ambiguous):
        abstentions.append({"reason": "ambiguous_alias_scope", "start": start})
    return [item for item in matches if (item["start"], item["end"]) not in ambiguous], abstentions
def extraction_request(messages: list[dict], tools: list[dict], *, trusted_alias_tools: tuple[str, ...] = ()) -> dict:
    'Optional policy extractor input; candidate output is always reverified.'
    frame = current_frame(messages, tools)
    return {"instruction": "Return JSON only: {user_version, constraints}. Each constraint has "
            "tool, field, op (eq/ne), value, start, end, quote. Extract only literal "
            "Tool.field = JSON_VALUE or != statements from the CURRENT user text. "
            "field alone is allowed only if unique in the current catalog. Offsets are "
            "Python Unicode character offsets. Do not infer aliases, negation, intent, "
            "or use observations/history/rewards. For trusted_alias_tools only, exact "
            "phrases in x-task-contract-aliases may map to the schema-declared value; "
            "never add an alias. Return no constraints when uncertain.",
            "user_version": frame["user_version"], "user_text": frame["user_text"],
            "tools": frame["catalog"], "trusted_alias_tools": list(trusted_alias_tools)}
def parse_proposals(text: str) -> dict:
    value = strict_json(text)
    if not isinstance(value, dict) or set(value) != {"user_version", "constraints"}:
        raise ValueError("Expected user_version and constraints only")
    if not isinstance(value["user_version"], str) or not isinstance(value["constraints"], list):
        raise ValueError("Malformed proposal envelope")
    if len(value["constraints"]) > MAX_PROPOSALS:
        raise ValueError("Too many constraint proposals")
    return value
@dataclass(frozen=True)
class Contract:
    'Serialized payload is rechecked for version, scope and evidence on use.'
    payload: str
def compile_contract(messages: list[dict], tools: list[dict], proposals: dict | str | None = None,
                     *, trusted_alias_tools: tuple[str, ...] = ()) -> Contract:
    frame = current_frame(messages, tools)
    candidates = _bindings(frame)
    if not isinstance(trusted_alias_tools, (list, tuple)) or any(not isinstance(item, str) for item in trusted_alias_tools):
        raise ValueError("trusted_alias_tools must be an explicit tool-name sequence")
    trusted_alias_tools = tuple(sorted(set(trusted_alias_tools)))
    alias_candidates, rejected = _alias_bindings(frame, trusted_alias_tools)
    candidates += alias_candidates
    if proposals is not None:
        envelope = parse_proposals(proposals if isinstance(proposals, str) else json.dumps(proposals))
        if envelope["user_version"] != frame["user_version"]:
            candidates = []
            rejected.append({"reason": "stale_user_version"})
        else:
            matched = []
            keys = {"tool", "field", "op", "value", "start", "end", "quote"}
            for item in envelope["constraints"]:
                if not isinstance(item, dict) or set(item) != keys:
                    rejected.append({"reason": "malformed_proposal"})
                    continue
                if any(isinstance(item[key], bool) or not isinstance(item[key], int) for key in ("start", "end")):
                    rejected.append({"reason": "invalid_span"})
                    continue
                found = next((binding for binding in candidates
                              if all(_equal(item[key], binding[key]) for key in keys)), None)
                if found is None:
                    rejected.append({"reason": "unsupported_or_inexact_evidence"})
                elif found not in matched:
                    matched.append(found)
            candidates = matched
    verified = []
    for item in candidates:
        name, field = item["tool"], item["field"]
        if item["scope_evidence"] == "unsupported_context":
            rejected.append({"reason": "negated_historical_or_quoted_binding", "start": item["start"]})
            continue
        schema = frame["catalog"].get(name)
        if schema is None or field not in schema.get("properties", {}):
            rejected.append({"reason": "unknown_or_ambiguous_tool_scope", "start": item["start"]})
            continue
        bad, uncertain = _schema_check(item["value"], schema["properties"][field])
        if bad or uncertain:
            rejected.append({"reason": "value_not_verified_against_schema", "start": item["start"]})
            continue
        verified.append(item)
    # Contradictory instructions are not silently resolved by mention order.
    ambiguous = set()
    for first in verified:
        for second in verified:
            if (first["tool"], first["field"]) != (second["tool"], second["field"]):
                continue
            if ((first["op"] == second["op"] == "eq" and not _equal(first["value"], second["value"]))
                    or (first["op"] != second["op"] and _equal(first["value"], second["value"]))):
                ambiguous.add((first["tool"], first["field"]))
    for name, field in sorted(ambiguous):
        rejected.append({"reason": "contradictory_current_requirements", "tool": name, "field": field})
    verified = [item for item in verified if (item["tool"], item["field"]) not in ambiguous]
    payload = {"format": FORMAT, "user_version": frame["user_version"],
               "catalog_sha256": frame["catalog_sha256"], "constraints": verified,
               "abstentions": rejected, "trusted_alias_tools": list(trusted_alias_tools),
               "scope": "latest_user_literals_and_operator_trusted_schema_aliases"}
    return Contract(json.dumps(payload, ensure_ascii=False, sort_keys=True))
def contract_report(contract: Contract) -> dict:
    if not isinstance(contract, Contract):
        raise ValueError("Expected a compiled Contract")
    value = strict_json(contract.payload)
    required = {"format", "user_version", "catalog_sha256", "constraints", "abstentions", "trusted_alias_tools", "scope"}
    if not isinstance(value, dict) or set(value) != required or value["format"] != FORMAT:
        raise ValueError("Malformed contract envelope")
    if not all(isinstance(value[key], str) and re.fullmatch(r"[a-f0-9]{64}", value[key])
               for key in ("user_version", "catalog_sha256")):
        raise ValueError("Malformed contract identity")
    if (not isinstance(value["constraints"], list) or len(value["constraints"]) > MAX_PROPOSALS
            or not isinstance(value["abstentions"], list)
            or not isinstance(value["trusted_alias_tools"], list)
            or not all(isinstance(name, str) for name in value["trusted_alias_tools"])):
        raise ValueError("Malformed contract collections")
    keys = {"tool", "field", "op", "value", "start", "end", "quote", "scope_evidence"}
    for item in value["constraints"]:
        if (not isinstance(item, dict) or set(item) != keys
                or not all(isinstance(item[key], str) for key in ("tool", "field", "quote", "scope_evidence"))
                or item["op"] not in ("eq", "ne")
                or any(type(item[key]) is not int for key in ("start", "end"))
                or not 0 <= item["start"] < item["end"] <= MAX_TEXT):
            raise ValueError("Malformed contract constraint")
    return value
def check_call(messages: list[dict], tools: list[dict], tool_name: str,
               arguments: dict | str, *, contract: Contract | None = None,
               block_unknown: bool = False, trusted_alias_tools: tuple[str, ...] = ()) -> dict:
    'Check one call without reading rewards, answers, or running the tool.'
    if not isinstance(block_unknown, bool):
        raise ValueError("block_unknown must be boolean")
    frame = current_frame(messages, tools)
    active = contract_report(compile_contract(messages, tools, trusted_alias_tools=trusted_alias_tools))
    provenance_error = None
    if contract is not None:
        try:
            provided = contract_report(contract)
            if (provided.get("user_version") != frame["user_version"]
                    or provided.get("catalog_sha256") != frame["catalog_sha256"]
                    or provided.get("trusted_alias_tools", []) != sorted(set(trusted_alias_tools))):
                provenance_error = "stale_contract"
            else:
                # Supplied contracts cannot omit a contradictory claim or forge one.
                envelope = {"user_version": provided["user_version"], "constraints": [
                    {key: item[key] for key in ("tool", "field", "op", "value", "start", "end", "quote")}
                    for item in provided["constraints"]]}
                verified = contract_report(compile_contract(messages, tools, envelope, trusted_alias_tools=trusted_alias_tools))
                if verified["abstentions"] or verified["constraints"] != provided["constraints"]:
                    provenance_error = "unverified_contract"
        except (ValueError, TypeError, KeyError):
            provenance_error = "unverified_contract"
        # Current deterministic evidence is authoritative; an extractor cannot erase another current.
    conflicts, uncertain = [], []
    try:
        values = strict_json(arguments) if isinstance(arguments, str) else arguments
        _finite(values)
        if not isinstance(values, dict):
            raise ValueError("Arguments must be a JSON object")
    except (ValueError, TypeError, RecursionError):
        return {"decision": "conflict", "blocked": True, "reason": "malformed_arguments",
                "conflicts": ["invalid_argument_object"], "checked_constraints": 0}
    schema = frame["catalog"].get(tool_name)
    if schema is None:
        conflicts.append("tool_not_in_current_catalog")
    else:
        conflicts, uncertain = _schema_check(values, schema)
    checked = 0
    if provenance_error:
        uncertain.append(provenance_error)
    for item in ([] if provenance_error else active["constraints"]):
        if item["tool"] != tool_name:
            continue
        checked += 1
        field = item["field"]
        if field not in values:
            conflicts.append(f"{field}: explicit requirement missing")
        elif (_equal(values[field], item["value"]) if item["op"] == "ne"
              else not _equal(values[field], item["value"])):
            conflicts.append(f"{field}: conflicts with current {item['op']} requirement")
    if active["abstentions"]:
        uncertain.append("some_current_evidence_unverified")
    decision = "conflict" if conflicts else "unknown" if uncertain or not checked else "accept"
    return {"decision": decision, "blocked": bool(conflicts) or (decision == "unknown" and block_unknown),
            "reason": "explicit_conflict" if conflicts else provenance_error or ("insufficient_evidence" if decision == "unknown" else "explicit_checks_passed"),
            "conflicts": conflicts, "uncertainties": uncertain, "checked_constraints": checked,
            "user_version": frame["user_version"], "catalog_sha256": frame["catalog_sha256"],
            "semantic_correctness_certified": False}
def memory_applicability(memory_contract: Contract, messages: list[dict], tools: list[dict],
                         *, trusted_alias_tools: tuple[str, ...] = ()) -> dict:
    'Compare recorded explicit assumptions; never enforce them as new goals.'
    prior = contract_report(memory_contract)
    current = contract_report(compile_contract(messages, tools, trusted_alias_tools=trusted_alias_tools))
    if prior.get("format") != FORMAT or prior.get("catalog_sha256") != current["catalog_sha256"]:
        return {"decision": "unknown", "reason": "schema_or_format_changed", "reuse_authorized": False}
    old = prior.get("constraints", [])
    now = current["constraints"]
    if not old or not now or current["abstentions"] or prior.get("abstentions"):
        return {"decision": "unknown", "reason": "insufficient_explicit_assumptions", "reuse_authorized": False}
    compared, mismatches = 0, []
    for earlier in old:
        candidates = [item for item in now if (item["tool"], item["field"]) == (earlier["tool"], earlier["field"])]
        for item in candidates:
            compared += 1
            conflict = (earlier["op"] == item["op"] == "eq" and not _equal(earlier["value"], item["value"]))
            conflict |= earlier["op"] != item["op"] and _equal(earlier["value"], item["value"])
            if conflict:
                mismatches.append({"tool": item["tool"], "field": item["field"]})
    covered = all(any(item["tool"] == previous["tool"] and item["field"] == previous["field"]
                      and item["op"] == previous["op"] and _equal(item["value"], previous["value"])
                      for item in now) for previous in old)
    decision = "inapplicable" if mismatches else "compatible_explicit_assumptions" if covered else "unknown"
    return {"decision": decision, "reason": "explicit_assumptions_only", "compared": compared,
            "mismatches": mismatches, "reuse_authorized": False,
            "warning": "Compatibility does not prove utility or cover unrecorded preconditions"}

def relative(root, name):
    if not isinstance(name, str) or not name or '\\' in name or PureWindowsPath(name).drive:
        raise ValueError('Use a nonempty portable relative path')
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Relative artifact paths cannot escape their declared root')
    for part in path.parts:
        if (part.endswith(('.', ' ')) or any(ord(char) < 32 or char in '<>:"|?*' for char in part)
                or re.fullmatch(r'(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?', part, flags=re.IGNORECASE)):
            raise ValueError('Artifact paths must use portable file and directory names')
    base = Path(root).resolve()
    result = base/path
    if not result.resolve().is_relative_to(base):
        raise ValueError('Resolved artifact path leaves its declared root')
    return result

def publish(path, text):
    if not isinstance(text, (str, bytes)):
        raise ValueError('Published content must be text or bytes')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.evocond-', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(text if isinstance(text, bytes) else (text+'\n').encode('utf-8'))
            stream.flush()
            os.fsync(stream.fileno())
        # Linking publishes the complete artifact without replacing a prior result.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
