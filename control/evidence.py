# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Explicit, episode-local dataflow receipts, never value-equality guesswork.'
from __future__ import annotations
from copy import deepcopy
import re
from control.isolation import ModelToolInputError, canonical_hash, finite_inputs
FORMAT = "searl_explicit_dataflow_v1"
REFERENCE = "__searl_result_ref__"
IDENTITY = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
VERSION = re.compile(r"[0-9a-f]{64}\Z")
REFERENCE_NOTE = ('To pass an earlier result into a tool argument, use '
                  '{"__searl_result_ref__":{"call_id":"the returned call_id","path":"/field"}}. '
                  'The path is a JSON Pointer into execution_result; use "" for the whole result. '
                  'Only successful calls from this episode may be referenced.')
def _json(value):
    finite_inputs({"value": value})
    return deepcopy(value)
def _escape(value):
    return str(value).replace("~", "~0").replace("/", "~1")
def _pointer(value, path):
    if not isinstance(path, str) or len(path) > 1024 or (path and not path.startswith("/")):
        raise ModelToolInputError("Invalid result reference JSON Pointer")
    if not path:
        return value
    current = value
    for token in path[1:].split("/"):
        if re.search(r"~(?![01])", token):
            raise ModelToolInputError("Invalid result reference escape")
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and re.fullmatch(r"0|[1-9][0-9]*", token):
            if len(token) > 9 or int(token) >= len(current):
                raise ModelToolInputError("Result reference index does not exist")
            current = current[int(token)]
        else:
            raise ModelToolInputError("Result reference path does not exist")
    return current
def _record_hash(record):
    return canonical_hash({key: value for key, value in record.items() if key != "sha256"})
def validate_record(record):
    if (not isinstance(record, dict) or record.get("format") != FORMAT
            or not isinstance(record.get("call_id"), str) or IDENTITY.fullmatch(record["call_id"]) is None
            or not isinstance(record.get("name"), str) or not record["name"]
            or type(record.get("success")) is not bool or not isinstance(record.get("arguments"), dict)
            or type(record.get("result_available")) is not bool
            or not isinstance(record.get("bindings"), list)):
        raise ValueError("Invalid trusted invocation record")
    version = record.get("tool_version")
    if version is not None and (not isinstance(version, str) or VERSION.fullmatch(version) is None):
        raise ValueError("Invalid immutable tool version in invocation record")
    _json(record["arguments"])
    _json(record["result"])
    if (record.get("result_sha256") != canonical_hash(record["result"])
            or record.get("arguments_sha256") != canonical_hash(record["arguments"])
            or record.get("sha256") != _record_hash(record)):
        raise ValueError("Invocation ledger integrity mismatch")
def call_record(call_id, name, tool_version, success, arguments, result, bindings=(), *, result_available=True):
    'Trusted post-execution record of the exact resolved arguments and result.'
    if type(result_available) is not bool or (not result_available and result is not None):
        raise ValueError("An unavailable reference result must be explicitly represented as None")
    record = {"format": FORMAT, "call_id": call_id, "name": name,
              "tool_version": tool_version, "success": success, "arguments": _json(arguments),
              "result_available": result_available,
              "result": _json(result), "bindings": deepcopy(list(bindings)),
              "result_sha256": canonical_hash(result), "arguments_sha256": canonical_hash(arguments)}
    record["sha256"] = _record_hash(record)
    validate_record(record)
    return record
def resolve_arguments(arguments, ledger=()):
    "Resolve explicit references using only this episode's trusted prior calls."
    try:
        finite_inputs(arguments)
    except (ValueError, TypeError, RecursionError) as error:
        raise ModelToolInputError("Tool arguments must be a bounded finite JSON object") from error
    if not isinstance(ledger, (list, tuple)):
        raise ValueError("Trusted invocation ledger must be ordered")
    known = {}
    for record in ledger:
        validate_record(record)  # Corrupt executor state is not a model error.
        if record["call_id"] in known:
            raise ValueError("Duplicate call identity in trusted invocation ledger")
        known[record["call_id"]] = record
    bindings = []
    def walk(value, target_path):
        if isinstance(value, dict) and REFERENCE in value:
            ref = value[REFERENCE]
            if (set(value) != {REFERENCE} or not isinstance(ref, dict)
                    or set(ref) != {"call_id", "path"} or not isinstance(ref["call_id"], str)):
                raise ModelToolInputError("Malformed explicit result reference")
            source = known.get(ref["call_id"])
            if source is None or not source["success"] or not source["result_available"]:
                raise ModelToolInputError("Result reference must identify a successful call in this episode")
            resolved = deepcopy(_pointer(source["result"], ref["path"]))
            bindings.append({"source_call_id": source["call_id"], "source_tool_version": source["tool_version"],
                             "source_result_sha256": source["result_sha256"], "source_record_sha256": source["sha256"],
                             "source_path": ref["path"], "target_path": target_path,
                             "value_sha256": canonical_hash(resolved)})
            return resolved
        if isinstance(value, dict):
            return {key: walk(item, target_path+"/"+_escape(key)) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item, target_path+"/"+str(i)) for i, item in enumerate(value)]
        return value
    resolved = walk(arguments, "")
    try:
        finite_inputs(resolved)
    except (ValueError, TypeError, RecursionError) as error:
        raise ModelToolInputError("Resolved arguments exceed the input size limit") from error
    return resolved, bindings
def transition_feedback(source, target):
    'Classify observed explicit dataflow, not causal task-level contribution.'
    validate_record(source)
    validate_record(target)
    if (source["call_id"] == target["call_id"] or not source["success"] or not source["result_available"]
            or source["tool_version"] is None or target["tool_version"] is None):
        return "unknown"
    for binding in target["bindings"]:
        if not isinstance(binding, dict):
            raise ValueError("Malformed trusted binding receipt")
        if binding.get("source_call_id") != source["call_id"]:
            continue
        if (binding.get("source_tool_version") != source["tool_version"]
                or binding.get("source_result_sha256") != source["result_sha256"]
                or binding.get("source_record_sha256") != source["sha256"]):
            raise ValueError("Binding source no longer matches the invocation ledger")
        try:
            original = _pointer(source["result"], binding["source_path"])
            actual = _pointer(target["arguments"], binding["target_path"])
        except (ModelToolInputError, KeyError) as error:
            raise ValueError("Binding points outside its trusted source/target") from error
        if canonical_hash(original) != binding.get("value_sha256") or canonical_hash(actual) != binding["value_sha256"]:
            raise ValueError("Binding no longer matches the actually executed arguments")
        return "verified" if target["success"] else "failed"
    return "unknown"
def reference_parameters(parameters):
    'Expose the shared reference syntax without weakening resolved validation.'
    reference = {"type": "object", "properties": {REFERENCE: {"type": "object", "properties": {
        "call_id": {"type": "string"}, "path": {"type": "string"}},
        "required": ["call_id", "path"], "additionalProperties": False}},
        "required": [REFERENCE], "additionalProperties": False}
    result = deepcopy(parameters)
    result["properties"] = {name: {"anyOf": [schema, deepcopy(reference)]}
                            for name, schema in result.get("properties", {}).items()}
    return result

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
