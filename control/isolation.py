# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Explicit deployment attestation and bounded SandboxFusion HTTP requests.'
from __future__ import annotations
import ast
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import http.client
import json
import math
from pathlib import Path
import re
import secrets
import time
from urllib.parse import urlsplit
CONFIG_FORMAT = "searl_isolation_config_v1"
RECEIPT_FORMAT = "searl_isolation_receipt_v1"
CHECKS = ("process_isolation", "network_egress_blocked", "no_host_mounts",
          "resource_limits", "enforced_timeout", "output_limit")
NAME = re.compile(r"^[A-Za-z_][A-Za-z_0-9]{0,127}$")
class ModelToolInputError(ValueError):
    'Invalid model-supplied request, recoverable as a failed tool observation.'
class SandboxInfrastructureError(RuntimeError):
    'Untrusted/failed executor transport; abort rather than train on false failure.'
def validate_timeout(value, maximum=60):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
        raise ModelToolInputError("Execution timeout exceeds the inspected limit")
    return float(value)
def prepare_python_source(code):
    'Parse only; the resulting program is still executed exclusively remotely.'
    if not isinstance(code, str) or not 0 < len(code.encode()) <= 32768:
        raise ModelToolInputError("Python source must be nonempty and at most 32 KiB")
    try:
        tree = ast.parse(code)
    except (ValueError, SyntaxError, RecursionError) as error:
        raise ModelToolInputError("Python source has invalid syntax") from error
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        expression = tree.body[-1].value
        if not (isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name)
                and expression.func.id == "print"):
            tree.body[-1] = ast.Expr(value=ast.Call(func=ast.Name(id="print", ctx=ast.Load()),
                                                 args=[expression], keywords=[]))
            ast.fix_missing_locations(tree)
            return ast.unparse(tree)
    return code
def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()
def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result
def strict_json(text):
    def reject(value):
        raise ValueError("Non-finite JSON number")
    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("Non-finite JSON number")
        return parsed
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=reject, parse_float=number)
def _time(value):
    if not isinstance(value, str):
        raise ValueError("Inspection timestamps must be ISO-8601 strings")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Inspection timestamps must include a timezone")
    return stamp
def validate_deployment(config, receipt, *, now=None):
    if not isinstance(config, dict) or config.get("format") != CONFIG_FORMAT:
        raise ValueError("Missing explicit isolated deployment configuration")
    if not isinstance(receipt, dict) or receipt.get("format") != RECEIPT_FORMAT:
        raise ValueError("Missing independently inspected deployment receipt")
    endpoint = config.get("endpoint", "")
    if not isinstance(endpoint, str):
        raise ValueError("Invalid SandboxFusion endpoint")
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment):
        raise ValueError("Endpoint must contain only an HTTP(S) origin")
    if config.get("isolation") not in {"container", "microvm"}:
        raise ValueError("Host execution is forbidden")
    if not isinstance(config.get("deployment_id"), str) or not config["deployment_id"].strip():
        raise ValueError("Deployment identity is required")
    for key, lower, upper in (("max_run_seconds", 0, 60),
                              ("memory_limit_mb", 15, 16384),
                              ("max_output_bytes", 1023, 1048576)):
        value = config.get(key)
        if type(value) is not int or not lower < value <= upper:
            raise ValueError("Invalid isolated resource limit: " + key)
    if (receipt.get("config_sha256") != canonical_hash(config)
            or receipt.get("endpoint") != endpoint
            or receipt.get("deployment_id") != config["deployment_id"]):
        raise ValueError("Inspection receipt does not match the deployment configuration")
    if not isinstance(receipt.get("inspector"), str) or not receipt["inspector"].strip():
        raise ValueError("Independent inspector identity is missing")
    inspected, expiry = _time(receipt.get("inspected_at")), _time(receipt.get("valid_until"))
    current = now or datetime.now(timezone.utc)
    if not inspected <= current < expiry:
        raise ValueError("Inspection receipt is expired or not yet valid")
    checks = receipt.get("checks")
    if not isinstance(checks, dict) or any(checks.get(key) is not True for key in CHECKS):
        raise ValueError("All isolation inspection checks must explicitly pass")
    return canonical_hash(receipt)
def finite_inputs(inputs):
    if not isinstance(inputs, dict) or any(not isinstance(key, str) for key in inputs):
        raise ValueError("Tool arguments must be a JSON object with string keys")
    def visit(value, depth=0):
        if depth > 24:
            raise ValueError("Tool argument nesting exceeds limit")
        if value is None or type(value) in {bool, str, int}:
            return
        if type(value) is float and math.isfinite(value):
            return
        if type(value) is list:
            for child in value:
                visit(child, depth + 1)
            return
        if type(value) is dict and all(type(key) is str for key in value):
            for child in value.values():
                visit(child, depth + 1)
            return
        raise ValueError("Arguments must contain only finite JSON values")
    visit(inputs)
    encoded = json.dumps(inputs, allow_nan=False)
    if len(encoded.encode()) > 65536:
        raise ValueError("Tool arguments exceed 64 KiB")
    return encoded
def function_schema(source, name):
    if not isinstance(name, str) or NAME.fullmatch(name) is None:
        raise ValueError("Invalid generated-tool name")
    if not isinstance(source, str) or not 0 < len(source.encode()) <= 32768:
        raise ValueError("Generated source exceeds 32 KiB or is empty")
    try:
        definitions = [node for node in ast.parse(source).body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    except (SyntaxError, RecursionError) as error:
        raise ValueError("Invalid generated Python source") from error
    if len(definitions) != 1:
        raise ValueError("Source must define its declared function exactly once")
    args = definitions[0].args
    if args.posonlyargs or args.vararg:
        raise ValueError("Generated tools require keyword-callable parameters")
    positional_required = len(args.args) - len(args.defaults)
    required = [arg.arg for arg in args.args[:positional_required]]
    required.extend(arg.arg for arg, default in zip(args.kwonlyargs, args.kw_defaults) if default is None)
    aliases = {"str": "string", "int": "integer", "float": "number", "bool": "boolean",
               "list": "array", "dict": "object"}
    properties = {}
    for arg in [*args.args, *args.kwonlyargs]:
        annotation = arg.annotation
        kind = aliases.get(annotation.id) if isinstance(annotation, ast.Name) else None
        properties[arg.arg] = {"type": kind} if kind else {}
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": args.kwarg is not None}
def validate_arguments(source, name, inputs):
    finite_inputs(inputs)
    schema = function_schema(source, name)
    if set(schema["required"]) - inputs.keys():
        raise ValueError("Missing required generated-tool argument")
    if not schema["additionalProperties"] and inputs.keys() - schema["properties"].keys():
        raise ValueError("Unknown generated-tool argument")
    checks = {"string": lambda x: type(x) is str, "integer": lambda x: type(x) is int,
              "number": lambda x: type(x) in {int, float}, "boolean": lambda x: type(x) is bool,
              "array": lambda x: type(x) is list, "object": lambda x: type(x) is dict}
    for key, value in inputs.items():
        kind = schema["properties"].get(key, {}).get("type")
        if kind and not checks[kind](value):
            raise ValueError("Wrong generated-tool argument type: " + key)
    return schema
class VerifiedSandboxFusion:
    'Bound requests to a separately inspected isolated service; no retries.'
    def __init__(self, config, receipt):
        self.config, self.receipt = deepcopy(config), deepcopy(receipt)
        self.receipt_sha256 = validate_deployment(config, receipt)
        self.isolation = config["isolation"]
        self.isolation_verified = True
    def _post(self, payload, timeout):
        parsed = urlsplit(self.config["endpoint"])
        connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        deadline = time.monotonic() + timeout
        connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
        cap = self.config["max_output_bytes"] * 2 + 16384
        try:
            connection.request("POST", "/run_code", json.dumps(payload, allow_nan=False).encode(),
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError("SandboxFusion HTTP status " + str(response.status))
            data = bytearray()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("SandboxFusion response deadline exceeded")
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                chunk = response.read1(min(16384, cap + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > cap:
                    raise ValueError("SandboxFusion HTTP output exceeds limit")
            return strict_json(data.decode("utf-8"))
        finally:
            connection.close()
    def run_source(self, source, timeout=15.0):
        validate_deployment(self.config, self.receipt)
        if not isinstance(source, str) or not 0 < len(source.encode()) <= 131072:
            raise ModelToolInputError("Invalid sandbox source size")
        timeout = validate_timeout(timeout, self.config["max_run_seconds"])
        try:
            payload = {"code": source, "language": "python", "run_timeout": timeout,
                       "compile_timeout": min(timeout, 10), "memory_limit_MB": self.config["memory_limit_mb"],
                       "files": {}, "fetch_files": []}
            data = self._post(payload, float(timeout) + min(timeout, 10) + 5)
            if not isinstance(data, dict) or not isinstance(data.get("run_result"), dict):
                raise ValueError("Malformed SandboxFusion response")
            result = data["run_result"]
            stdout, stderr = result.get("stdout", ""), result.get("stderr", "")
            if not isinstance(stdout, str) or not isinstance(stderr, str):
                raise ValueError("Malformed sandbox output")
            if len(stdout.encode()) + len(stderr.encode()) > self.config["max_output_bytes"]:
                raise ValueError("SandboxFusion output exceeds inspected limit")
            success = (data.get("status") == "Success" and result.get("status") == "Finished"
                       and type(result.get("return_code")) is int and result["return_code"] == 0)
            return {"run_success": success, "status": str(result.get("status", "unknown")),
                    "stdout": stdout, "stderr": stderr,
                    "error": "" if success else (stderr or "Isolated execution failed")[:2000],
                    "isolation_receipt_sha256": self.receipt_sha256}
        except Exception as error:
            # Never expose transport details or mislabel executor failure as model error.
            raise SandboxInfrastructureError("Sandbox transport or response validation failed") from None
    def run_function(self, source, name, inputs, timeout=15.0):
        validate_arguments(source, name, inputs)
        marker = "__SEARL_RESULT_" + secrets.token_hex(16) + "__"
        script = ("import asyncio, json, sys\n"
                  "_scope = {'__name__': '__generated_tool__'}\n"
                  "exec(compile(" + repr(source) + ", '<isolated-tool>', 'exec'), _scope)\n"
                  "_value = _scope[" + repr(name) + "](**json.loads(" + repr(finite_inputs(inputs)) + "))\n"
                  "if asyncio.iscoroutine(_value): _value = asyncio.run(_value)\n"
                  "print(" + repr(marker) + " + json.dumps(_value, ensure_ascii=False, allow_nan=False))\n")
        result = self.run_source(script, timeout)
        if not result["run_success"]:
            return result
        lines = [line[len(marker):] for line in result["stdout"].splitlines() if line.startswith(marker)]
        try:
            if len(lines) != 1:
                raise ValueError("Missing or ambiguous generated result")
            value = strict_json(lines[0])
        except ValueError as error:
            return {**result, "run_success": False, "error": str(error)}
        return {**result, "result": value,
                "stdout": "\n".join(line for line in result["stdout"].splitlines() if not line.startswith(marker))}
def load_verified_backend(config_path, receipt_path):
    config = strict_json(Path(config_path).read_text(encoding="utf-8"))
    receipt = strict_json(Path(receipt_path).read_text(encoding="utf-8"))
    return VerifiedSandboxFusion(config, receipt)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
