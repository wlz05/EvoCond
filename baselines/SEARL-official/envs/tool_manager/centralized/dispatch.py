# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Persistent virtual tools. Source is data; only the isolated backend runs it."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import threading

from control.dynamic import admit, empty_registry, invoke, validate, _check_spec, resolve_tool_version
from control.isolation import (ModelToolInputError, canonical_hash, function_schema,
                                    load_verified_backend, strict_json, validate_arguments, validate_timeout)
from control.evidence import resolve_arguments, reference_parameters

RESERVED = {"search", "execute_python_code", "create_and_execute_mcp", "answer",
            "search-search", "execute_python_code-execute_python_code"}
MAX_REGISTRY_BYTES = 32 * 1024 * 1024

class _ProbeRecorder:
    def __init__(self, backend):
        self.backend, self.last_result, self.probe_result = backend, None, None
        self.isolation = backend.isolation
        self.isolation_verified = backend.isolation_verified

    def run_function(self, *args):
        self.last_result = self.backend.run_function(*args)
        if self.probe_result is None:
            self.probe_result = self.last_result
        return self.last_result

class IsolatedToolDispatcher:
    def __init__(self, registry_path, backend, *, initial_registry=None):
        self.path = Path(registry_path).resolve()
        if self.path.suffix != ".json":
            raise ValueError("Virtual registry must be a JSON file, never Python source")
        self.backend = backend
        self._thread_lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            if self.path.exists():
                if initial_registry is not None:
                    raise FileExistsError("Refusing to overwrite an existing virtual registry")
                self._read()
            else:
                self._write(initial_registry if initial_registry is not None else empty_registry())

    @classmethod
    def from_environment(cls):
        required = ("SEARL_DYNAMIC_REGISTRY", "SEARL_ISOLATION_CONFIG", "SEARL_ISOLATION_RECEIPT")
        if any(not os.environ.get(key) for key in required):
            raise RuntimeError("Explicit dynamic registry and inspected sandbox configuration are required")
        backend = load_verified_backend(os.environ[required[1]], os.environ[required[2]])
        return cls(os.environ[required[0]], backend)

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            lock = self.path.with_suffix(self.path.suffix + ".lock")
            try:
                fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError as error:
                raise RuntimeError("Virtual registry is locked; inspect the owning process before recovery") from error
            try:
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                yield
            finally:
                lock.unlink()

    def _read(self):
        if self.path.stat().st_size > MAX_REGISTRY_BYTES:
            raise ValueError("Virtual registry exceeds size limit")
        registry = strict_json(self.path.read_text(encoding="utf-8"))
        validate(registry)
        if RESERVED.intersection(registry["aliases"]):
            raise ValueError("Virtual registry shadows a built-in tool")
        return registry

    def _write(self, registry):
        validate(registry)
        if RESERVED.intersection(registry["aliases"]):
            raise ValueError("Virtual tool cannot replace a built-in tool")
        encoded = json.dumps(registry, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
        if len(encoded) > MAX_REGISTRY_BYTES:
            raise ValueError("Virtual registry exceeds size limit")
        fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def reload(self):
        with self._thread_lock:
            return deepcopy(self._read())

    @property
    def registry(self):
        return self.reload()

    def replace_registry(self, registry, expected_sha256):
        """CAS persistence for trusted train-only feedback, not actor tool input."""
        with self._locked():
            if canonical_hash(self._read()) != expected_sha256:
                raise RuntimeError("Registry changed while feedback was computed; reload and retry")
            self._write(deepcopy(registry))

    def transform_registry(self, mutator):
        """Atomically apply a trusted host feedback function, never actor code."""
        if not callable(mutator):
            raise ValueError("Registry transformation requires a trusted callable")
        with self._locked():
            updated = mutator(deepcopy(self._read()))
            self._write(updated)
            return deepcopy(updated)

    def fork(self, registry_path):
        """Create a private episode snapshot; evaluation never changes train state."""
        return type(self)(registry_path, self.backend, initial_registry=self.registry)

    def has_tool(self, name):
        return isinstance(name, str) and name in self.registry["aliases"]

    def schemas(self):
        registry = self.registry
        result = []
        for name, version in sorted(registry["aliases"].items()):
            spec = registry["semantic"][version]
            result.append({"type": "function", "function": {
                "name": name, "description": spec["description"],
                "parameters": reference_parameters(function_schema(spec["code"], spec["name"]))},
                "tool_version": version})
        return result

    def create(self, spec, inputs, *, task_id, timeout=15.0, encoder=None,
               merge_with=None, equivalence_inputs=None, ledger=(), capability=False):
        inputs, bindings = resolve_arguments(inputs, ledger)
        try:
            _check_spec(spec)
            if spec["name"] in RESERVED:
                raise ValueError("Generated tool cannot shadow a built-in tool")
            validate_arguments(spec["code"], spec["name"], inputs)
            timeout = validate_timeout(timeout, getattr(self.backend, 'config', {}).get('max_run_seconds', 60))
        except (ValueError, TypeError, SyntaxError, RecursionError) as error:
            raise ModelToolInputError("Invalid generated function, arguments, metadata, or timeout") from error
        if not isinstance(task_id, str) or not task_id.strip():
            raise RuntimeError("Independent task identity is required")
        recorder = _ProbeRecorder(self.backend)
        with self._locked():
            registry = self._read()
            if capability:
                from control.identity import admit as capability_admit
                updated, decision = capability_admit(registry, spec, recorder, probe_inputs=inputs,
                    task_id=task_id, timeout=timeout, encoder=encoder, ledger=ledger,
                    equivalence_inputs=equivalence_inputs)
            else:
                updated, decision = admit(registry, spec, recorder, probe_inputs=inputs,
                    task_id=task_id, timeout=timeout, encoder=encoder,
                    merge_with=merge_with, equivalence_inputs=equivalence_inputs)
            if decision["decision"] == "already_admitted":
                execution = invoke(updated, spec["name"], inputs, self.backend,
                                   timeout=timeout, expected_version=decision["version"])
            else:
                execution = (recorder.last_result if capability else recorder.probe_result) or {"run_success": False, "error": "Probe result missing"}
            success = execution.get("run_success") is True and decision["decision"] in {
                "already_admitted", "new_versioned_node", "merge_verified"}
            if success:
                updated.setdefault("dispatch_receipts", []).append({
                    "task_id": task_id, "tool_version": decision["version"],
                    "source_sha256": sha256(spec["code"].encode()).hexdigest(),
                    "isolation_receipt_sha256": execution.get("isolation_receipt_sha256"),
                    "decision": decision["decision"]})
                updated["dispatch_receipts"] = updated["dispatch_receipts"][-10000:]
            self._write(updated)
            return {"creation_success": success, "run_success": execution.get("run_success") is True,
                    "registered": success, "tool_version": decision["version"],
                    "registry_sha256": canonical_hash(updated), "decision": decision["decision"],
                    "admission_receipt": decision,
                    "execution_result": execution.get("result"),
                    "stdout": execution.get("stdout", ""), "stderr": execution.get("stderr", ""),
                    "error": execution.get("error", ""),
                    "resolved_arguments": inputs, "input_bindings": bindings,
                    "isolation_receipt_sha256": execution.get("isolation_receipt_sha256")}

    def invoke(self, name, inputs, *, expected_version=None, timeout=15.0, ledger=()):
        # The immutable snapshot prevents concurrent alias changes during this call.
        registry = self.registry  # Corrupt state must never be treated as model input.
        inputs, bindings = resolve_arguments(inputs, ledger)
        try:
            version = resolve_tool_version(registry, name)
            if expected_version is not None and expected_version != version:
                raise ValueError("Tool alias changed; retrieve its current version")
            spec = registry['semantic'][version]
            validate_arguments(spec['code'], spec['name'], inputs)
            timeout = validate_timeout(timeout, getattr(self.backend, 'config', {}).get('max_run_seconds', 60))
        except (ValueError, TypeError, SyntaxError, RecursionError) as error:
            raise ModelToolInputError("Invalid tool name, version, arguments, or timeout; retrieve again") from error
        result = invoke(registry, name, inputs, self.backend, timeout=timeout,
                        expected_version=expected_version)
        return {**result, "resolved_arguments": inputs, "input_bindings": bindings}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
