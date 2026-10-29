# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""CPU-only transport mocks; no generated source executes in these tests."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_root))
_path = Path(__file__).resolve().parents[1] / "envs" / "tool_manager" / "centralized" / "dispatch.py"
_spec = spec_from_file_location("isolated_dispatch", _path)
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)
IsolatedToolDispatcher = _module.IsolatedToolDispatcher
from control.isolation import canonical_hash

class MockSandbox:
    isolation = "container"
    isolation_verified = True

    def __init__(self):
        self.calls = []
        self.fail = False

    def run_function(self, source, name, inputs, timeout):
        self.calls.append((source, name, inputs, timeout))
        return {"run_success": not self.fail, "result": inputs.get("x"),
                "error": "mock failure" if self.fail else "",
                "isolation_receipt_sha256": "CPU_MOCK_NOT_DEPLOYMENT"}

def tool(name="identity"):
    return {"name": name, "description": "Return a value", "arguments": "x: int",
            "returns": "integer", "side_effects": "none", "code": f"def {name}(x: int): return x"}

class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "registry.json"
        self.backend = MockSandbox()
        self.dispatch = IsolatedToolDispatcher(self.path, self.backend)

    def test_admit_call_and_reload_survive_restart_without_source_import(self):
        candidate = tool()
        candidate["code"] = "raise RuntimeError('MUST_NOT_RUN_ON_HOST')\n" + candidate["code"]
        result = self.dispatch.create(candidate, {"x": 7}, task_id="q1")
        self.assertTrue(result["creation_success"])
        self.assertTrue(result["registered"])
        self.assertEqual(result["execution_result"], 7)
        reloaded = IsolatedToolDispatcher(self.path, self.backend)
        self.assertTrue(reloaded.has_tool("identity"))
        self.assertEqual(reloaded.invoke("identity", {"x": 9})["result"], 9)
        self.assertEqual(len(self.backend.calls), 2)
        self.assertEqual(list(Path(self.temporary.name).glob("*.py")), [])
        self.assertEqual(reloaded.schemas()[0]["function"]["parameters"]["required"], ["x"])

    def test_failed_probe_is_not_callable(self):
        self.backend.fail = True
        result = self.dispatch.create(tool(), {"x": 7}, task_id="q1")
        self.assertFalse(result["creation_success"])
        self.assertFalse(result["registered"])
        self.assertFalse(self.dispatch.has_tool("identity"))
        with self.assertRaises(ValueError):
            self.dispatch.invoke("identity", {"x": 7})

    def test_invalid_inputs_do_not_reach_backend(self):
        from control.isolation import ModelToolInputError
        with self.assertRaises(ModelToolInputError):
            self.dispatch.create(tool(), {"x": "wrong"}, task_id="q1")
        self.assertFalse(self.dispatch.has_tool("identity"))
        self.assertEqual(self.backend.calls, [])

    def test_model_input_errors_are_distinct_from_corrupt_registry_or_transport(self):
        from control.isolation import ModelToolInputError, SandboxInfrastructureError
        self.dispatch.create(tool(), {"x": 7}, task_id="q1")
        with self.assertRaises(ModelToolInputError):
            self.dispatch.invoke("identity", {"x": "bad"})
        with patch.object(self.backend, "run_function", side_effect=SandboxInfrastructureError("unavailable")), \
                self.assertRaises(SandboxInfrastructureError):
            self.dispatch.invoke("identity", {"x": 7})
        self.path.write_text('{}', encoding='utf-8')
        with self.assertRaises(ValueError) as caught:
            self.dispatch.invoke("identity", {"x": 7})
        self.assertNotIsInstance(caught.exception, ModelToolInputError)

    def test_explicit_reference_is_resolved_before_actual_signature_validation(self):
        from control.evidence import call_record, REFERENCE
        first = self.dispatch.create(tool(), {"x": 7}, task_id="q1")
        record = call_record("call1", "identity", first["tool_version"], True,
                             {"x": 7}, {"value": 7})
        ref = {"x": {REFERENCE: {"call_id": "call1", "path": "/value"}}}
        result = self.dispatch.invoke("identity", ref, ledger=[record])
        self.assertEqual(result["resolved_arguments"], {"x": 7})
        self.assertEqual(self.backend.calls[-1][2], {"x": 7})
        self.assertEqual(result["input_bindings"][0]["source_call_id"], "call1")
        with self.assertRaises(ValueError):
            self.dispatch.invoke("identity", ref, ledger=[])

    def test_unverified_backend_refused(self):
        self.backend.isolation_verified = False
        with self.assertRaises(RuntimeError):
            self.dispatch.create(tool(), {"x": 7}, task_id="q1")
        self.assertEqual(self.backend.calls, [])

    def test_builtin_shadowing_is_refused(self):
        for name in ("search", "execute_python_code", "create_and_execute_mcp"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.dispatch.create(tool(name), {"x": 7}, task_id="q1")
        self.assertEqual(self.backend.calls, [])

    def test_version_pin_and_alias_update(self):
        first = self.dispatch.create(tool(), {"x": 7}, task_id="q1")
        changed = tool()
        changed["description"] = "A new version"
        second = self.dispatch.create(changed, {"x": 7}, task_id="q2")
        self.assertNotEqual(first["tool_version"], second["tool_version"])
        with self.assertRaises(ValueError):
            self.dispatch.invoke("identity", {"x": 3}, expected_version=first["tool_version"])
        self.assertEqual(len(self.dispatch.registry["semantic"]), 2)

    def test_repeated_create_runs_exactly_one_probe(self):
        self.dispatch.create(tool(), {"x": 7}, task_id="q1")
        result = self.dispatch.create(tool(), {"x": 8}, task_id="q2")
        self.assertTrue(result["registered"])
        self.assertEqual(result["execution_result"], 8)
        self.assertEqual(len(self.backend.calls), 2)

    def test_evaluation_fork_does_not_mutate_train(self):
        self.dispatch.create(tool(), {"x": 7}, task_id="train1")
        before = self.path.read_bytes()
        fork_path = self.path.parent / "validation" / "episode1.json"
        fork = self.dispatch.fork(fork_path)
        fork.create(tool("new_tool"), {"x": 8}, task_id="validation1")
        self.assertFalse(self.dispatch.has_tool("new_tool"))
        self.assertTrue(fork.has_tool("new_tool"))
        self.assertEqual(self.path.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            self.dispatch.fork(fork_path)

    def test_cas_rejects_stale_feedback(self):
        initial = self.dispatch.registry
        self.dispatch.create(tool(), {"x": 7}, task_id="q1")
        with self.assertRaises(RuntimeError):
            self.dispatch.replace_registry(initial, canonical_hash(initial))
        current = self.dispatch.registry
        new = deepcopy(current)
        new["feedback_note"] = "train only"
        self.dispatch.replace_registry(new, canonical_hash(current))
        self.assertEqual(self.dispatch.registry["feedback_note"], "train only")

    def test_lock_is_never_removed_blindly(self):
        lock = self.path.with_suffix(".json.lock")
        lock.write_text("1234", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.dispatch.create(tool(), {"x": 7}, task_id="q1")
        self.assertTrue(lock.exists())
        self.assertEqual(self.backend.calls, [])

    def test_atomic_feedback_transform_and_rollback(self):
        self.dispatch.create(tool(), {"x": 7}, task_id="q1")
        updated = self.dispatch.transform_registry(lambda old: {**old, "feedback_note": "observed"})
        self.assertEqual(updated, self.dispatch.registry)
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.dispatch.transform_registry(lambda old: {"corrupt": True})
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(self.path.with_suffix(".json.lock").exists())

    def test_tampered_persisted_source_fails_on_reload(self):
        self.dispatch.create(tool(), {"x": 7}, task_id="q1")
        registry = self.dispatch.registry
        version = registry["aliases"]["identity"]
        registry["semantic"][version]["code"] = "def identity(x): return 999"
        self.path.write_text(json.dumps(registry), encoding="utf-8")
        with self.assertRaises(ValueError):
            IsolatedToolDispatcher(self.path, self.backend)

    def test_missing_environment_never_uses_uninspected_defaults(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            IsolatedToolDispatcher.from_environment()

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
