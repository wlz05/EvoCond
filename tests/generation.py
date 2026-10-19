# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import unittest
from unittest.mock import patch
import sys
import types
from copy import deepcopy
from control.dynamic import SandboxFusionBackend, admit, empty_registry, invoke, update_edge, validate
class FakeSandbox:
    isolation = "container"
    isolation_verified = True
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []
    def run_function(self, source, name, inputs, timeout):
        self.calls.append((source, name, inputs, timeout))
        return {"run_success": not self.fail, "result": inputs.get("value")}
class FakeEncoder:
    def encode(self, text):
        return [1.0, 0.0]
def tool(name, code=None):
    return {"name": name, "description": "Return a supplied value",
            "arguments": {"value": "integer"}, "returns": "integer", "side_effects": "none",
            "code": code or f"def {name}(value):\n    return value\n"}
class DynamicToolTests(unittest.TestCase):
    def test_sandbox_fusion_adapter_never_runs_unverified(self):
        backend = SandboxFusionBackend("http://sandbox:8080", isolation="container")
        with self.assertRaises(RuntimeError):
            backend.run_function(tool("identity")["code"], "identity", {"value": 3}, 5)
    def test_sandbox_fusion_response_is_checked(self):
        backend = SandboxFusionBackend("http://sandbox:8080", isolation="container",
                                       isolation_verified=True)
        with self.assertRaises(RuntimeError):
            backend.run_function(tool("identity")["code"], "identity", {"value": 3}, 5)
    def test_backend_isolation_is_mandatory(self):
        backend = FakeSandbox()
        backend.isolation_verified = False
        with self.assertRaises(RuntimeError):
            admit(empty_registry(), tool("identity"), backend, probe_inputs={"value": 3}, task_id="q1")
        self.assertEqual(backend.calls, [])
    def test_failure_quarantines_and_never_exposes_tool(self):
        backend = FakeSandbox(fail=True)
        registry, event = admit(empty_registry(), tool("identity"), backend,
                                probe_inputs={"value": 3}, task_id="q1")
        self.assertEqual(event["decision"], "quarantine")
        self.assertNotIn("identity", registry["aliases"])
        with self.assertRaises(ValueError):
            invoke(registry, "identity", {"value": 3}, backend)
    def test_successful_admission_and_invocation(self):
        backend = FakeSandbox()
        registry, event = admit(empty_registry(), tool("identity"), backend,
                                probe_inputs={"value": 3}, task_id="q1")
        self.assertEqual(event["decision"], "new_versioned_node")
        self.assertEqual(invoke(registry, "identity", {"value": 4}, backend)["result"], 4)
        validate(registry)
        registry["semantic"][event["version"]]["code"] = "tampered"
        with self.assertRaises(ValueError):
            validate(registry)
    def test_admitted_identity_covers_description_and_both_contract_stores(self):
        registry, event = admit(empty_registry(), tool("identity"), FakeSandbox(),
                                probe_inputs={"value": 3}, task_id="q1")
        version = event["version"]
        for key, value in (("description", "A changed purpose"), ("returns", "string"),
                           ("side_effects", "writes files"), ("arguments", {"value": "string"})):
            with self.subTest(key=key):
                modified = deepcopy(registry)
                modified["semantic"][version][key] = value
                with self.assertRaises(ValueError):
                    validate(modified)
        modified = deepcopy(registry)
        modified["structure"]["nodes"][version]["arguments"]["value"] = "string"
        with self.assertRaises(ValueError):
            validate(modified)
    def test_admission_does_not_retain_mutable_caller_contract(self):
        spec = tool("identity")
        registry, event = admit(empty_registry(), spec, FakeSandbox(),
                                probe_inputs={"value": 3}, task_id="q1")
        spec["arguments"]["value"] = "changed after admission"
        self.assertEqual(registry["semantic"][event["version"]]["arguments"], {"value": "integer"})
        validate(registry)
    def test_invocation_receipt_pins_executed_version_and_rejects_alias_race(self):
        backend = FakeSandbox()
        registry, old = admit(empty_registry(), tool("identity"), backend,
                               probe_inputs={"value": 1}, task_id="old")
        receipt = invoke(registry, "identity", {"value": 1}, backend, expected_version=old["version"])
        self.assertEqual(receipt["tool_version"], old["version"])
        self.assertEqual(receipt["invoked_name"], "identity")
        changed = tool("identity")
        changed["description"] = "Return a value under a new description"
        registry, new = admit(registry, changed, backend, probe_inputs={"value": 1}, task_id="new")
        calls = len(backend.calls)
        with self.assertRaises(ValueError):
            invoke(registry, "identity", {"value": 1}, backend, expected_version=old["version"])
        self.assertEqual(len(backend.calls), calls)
        self.assertEqual(invoke(registry, "identity", {"value": 1}, backend)["tool_version"], new["version"])
    def test_backend_cannot_spoof_invocation_identity_receipt(self):
        backend = FakeSandbox()
        registry, event = admit(empty_registry(), tool("identity"), backend,
                                probe_inputs={"value": 1}, task_id="q1")
        backend.run_function = lambda *args: {"run_success": True, "tool_version": "spoofed", "invoked_name": "other"}
        receipt = invoke(registry, "identity", {"value": 1}, backend)
        self.assertEqual(receipt["tool_version"], event["version"])
        self.assertEqual(receipt["invoked_name"], "identity")
    def test_merge_requires_matching_contract_and_behavior(self):
        backend = FakeSandbox()
        registry, event = admit(empty_registry(), tool("identity"), backend,
                                probe_inputs={"value": 3}, task_id="q1")
        quarantined, pending = admit(registry, tool("identity_alt"), backend,
                                     probe_inputs={"value": 3}, task_id="q2", encoder=FakeEncoder())
        self.assertEqual(pending["decision"], "quarantine")
        self.assertNotIn("identity_alt", quarantined["aliases"])
        merged, decision = admit(registry, tool("identity_alt"), backend,
                                 probe_inputs={"value": 3}, task_id="q2",
                                 merge_with=event["version"],
                                 equivalence_inputs=[{"value": 1}, {"value": 2}],
                                 encoder=FakeEncoder())
        self.assertEqual(decision["decision"], "merge_verified")
        self.assertEqual(merged["aliases"]["identity_alt"], event["version"])
        self.assertEqual(len(merged["semantic"]), 1)
    def test_repeated_probe_is_not_independent_merge_evidence(self):
        backend = FakeSandbox()
        registry, event = admit(empty_registry(), tool("identity"), backend,
                                probe_inputs={"value": 3}, task_id="q1")
        with self.assertRaises(ValueError):
            admit(registry, tool("identity_alt"), backend, probe_inputs={"value": 3}, task_id="q2",
                  merge_with=event["version"], equivalence_inputs=[{"value": 1}, {"value": 1}],
                  encoder=FakeEncoder())
        merged, decision = admit(registry, tool("identity_alt"), backend,
                                 probe_inputs={"value": 3}, task_id="q2", merge_with=event["version"],
                                 equivalence_inputs=[{"value": 1}, {"value": 2}, {"value": 1}],
                                 encoder=FakeEncoder())
        self.assertEqual(decision["behavioral_probes"], 2)
        self.assertEqual(len(set(decision["behavioral_probe_sha256"])), 2)
        validate(merged)
    def test_edge_updates_are_train_only_and_independent(self):
        backend = FakeSandbox()
        registry, _ = admit(empty_registry(), tool("first"), backend,
                            probe_inputs={"value": 1}, task_id="q1")
        registry, _ = admit(registry, tool("second"), backend,
                            probe_inputs={"value": 1}, task_id="q1")
        with self.assertRaises(ValueError):
            update_edge(registry, source_name="first", target_name="second",
                        task_id="q2", split="test", direct_feedback="verified", task_success=True)
        first = update_edge(registry, source_name="first", target_name="second",
                            task_id="q1", split="train", direct_feedback="verified", task_success=True)
        duplicate = update_edge(first, source_name="first", target_name="second",
                                task_id="q1", split="train", direct_feedback="verified", task_success=True)
        self.assertEqual(first, duplicate)
        with self.assertRaises(ValueError):
            update_edge(first, source_name="first", target_name="second",
                        task_id="q1", split="train", direct_feedback="failed", task_success=False)
        no_blame = update_edge(first, source_name="first", target_name="second",
                               task_id="q3", split="train", direct_feedback="verified", task_success=False)
        self.assertEqual(first, no_blame)
    def test_legacy_updater_cannot_silently_write_v2_or_accept_bad_rates(self):
        from control.edges import migrate_legacy_edges
        registry, _ = admit(empty_registry(), tool("first"), FakeSandbox(),
                            probe_inputs={"value": 1}, task_id="q1")
        registry, _ = admit(registry, tool("second"), FakeSandbox(),
                            probe_inputs={"value": 1}, task_id="q1")
        args = dict(source_name="first", target_name="second", task_id="q2", split="train",
                    direct_feedback="verified", task_success=True)
        for rate in (-1, 0, 2, float("nan"), True):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                update_edge(registry, learning_rate=rate, **args)
        old = update_edge(registry, **args)
        migrated = migrate_legacy_edges(old)
        with self.assertRaises(ValueError):
            update_edge(migrated, **args)
        self.assertEqual(old["structure"]["edges"], migrated["structure"]["edges"])
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
