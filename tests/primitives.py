# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from copy import deepcopy
from hashlib import sha256
import unittest
from control.dynamic import admit, empty_registry
from control.runtime import execution_context_fingerprint
from control.primitive import creation_note, profile, validate_mode
from experiment import teacher_options
from tests.generation import FakeSandbox, tool
class ToolAbstractionTests(unittest.TestCase):
    def test_complex_parameterized_code_is_not_declared_bad(self):
        spec = tool("sum_positive", "def sum_positive(values):\n    total = 0\n    for value in values:\n        if value > 0:\n            total += value\n    return total\n")
        report = profile(spec)
        self.assertEqual(report["complexity"]["loops"], 1)
        self.assertEqual(report["complexity"]["branches"], 1)
        self.assertEqual(report["input_diagnostics"]["lexically_unused_parameters"], [])
        self.assertFalse(report["generality_certified"])
    def test_short_constant_answer_is_not_certified_general(self):
        report = profile(tool("constant", "def constant(value):\n    return 42\n"))
        self.assertEqual(report["input_diagnostics"]["lexically_unused_parameters"], ["value"])
        self.assertEqual(report["input_diagnostics"]["numeric_literals"], 1)
        self.assertFalse(report["generality_certified"])
        self.assertEqual(report["behavioral_cross_task_tests"], 0)
    def test_source_diagnostics_never_execute_and_include_keyword_only_args(self):
        spec = tool("unsafe", "raise RuntimeError('must not execute')\ndef unsafe(x, /, *, limit, **options):\n    return x\n")
        report = profile(spec)
        self.assertEqual(report["input_diagnostics"]["lexically_unused_parameters"], ["limit", "options"])
        self.assertEqual(report["source_sha256"], sha256(spec["code"].encode()).hexdigest())
    def test_profile_attached_on_admission_does_not_rewrite_source_or_version(self):
        spec = tool("identity")
        before = deepcopy(spec)
        registry, event = admit(empty_registry(), spec, FakeSandbox(), probe_inputs={"value": 1}, task_id="train-1")
        saved = registry["semantic"][event["version"]]
        self.assertEqual(saved["code"], before["code"])
        self.assertEqual(saved["abstraction_profile"], profile(spec))
        self.assertEqual(spec, before)
    def test_creation_intervention_and_structure_budget_are_fingerprinted(self):
        legacy = execution_context_fingerprint({})
        self.assertEqual(legacy, execution_context_fingerprint({"tool_construction": "legacy", "structure_policy": {"mode": "dense"}}))
        self.assertNotEqual(legacy, execution_context_fingerprint({"tool_construction": "portable"}))
        self.assertNotEqual(legacy, execution_context_fingerprint({"structure_policy": {"mode": "none"}}))
        self.assertEqual(creation_note(), "")
        self.assertIn("parameters", creation_note("portable").replace("arguments", "parameters"))
        self.assertIn("longer algorithm is allowed", creation_note("portable"))
        with self.assertRaises(ValueError):
            validate_mode("shortest")
        with self.assertRaises(ValueError):
            teacher_options({"tool_construction": "shortest"})
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
