# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'CPU tests of explicit data dependencies; no tool execution or model calls.'
from copy import deepcopy
import unittest
from control.isolation import ModelToolInputError, canonical_hash
from control.evidence import (REFERENCE, call_record, reference_parameters,
                                               resolve_arguments, transition_feedback, validate_record)
def reference(call_id="call1", path="/value"):
    return {REFERENCE: {"call_id": call_id, "path": path}}
def source(**kwargs):
    return call_record("call1", "producer", "a"*64, kwargs.get("success", True),
                       {"seed": 7}, kwargs.get("result", {"value": 7}))
class TransitionTests(unittest.TestCase):
    def test_resolve_exact_pointer_and_record_actual_arguments(self):
        first = source()
        inputs, bindings = resolve_arguments({"x": reference()}, [first])
        self.assertEqual(inputs, {"x": 7})
        second = call_record("call2", "consumer", "b"*64, True, inputs, 14, bindings)
        self.assertEqual(transition_feedback(first, second), "verified")
        self.assertEqual(bindings[0]["source_record_sha256"], first["sha256"])
    def test_equal_values_without_reference_are_not_dependency_evidence(self):
        first = source()
        for value in (7, True, None, "", 0):
            first = source(result={"value": value})
            second = call_record("call2", "consumer", "b"*64, True, {"x": value}, value)
            self.assertEqual(transition_feedback(first, second), "unknown")
    def test_cross_episode_unknown_failed_or_future_call_is_rejected(self):
        for ledger, ref in (([], reference()), ([source()], reference("future")),
                            ([source(success=False)], reference())):
            with self.subTest(ledger=bool(ledger), ref=ref), self.assertRaises(ModelToolInputError):
                resolve_arguments({"x": ref}, ledger)
    def test_nested_arrays_and_json_pointer_escapes(self):
        first = source(result={"a/b": {"~key": [False, {"n": 9}]}})
        arguments = {"nested": [reference(path="/a~1b/~0key/1/n")]}
        resolved, bindings = resolve_arguments(arguments, [first])
        self.assertEqual(resolved, {"nested": [9]})
        self.assertEqual(bindings[0]["target_path"], "/nested/0")
        second = call_record("call2", "consumer", "b"*64, True, resolved, None, bindings)
        self.assertEqual(transition_feedback(first, second), "verified")
    def test_whole_result_reference(self):
        first = source(result=[1, 2, 3])
        resolved, _ = resolve_arguments({"x": reference(path="")}, [first])
        self.assertEqual(resolved, {"x": [1, 2, 3]})
    def test_nonreferenceable_call_keeps_sequence_without_fabricating_null_result(self):
        unavailable = call_record("call1", "producer", "a"*64, True, {}, None, result_available=False)
        with self.assertRaises(ModelToolInputError):
            resolve_arguments({"x": reference(path="")}, [unavailable])
        genuine_null = call_record("call1", "producer", "a"*64, True, {}, None)
        self.assertEqual(resolve_arguments({"x": reference(path="")}, [genuine_null])[0], {"x": None})
    def test_malformed_references_do_not_downgrade_to_literal_objects(self):
        invalid = [None, {}, {"call_id": "call1"}, {"call_id": ["call1"], "path": ""},
                   {"call_id": "call1", "path": "value"}, {"call_id": "call1", "path": "/~2"},
                   {"call_id": "call1", "path": "/missing"}]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ModelToolInputError):
                resolve_arguments({"x": {REFERENCE: item}}, [source()])
        with self.assertRaises(ModelToolInputError):
            resolve_arguments({"x": {**reference(), "literal": 2}}, [source()])
    def test_duplicate_or_tampered_ledger_is_fatal_not_model_failure(self):
        first = source()
        corrupt = deepcopy(first)
        corrupt["result"]["value"] = 100
        for ledger in ([first, first], [corrupt]):
            with self.assertRaises(ValueError) as caught:
                resolve_arguments({"x": reference()}, ledger)
            self.assertNotIsInstance(caught.exception, ModelToolInputError)
    def test_target_failure_has_direct_dataflow_but_unrelated_failure_does_not(self):
        first = source()
        resolved, bindings = resolve_arguments({"x": reference()}, [first])
        target = call_record("call2", "consumer", "b"*64, False, resolved, None, bindings)
        self.assertEqual(transition_feedback(first, target), "failed")
        unrelated = call_record("call2", "consumer", "b"*64, False, {"x": 7}, None)
        self.assertEqual(transition_feedback(first, unrelated), "unknown")
    def test_wrong_source_version_or_executed_value_is_fatal(self):
        first = source()
        args, bindings = resolve_arguments({"x": reference()}, [first])
        for changed in ({**bindings[0], "source_tool_version": "c"*64},
                        {**bindings[0], "value_sha256": canonical_hash(999)}):
            target = call_record("call2", "consumer", "b"*64, True, args, 14, [changed])
            with self.assertRaises(ValueError):
                transition_feedback(first, target)
        target = call_record("call2", "consumer", "b"*64, True, {"x": 999}, 14, bindings)
        with self.assertRaises(ValueError):
            transition_feedback(first, target)
    def test_builtin_results_are_usable_but_not_versioned_edge_credit(self):
        first = call_record("call1", "search", None, True, {"query": "facts"}, {"value": 7})
        args, bindings = resolve_arguments({"x": reference()}, [first])
        target = call_record("call2", "consumer", "b"*64, True, args, 14, bindings)
        self.assertEqual(transition_feedback(first, target), "unknown")
    def test_schema_advertises_references_without_mutating_original(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
        advertised = reference_parameters(schema)
        self.assertEqual(schema["properties"]["x"], {"type": "integer"})
        self.assertEqual(advertised["properties"]["x"]["anyOf"][0], {"type": "integer"})
        self.assertIn(REFERENCE, advertised["properties"]["x"]["anyOf"][1]["properties"])
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
