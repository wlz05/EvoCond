# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import math
import unittest
from copy import deepcopy
from control.dynamic import admit, empty_registry, invoke
from control.edges import observe_transition
from control.ranking import (VIEW_SCHEMA, active_tools, build_alignment_pairs,
                                      describe, rank, prepare_alignment, validate_alignment)
from tests.generation import FakeEncoder, FakeSandbox, tool
class Encoder:
    encoder_id = "fixture-encoder-384-v1"
    def encode(self, text):
        return [0., 1.] + [0.]*382 if "unrelated" in text else [1., 0.] + [0.]*382
def alignment(dim=384):
    return {"format": "evocond_searl_alignment_v1", "dimension": dim,
            "encoder_id": Encoder.encoder_id, "view_schema": VIEW_SCHEMA,
            "matrix": [[float(i == j) for j in range(dim)] for i in range(dim)]}
def registry():
    result = empty_registry()
    for name in ("first", "second"):
        result, _ = admit(result, tool(name), FakeSandbox(), probe_inputs={"value": 1}, task_id="admit")
    return result
class RetrievalV2Tests(unittest.TestCase):
    def test_retired_merged_alias_keeps_real_previous_version(self):
        memory, _ = admit(empty_registry(), tool("first"), FakeSandbox(),
                          probe_inputs={"value": 1}, task_id="admit")
        old = memory["aliases"]["first"]
        memory, _ = admit(memory, tool("renamed"), FakeSandbox(), probe_inputs={"value": 1},
                          task_id="merge", encoder=FakeEncoder(), merge_with=old,
                          equivalence_inputs=[{"value": 1}, {"value": 2}])
        memory, _ = admit(memory, tool("renamed", "def renamed(value):\n    return value + 3\n"),
                          FakeSandbox(), probe_inputs={"value": 1}, task_id="upgrade")
        view = describe(memory, Encoder(), alignment(), query="question", previous_tool="renamed",
                        previous_version=old)
        self.assertEqual(view["visible_state"]["previous_version"], old)
    def test_query_and_passage_encoder_paths_match_collection_and_serving(self):
        class AsymmetricEncoder(Encoder):
            def __init__(self):
                self.queries, self.passages = [], []
            def encode_query(self, text):
                self.queries.append(text)
                return [1., 0.] + [0.]*382
            def encode(self, text):
                self.passages.append(text)
                return super().encode(text)
        encoder = AsymmetricEncoder()
        build_alignment_pairs(registry(), encoder, query_id="fit", query="visible subtask")
        describe(registry(), encoder, alignment(), query="visible subtask")
        self.assertEqual(encoder.queries, ["visible subtask", "visible subtask"])
        self.assertNotIn("visible subtask", encoder.passages)
    def test_real_encoder_dimension_and_untrained_semantic_fallback(self):
        before = registry()
        frozen = deepcopy(before)
        result = rank(before, Encoder(), alignment(), None, query="question", previous_tool="first")
        self.assertEqual(result["teacher"]["structure_weight"], 0)
        self.assertEqual(len(result["query_embedding"]), 384)
        self.assertEqual(before, frozen)
        self.assertTrue(all(row["score"] == row["semantic_score"] for row in result["tools"]))
    def test_context_shift_removes_old_transition_evidence(self):
        memory = observe_transition(registry(), source_name="first", target_name="second",
                                    task_id="q1", split="train", context_vector=Encoder().encode("old"),
                                    encoder_id=Encoder.encoder_id, direct_feedback="verified", task_success=True)
        old = describe(memory, Encoder(), alignment(), query="old", previous_tool="first")
        new = describe(memory, Encoder(), alignment(), query="unrelated", previous_tool="first")
        old_edge = next(r["edge"] for r in old["candidates"] if r["name"] == "second")
        new_edge = next(r["edge"] for r in new["candidates"] if r["name"] == "second")
        self.assertGreater(old_edge["score"], 0)
        self.assertEqual(new_edge["score"], 0)
    def test_no_stale_ranked_version_can_be_invoked(self):
        memory = registry()
        stale = memory["aliases"]["first"]
        memory, _ = admit(memory, tool("first", "def first(value):\n    return value + 1\n"),
                          FakeSandbox(), probe_inputs={"value": 1}, task_id="upgrade")
        rows = active_tools(memory)
        self.assertNotIn(stale, {row["version"] for row in rows})
        backend = FakeSandbox()
        for row in rank(memory, Encoder(), alignment(), None, query="question")["tools"]:
            invoke(memory, row["name"], {"value": 1}, backend)
            self.assertEqual(backend.calls[-1][0], memory["semantic"][row["version"]]["code"])
    def test_fixed_action_is_global_and_excludes_unavailable_candidate(self):
        memory = registry()
        allowed = [memory["aliases"]["second"]]
        result = rank(memory, Encoder(), alignment(), None, query="question", fixed_alpha=.5,
                      eligible_versions=allowed)
        self.assertEqual(len(result["tools"]), 1)
        self.assertAlmostEqual(result["tools"][0]["score"], .75)
        with self.assertRaises(ValueError):
            rank(memory, Encoder(), alignment(), None, query="question", eligible_versions=["missing"])
    def test_snapshot_validation_rejects_encoder_changes_and_nonfinite_map(self):
        bad = alignment()
        bad["encoder_id"] = "changed"
        with self.assertRaises(ValueError):
            rank(registry(), Encoder(), bad, None, query="question")
        bad = alignment()
        bad["matrix"][0][0] = math.nan
        with self.assertRaises(ValueError):
            rank(registry(), Encoder(), bad, None, query="question")
    def test_alignment_never_fits_calibration_questions(self):
        pair = {"query_id": "q2", "tool_version": "v", "semantic": [1., 0.],
                "structural": [1., 0.], "encoder_id": Encoder.encoder_id,
                "view_schema": VIEW_SCHEMA}
        with self.assertRaises(ValueError):
            prepare_alignment([pair], encoder_id=Encoder.encoder_id, train_ids={"q1"},
                              calibration_ids={"q2"}, validation_ids={"q3"}, test_ids={"q4"})
    def test_alignment_collection_and_serving_encode_the_same_structure_views(self):
        class RecordingEncoder(Encoder):
            def __init__(self):
                self.texts = []
            def encode(self, text):
                self.texts.append(text)
                return super().encode(text)
        encoder = RecordingEncoder()
        memory = observe_transition(registry(), source_name="first", target_name="second",
                                    task_id="train-1", split="train", context_vector=encoder.encode("task"),
                                    encoder_id=encoder.encoder_id, direct_feedback="verified", task_success=True)
        before = deepcopy(memory)
        pairs = build_alignment_pairs(memory, encoder, query_id="q1", query="task")
        encoder.texts.clear()
        view = describe(memory, encoder, alignment(), query="task", previous_tool="first")
        served_texts = [text for text in encoder.texts if text.startswith("Input contract:")]
        self.assertCountEqual([row["structural_text"] for row in pairs], served_texts)
        self.assertTrue(any("Supported predecessors: first" in text for text in served_texts))
        self.assertTrue(any("Supported successors: second" in text for text in served_texts))
        self.assertTrue(all(row["view_schema"] == VIEW_SCHEMA for row in pairs))
        self.assertEqual({row["tool_version"] for row in pairs},
                         {row["version"] for row in view["candidates"]})
        self.assertEqual(memory, before)
        fitted = prepare_alignment(pairs, encoder_id=encoder.encoder_id, train_ids={"q1"},
                                   calibration_ids={"q2"}, validation_ids={"q3"}, test_ids={"q4"})
        validate_alignment(fitted, encoder_id=encoder.encoder_id)
        self.assertEqual(fitted["validation_ids"], ["q3"])
        self.assertEqual(fitted["calibration_ids"], ["q2"])
    def test_old_or_malformed_alignment_is_rejected_before_serving(self):
        invalid = [alignment(), alignment(), alignment(), alignment()]
        invalid[0].pop("view_schema")
        invalid[1].pop("matrix")
        invalid[2]["matrix"][0][0] = math.inf
        invalid[3]["format"] = "wrong-format"
        for artifact in invalid:
            with self.subTest(keys=artifact.keys()):
                with self.assertRaises(ValueError):
                    validate_alignment(artifact)
        pairs = build_alignment_pairs(registry(), Encoder(), query_id="q1", query="task")
        pairs[0]["view_schema"] = "predecessor-only-legacy-view"
        with self.assertRaisesRegex(ValueError, "shared structural view"):
            prepare_alignment(pairs, encoder_id=Encoder.encoder_id, train_ids={"q1"},
                              calibration_ids={"q2"}, validation_ids={"q3"}, test_ids={"q4"})
    def test_previous_invocation_keeps_observed_version_after_same_name_upgrade(self):
        memory = observe_transition(registry(), source_name="first", target_name="second",
                                    task_id="train-1", split="train", context_vector=Encoder().encode("task"),
                                    encoder_id=Encoder.encoder_id, direct_feedback="verified", task_success=True)
        observed_version = memory["aliases"]["first"]
        memory, _ = admit(memory, tool("first", "def first(value):\n    return value + 1\n"),
                          FakeSandbox(), probe_inputs={"value": 1}, task_id="upgrade")
        real = describe(memory, Encoder(), alignment(), query="task", previous_tool="first",
                        previous_version=observed_version)
        current_alias = describe(memory, Encoder(), alignment(), query="task", previous_tool="first")
        real_edge = next(row["edge"] for row in real["candidates"] if row["name"] == "second")
        alias_edge = next(row["edge"] for row in current_alias["candidates"] if row["name"] == "second")
        self.assertGreater(real_edge["score"], 0)
        self.assertEqual(alias_edge["score"], 0)
        self.assertEqual(real["visible_state"]["previous_version"], observed_version)
        self.assertNotEqual(real["visible_state_hash"], current_alias["visible_state_hash"])
        self.assertNotIn("context_hash", real)
        self.assertEqual(real["visible_state_kind"], "retrieval_summary_not_full_continuation_snapshot")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            describe(memory, Encoder(), alignment(), query="task", previous_tool="second",
                     previous_version=observed_version)
    def test_large_finite_embeddings_do_not_overflow_and_scores_stay_bounded(self):
        class LargeEncoder(Encoder):
            def encode(self, text):
                return [1e300, 1e300] + [0.0]*382
        result = rank(registry(), LargeEncoder(), alignment(), None, query="task")
        self.assertTrue(all(math.isfinite(value) for value in result["query_embedding"]))
        for row in result["candidates"]:
            self.assertTrue(-1 <= row["semantic_score"] <= 1)
            self.assertTrue(-1 <= row["structural_score"] <= 1)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
