# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from copy import deepcopy
import unittest
from control.dynamic import admit, empty_registry
from control.edges import observe_policy_transition, observe_transition
from control.ranking import build_alignment_pairs, describe, prepare_alignment
from control.budget import policy, select
from abstraction import audit
from tests.generation import FakeSandbox, tool
SPARSE = {"mode": "sparse", "max_incident_edges": 1, "min_matching_tasks": 2,
          "min_abs_score": 0.0, "selection": "score", "seed": 42}
class Encoder:
    encoder_id = "structure-fixture-v1"
    def __init__(self):
        self.texts = []
    def encode(self, text):
        self.texts.append(text)
        return [0., 1.] if text == "shifted" else [1., 0.]
def memory():
    registry = empty_registry()
    for name in ("first", "second", "third"):
        registry, _ = admit(registry, tool(name), FakeSandbox(), probe_inputs={"value": 1}, task_id="admit")
    for name in ("second", "third"):
        for index in range(2):
            registry = observe_transition(registry, source_name="first", target_name=name,
                task_id=f"{name}-{index}", split="train", context_vector=[1., 0.],
                encoder_id=Encoder.encoder_id, direct_feedback="verified", task_success=True)
    return registry
def fit(registry, value=SPARSE):
    pairs = build_alignment_pairs(registry, Encoder(), query_id="fit", query="task", structure_policy=value)
    return prepare_alignment(pairs, encoder_id=Encoder.encoder_id, train_ids={"fit"},
                             calibration_ids={"calibration"}, validation_ids={"val"}, test_ids={"test"})
class StructureBudgetTests(unittest.TestCase):
    def test_policy_rejects_implicit_or_invalid_sparse_parameters(self):
        invalid = [{"mode": "sparse"}, {"mode": "dense", "seed": 1}, {"mode": "unknown"},
                   {**SPARSE, "max_incident_edges": True}, {**SPARSE, "min_matching_tasks": 0},
                   {**SPARSE, "min_abs_score": float("nan")}, {**SPARSE, "selection": "answer"}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                policy(value)
        self.assertEqual(policy(), {"mode": "dense"})
    def test_signed_budget_is_deterministic_and_read_only(self):
        views = {"a->b": {"matching_tasks": 2, "score": -.8, "support": 2},
                 "a->c": {"matching_tasks": 2, "score": .3, "support": 2},
                 "b->c": {"matching_tasks": 2, "score": .2, "support": 2}}
        before = deepcopy(views)
        selected, receipt = select(views, SPARSE, allowed_versions=["a", "b", "c"])
        self.assertEqual(set(selected), {"a->b"})
        self.assertEqual(receipt["negative_edges"], 1)
        self.assertEqual(receipt["max_incident_edges_observed"], 1)
        self.assertFalse(receipt["hard_prerequisites_changed"])
        self.assertEqual(select(dict(reversed(list(views.items()))), SPARSE,
                                allowed_versions=["a", "b", "c"]), (selected, receipt))
        self.assertEqual(views, before)
    def test_hash_is_a_seeded_budget_control(self):
        views = {f"a->{key}": {"matching_tasks": 2, "score": .5, "support": 2} for key in "bcd"}
        spec = {**SPARSE, "selection": "hash"}
        first = select(views, spec, allowed_versions=list("abcd"))
        self.assertEqual(first, select(views, spec, allowed_versions=list("abcd")))
        self.assertEqual(first[1]["selected_edges"], 1)
        self.assertNotEqual(first[1]["policy_hash"], select(views, {**spec, "seed": 43}, allowed_versions=list("abcd"))[1]["policy_hash"])
    def test_zero_budget_and_none_do_not_delete_history(self):
        registry = memory()
        before = deepcopy(registry)
        for value in ({"mode": "none"}, {**SPARSE, "max_incident_edges": 0}):
            view = describe(registry, Encoder(), fit(registry, value), query="task", previous_tool="first", structure_policy=value)
            self.assertEqual(view["structure_receipt"]["selected_edges"], 0)
            self.assertTrue(all(row["edge"]["score"] == 0 for row in view["candidates"]))
            self.assertEqual(len(view["candidates"]), 3)
        self.assertEqual(registry, before)
    def test_collection_and_serving_use_identical_budgeted_text(self):
        registry, encoder = memory(), Encoder()
        pairs = build_alignment_pairs(registry, encoder, query_id="fit", query="task", structure_policy=SPARSE)
        encoder.texts.clear()
        view = describe(registry, encoder, fit(registry), query="task", previous_tool="first", structure_policy=SPARSE)
        self.assertCountEqual([row["structural_text"] for row in pairs],
                              [text for text in encoder.texts if text.startswith("Input contract:")])
        self.assertEqual(view["structure_receipt"]["selected_edges"], 1)
        self.assertTrue(all(row["structure_policy"] == SPARSE for row in pairs))
    def test_old_or_other_view_fit_cannot_silently_serve_sparse(self):
        registry = memory()
        with self.assertRaisesRegex(ValueError, "structure policy"):
            describe(registry, Encoder(), fit(registry, {"mode": "dense"}), query="task", structure_policy=SPARSE)
        with self.assertRaisesRegex(ValueError, "structure policy"):
            describe(registry, Encoder(), fit(registry), query="task")
        pairs = build_alignment_pairs(registry, Encoder(), query_id="fit", query="task", structure_policy=SPARSE)
        pairs[0]["structure_policy"] = {"mode": "none"}
        with self.assertRaisesRegex(ValueError, "mix historical"):
            prepare_alignment(pairs, encoder_id=Encoder.encoder_id, train_ids={"fit"},
                              calibration_ids={"cal"}, validation_ids={"val"}, test_ids={"test"})
    def test_repeated_rollouts_do_not_promote_one_question(self):
        registry = memory()
        # Use an existing pair under a new encoder/context and one base task.
        for rollout in range(8):
            registry = observe_policy_transition(registry, source_name="second", target_name="third",
                task_id="single-question", split="train", context_vector=[0., 1.], encoder_id=Encoder.encoder_id,
                policy_step=0, rollout_id=str(rollout), feedback=["verified"], task_success=True)
        view = describe(registry, Encoder(), fit(registry), query="shifted", structure_policy=SPARSE)
        self.assertEqual(view["structure_receipt"]["selected_edges"], 0)
    def test_unknown_coactivation_is_not_evidence_and_audit_leaves_state_frozen(self):
        registry = memory()
        for index in range(4):
            registry = observe_transition(registry, source_name="second", target_name="third",
                task_id=f"unknown-{index}", split="train", context_vector=[0., 1.], encoder_id=Encoder.encoder_id,
                direct_feedback="unknown", task_success=True)
        before = deepcopy(registry)
        report = audit(registry, context_vector=[0., 1.], encoder_id=Encoder.encoder_id)
        self.assertEqual(report["structure_conditions"]["sparse_score"]["selected_edges"], 0)
        self.assertFalse(report["accuracy_measured"])
        self.assertFalse(report["generated_code_executed"])
        self.assertEqual(registry, before)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
