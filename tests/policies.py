# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import copy
from hashlib import sha256
import json
import unittest
from control.teacher import (ACTIONS, FEATURE_NAMES, PROVENANCE_FIELDS,
                                    decide, fit, query_novelty, validate)
def digest(value):
    return sha256(str(value).encode()).hexdigest()
def features():
    result = {name: 0.0 for name in FEATURE_NAMES}
    result.update(semantic_top_score=0.7, structural_top_score=0.6,
                  semantic_margin=0.1, structural_margin=0.2, rank_agreement=0.5)
    return result
def pair(identity, *, gains=None, seed=11):
    provenance = {name: digest(name) for name in PROVENANCE_FIELDS}
    provenance["encoder_id"] = "test-encoder-v1"
    provenance["context_hash"] = digest(identity)
    gains = gains if gains is not None else {0.5: 0.6}
    actions = []
    for alpha in ACTIONS:
        reward = 0.2+gains.get(alpha, 0.0)
        actions.append({**provenance, "alpha": alpha, "reward": reward,
                        "success": reward >= 0.5, "seed": seed,
                        "trace_hash": digest((identity, seed, alpha)), "executed": True,
                        "action_scope": "next_retrieval_only"})
    return {"schema": "evocond_teacher_pairs_v2", "query_id": identity,
            "decision_id": "initial", "seed": seed, **provenance,
            "query_embedding": [1.0, 0.0, 0.0], "features": features(), "actions": actions}
def records():
    return [pair(f"q{i}") for i in range(6)]
def train(rows=None):
    return fit(records() if rows is None else rows,
               train_ids={"q0", "q1", "q2"}, calibration_ids={"q3", "q4", "q5"},
               validation_ids={"v1"}, test_ids={"t1"})
def live_decide(model, values, **overrides):
    context = {"query_embedding": [1.0, 0.0, 0.0], "encoder_id": "test-encoder-v1",
               "policy_hash": digest("policy_hash"), "alignment_hash": digest("alignment_hash")}
    context.update(overrides)
    return decide(model, values, **context)
class TeacherPolicyTests(unittest.TestCase):
    def test_selects_executed_intermediate_action_without_edge_evidence(self):
        model, report = train()
        model = json.loads(json.dumps(model))
        validate(model)
        decision = live_decide(model, features(), query_embedding=[1.0, 0.0, 0.0],
                          encoder_id="test-encoder-v1")
        self.assertEqual(decision["structure_weight"], 0.5)
        self.assertGreater(decision["lower_bound"], 0.5)
        self.assertEqual(report["fit_questions"], 3)
        self.assertEqual(features()["edge_reliability"], 0.0)
    def test_negative_action_benefit_always_falls_back(self):
        rows = [pair(f"q{i}", gains={alpha: -0.1 for alpha in ACTIONS[1:]}) for i in range(6)]
        model, _ = train(rows)
        self.assertEqual(live_decide(model, features())["structure_weight"], 0.0)
    def test_calibration_detects_overoptimistic_training_gain(self):
        rows = records()
        rows[3:] = [pair(f"q{i}", gains={0.5: -0.2}) for i in range(3, 6)]
        model, _ = train(rows)
        self.assertGreater(model["action_models"]["0.5"]["calibration_radius"], 0.7)
        self.assertEqual(live_decide(model, features())["structure_weight"], 0.0)
    def test_novelty_uses_only_fit_questions_and_cannot_be_overridden_by_zero(self):
        model, _ = train()
        self.assertEqual({item["query_id"] for item in model["query_prototypes"]}, {"q0", "q1", "q2"})
        self.assertEqual(query_novelty(model, [0.0, 1.0, 0.0], "test-encoder-v1"), 1.0)
        result = live_decide(model, features(), query_embedding=[0.0, 1.0, 0.0], encoder_id="test-encoder-v1")
        self.assertEqual(result["reason"], "novel_query_outside_training_support")
        self.assertEqual(result["structure_weight"], 0.0)
    def test_fit_novelty_excludes_same_question(self):
        rows = records()
        for i in range(3):
            rows[i]["query_embedding"] = [float(j == i) for j in range(3)]
        model, _ = train(rows)
        self.assertEqual(model["feature_bounds"]["query_novelty"], [1.0, 1.0])
        self.assertEqual(rows[0]["features"]["query_novelty"], 0.0)
    def test_unsupported_intermediate_action_is_not_invented(self):
        rows = records()
        for row in rows:
            row["actions"] = [action for action in row["actions"] if action["alpha"] in (0.0, 1.0)]
        model, _ = train(rows)
        self.assertEqual(set(model["action_models"]), {"1.0"})
        self.assertEqual(live_decide(model, features())["structure_weight"], 0.0)
    def test_repeated_seeds_do_not_replace_independent_questions(self):
        rows = [pair("q0", seed=i) for i in range(10)]+records()[3:]
        with self.assertRaisesRegex(ValueError, "independent"):
            train(rows)
    def test_provenance_and_observed_execution_are_mandatory(self):
        changes = (("context_hash", digest("other")), ("candidate_hash", digest("other")),
                   ("alignment_hash", digest("other")), ("policy_hash", digest("other")),
                   ("seed", 12), ("executed", False), ("action_scope", "whole_episode"))
        for key, value in changes:
            with self.subTest(key=key):
                rows = records()
                rows[0]["actions"][1][key] = value
                with self.assertRaises(ValueError):
                    train(rows)
    def test_held_out_rows_and_split_overlap_are_rejected(self):
        rows = records()
        rows[0] = pair("v1")
        with self.assertRaises(ValueError):
            train(rows)
        with self.assertRaises(ValueError):
            fit(records(), train_ids={"q0", "q1", "q2"}, calibration_ids={"q2", "q3", "q4"},
                validation_ids=set(), test_ids=set())
    def test_extra_outcome_feature_and_incompatible_live_versions_rejected(self):
        rows = records()
        rows[0]["features"]["answer_correct"] = 1.0
        with self.assertRaises(ValueError):
            train(rows)
        model, _ = train()
        for keyword, value in (("encoder_id", "other"), ("policy_hash", digest("other")),
                               ("alignment_hash", digest("other"))):
            with self.subTest(keyword=keyword), self.assertRaises(ValueError):
                live_decide(model, features(), **{keyword: value})
    def test_feature_shift_abstains_and_corrupt_model_rejected(self):
        model, _ = train()
        values = features()
        values["goal_changed"] = 1.0
        self.assertEqual(live_decide(model, values)["reason"], "features_outside_training_support")
        broken = copy.deepcopy(model)
        broken["query_prototypes"][0]["query_id"] = "v1"
        with self.assertRaises(ValueError):
            validate(broken)
    def test_large_finite_embeddings_normalize_without_squared_overflow(self):
        model, _ = train()
        self.assertEqual(query_novelty(model, [1e308, 0.0, 0.0], "test-encoder-v1"), 0.0)
        self.assertAlmostEqual(query_novelty(model, [1e308, 1e308, 0.0], "test-encoder-v1"),
                               1.0-2**-0.5)
        for embedding in ([1.7e308, 1.7e308, 0.0], [10**1000, 0, 0]):
            with self.subTest(embedding_kind=type(embedding[0]).__name__), self.assertRaises(ValueError):
                query_novelty(model, embedding, "test-encoder-v1")
    def test_nonfinite_accumulated_prediction_falls_back(self):
        model, _ = train()
        model["action_models"]["0.5"]["weights"] = [1e308]*(len(FEATURE_NAMES)+1)
        decision = live_decide(model, features())
        self.assertEqual(decision["reason"], "nonfinite_action_prediction")
        self.assertEqual(decision["structure_weight"], 0.0)
    def test_equal_supported_actions_choose_smaller_structure_weight(self):
        rows = [pair(f"q{i}", gains={0.25: 0.6, 0.5: 0.6}) for i in range(6)]
        model, _ = train(rows)
        self.assertEqual(live_decide(model, features())["structure_weight"], 0.25)
        model["action_models"] = dict(reversed(list(model["action_models"].items())))
        self.assertEqual(live_decide(model, features())["structure_weight"], 0.25)
    def test_live_query_and_frozen_versions_cannot_be_omitted(self):
        model, _ = train()
        for name in ("query_embedding", "encoder_id", "policy_hash", "alignment_hash"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                live_decide(model, features(), **{name: None})
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
