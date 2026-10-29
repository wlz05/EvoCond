# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from copy import deepcopy
import unittest
from control.pairing import (ACTION_SCOPE, PairCollectionError,
                                  collect_teacher_policy_pairs, snapshot_hash)
from control.teacher import ACTIONS, FEATURE_NAMES, PROVENANCE_FIELDS, decide, fit
def inputs():
    return {"query_id": "q-train-1", "decision_id": "turn-2", "split": "train",
            "state": {"question": "Find an address", "history": ["searched"]},
            "registry": {"tools": {"search": {"version": "v1"}}, "edges": []},
            "candidates": [{"name": "lookup", "version": "v2"}],
            "alignment": {"version": "alignment-1", "matrix": [[1.0]]},
            "features": {name: 0.0 for name in FEATURE_NAMES},
            "query_embedding": [1.0] + [0.0] * 383,
            "encoder_id": "sentence-transformers/all-MiniLM-L6-v2@pinned",
            "policy_hash": "a" * 64, "seeds": [4, 9]}
def receipt(request):
    return {"executed": True, "action_scope": ACTION_SCOPE,
            "alpha": request.alpha, "seed": request.seed,
            "reward": 1.0 if request.alpha == 0.25 else 0.0,
            "success": request.alpha == 0.25,
            "trace_hash": snapshot_hash([request.query_id, request.seed, request.alpha]),
            **{name: getattr(request, name) for name in PROVENANCE_FIELDS}}
class PairingTests(unittest.TestCase):
    def test_all_actions_reset_to_the_same_state_with_common_seeds(self):
        source = inputs()
        untouched = deepcopy(source)
        calls = []
        def evaluate(request):
            self.assertEqual(request.context_hash, snapshot_hash(request.state))
            self.assertEqual(request.registry_hash, snapshot_hash(request.registry))
            self.assertEqual(request.candidate_hash, snapshot_hash(request.candidates))
            self.assertEqual(request.alignment_hash, snapshot_hash(request.alignment))
            self.assertEqual(request.state["history"], ["searched"])
            self.assertEqual(request.registry["edges"], [])
            self.assertEqual(len(request.query_embedding), 384)
            calls.append((request.seed, request.alpha))
            result = receipt(request)
            request.state["history"].append("branch completed")
            request.registry["edges"].append("local observation")
            request.candidates[0]["name"] = "branch-local candidate"
            request.alignment["matrix"][0][0] = 5.0
            request.features[next(iter(request.features))] = 9.0
            request.query_embedding[0] = 7.0
            return result
        records = collect_teacher_policy_pairs(evaluator=evaluate, **source)
        self.assertEqual(source, untouched)
        self.assertEqual(calls, [(seed, alpha) for seed in [4, 9] for alpha in ACTIONS])
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record["schema"], "evocond_teacher_pairs_v2")
            self.assertEqual(record["features"], untouched["features"])
            self.assertEqual(record["query_embedding"], untouched["query_embedding"])
            self.assertEqual([row["alpha"] for row in record["actions"]], list(ACTIONS))
            self.assertTrue(all(row["executed"] is True for row in record["actions"]))
            self.assertEqual(sum(row["success"] for row in record["actions"]), 1)
    def test_different_start_or_policy_receipt_cannot_be_paired(self):
        for field in PROVENANCE_FIELDS:
            with self.subTest(field=field):
                def evaluate(request):
                    result = receipt(request)
                    result[field] = "other-encoder" if field == "encoder_id" else "b" * 64
                    return result
                with self.assertRaisesRegex(PairCollectionError, field):
                    collect_teacher_policy_pairs(evaluator=evaluate, **inputs())
    def test_only_actually_executed_requested_actions_supply_rewards(self):
        changes = [{"executed": False}, {"alpha": 0.75}, {"seed": 100},
                   {"action_scope": "whole_episode"}, {"success": 1},
                   {"reward": float("nan")}, {"reward": 1.1}, {"reward": 10**1000},
                   {"trace_hash": "missing"}]
        for change in changes:
            with self.subTest(change=change):
                def evaluate(request):
                    return {**receipt(request), **change}
                with self.assertRaises(PairCollectionError):
                    collect_teacher_policy_pairs(evaluator=evaluate, **inputs())
    def test_worker_failure_does_not_become_a_zero_reward_example(self):
        calls = []
        def evaluate(request):
            calls.append(request.alpha)
            if request.alpha == 0.5:
                raise TimeoutError("worker unavailable")
            return receipt(request)
        with self.assertRaisesRegex(PairCollectionError, "worker unavailable"):
            collect_teacher_policy_pairs(evaluator=evaluate, **inputs())
        self.assertEqual(calls, [0.0, 0.25, 0.5])
    def test_validation_and_test_never_invoke_the_evaluator(self):
        for split in ("validation", "test"):
            source = inputs()
            source["split"] = split
            calls = []
            with self.assertRaisesRegex(ValueError, "training-only"):
                collect_teacher_policy_pairs(evaluator=lambda request: calls.append(request), **source)
            self.assertEqual(calls, [])
    def test_feature_schema_embedding_and_seed_checks(self):
        invalid = [{"features": {"unknown": 0.0}}, {"query_embedding": [0.0, 0.0]},
                   {"query_embedding": [float("inf")]}, {"seeds": [1, 1]},
                   {"query_embedding": [10**1000]}, {"seeds": [True]},
                   {"policy_hash": "not-a-checkpoint"}]
        for update in invalid:
            with self.subTest(update=update):
                with self.assertRaises(ValueError):
                    collect_teacher_policy_pairs(evaluator=receipt, **{**inputs(), **update})
    def test_out_of_band_shared_artifact_mutation_is_detected(self):
        source = inputs()
        def evaluate(request):
            source["registry"]["edges"].append("unapproved shared commit")
            return receipt(request)
        with self.assertRaisesRegex(PairCollectionError, "shared pre-decision artifacts"):
            collect_teacher_policy_pairs(evaluator=evaluate, **source)
    def test_collected_receipts_fit_the_action_teacher_without_schema_conversion(self):
        records = []
        for index in range(6):
            source = inputs()
            source["query_id"] = f"q{index}"
            records.extend(collect_teacher_policy_pairs(evaluator=receipt, **source))
        model, report = fit(records, train_ids={"q0", "q1", "q2"},
                            calibration_ids={"q3", "q4", "q5"},
                            validation_ids={"v1"}, test_ids={"t1"})
        result = decide(model, inputs()["features"],
                        query_embedding=inputs()["query_embedding"],
                        encoder_id=inputs()["encoder_id"],
                        policy_hash=inputs()["policy_hash"],
                        alignment_hash=records[0]["alignment_hash"])
        self.assertEqual(report["matched_decisions"], 12)
        self.assertEqual(result["structure_weight"], 0.25)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
