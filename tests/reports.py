# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import unittest
from copy import deepcopy
from report import paired_summary
from judge import StrictAuthorJudge, individual_scores, author_helpers
from tests.checkout import official_root
def rows(values):
    return [{"question_sha256": f"{q:064x}", "sample_index": sample, "success": bool(value)}
            for q, outcomes in enumerate(values) for sample, value in enumerate(outcomes)]
class PairedReportTests(unittest.TestCase):
    def test_pair_counts_include_losses(self):
        report = paired_summary({"a": rows([[0], [1], [1]]), "b": rows([[1], [0], [1]])})
        pair = report["pairs"]["b_minus_a"]
        self.assertEqual((pair["higher_question_mean"], pair["lower_question_mean"]), (1, 1))
        self.assertEqual(pair["mean_accuracy_delta"], 0)
        self.assertTrue(pair["binary_paired_gain_loss"])
    def test_repeated_samples_cluster_by_question(self):
        arms = {"a": rows([[0, 1, 0], [1, 1, 1]]), "b": rows([[1, 1, 0], [1, 1, 1]])}
        report = paired_summary(arms)
        self.assertEqual(report["questions"], 2)
        self.assertEqual(report["samples"], 6)
        self.assertAlmostEqual(report["pairs"]["b_minus_a"]["mean_accuracy_delta"], 1/6)
        self.assertFalse(report["pairs"]["b_minus_a"]["binary_paired_gain_loss"])
        self.assertEqual(report, paired_summary(arms))
    def test_missing_or_duplicate_rows_are_not_silently_dropped(self):
        a = rows([[1], [0]])
        for b in (a[:1], a+[a[0]], rows([[1, 0], [1]])):
            with self.assertRaises(ValueError):
                paired_summary({"a": a, "b": b})
    def test_replay_includes_missing_answer_without_another_model_call(self):
        class Backend:
            batch_size = 8
            def _build_prompt(self, prediction, reference, question):
                return [prediction, reference, question]
            def _call_llm_batch(self, messages):
                return ["1"] * len(messages)
        metric = author_helpers(official_root())["evaluation_metrics_HLE"]
        samples = [{"question": "Q", "ground_truth": "A", "output": output} for output in
                   ("<answer>\\boxed{A}</answer>", "no answer", "<answer>A</answer>")]
        judge = StrictAuthorJudge(Backend())
        aggregate = metric(samples, judge)
        scores = individual_scores(samples, judge.calls, metric)
        self.assertEqual([row["success"] for row in scores], [True, False, False])
        self.assertEqual([row["sample_index"] for row in scores], [0, 1, 2])
        self.assertEqual(sum(row["success"] for row in scores), aggregate[2])
        self.assertEqual(len(judge.calls), 1)
        bad = deepcopy(judge.calls)
        bad[0]["prediction_sha256"] = "0"*64
        with self.assertRaises(ValueError):
            individual_scores(samples, bad, metric)
        with self.assertRaises(ValueError):
            individual_scores(samples, [], metric)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
