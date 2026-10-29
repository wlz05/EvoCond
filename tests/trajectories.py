# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from control.trajectory import episode_record, load_rollouts
class PolicyDataTest(unittest.TestCase):
    def load(self, outcomes: list[bool], mutate=None) -> tuple[list[dict], dict]:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for rollout, success in enumerate(outcomes):
                result = root / f"result_{rollout}.json"
                trace = root / f"trace_{rollout}.jsonl"
                result.write_text(json.dumps({"results": [{"status": "completed",
                    "evaluation": {"success": success}}], "stats": {"scores": {
                    "cost_metrics": {"total_agent_cost": 100.0}}}}), encoding="utf-8")
                trace.write_text(json.dumps({"action": "state_training_observation", "policy_attempts": [
                    {"prompt_ids": [1, 2], "completion_ids": [3, 4], "accepted": True}]})+"\n", encoding="utf-8")
                rows.append(episode_record(result, trace, 0, trace.stat().st_size, "evocond",
                                           "q17/none", rollout, "model", "tokenizer", "a"*64))
            if mutate:
                mutate(rows)
            path = root / "rollouts.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows)+"\n", encoding="utf-8")
            return load_rollouts(path, "evocond", "model", "tokenizer")
    def test_grouped_advantage(self) -> None:
        rows, report = self.load([True, False])
        self.assertEqual(report["nonzero_advantages"], 2)
        self.assertAlmostEqual(rows[0]["advantage"], 1.0)
        self.assertAlmostEqual(rows[1]["advantage"], -1.0)
    def test_test_split_is_rejected(self) -> None:
        def mutate(rows):
            rows[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "split or arm"):
            self.load([True, False], mutate)
    def test_tampered_acceptance_is_rejected(self) -> None:
        def mutate(rows):
            rows[0]["turns"][0]["accepted"] = False
        with self.assertRaisesRegex(ValueError, "Reward or policy tokens"):
            self.load([True, False], mutate)
    def test_duplicate_rollout_is_rejected(self) -> None:
        def mutate(rows):
            rows[1]["rollout"] = 0
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load([True, False], mutate)
    def test_mixed_sources_are_rejected(self) -> None:
        def mutate(rows):
            rows[1]["query_sha256"] = "b"*64
        with self.assertRaisesRegex(ValueError, "Mixed query"):
            self.load([True, False], mutate)
    def test_reward_tampering_rejected(self) -> None:
        def mutate(rows):
            rows[0]["reward"] = 9.0
        with self.assertRaisesRegex(ValueError, "Reward or policy tokens"):
            self.load([True, False], mutate)
    def test_all_rejected_attempts_still_form_failure_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result, trace = root / "result.json", root / "trace.jsonl"
            result.write_text(json.dumps({"results": [{"status": "completed", "evaluation": {"success": False}}],
                                          "stats": {"scores": {"cost_metrics": {"total_agent_cost": 0.0}}}}), encoding="utf-8")
            trace.write_text(json.dumps({"action": "state_training_observation", "policy_attempts": [
                {"prompt_ids": [1], "completion_ids": [2], "accepted": False}]})+"\n", encoding="utf-8")
            row = episode_record(result, trace, 0, trace.stat().st_size, "evocond", "q17/none", 0,
                                 "model", "tokenizer", "a"*64)
            self.assertEqual(row["reward"], 0.0)
            self.assertFalse(row["turns"][0]["accepted"])
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
