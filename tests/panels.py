# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Read-only and synthetic-result checks for the paired panel runner.'
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from panel import PROJECT, bench_args, completed, server_args, summarize
class PanelTests(unittest.TestCase):
    def test_arm_commands_keep_exact_and_family_separate(self) -> None:
        args = argparse.Namespace(model="assets/model", port=8000, memory="runs/memory.json", queries="assets/queries.json", threads=1, tokens=2048, refinement=2, sample="runs/sample.json")
        root = PROJECT / "runs" / "panel_test"
        exact = server_args("joint_exact", args, root / "exact.jsonl", 0.5)
        family = server_args("joint_family", args, root / "family.jsonl", 0.5)
        baseline = server_args("baseline", args, root / "baseline.jsonl", 0.5)
        self.assertEqual(exact[-1], "exact")
        self.assertEqual(family[-1], "family")
        self.assertNotIn("--memory", baseline)
        for command, name in ((exact, "exact.jsonl"), (family, "family.jsonl"), (baseline, "baseline.jsonl")):
            trace = command[command.index("--trace") + 1]
            self.assertFalse(Path(trace).is_absolute())
            self.assertEqual((PROJECT / trace).resolve(), root / name)
        command = bench_args("joint_family", "preference_change", 23, args, root / "q23", 0.5)
        self.assertIn("--blocker", command)
        self.assertEqual(command[command.index("--expect-arm") + 1], "joint")
        self.assertEqual(command[command.index("--memory-backoff") + 1], "family")
    def test_summary_uses_all_paired_questions_and_preserves_failures(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "protocol.json").write_text("{}", encoding="utf-8")
            state = {"arms": ["baseline", "filter"], "modes": ["none"], "indices": [1, 2]}
            for arm, rows in {
                "baseline": [(False, 10, 100), (True, 4, 40)],
                "filter": [(True, 5, 70), (False, 8, 80)],
            }.items():
                for index, (success, steps, cost) in zip(state["indices"], rows):
                    destination = root / arm / "none" / f"q{index}"
                    destination.mkdir(parents=True)
                    data = {"results": [{"status": "completed", "evaluation": {"success": success}, "total_steps": steps}], "stats": {"scores": {"cost_metrics": {"total_agent_cost": cost}}}}
                    (destination / "results_test.json").write_text(json.dumps(data), encoding="utf-8")
            report = summarize(root, state)
            self.assertEqual(report["conditions"]["baseline"]["none"]["correct"], 1)
            paired = report["paired"]["filter_minus_baseline"]["none"]
            self.assertEqual((paired["gain"], paired["loss"]), (1, 1))
            self.assertEqual(paired["mean_step_delta_all"], -0.5)
            self.assertEqual(paired["mean_cost_delta_all"], 5.0)
    def test_empty_output_directory_can_resume_without_overwrite(self) -> None:
        with TemporaryDirectory() as folder:
            destination = Path(folder) / "q23"
            destination.mkdir()
            self.assertIsNone(completed(destination))
            (destination / "unexpected.txt").write_text("retain", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                completed(destination)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
