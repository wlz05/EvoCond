# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from control.gate import (ARMS, DIMENSION, calibrate, decide, evaluate,
                               features, fit, initial_user, normalize, settings,
                               utility, validate_model)
import gate as workflow
from panel import PROJECT
def example(index: int, text: str, best: str = "joint_family") -> dict:
    outcomes = {a: {"success": True, "cost": 100.0, "steps": 6} for a in ARMS}
    outcomes[best] = {"success": True, "cost": 50.0, "steps": 3}
    return {"index": index, "mode": "none", "text": text, "outcomes": outcomes}
class TaskGateTests(unittest.TestCase):
    def test_feature_normalization_removes_instance_numbers(self):
        self.assertEqual(features("go to <LocationPreference01234> at <TimeInfo333>"),
                         features("go to <LocationPreference98765> at <TimeInfo999>"))
        self.assertNotEqual(features("avoid costly live search"), features("must use live search"))
        self.assertNotEqual(features("budget 100 dollars"), features("budget 1000 dollars"))
    def test_only_initial_visible_user_is_read(self):
        messages = [{"role": "system", "content": "hidden"}, {"role": "user", "content": "initial query"},
                    {"role": "tool", "content": "answer 19923"}, {"role": "user", "content": "future changed goal"}]
        self.assertEqual(initial_user(messages), "initial query")
        self.assertEqual(initial_user([{"role": "assistant", "content": "answer"}]), "")
    def test_supervised_training_changes_parameters_and_is_deterministic(self):
        rows = [example(i, "find a quiet hotel near the station") for i in range(6)]
        a, b = fit(rows), fit(rows)
        self.assertEqual(a, b)
        self.assertGreater(sum(abs(x) for x in a["weights"]["joint_family"]), 0)
        a["threshold"] = 0.0
        validate_model(a)
        self.assertEqual(decide(a, rows[0]["text"])["arm"], "joint_family")
    def test_missing_validation_support_defers_not_fake_success(self):
        rows = [example(0, "a hotel near a station")]
        model = fit(rows)
        report = calibrate(model, [example(20, rows[0]["text"])])
        self.assertIsNone(report["selected_threshold"])
        self.assertEqual(decide(model, rows[0]["text"])["arm"], "filter")
    def test_validation_can_admit_useful_gate(self):
        text = "find a quiet hotel near a station"
        model = fit([example(i, text) for i in range(6)])
        report = calibrate(model, [example(i, text) for i in range(20, 24)])
        self.assertIsNotNone(report["selected_threshold"])
        result = evaluate(model, [example(30, text)])
        self.assertEqual(result["loss"], 0)
        self.assertEqual(result["selected"], 1)
    def test_correctness_regression_on_validation_rejects_gate(self):
        text = "find a quiet hotel near a station"
        model = fit([example(i, text) for i in range(6)])
        rows = [example(i, text) for i in range(20, 24)]
        for row in rows:
            row["outcomes"]["joint_family"]["success"] = False
        calibrate(model, rows)
        self.assertIsNone(model["threshold"])
    def test_similarity_abstention_and_empty_input(self):
        text = "find a quiet hotel near a station"
        model = fit([example(1, text)])
        model.update(threshold=0.0, min_similarity=1.0)
        self.assertEqual(decide(model, "solve a geometric theorem")['arm'], "filter")
        self.assertEqual(decide(model, "")['reason'], "empty_initial_user")
    def test_wrong_answer_cannot_win_by_cheap_cost_on_one_task(self):
        base = {"success": True, "cost": 100.0, "steps": 20}
        bad = {"success": False, "cost": 0, "steps": 0}
        self.assertLess(utility(bad, base), -0.8)
    def test_settings_do_not_leak_memory_into_filter(self):
        memory = object()
        self.assertEqual(settings("filter", memory), ("filter", None, "exact"))
        self.assertEqual(settings("evocond", memory), ("full", None, "exact"))
        self.assertEqual(settings("joint_exact", memory), ("full", memory, "exact"))
        self.assertEqual(settings("joint_family", memory), ("full", memory, "family"))
    def test_invalid_parameters_fail_closed(self):
        model = fit([example(1, "find hotel")])
        model["weights"]["evocond"][0] = float("nan")
        with self.assertRaises(ValueError):
            validate_model(model)
    def test_artifacts_cannot_be_overwritten(self):
        with TemporaryDirectory(dir=PROJECT) as folder:
            path = Path(folder) / "frozen.json"
            workflow.write_new(path, {"x": 1})
            workflow.write_new(path, {"x": 1})
            with self.assertRaises(FileExistsError):
                workflow.write_new(path, {"x": 2})
            self.assertEqual(json.loads(path.read_text())["x"], 1)
class GateWorkflowTests(unittest.TestCase):
    def fixture(self, root: Path):
        queries = []
        for task in sorted(workflow.TASKS):
            for i in range(8):
                queries.append({"task": task, "query_id": f"<Query{len(queries) + 1}>",
                                "user_requirements": f"{task} distinct {chr(97 + i)} request"})
        workflow.write_new(root / "queries.json", queries)
        workflow.write_new(root / "memory.json", {"metadata": {"source": "costbench_traces_v1",
                           "train_indices": [0, 1], "query_sha256": workflow.digest(root / "queries.json")}})
        workflow.write_new(root / "model" / "config.json", {"model_type": "synthetic"})
        (root / "prior").mkdir()
        return argparse.Namespace(root=str(root / "gate"), queries=str(root / "queries.json"),
            memory=str(root / "memory.json"), model=str(root / "model"), exclude_root=str(root / "prior"),
            train_per_task=1, validation_per_task=1, test_per_task=1, tokens=20, refinement=2,
            port=8000, seed=20260927)
    def populate(self, root: Path, name: str, manifest: dict):
        protocol = {k: manifest[k] for k in ("query_sha256", "memory_sha256", "model_config_sha256",
                                            "tokens", "refinement", "memory_extra_cost", "arms", "modes")}
        protocol.update(indices=manifest["splits"][name]["indices"], threads=1,
                        sample_sha256=workflow.digest(root / name / "sample.json"),
                        code_sha256={"serve.py": manifest["source_sha256"]["serve.py"]})
        workflow.write_new(root / name / "panel/protocol.json", protocol)
        for index in manifest["splits"][name]["indices"]:
            # Letters keep prompts distinct after instance-number removal.
            text = "pick a hotel " + chr(97 + index // 26) + chr(97 + index % 26)
            for mode in manifest["modes"]:
                for arm in ARMS:
                    data = {"results": [{"status": "completed", "total_steps": 4,
                        "evaluation": {"success": True},
                        "conversation_history": [{"role": "user", "content": text}]}],
                        "stats": {"scores": {"cost_metrics": {"total_agent_cost": 100}}}}
                    workflow.write_new(root / name / "panel" / arm / mode / f"q{index}" / "results_test.json", data)
    def test_prepare_read_only_then_resume_frozen_disjoint_splits(self):
        with TemporaryDirectory(dir=PROJECT) as folder:
            args = self.fixture(Path(folder))
            manifest = workflow.make_manifest(args)
            root = Path(args.root)
            self.assertFalse(root.exists())
            assigned = [i for s in manifest["splits"].values() for i in s["indices"]]
            self.assertEqual(len(assigned), 18)
            self.assertEqual(len(set(assigned)), 18)
            self.assertFalse({0, 1} & set(assigned))
            workflow.initialize(root, manifest)
            self.assertEqual(workflow.make_manifest(args), manifest)
            args.tokens += 1
            with self.assertRaises(ValueError):
                workflow.make_manifest(args)
    def test_no_silent_reuse_when_unused_pool_is_too_small(self):
        with TemporaryDirectory(dir=PROJECT) as folder:
            args = self.fixture(Path(folder))
            args.train_per_task = 100
            with self.assertRaises(ValueError):
                workflow.make_manifest(args)
    def test_training_never_reads_test_and_keeps_gate_frozen(self):
        with TemporaryDirectory(dir=PROJECT) as folder:
            args = self.fixture(Path(folder))
            root = Path(args.root)
            manifest = workflow.make_manifest(args)
            workflow.initialize(root, manifest)
            self.populate(root, "train", manifest)
            self.populate(root, "validation", manifest)
            # Any accidental test loading will hit invalid JSON.
            test_dir = root / "test" / "poison"
            test_dir.mkdir()
            (test_dir / "results_bad.json").write_text("not json")
            model = workflow.train(root, manifest)
            self.assertIsNone(model["threshold"])
            self.assertEqual(workflow.train(root, manifest), model)
            self.assertNotIn("test_result_sha256", model["provenance"])
    def test_mismatched_initial_inputs_are_not_pairable(self):
        with TemporaryDirectory(dir=PROJECT) as folder:
            args = self.fixture(Path(folder))
            root = Path(args.root)
            manifest = workflow.make_manifest(args)
            workflow.initialize(root, manifest)
            self.populate(root, "train", manifest)
            i = manifest["splits"]["train"]["indices"][0]
            path = root / "train/panel/joint_family/none" / f"q{i}/results_test.json"
            data = json.loads(path.read_text())
            data["results"][0]["conversation_history"][0]["content"] = "different problem"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                workflow.load_examples(root, "train", manifest)
    def test_missing_outcome_is_not_silently_wrong_or_zero_cost(self):
        with TemporaryDirectory(dir=PROJECT) as folder:
            args = self.fixture(Path(folder))
            root = Path(args.root)
            manifest = workflow.make_manifest(args)
            workflow.initialize(root, manifest)
            self.populate(root, "train", manifest)
            i = manifest["splits"]["train"]["indices"][0]
            path = root / "train/panel/filter/none" / f"q{i}/results_test.json"
            data = json.loads(path.read_text())
            data["results"][0]["evaluation"] = {}
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                workflow.load_examples(root, "train", manifest)
    def test_completed_live_test_aggregation_and_resume_without_gpu(self):
        with TemporaryDirectory(dir=PROJECT) as folder:
            args = self.fixture(Path(folder))
            root = Path(args.root)
            manifest = workflow.make_manifest(args)
            workflow.initialize(root, manifest)
            for name in ("train", "validation"):
                self.populate(root, name, manifest)
            model = workflow.train(root, manifest)
            self.populate(root, "test", manifest)
            for mode in manifest["modes"]:
                for i in manifest["splits"]["test"]["indices"]:
                    source = root / "test/panel/filter" / mode / f"q{i}/results_test.json"
                    workflow.write_new(root / "live_gate" / mode / f"q{i}/results_test.json", json.loads(source.read_text()))
            workflow.live_test(args, manifest, model)
            report = json.loads((root / "test_report.json").read_text())
            self.assertEqual(report["conditions"]["gate"]["overall"]["correct"], 12)
            self.assertEqual(report["gate_selections"], {"filter": 12})
            self.assertEqual(report["replay_disagreements"], 0)
            self.assertEqual(report["validation_selected_static"], "filter")
            workflow.live_test(args, manifest, model)
    def test_no_refitting_after_test_started_without_frozen_gate(self):
        with TemporaryDirectory(dir=PROJECT) as folder:
            args = self.fixture(Path(folder))
            root = Path(args.root)
            manifest = workflow.make_manifest(args)
            workflow.initialize(root, manifest)
            for name in ("train", "validation", "test"):
                self.populate(root, name, manifest)
            with self.assertRaises(ValueError):
                workflow.train(root, manifest)
    def test_wrong_panel_budget_cannot_be_used_for_training(self):
        with TemporaryDirectory(dir=PROJECT) as folder:
            args = self.fixture(Path(folder))
            root = Path(args.root)
            manifest = workflow.make_manifest(args)
            workflow.initialize(root, manifest)
            self.populate(root, "train", manifest)
            path = root / "train/panel/protocol.json"
            data = json.loads(path.read_text())
            data["tokens"] += 100
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                workflow.load_examples(root, "train", manifest)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
