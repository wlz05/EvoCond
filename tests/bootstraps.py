# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'CPU-only bootstrap orchestration checks; fixtures are not real rollouts.'
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import bootstrap as bootstrap
import experiment as experiment
def row(question, index=0):
    return {"question": question, "prompt": [{"role": "user", "content": question}],
            "data_source": "arpo", "reward_model": {"style": "rule", "ground_truth": {"target": ["fixture"]}},
            "extra_info": {"index": index}}
class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.partition = {"fit_queries": 4, "calibration_queries": 2, "seed": 42}
        self.training = [row(f"training {i}", i) for i in range(10)]
    def protocol(self):
        frozen = {"config": {"run_id": "cpu", "model": str(self.root / "model"),
                  "official_root": str(self.root / "official"), "encoder": str(self.root / "encoder"),
                  "isolation_config": str(self.root / "isolation"), "isolation_receipt": str(self.root / "receipt"),
                  "search_url": "http://localhost:8000/retrieve",
                  "datasets": {s: str(self.root / (s+".parquet")) for s in ("train", "validation", "test")},
                  "budget": experiment.settings({"gpus": 1, "batch_size": 2, "rollouts": 1}, collection=True)},
                  "fingerprints": {"model": {"sha256": "a" * 64}, "official_source": {"sha256": "b" * 64}}}
        return {"experiment": frozen, "input_config": {"encoder_id": "fixture", "seeds": [42, 43], "capture_turn": 2},
                "splits": bootstrap.partitions(self.training, [row("validation")], [row("test")], self.partition)}
    def test_partition_is_training_only_disjoint_and_order_invariant(self):
        a = bootstrap.partitions(self.training, [row("validation")], [row("test")], self.partition)
        b = bootstrap.partitions(list(reversed(self.training)), [row("validation")], [row("test")], self.partition)
        self.assertEqual(a, b)
        self.assertEqual(len(a["teacher_fit"]), 4)
        self.assertEqual(len(a["teacher_calibration"]), 2)
        self.assertFalse(set(a["train"]) & set(a["test"]))
    def test_partition_refuses_insufficient_data_and_duplicate_questions(self):
        with self.assertRaises(ValueError):
            bootstrap.partitions(self.training[:5], [row("validation")], [row("test")], self.partition)
        with self.assertRaises(ValueError):
            bootstrap.partitions(self.training+[row("TRAINING  0")], [row("validation")], [row("test")], self.partition)
    def test_no_training_on_validation_or_test_even_unicode_equivalent(self):
        for name in ("validation", "test"):
            other = {"validation": [row("validation")], "test": [row("test")]}
            other[name] = [row('\uff34\uff32\uff21\uff29\uff2e\uff29\uff2e\uff27  0')]
            with self.subTest(name=name), self.assertRaises(ValueError):
                bootstrap.partitions(self.training, other["validation"], other["test"], self.partition)
    def test_bad_partition_counts(self):
        for spec in ({}, {**self.partition, "fit_queries": True}, {**self.partition, "calibration_queries": 0}, {**self.partition, "seed": -1}):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                bootstrap.partitions(self.training, [row("v")], [row("t")], spec)
    def test_warmup_and_collection_do_not_load_fitted_teacher(self):
        protocol = self.protocol()
        warm = bootstrap.runtime_config(protocol, self.root, "warmup")
        collect = bootstrap.runtime_config(protocol, self.root, "collect")
        self.assertFalse(warm["collection"])
        self.assertTrue(warm["bootstrap_warmup"])
        self.assertNotIn("alignment_path", warm)
        self.assertTrue(collect["collection"])
        self.assertNotIn("teacher_path", collect)
        self.assertNotEqual(warm["registry_path"], collect["registry_path"])
        self.assertTrue(collect["value_contracts"])
        self.assertTrue(collect["sample_tool_turns"])
    def test_abstraction_options_propagate_to_warmup_and_continuation(self):
        protocol = self.protocol()
        spec = {"mode": "sparse", "max_incident_edges": 2, "min_matching_tasks": 2,
                "min_abs_score": 0., "selection": "score", "seed": 42}
        protocol["input_config"].update(tool_construction="portable", structure_policy=spec, reuse_boundary_mode="explicit")
        for stage in ("warmup", "collect"):
            config = bootstrap.runtime_config(protocol, self.root, stage)
            self.assertEqual(config["tool_construction"], "portable")
            self.assertEqual(config["structure_policy"], spec)
            self.assertEqual(config["reuse_boundary_mode"], "explicit")
    def test_recorded_registry_does_not_use_one_mb_proposal_limit(self):
        path = self.root / 'snapshots.jsonl'
        path.write_text(json.dumps({'registry': 'x'*1100000})+'\n', encoding='utf-8')
        self.assertEqual(len(list(bootstrap.artifact_rows(path))[0]['registry']), 1100000)
        with self.assertRaises(ValueError):
            list(bootstrap.artifact_rows(path, max_characters=100))
        for value in ('{"k":1,"k":2}\n', '{"k":NaN}\n', '\n', '[]\n'):
            path.write_text(value, encoding='utf-8')
            with self.assertRaises(ValueError):
                list(bootstrap.artifact_rows(path))
    def test_actual_command_uses_official_collection_not_optimizer(self):
        protocol = self.protocol()
        for stage, mode in (("warmup", "warmup"), ("collect", "pairs")):
            cmd, env = bootstrap.command(protocol, self.root, stage)
            self.assertEqual(cmd[:2], ["bash", "main_searl.sh"])
            self.assertIn('++trainer.teacher_collect_mode='+json.dumps(mode), cmd)
            self.assertIn('++actor_rollout_ref.rollout.n=1', cmd)
            self.assertEqual(sum(arg.startswith('++actor_rollout_ref.rollout.n=') for arg in cmd), 1)
            self.assertIn('++trainer.val_before_train=false', cmd)
            self.assertEqual(json.loads(env["TRAIN_FILES"]), str(self.root / ("teacher_fit.parquet" if stage == "warmup" else "teacher_all.parquet")))
    def test_dry_run_never_creates_stage_or_checks_gpu(self):
        with patch.object(experiment, "readiness") as gpu, patch.object(bootstrap.subprocess, "run") as run:
            report = bootstrap.launch(self.root, self.protocol(), "warmup")
        self.assertFalse(report["qwen_updates"])
        self.assertFalse((self.root / "warmup").exists())
        gpu.assert_not_called()
        run.assert_not_called()
    def test_existing_stage_is_never_overwritten(self):
        (self.root / "warmup").mkdir()
        with self.assertRaises(FileExistsError):
            bootstrap.launch(self.root, self.protocol(), "warmup", True)
    def test_same_gpu_lock_blocks_other_launch_without_removing_it(self):
        lock = self.root / ".launch.lock"
        lock.write_text("other process", encoding="utf-8")
        with patch.object(experiment, "readiness"), patch.object(bootstrap.subprocess, "run") as run, self.assertRaises(FileExistsError):
            bootstrap.launch(self.root, self.protocol(), "warmup", True)
        run.assert_not_called()
        self.assertEqual(lock.read_text(), "other process")
    def test_nonzero_process_exit_preserves_artifacts_but_releases_lock(self):
        class Failure:
            returncode = 9
        with patch.object(experiment, "readiness"), patch.object(experiment, "environment_identity", return_value={"fixture": True}), \
                patch.object(bootstrap.subprocess, "run", return_value=Failure()), self.assertRaisesRegex(RuntimeError, "exited 9"):
            bootstrap.launch(self.root, self.protocol(), "warmup", True)
        self.assertFalse((self.root / ".launch.lock").exists())
        self.assertEqual(experiment.read(self.root / "warmup_result.json")["exit_code"], 9)
        self.assertTrue((self.root / "warmup_launch.json").exists())
    def test_environment_change_blocks_process(self):
        experiment.write_new(self.root / "environment.json", {"fixture": 1})
        with patch.object(experiment, "readiness"), patch.object(experiment, "environment_identity", return_value={"fixture": 2}), \
                patch.object(bootstrap.subprocess, "run") as run, self.assertRaisesRegex(ValueError, "environment changed"):
            bootstrap.launch(self.root, self.protocol(), "warmup", True)
        run.assert_not_called()
        self.assertFalse((self.root / ".launch.lock").exists())
    def test_completed_artifact_modification_is_detected(self):
        protocol = self.protocol()
        experiment.write_new(self.root / "protocol.json", protocol)
        experiment.write_new(self.root / "warmup_launch.json", {"protocol_sha256": experiment.canonical_hash(protocol)})
        experiment.write_new(self.root / "warmup/audit.json", {"complete": True})
        experiment.write_new(self.root / "warmup/registry.json", {"fixture": True})
        (self.root / "warmup/snapshots.jsonl").write_text('{"fixture":true}\n', encoding="utf-8")
        experiment.write_new(self.root / "warmup_result.json", {"exit_code": 0, "outputs": {
            name: experiment.file_digest(self.root / "warmup" / name) for name in ("audit.json", "registry.json", "snapshots.jsonl")}})
        self.assertTrue(bootstrap.completed(self.root, "warmup")["complete"])
        (self.root / "warmup/snapshots.jsonl").write_text('{"fixture":false}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            bootstrap.completed(self.root, "warmup")
    def test_preparation_dry_run_has_no_files(self):
        output = self.root / "new"
        config = self.root / "config.json"
        config.write_text('{}', encoding="utf-8")
        with patch.object(bootstrap, "freeze", return_value=(self.protocol(), self.training)), redirect_stdout(io.StringIO()):
            bootstrap.main(["prepare", "--root", str(output), "--config", str(config)])
        self.assertFalse(output.exists())
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
