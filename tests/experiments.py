# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'CPU-only launcher checks. Tiny files are not real models or checkpoints.'
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import experiment as experiment
class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
    def model(self):
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
        (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (model / "tokenizer.json").write_text("{}", encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"CPU_FIXTURE_NOT_MODEL_WEIGHTS")
        return model
    def frozen(self):
        return {"claim": "CPU fixture only", "config": {
            "run_id": "unit", "model": str(self.root / "model"),
            "official_root": str(self.root / "official"), "encoder": str(self.root / "encoder"),
            "isolation_config": str(self.root / "config.json"), "isolation_receipt": str(self.root / "receipt.json"),
            "search_url": "http://localhost:9000/retrieve", "teacher": {"value_contracts": True},
            "datasets": {s: str(self.root / (s+".parquet")) for s in ("train", "validation", "test")},
            "budget": experiment.settings({})},
            "fingerprints": {"model": {"sha256": "a" * 64}, "official_source": {"sha256": "b" * 64}}}
    def checkpoint(self, arm="evocond"):
        folder = self.root / arm / "checkpoints" / "global_step_5"
        (folder / "actor").mkdir(parents=True)
        for name in ("data.pt", "method_runtime/runtime.json", "method_runtime/graphrag.json", "method_runtime/subtasks.graphml"):
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("CPU_FIXTURE", encoding="utf-8")
        return folder
    def test_weight_fingerprint_changes_with_weight_bytes(self):
        model = self.model()
        first = experiment.directory_digest(model, model=True)
        (model / "model.safetensors").write_bytes(b"different")
        second = experiment.directory_digest(model, model=True)
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertIn("tokenizer.json", second["files"])
    def test_config_accepts_bom_but_rejects_duplicate_or_nonfinite_fields(self):
        config = self.root/'config.json'
        config.write_text('\ufeff{"format":"fixture"}\r\n', encoding='utf-8')
        self.assertEqual(experiment.read(config), {'format': 'fixture'})
        for value in ('{"seed":1,"seed":2}', '{"score":NaN}'):
            config.write_text(value, encoding='utf-8')
            with self.assertRaises(ValueError):
                experiment.read(config)
    def test_relative_paths_use_explicit_workspace(self):
        config = self.root/'config.json'
        config.write_text('{}', encoding='utf-8')
        self.assertEqual(experiment.resolved('config.json', self.root), str(config))
    def test_prepare_from_other_directory_uses_explicit_workspace(self):
        frozen = self.frozen()
        (self.root/'input.json').write_text('{}', encoding='utf-8')
        with patch.object(sys, 'argv', ['experiment.py', 'prepare', '--workspace', str(self.root),
                '--root', 'outputs/new', '--config', 'input.json']), \
                patch.object(experiment, 'freeze', return_value=frozen) as freeze, \
                redirect_stdout(io.StringIO()):
            experiment.main()
        freeze.assert_called_once_with({}, base=self.root)
        self.assertFalse((self.root/'outputs').exists())
    def test_model_shard_index_missing_file_is_refused(self):
        model = self.model()
        (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"p": "missing.safetensors"}}), encoding="utf-8")
        with self.assertRaises(ValueError):
            experiment.directory_digest(model, model=True)
    def test_tokenizer_configuration_alone_is_not_a_tokenizer(self):
        model = self.model()
        (model / "tokenizer.json").unlink()
        with self.assertRaises(ValueError):
            experiment.directory_digest(model, model=True)
        (model / "vocab.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            experiment.directory_digest(model, model=True)
        (model / "merges.txt").write_text("CPU_FIXTURE", encoding="utf-8")
        self.assertEqual(len(experiment.directory_digest(model, model=True)["sha256"]), 64)
    def test_wrong_model_type_is_refused(self):
        model = self.model()
        (model / "config.json").write_text('{"model_type":"other"}', encoding="utf-8")
        with self.assertRaises(ValueError):
            experiment.directory_digest(model, model=True)
    def test_budget_rejects_unknown_invalid_and_indivisible(self):
        for change in ({"typo": 1}, {"gpus": True}, {"batch_size": 3}, {"epochs": 0},
                       {"max_turns": -1}, {"learning_rate": float("nan")}, {"learning_rate": True}, {"rollouts": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                experiment.settings(change)
    def test_contextual_budget_requires_complete_windows_and_aligned_frequencies(self):
        from control.stage import settings
        budget = experiment.settings({'gpus': 4, 'batch_size': 4, 'save_freq': 2, 'validation_freq': 2})
        options = settings({})
        self.assertEqual(experiment.window_budget(budget, options, 8)['actor_updates_per_epoch'], 1)
        for count in (7, 12):
            with self.assertRaises(ValueError):
                experiment.window_budget(budget, options, count)
        with self.assertRaises(ValueError):
            experiment.window_budget({**budget, 'save_freq': 1}, options, 8)
    def test_contextual_example_has_relative_paths_and_explicit_schedule(self):
        config = experiment.read(experiment.HERE/'configs/contextual.json')
        from control.stage import settings
        options = settings(config['contextual'])
        for key in ('official_root', 'model', 'encoder', 'isolation_config', 'isolation_receipt'):
            self.assertFalse(Path(config[key]).is_absolute())
        self.assertEqual(options['frozen_batches'], 2)
        self.assertNotIn('teacher', config)
    def test_method_fingerprint_excludes_embedded_upstream_and_runs(self):
        (self.root/'main.py').write_text('value = 1', encoding='utf-8')
        expected = experiment.directory_digest(self.root)
        folder = self.root/'baselines/SEARL-official'
        folder.mkdir(parents=True)
        (folder/'main.py').write_text('value = 2', encoding='utf-8')
        self.assertEqual(experiment.directory_digest(self.root), expected)
    def test_three_arms_receive_same_training_budget_and_private_outputs(self):
        frozen = self.frozen()
        commands, environments = {}, {}
        for arm in experiment.ARMS:
            commands[arm], environments[arm] = experiment.launch_spec(frozen, self.root, arm, attempt="cpu")
        for parameter in ("actor_rollout_ref.actor.optim.lr", "actor_rollout_ref.rollout.n", "data.train_batch_size",
                          "actor_rollout_ref.rollout.max_turns", "trainer.total_epochs"):
            values = [next(value for value in commands[arm] if value.startswith("++"+parameter+"=")) for arm in experiment.ARMS]
            self.assertEqual(len(set(values)), 1)
        for key in ("RESULT_DIR", "LOG_PATH", "PROJECT_NAME", "SEARL_DYNAMIC_REGISTRY", "WANDB_DIR"):
            self.assertEqual(len({env[key] for env in environments.values()}), 3)
        self.assertEqual({env["MODEL_PATH"] for env in environments.values()}, {frozen["config"]["model"]})
        self.assertEqual({env["RL_FACTORY_EMBEDDING_MODEL"] for env in environments.values()}, {frozen["config"]["encoder"]})
        namespaces = [next(value for value in commands[arm] if value.startswith("++ray_init.namespace=")) for arm in experiment.ARMS]
        self.assertEqual(len(set(namespaces)), 3)
    def test_runtime_registries_are_arm_private(self):
        configurations = experiment.runtimes(self.frozen(), self.root)
        self.assertEqual(len({row["registry_path"] for row in configurations.values()}), 3)
        for arm, row in configurations.items():
            self.assertEqual(row["arm"], arm)
            self.assertTrue(Path(row["registry_path"]).is_relative_to(self.root / arm))
    def test_resume_requires_matching_arm_and_complete_runtime(self):
        checkpoint = self.checkpoint()
        self.assertEqual(experiment.checkpoint(self.root, "evocond", checkpoint), str(checkpoint))
        with self.assertRaises(ValueError):
            experiment.checkpoint(self.root, "teacher", checkpoint)
        (checkpoint / "method_runtime/runtime.json").unlink()
        with self.assertRaises(ValueError):
            experiment.checkpoint(self.root, "evocond", checkpoint)
    def test_checkpoint_name_must_be_an_explicit_optimizer_step(self):
        folder = self.root / "evocond" / "checkpoints" / "latest"
        folder.mkdir(parents=True)
        with self.assertRaises(ValueError):
            experiment.checkpoint(self.root, "evocond", folder)
    def test_existing_checkpoint_requires_explicit_resume(self):
        self.checkpoint()
        with self.assertRaises(ValueError):
            experiment.launch_spec(self.frozen(), self.root, "evocond")
    def test_evaluate_requires_checkpoint_and_uses_selected_split(self):
        with self.assertRaises(ValueError):
            experiment.launch_spec(self.frozen(), self.root, "evocond", stage="evaluate")
        checkpoint = self.checkpoint()
        command, env = experiment.launch_spec(self.frozen(), self.root, "evocond", stage="evaluate", split="test", resume=checkpoint)
        self.assertEqual(json.loads(env["VAL_FILES"]), self.frozen()["config"]["datasets"]["test"])
        self.assertIn("++trainer.val_only=true", command)
    def test_training_cannot_mislabel_validation_as_test(self):
        with self.assertRaisesRegex(ValueError, "validation only"):
            experiment.launch_spec(self.frozen(), self.root, "searl", stage="train", split="test")
    def test_real_vllm_output_budget_matches_response_budget(self):
        command, _ = experiment.launch_spec(self.frozen(), self.root, "searl")
        self.assertIn("++actor_rollout_ref.rollout.max_tokens=2048", command)
        self.assertIn("++actor_rollout_ref.rollout.max_model_len=32768", command)
    def test_prepare_dry_run_has_no_side_effects(self):
        output = self.root / "not-created"
        frozen = self.frozen()
        config = self.root / "input.json"
        config.write_text("{}", encoding="utf-8")
        with patch.object(sys, "argv", ['experiment.py', "prepare", "--root", str(output), "--config", str(config)]), \
                patch.object(experiment, "freeze", return_value=frozen), \
                patch.object(experiment.subprocess, "run") as launch, redirect_stdout(io.StringIO()):
            experiment.main()
        self.assertFalse(output.exists())
        launch.assert_not_called()
    def test_train_dry_run_does_not_touch_gpu_or_start_trainer(self):
        frozen = self.frozen()
        experiment.write_new(self.root / "protocol.json", frozen)
        experiment.write_new(self.root / "evocond/runtime_config.json", experiment.runtimes(frozen, self.root)["evocond"])
        with patch.object(sys, "argv", ['experiment.py', "train", "--root", str(self.root), "--arm", "evocond"]), \
                patch.object(experiment, "freeze", return_value=frozen), \
                patch.object(experiment, "readiness") as gpu, \
                patch.object(experiment.subprocess, "run") as launch, redirect_stdout(io.StringIO()):
            experiment.main()
        gpu.assert_not_called()
        launch.assert_not_called()
        self.assertFalse((self.root / "evocond/.launch.lock").exists())
        self.assertFalse((self.root / ".launch.lock").exists())
    def test_global_gpu_lock_blocks_a_different_arm(self):
        frozen = self.frozen()
        experiment.write_new(self.root / "protocol.json", frozen)
        experiment.write_new(self.root / "teacher/runtime_config.json", experiment.runtimes(frozen, self.root)["teacher"])
        lock = self.root / ".launch.lock"
        lock.write_text('{"arm":"searl","pid":1234}', encoding="utf-8")
        with patch.object(sys, "argv", ['experiment.py', "train", "--root", str(self.root), "--arm", "teacher", "--run"]), \
                patch.object(experiment, "freeze", return_value=frozen), \
                patch.object(experiment, "readiness"), \
                patch.object(experiment.subprocess, "run") as launch, \
                redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            experiment.main()
        launch.assert_not_called()
        self.assertTrue(lock.exists())
        self.assertEqual(json.loads(lock.read_text())["arm"], "searl")
    def test_launch_owns_and_releases_only_its_global_lock(self):
        frozen = self.frozen()
        experiment.write_new(self.root / "protocol.json", frozen)
        experiment.write_new(self.root / "evocond/runtime_config.json", experiment.runtimes(frozen, self.root)["evocond"])
        def process(*args, **kwargs):
            self.assertEqual(json.loads((self.root / ".launch.lock").read_text())["arm"], "evocond")
            raise RuntimeError("CPU mock launch failure")
        with patch.object(sys, "argv", ['experiment.py', "train", "--root", str(self.root), "--arm", "evocond", "--run"]), \
                patch.object(experiment, "freeze", return_value=frozen), \
                patch.object(experiment, "readiness"), \
                patch.object(experiment, "environment_identity", return_value={"fixture": True}), \
                patch.object(experiment.subprocess, "run", side_effect=process) as launch, \
                redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "CPU mock launch failure"):
            experiment.main()
        launch.assert_called_once()
        self.assertFalse((self.root / ".launch.lock").exists())
    def test_teacher_cannot_reuse_unrelated_split_ids(self):
        identities = {"train": {"a", "b"}, "validation": {"c"}, "test": {"d"}}
        teacher = {"train_ids": ["a"], "calibration_ids": ["b"], "validation_ids": ["c"], "test_ids": ["d"]}
        experiment.validate_teacher_splits(teacher, teacher, identities)
        for key, value in (("train_ids", ["c"]), ("validation_ids", ["d"]), ("test_ids", ["unknown"])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                experiment.validate_teacher_splits({**teacher, key: value}, teacher, identities)
    def test_environment_drift_never_launches_and_releases_lock(self):
        frozen = self.frozen()
        experiment.write_new(self.root / "protocol.json", frozen)
        experiment.write_new(self.root / "evocond/runtime_config.json", experiment.runtimes(frozen, self.root)["evocond"])
        experiment.write_new(self.root / "environment.json", {"versions": "original"})
        with patch.object(sys, "argv", ['experiment.py', "train", "--root", str(self.root), "--arm", "evocond", "--run"]), \
                patch.object(experiment, "freeze", return_value=frozen), patch.object(experiment, "readiness"), \
                patch.object(experiment, "environment_identity", return_value={"versions": "changed"}), \
                patch.object(experiment.subprocess, "run") as launch, redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(ValueError, "environment changed"):
            experiment.main()
        launch.assert_not_called()
        self.assertFalse((self.root / ".launch.lock").exists())
    def test_changed_protocol_stops_before_launch(self):
        frozen = self.frozen()
        experiment.write_new(self.root / "protocol.json", frozen)
        changed = deepcopy(frozen)
        changed["fingerprints"]["model"]["sha256"] = "b" * 64
        with patch.object(sys, "argv", ['experiment.py', "train", "--root", str(self.root), "--arm", "evocond", "--run"]), \
                patch.object(experiment, "freeze", return_value=changed), \
                patch.object(experiment.subprocess, "run") as launch, self.assertRaises(ValueError):
            experiment.main()
        launch.assert_not_called()
    def test_test_split_requires_explicit_unseal(self):
        experiment.write_new(self.root / "protocol.json", self.frozen())
        with patch.object(sys, "argv", ['experiment.py', "evaluate", "--root", str(self.root), "--arm", "evocond", "--split", "test"]), \
                redirect_stdout(io.StringIO()), patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            experiment.main()
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
