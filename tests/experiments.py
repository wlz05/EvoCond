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
            "search_url": "http://localhost:9000/retrieve", "teacher": {
                "value_contracts": True, "teacher_path": str(self.root / "teacher.json"),
                "alignment_path": str(self.root / "alignment.json"), "encoder_id": "c" * 64,
                "teacher_policy_mode": "exact_policy"},
            "datasets": {s: str(self.root / (s+".parquet")) for s in ("train", "validation", "test")},
            "budget": experiment.settings({})},
            "fingerprints": {"model": {"sha256": "a" * 64}, "official_source": {"sha256": "b" * 64}}}
    def checkpoint(self, arm="evocond", world_size=8):
        folder = self.root / arm / "checkpoints" / "global_step_5"
        (folder / "actor").mkdir(parents=True)
        (folder/'actor/fsdp_config.json').write_text(json.dumps(dict(FSDP_version=1, world_size=world_size)), encoding='utf-8')
        for prefix in ('model', 'optim', 'extra_state'):
            for rank in range(world_size):
                (folder/f'actor/{prefix}_world_size_{world_size}_rank_{rank}.pt').write_bytes(b'CPU_SHARD_FIXTURE')
        for name in ("data.pt", "method_runtime/runtime.json", "method_runtime/graphrag.json", "method_runtime/subtasks.graphml"):
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("CPU_FIXTURE", encoding="utf-8")
        runtime = experiment.runtimes(self.frozen(), self.root)[arm]
        state = dict(format=experiment.RUNTIME_FORMAT, arm=arm, run_id=runtime['run_id'],
            policy_step=5, config_hash=experiment.canonical_hash(runtime))
        (folder/'method_runtime/runtime.json').write_text(json.dumps(state), encoding='utf-8')
        (folder/'method_runtime/graphrag.json').write_text(json.dumps(dict(execution_history={},
            communities={}, community_summaries={}, global_summary='')), encoding='utf-8')
        (folder/'method_runtime/subtasks.graphml').write_text(
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"><graph edgedefault="directed"/></graphml>',
            encoding='utf-8')
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
        self.assertNotIn('maximum_windows', config['contextual'])
        self.assertNotIn('teacher', config)
    def test_method_fingerprint_excludes_embedded_upstream_and_runs(self):
        (self.root/'main.py').write_text('value = 1', encoding='utf-8')
        expected = experiment.directory_digest(self.root)
        folder = self.root/'baselines/SEARL-official'
        folder.mkdir(parents=True)
        (folder/'main.py').write_text('value = 2', encoding='utf-8')
        self.assertEqual(experiment.directory_digest(self.root), expected)

    def test_method_fingerprint_ignores_disposable_test_dependencies(self):
        (self.root/'main.py').write_text('value = 1', encoding='utf-8')
        expected = experiment.directory_digest(self.root)
        folder = self.root/'.test-tmp/arrow'
        folder.mkdir(parents=True)
        (folder/'temporary.py').write_text('value = 2', encoding='utf-8')
        self.assertEqual(experiment.directory_digest(self.root), expected)

    def test_encoder_inside_a_parent_cache_directory_is_valid(self):
        folder = self.root/'.cache/encoder'
        folder.mkdir(parents=True)
        (folder/'model.safetensors').write_bytes(b'CPU_ENCODER_FIXTURE')
        self.assertEqual(len(experiment.directory_digest_encoder(folder)), 64)
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

    def test_launch_pins_native_roots_and_prompt_mode_from_dirty_shell(self):
        frozen = self.frozen()
        with patch.dict(experiment.os.environ, {'RL_FACTORY_ROOT': 'other-project', 'SEARL_OFFICIAL_ROOT': 'other-source',
                'DYNAMIC_PROMPT': 'True', 'PYTHONPATH': 'unrelated-source'}):
            command, environment = experiment.launch_spec(frozen, self.root, 'evocond', attempt='cpu')
        self.assertEqual(environment['RL_FACTORY_ROOT'], frozen['config']['official_root'])
        self.assertEqual(environment['SEARL_OFFICIAL_ROOT'], frozen['config']['official_root'])
        self.assertEqual(environment['DYNAMIC_PROMPT'], 'False')
        self.assertNotIn('unrelated-source', environment['PYTHONPATH'])
        self.assertIn('++actor_rollout_ref.env.dynamic_prompt=false', command)
    def test_native_entry_receives_default_and_custom_training_limits(self):
        defaults = experiment.settings({})
        expected = dict(gpus=8, batch_size=384, rollouts=8, epochs=1, max_prompt=4096,
            max_response=2048, max_context=32768, max_turns=6, learning_rate=1e-6,
            temperature=.7, top_p=.98, repetition_penalty=1.05)
        self.assertEqual({key: defaults[key] for key in expected}, expected)
        changes = dict(batch_size=192, rollouts=4, epochs=2, max_prompt=8192, max_response=1536,
            max_turns=4, learning_rate=2e-6, temperature=.6, top_p=.9, repetition_penalty=1.1)
        mapping = {'data.train_batch_size': 'batch_size', 'data.max_prompt_length': 'max_prompt',
            'data.max_response_length': 'max_response', 'actor_rollout_ref.rollout.max_tokens': 'max_response',
            'actor_rollout_ref.rollout.max_model_len': 'max_context', 'actor_rollout_ref.rollout.n': 'rollouts',
            'actor_rollout_ref.rollout.max_turns': 'max_turns', 'actor_rollout_ref.env.max_turns': 'max_turns',
            'actor_rollout_ref.actor.optim.lr': 'learning_rate', 'actor_rollout_ref.rollout.temperature': 'temperature',
            'actor_rollout_ref.rollout.top_p': 'top_p', 'actor_rollout_ref.rollout.repetition_penalty': 'repetition_penalty',
            'trainer.total_epochs': 'epochs', 'trainer.n_gpus_per_node': 'gpus'}
        for overrides in ({}, changes):
            for arm in experiment.ARMS:
                with self.subTest(overrides=overrides, arm=arm):
                    frozen = self.frozen()
                    budget = frozen['config']['budget'] = experiment.settings(overrides)
                    with patch.dict(experiment.os.environ, {'MAX_PROMPT_LENGTH': '1', 'MAX_RESPONSE_LENGTH': '2',
                            'SEARL_GPUS_PER_NODE': '1'}):
                        command, env = experiment.launch_spec(frozen, self.root, arm)
                    self.assertEqual(command[:2], ['bash', 'main_searl.sh'])
                    pairs = [item[2:].split('=', 1) for item in command[2:]]
                    params = {key: json.loads(value) for key, value in pairs}
                    self.assertEqual(len(params), len(pairs))
                    for key, option in mapping.items():
                        self.assertEqual(params[key], budget[option], key)
                    self.assertEqual(params['trainer.nnodes'], 1)
                    tokens = budget['max_turns']*(budget['max_prompt']+budget['max_response'])
                    for key in ('actor_rollout_ref.actor.ppo_max_token_len_per_gpu',
                            'actor_rollout_ref.ref.log_prob_max_token_len_per_gpu',
                            'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu'):
                        self.assertEqual(params[key], tokens)
                    self.assertIs(params['actor_rollout_ref.env.enable_thinking'], False)
                    self.assertEqual(env['MAX_PROMPT_LENGTH'], str(budget['max_prompt']))
                    self.assertEqual(env['MAX_RESPONSE_LENGTH'], str(budget['max_response']))
                    self.assertEqual(env['SEARL_GPUS_PER_NODE'], '8')
                    for key in ('actor_rollout_ref.rollout.tensor_model_parallel_size',
                            'actor_rollout_ref.rollout.gpu_memory_utilization',
                            'actor_rollout_ref.actor.ppo_mini_batch_size',
                            'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu',
                            'actor_rollout_ref.actor.fsdp_config.param_offload'):
                        self.assertNotIn(key, params)
    def test_evaluation_uses_the_same_native_length_limits(self):
        frozen = self.frozen()
        checkpoint = self.checkpoint()
        command, env = experiment.launch_spec(frozen, self.root, 'evocond', stage='evaluate', resume=checkpoint)
        self.assertIn('++data.max_prompt_length=4096', command)
        self.assertIn('++data.max_response_length=2048', command)
        self.assertEqual((env['MAX_PROMPT_LENGTH'], env['MAX_RESPONSE_LENGTH']), ('4096', '2048'))
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
    def test_checkpoint_requires_every_rank_and_all_three_state_families(self):
        folder = self.checkpoint()
        for prefix in ('model', 'optim', 'extra_state'):
            path = folder/f'actor/{prefix}_world_size_8_rank_7.pt'
            original = path.read_bytes()
            with self.subTest(prefix=prefix):
                path.unlink()
                with self.assertRaisesRegex(ValueError, 'shards'):
                    experiment.checkpoint(self.root, 'evocond', folder)
                path.write_bytes(b'')
                with self.assertRaisesRegex(ValueError, 'shards'):
                    experiment.checkpoint(self.root, 'evocond', folder)
                path.write_bytes(original)
        extra = folder/'actor/model_world_size_4_rank_0.pt'
        extra.write_bytes(b'OTHER_WORLD_SIZE')
        with self.assertRaisesRegex(ValueError, 'mixed'):
            experiment.checkpoint(self.root, 'evocond', folder)
        self.assertEqual(extra.read_bytes(), b'OTHER_WORLD_SIZE')
    def test_checkpoint_metadata_and_empty_paired_files_fail_before_launch(self):
        folder = self.checkpoint()
        metadata = folder/'actor/fsdp_config.json'
        original = metadata.read_bytes()
        for value in ({}, [], {'FSDP_version': True, 'world_size': 8},
                      {'FSDP_version': 1, 'world_size': True}, {'FSDP_version': 1, 'world_size': 0}):
            metadata.write_text(json.dumps(value), encoding='utf-8')
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'metadata'):
                experiment.checkpoint(self.root, 'evocond', folder)
        metadata.write_bytes(original)
        for name in ('data.pt', 'method_runtime/graphrag.json', 'method_runtime/subtasks.graphml'):
            target = folder/name
            payload = target.read_bytes()
            target.write_bytes(b'')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'missing or invalid'):
                experiment.launch_spec(self.frozen(), self.root, 'evocond', resume=folder)
            target.write_bytes(payload)
    def test_checkpoint_rejects_nonempty_corrupt_graph_cache_before_launch(self):
        folder = self.checkpoint()
        cache = folder/'method_runtime/graphrag.json'
        original = cache.read_bytes()
        for text in ('{"execution_history":', '[]', '{}',
                json.dumps(dict(execution_history={}, communities={'c': [1]},
                    community_summaries={}, global_summary=''))):
            cache.write_text(text, encoding='utf-8')
            with self.subTest(text=text), self.assertRaises(ValueError):
                experiment.launch_spec(self.frozen(), self.root, 'evocond', resume=folder)
        cache.write_bytes(original)
        self.assertEqual(experiment.checkpoint(self.root, 'evocond', folder), str(folder))

    def test_checkpoint_rejects_truncated_and_inconsistent_graphml(self):
        folder = self.checkpoint()
        graph = folder/'method_runtime/subtasks.graphml'
        original = graph.read_bytes()
        bodies = ('<graph', '<graph edgedefault="undirected"/>', '<graph edgedefault="directed">'
            '<node id="a"/><node id="a"/></graph>', '<graph edgedefault="directed">'
            '<node id="a"/><edge source="a" target="missing"/></graph>')
        for body in bodies:
            graph.write_text('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">'+body+'</graphml>',
                encoding='utf-8')
            with self.subTest(body=body), self.assertRaisesRegex(ValueError, 'GraphML'):
                experiment.launch_spec(self.frozen(), self.root, 'evocond', resume=folder)
        graph.write_text('<?xml version="1.0" encoding="unsupported-encoding"?><graphml/>', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'GraphML'):
            experiment.checkpoint(self.root, 'evocond', folder)
        graph.write_bytes(original)
        self.assertEqual(experiment.checkpoint(self.root, 'evocond', folder), str(folder))

    def test_checkpoint_world_size_must_match_the_frozen_allocation(self):
        folder = self.checkpoint(world_size=4)
        self.assertEqual(experiment.checkpoint(self.root, 'evocond', folder), str(folder))
        with self.assertRaisesRegex(ValueError, 'GPU allocation'):
            experiment.launch_spec(self.frozen(), self.root, 'evocond', resume=folder)
    def test_runtime_checkpoint_step_and_configuration_cannot_be_relabelled(self):
        folder = self.checkpoint()
        target = folder/'method_runtime/runtime.json'
        state = json.loads(target.read_text(encoding='utf-8'))
        for change in ({'policy_step': 4}, {'policy_step': True}, {'arm': 'teacher'},
                       {'run_id': 'other'}, {'config_hash': 'f'*64}):
            target.write_text(json.dumps({**state, **change}), encoding='utf-8')
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'Runtime checkpoint'):
                experiment.launch_spec(self.frozen(), self.root, 'evocond', resume=folder)
        target.write_text(json.dumps(state), encoding='utf-8')
        self.assertEqual(experiment.checkpoint(self.root, 'evocond', folder), str(folder))
    def test_older_training_resume_preserves_later_checkpoints_but_evaluation_is_allowed(self):
        folder = self.checkpoint()
        later = folder.parent/'global_step_10'
        later.mkdir()
        sentinel = later/'incomplete.pt'
        sentinel.write_bytes(b'PRESERVE_EXISTING_BRANCH')
        with self.assertRaisesRegex(ValueError, 'Later checkpoints'):
            experiment.launch_spec(self.frozen(), self.root, 'evocond', resume=folder)
        command, _ = experiment.launch_spec(self.frozen(), self.root, 'evocond', stage='evaluate', resume=folder)
        self.assertIn('++trainer.val_only=true', command)
        self.assertEqual(sentinel.read_bytes(), b'PRESERVE_EXISTING_BRANCH')
    def test_nested_checkpoint_and_unknown_launch_stages_are_rejected(self):
        folder = self.root/'evocond/checkpoints/archive/global_step_5'
        folder.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'belong'):
            experiment.checkpoint(self.root, 'evocond', folder)
        for stage, split in (('typo', 'validation'), ('train', 'unknown')):
            with self.subTest(stage=stage, split=split), self.assertRaisesRegex(ValueError, 'stage'):
                experiment.launch_spec(self.frozen(), self.root, 'evocond', stage=stage, split=split)
    def test_status_does_not_offer_incomplete_checkpoints_as_recovery_candidates(self):
        complete = self.checkpoint()
        partial = complete.parent/'global_step_10'
        (partial/'actor').mkdir(parents=True)
        inventory = experiment.checkpoint_inventory(self.frozen(), self.root, 'evocond')
        self.assertEqual(inventory['checkpoints'], ['global_step_5'])
        self.assertEqual(set(inventory['incomplete_checkpoints']), {'global_step_10'})
        self.assertIn('missing or invalid', inventory['incomplete_checkpoints']['global_step_10'])
        self.assertTrue((partial/'actor').is_dir())
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

    def test_prepare_recovers_missing_configs_after_interrupted_publication(self):
        output, frozen = self.root/'prepared', self.frozen()
        original = experiment.write_new
        def interrupted(path, value):
            if path == output/'evocond/runtime_config.json':
                raise OSError('Interrupted publication')
            original(path, value)
        with patch.object(experiment, 'write_new', side_effect=interrupted), self.assertRaises(OSError):
            experiment.prepare(output, frozen, run=True)
        saved = (output/'protocol.json').read_bytes()
        result = experiment.prepare(output, frozen, run=True)
        self.assertTrue(result['prepared'])
        self.assertFalse(result['reused'])
        self.assertEqual(result['missing'], [])
        self.assertTrue(result['created'])
        self.assertEqual((output/'protocol.json').read_bytes(), saved)
        self.assertTrue(all((output/arm/'runtime_config.json').is_file() for arm in experiment.ARMS))
        self.assertTrue(experiment.prepare(output, frozen, run=True)['reused'])

    def test_prepare_validates_all_runtimes_before_publishing(self):
        output, frozen = self.root/'prepared', self.frozen()
        frozen['config']['teacher'].pop('teacher_path')
        with self.assertRaisesRegex(ValueError, 'fitted artifact'):
            experiment.prepare(output, frozen, run=True)
        self.assertFalse(output.exists())

    def test_prepare_preserves_conflicting_config_and_existing_execution_state(self):
        output, frozen = self.root/'prepared', self.frozen()
        experiment.prepare(output, frozen, run=True)
        config = output/'evocond/runtime_config.json'
        original = config.read_bytes()
        config.write_text('{}', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            experiment.prepare(output, frozen, run=True)
        self.assertEqual(config.read_text(encoding='utf-8'), '{}')
        config.unlink()
        state = config.parent/'runtime'
        state.mkdir()
        with self.assertRaisesRegex(FileExistsError, 'execution state'):
            experiment.prepare(output, frozen, run=True)
        self.assertFalse(config.exists())
        self.assertTrue(state.is_dir())
        config.write_bytes(original)

    def test_prepare_refuses_file_root_and_unowned_nonempty_directory(self):
        output, frozen = self.root/'prepared', self.frozen()
        output.write_text('preserve', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'directory'):
            experiment.prepare(output, frozen, run=True)
        self.assertEqual(output.read_text(encoding='utf-8'), 'preserve')
        output.unlink()
        output.mkdir()
        (output/'unrelated').write_text('preserve', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            experiment.prepare(output, frozen, run=True)
        self.assertFalse((output/'protocol.json').exists())
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
class AcceleratorContractTests(unittest.TestCase):
    def devices(self):
        return [dict(name='NVIDIA H100 80GB HBM3', memory=80*1024**3) for _ in range(8)]
    def test_eight_full_memory_h100_devices_match_the_study(self):
        devices = self.devices()
        checked = experiment.validate_accelerators(devices, 8, h100=True)
        self.assertEqual(checked, devices)
        checked[0]['name'] = 'changed'
        self.assertEqual(devices[0]['name'], 'NVIDIA H100 80GB HBM3')
    def test_study_rejects_partial_or_extra_visible_allocations(self):
        for count, requested in ((7, 8), (9, 8), (6, 6)):
            devices = (self.devices()+self.devices())[:count]
            with self.subTest(count=count, requested=requested), self.assertRaises(ValueError):
                experiment.validate_accelerators(devices, requested, h100=True)
    def test_study_rejects_mixed_models_and_reduced_memory(self):
        for update in (dict(name='NVIDIA H200'), dict(name='NVIDIA RTX 4090'),
                       dict(name='NVIDIA H100 MIG 1g.10gb'), dict(memory=10*1024**3)):
            devices = self.devices()
            devices[3].update(update)
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, 'full-memory H100'):
                experiment.validate_accelerators(devices, 8, h100=True)
    def test_hardware_metadata_requires_exact_types(self):
        for update in (dict(name=None), dict(name=''), dict(memory=True), dict(memory='80000000000'), dict(memory=0)):
            devices = self.devices()
            devices[0].update(update)
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, 'inventory'):
                experiment.validate_accelerators(devices, 8, h100=True)
        for gpus, flag in ((True, True), (0, True), (8, 1)):
            with self.subTest(gpus=gpus, flag=flag), self.assertRaises(ValueError):
                experiment.validate_accelerators(self.devices(), gpus, h100=flag)
    def test_legacy_allocations_keep_their_declared_gpu_model(self):
        devices = [dict(name='NVIDIA RTX 4090', memory=24*1024**3)]
        self.assertEqual(experiment.validate_accelerators(devices, 1), devices)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
