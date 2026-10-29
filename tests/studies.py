# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import experiment
import run
import study
import judge
import summary
from control import windows

ROOT = Path(__file__).resolve().parents[1]

class StudyTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT/'configs/contextual.json').read_text(encoding='utf-8'))

    def test_main_preserves_native_graph_and_names_cadence_control(self):
        values = study.matrix(self.config(), 'main')
        self.assertEqual(len(values), 12)
        for seed in study.SEEDS:
            native, pace, full = (values[f'{name}/seed{seed}'] for name in ('searl', 'pace', 'evocond'))
            self.assertNotIn('contextual', native)
            self.assertNotIn('contextual', pace)
            self.assertEqual(full['contextual']['variant'], 'full')
            self.assertEqual([value['design']['actor_batches'] for value in (native, pace, full)], [1, 2, 2])
            self.assertEqual(len({json.dumps(value['datasets'], sort_keys=True) for value in (native, pace, full)}), 1)
            self.assertEqual({value['model'] for value in (native, pace, full)}, {'assets/models/Qwen3-4B'})
        study.inventory(values)

    def test_ablation_changes_only_its_registered_recipe(self):
        values = study.matrix(self.config(), 'ablation', [42])
        baseline = values['evocond/seed42']['contextual']
        for name, changes in study.CHANGES.items():
            key = name+'/seed42'
            if key not in values:
                continue
            expected = {**baseline, **changes}
            self.assertEqual(values[key]['contextual'], expected)
        self.assertEqual(set(values), {name+'/seed42' for name in study.SUITES['ablation']})

    def test_relation_and_adoption_controls_form_a_matched_factorial(self):
        names = ('pooled', 'global', 'fixed', 'evocond')
        values = study.matrix(self.config(), 'ablation', conditions=names)
        self.assertEqual(len(values), 4*len(study.SEEDS))
        study.inventory(values)
        for seed in study.SEEDS:
            options = [values[f'{name}/seed{seed}']['contextual'] for name in names]
            self.assertEqual([(row['variant'], row['fixed_alpha']) for row in options],
                [('global', .5), ('global', None), ('full', .5), ('full', None)])
            # Both axes preserve the remaining execution settings.
            shared = [{key: value for key, value in row.items() if key not in ('variant', 'fixed_alpha')}
                for row in options]
            self.assertTrue(all(row == shared[0] for row in shared))

    def test_pooled_control_cannot_silently_learn_or_retain_conditions(self):
        config = study.matrix(self.config(), 'ablation', [42], ['pooled'])['pooled/seed42']
        for changes in ({'variant': 'full'}, {'fixed_alpha': None}):
            altered = deepcopy(config)
            altered['contextual'].update(changes)
            with self.assertRaisesRegex(ValueError, 'ablation factor'):
                study.setup(altered)

    def test_false_native_labels_and_duplicate_seeds_are_rejected(self):
        value = study.matrix(self.config(), seeds=[42])['searl/seed42']['design']
        with self.assertRaises(ValueError):
            study.validate({**value, 'native': False})
        for seeds in ([42, 42], [True], []):
            with self.assertRaises(ValueError):
                study.matrix(self.config(), seeds=seeds)

    def test_matrix_rejects_unmatched_model_and_never_overwrites(self):
        values = study.matrix(self.config(), seeds=[42])
        altered = deepcopy(values)
        altered['searl/seed42']['model'] = 'assets/models/other'
        with self.assertRaises(ValueError):
            study.inventory(altered)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            study.write(root, values)
            study.write(root, values)
            (root/'searl/seed42.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                study.write(root, values)
            self.assertEqual((root/'searl/seed42.json').read_text(), '{}')

    def test_tail_layout_keeps_all_ten_thousand_questions(self):
        for count in (1, 2):
            layout = windows.layout(10000, 256, count, tail='keep')
            self.assertEqual(layout['batches_per_epoch'], 40)
            self.assertEqual(layout['tail_questions'], 16)
            self.assertEqual(sum(layout['window_questions']), 10000)
            self.assertEqual(layout['actor_updates_per_epoch'], 40//count)
        self.assertEqual(windows.layout(10000, 256, 2, tail='keep')['window_questions'][-1], 272)

    def test_short_final_window_has_an_explicit_epoch_barrier(self):
        self.assertEqual(windows.layout(9, 4, 2, tail='keep')['window_questions'], [8, 1])
        self.assertFalse(windows.barrier(1, 3, 2))
        self.assertTrue(windows.barrier(2, 3, 2))
        self.assertTrue(windows.barrier(3, 3, 2))
        self.assertFalse(windows.barrier(4, 3, 2))
        self.assertEqual([windows.width(index, 3, 2) for index in range(1, 7)], [2, 2, 1, 2, 2, 1])
        for phase in (-1, None, True):
            with self.assertRaises(ValueError):
                windows.barrier(phase, 3, 2)

    def test_study_native_runtime_does_not_override_retrieval(self):
        from experiments import ExperimentTests
        fixture = ExperimentTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        frozen = fixture.frozen()
        design = study.matrix(self.config(), seeds=[42])['searl/seed42']['design']
        frozen['config'].pop('teacher')
        frozen['config']['design'] = design
        frozen['schedule'] = windows.layout(10000, 256, 1, tail='keep')
        runtime = experiment.runtimes(frozen, fixture.root)
        self.assertEqual(set(runtime), {'searl'})
        self.assertNotIn('contextual', runtime['searl'])
        self.assertEqual(runtime['searl']['actor_batches'], 1)

    def test_readme_runner_evaluation_requires_checkpoint_before_launch(self):
        with patch.object(run.subprocess, 'run') as launch, redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            run.main(['evaluate', '--conditions', 'evocond', '--seeds', '42'])
        launch.assert_not_called()

    def test_runner_dry_run_is_independent_of_calling_directory(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(run.subprocess, 'run') as launch:
            with patch('os.getcwd', return_value=folder), redirect_stdout(io.StringIO()) as output:
                run.main(['train', '--conditions', 'evocond', '--seeds', '42'])
            command = json.loads(output.getvalue())['command']
            self.assertEqual(Path(command[2]), ROOT/'experiment.py')
            self.assertIn('--workspace', command)
            launch.assert_not_called()

    def test_initial_actor_architecture_is_not_inferred_from_folder_name(self):
        config = dict(model_type='qwen3', hidden_size=2560, intermediate_size=9728,
            num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8, vocab_size=151936)
        study.actor(config)
        with self.assertRaises(ValueError):
            study.actor({**config, 'hidden_size': 4096})

    def test_shared_gpu_allocation_blocks_other_study_roots(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with experiment.allocation(root/'first', root, {'job': 'first'}, shared=True):
                with self.assertRaises(FileExistsError):
                    with experiment.allocation(root/'second', root, {'job': 'second'}, shared=True):
                        self.fail('A second eight-GPU job acquired the same allocation')
                self.assertFalse((root/'second/.launch.lock').exists())
            self.assertFalse((root/'runs/.gpu.lock').exists())

    def test_checkpoint_map_requires_unique_explicit_relative_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoints.json'
            path.write_text(json.dumps(dict(format='evocond_checkpoints_v1',
                checkpoints={'evocond/seed42': 'runs/main/step'})), encoding='utf-8')
            self.assertEqual(run.checkpoints(path), {'evocond/seed42': 'runs/main/step'})
            path.write_text(json.dumps(dict(format='evocond_checkpoints_v1',
                checkpoints={'evocond/seed42': '../outside'})), encoding='utf-8')
            with self.assertRaises(ValueError):
                run.checkpoints(path)

    def test_study_judge_supports_current_labels_and_counts_missing_zero(self):
        from tests.judging import JudgeTests
        fixture = JudgeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        manifest = fixture.setup_manifest(inputs=[])
        manifest.update(format=judge.STUDY_FORMAT, training_seed=42)
        manifest['arms'] = dict(searl=manifest['arms']['searl'], pace=manifest['arms']['evocond'], evocond=manifest['arms']['teacher'])
        fixture.save_manifest(manifest)
        protocol, rows = judge.preflight(fixture.manifest_path, fixture.author, fixture.model)
        self.assertEqual(set(rows), {'searl', 'pace', 'evocond'})
        for name in rows:
            self.assertEqual(len(rows[name]), 1)
            self.assertEqual(rows[name][0]['output'], '')
            self.assertEqual(protocol['arms'][name]['missing_scored_zero'], 1)
        self.assertEqual(protocol['training_seed'], 42)
        with self.assertRaises(ValueError):
            judge.arm_names({**manifest, 'samples_per_question': 8})

    def result_manifest(self, root):
        reports = []
        for seed in study.SEEDS:
            folder = root/str(seed)
            folder.mkdir()
            protocol = dict(split='test', training_seed=seed, samples_per_question=1,
                dataset_name='fixture', dataset={'sha256': 'a'*64}, reference_count=2,
                judge_model={'identity': 'fixed'}, parser='fixed', generation={'temperature': 0.},
                arms={'searl': {}, 'evocond': {}})
            (folder/'judge_protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
            result = dict(complete=True, protocol_sha256=judge.digest(protocol),
                arms={'searl': dict(n=2, correct=1, acc=.5), 'evocond': dict(n=2, correct=2, acc=1.)})
            path = folder/'judge_report.json'
            path.write_text(json.dumps(result), encoding='utf-8')
            reports.append(dict(benchmark='fixture', seed=seed, path=path.relative_to(root).as_posix(), sha256=judge.file_hash(path)))
        value = dict(format='evocond_results_v1', benchmarks=['fixture'], conditions=['searl', 'evocond'], seeds=list(study.SEEDS), reports=reports)
        path = root/'results.json'
        path.write_text(json.dumps(value), encoding='utf-8')
        return path, value

    def test_summary_reports_training_seed_variation_and_rejects_partial_grid(self):
        with tempfile.TemporaryDirectory() as folder:
            path, value = self.result_manifest(Path(folder))
            result = summary.aggregate(path)
            self.assertEqual(result['conditions']['evocond']['macro'], dict(mean=1., sample_std=0.))
            self.assertNotIn('conditional_adoption', result)
            value['reports'].pop()
            path.write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Missing'):
                summary.aggregate(path)

    def test_summary_computes_crossed_effects_from_complete_seed_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, value = self.result_manifest(root)
            value['conditions'] = ['pooled', 'global', 'fixed', 'evocond']
            for index, item in enumerate(value['reports']):
                target = root/item['path']
                protocol_path = target.parent/'judge_protocol.json'
                protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
                protocol.update(reference_count=20, arms={name: {} for name in value['conditions']})
                protocol_path.write_text(json.dumps(protocol), encoding='utf-8')
                counts = (2, 4, 6, 10+index)
                report = dict(complete=True, protocol_sha256=judge.digest(protocol), arms={
                    name: dict(n=20, correct=count, acc=count/20) for name, count in zip(value['conditions'], counts)})
                target.write_text(json.dumps(report), encoding='utf-8')
                item['sha256'] = judge.file_hash(target)
            path.write_text(json.dumps(value), encoding='utf-8')
            result = summary.aggregate(path)['conditional_adoption']
            self.assertEqual(result['unit'], 'fraction')
            effects = result['benchmarks']['fixture']
            self.assertAlmostEqual(effects['condition_at_fixed']['mean'], .2)
            self.assertAlmostEqual(effects['adoption_at_pooled']['mean'], .1)
            self.assertAlmostEqual(effects['interaction']['mean'], .175)
            self.assertAlmostEqual(effects['interaction']['per_seed']['42'], .1)
            self.assertAlmostEqual(effects['interaction']['per_seed']['3407'], .25)
            self.assertGreater(effects['interaction']['sample_std'], 0.)
            self.assertEqual(result['comparison'], 'independently_trained_conditions')
            self.assertAlmostEqual(effects['condition_average']['mean'], .2875)
            self.assertAlmostEqual(effects['adoption_average']['mean'], .1875)
            for name, coefficients in result['coefficients'].items():
                self.assertAlmostEqual(sum(coefficients.values()), 0., msg=name)

    def test_system_summary_separates_actor_cadence_from_matched_method(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, value = self.result_manifest(root)
            value['conditions'] = ['searl', 'pace', 'evocond']
            for index, item in enumerate(value['reports']):
                target = root/item['path']
                protocol_path = target.parent/'judge_protocol.json'
                protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
                protocol.update(reference_count=20, arms={name: {} for name in value['conditions']})
                protocol_path.write_text(json.dumps(protocol), encoding='utf-8')
                counts = (5, 7+index, 12)
                report = dict(complete=True, protocol_sha256=judge.digest(protocol), arms={
                    name: dict(n=20, correct=count, acc=count/20) for name, count in zip(value['conditions'], counts)})
                target.write_text(json.dumps(report), encoding='utf-8')
                item['sha256'] = judge.file_hash(target)
            path.write_text(json.dumps(value), encoding='utf-8')
            result = summary.aggregate(path)['system_comparison']
            effects = result['benchmarks']['fixture']
            self.assertEqual(result['comparison'], 'independently_trained_systems')
            self.assertAlmostEqual(effects['total']['mean'], .35)
            for seed in map(str, study.SEEDS):
                self.assertAlmostEqual(effects['cadence']['per_seed'][seed]+effects['matched_cadence']['per_seed'][seed],
                    effects['total']['per_seed'][seed])
            value['conditions'].remove('pace')
            for item in value['reports']:
                target = root/item['path']
                protocol_path = target.parent/'judge_protocol.json'
                protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
                protocol['arms'].pop('pace')
                protocol_path.write_text(json.dumps(protocol), encoding='utf-8')
                report = json.loads(target.read_text(encoding='utf-8'))
                report['arms'].pop('pace')
                report['protocol_sha256'] = judge.digest(protocol)
                target.write_text(json.dumps(report), encoding='utf-8')
                item['sha256'] = judge.file_hash(target)
            path.write_text(json.dumps(value), encoding='utf-8')
            self.assertNotIn('system_comparison', summary.aggregate(path))

    def test_summary_does_not_relabel_one_report_as_multiple_training_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            path, value = self.result_manifest(Path(folder))
            value['reports'][1].update(path=value['reports'][0]['path'], sha256=value['reports'][0]['sha256'])
            path.write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaises(ValueError):
                summary.aggregate(path)

if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
