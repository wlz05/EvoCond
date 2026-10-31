# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
import importlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import experiment
import run
import study
import summary
from judge import digest
from control import bandit, dynamic, stage, windows
from control.contracts import relative
from tests.bandits import records
from tests.fatigues import zero
from tests import stages as stage_fixtures, studies as study_fixtures, windows as window_fixtures
from tests.windows import batch, concatenate, tensor
from namespace import Initializers, install
from run import SEEDS, recipes
from source import RECEIPT, materialize, resources, reviewed
from experiment import directory_digest

class PublicationTests(unittest.TestCase):
    def test_runtime_resources_restore_exact_bytes_without_clobbering_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = materialize(root)
            self.assertGreater(len(report['missing']), 0)
            self.assertEqual(list(root.iterdir()), [])
            created = materialize(root, run=True)
            self.assertEqual(created['created'], len(report['missing']))
            for name, payload in resources().items():
                self.assertEqual((root/name).read_bytes(), payload, name)
            starter = json.loads((root/'configs/contextual.json').read_text(encoding='utf-8'))
            self.assertNotIn('maximum_windows', starter['contextual'])
            config = root/'configs/contextual.json'
            config.write_bytes(b'custom settings\r\n')
            self.assertEqual(materialize(root, run=True)['created'], 0)
            self.assertEqual(config.read_bytes(), b'custom settings\r\n')

    def test_runtime_resource_paths_are_checked_before_any_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch('source.resources', return_value={'valid.txt': b'ok', '../outside': b'bad'}):
                with self.assertRaises(ValueError):
                    materialize(root, run=True)
            self.assertEqual(list(root.iterdir()), [])

    def receipt(self, root, *, newline=b'\n'):
        baseline = root/Path(RECEIPT).parent
        baseline.mkdir(parents=True)
        (baseline/'value.py').write_bytes(b'number = 8\r\n')
        before = json.dumps(dict(format='evocond_vendored_source_v1', upstream_commit='pin',
            files={'value.py': sha256(b'number = 7\n').hexdigest()})).encode()+newline
        after = json.dumps(dict(format='evocond_vendored_source_v1', upstream_commit='pin',
            files={'value.py': sha256(b'number = 8\n').hexdigest(),
                'settings.yaml': sha256(b'number: 8\n').hexdigest()})).encode()+b'\n'
        (root/RECEIPT).write_bytes(before)
        values = {RECEIPT: after, (Path(RECEIPT).parent/'settings.yaml').as_posix(): b'number: 8\n'}
        return baseline, before, after, values

    def test_known_receipt_upgrade_checks_source_and_preserves_user_settings(self):
        for newline in (b'\n', b'\r\n'):
            with self.subTest(newline=newline), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                baseline, before, after, values = self.receipt(root, newline=newline)
                (root/'settings.json').write_bytes(b'custom settings\n')
                accepted = {sha256(before.replace(b'\r\n', b'\n')).hexdigest()}
                with patch('source.resources', return_value=values), patch('source.PREVIOUS', accepted):
                    plan = materialize(root)
                    self.assertEqual(plan['refresh'], [RECEIPT])
                    self.assertEqual(plan['updated'], 0)
                    self.assertEqual((root/RECEIPT).read_bytes(), before)
                    self.assertFalse((baseline/'settings.yaml').exists())
                    result = materialize(root, run=True)
                    self.assertEqual(result['updated'], 1)
                    self.assertEqual(result['created'], 1)
                    self.assertEqual((root/RECEIPT).read_bytes(), after)
                    self.assertEqual(reviewed(baseline, 'pin'), 'pin')
                    self.assertEqual(materialize(root, run=True)['updated'], 0)
                self.assertEqual((root/'settings.json').read_bytes(), b'custom settings\n')

    def test_receipt_upgrade_rejects_changed_source_before_any_publication(self):
        for existing in (True, False):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                baseline, before, _, values = self.receipt(root)
                if existing:
                    (baseline/'value.py').write_bytes(b'number = 9\n')
                else:
                    values[(Path(RECEIPT).parent/'settings.yaml').as_posix()] = b'number: 9\n'
                with patch('source.resources', return_value=values), patch('source.PREVIOUS', {sha256(before).hexdigest()}):
                    with self.assertRaisesRegex(ValueError, 'Vendored source changed'):
                        materialize(root, run=True)
                self.assertEqual((root/RECEIPT).read_bytes(), before)
                self.assertFalse((baseline/'settings.yaml').exists())

    def test_unknown_receipts_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, before, _, values = self.receipt(root)
            with patch('source.resources', return_value=values), patch('source.PREVIOUS', frozenset()):
                self.assertEqual(materialize(root, run=True)['updated'], 0)
            self.assertEqual((root/RECEIPT).read_bytes(), before)

    def test_receipt_replacement_failure_preserves_original_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, before, _, values = self.receipt(root)
            with patch('source.resources', return_value=values), patch('source.PREVIOUS', {sha256(before).hexdigest()}), \
                    patch('source.os.replace', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):
                    materialize(root, run=True)
            self.assertEqual((root/RECEIPT).read_bytes(), before)
            self.assertEqual(list(baseline.glob('.evocond-*.tmp')), [])
            with patch('source.resources', return_value=values), patch('source.PREVIOUS', {sha256(before).hexdigest()}):
                self.assertEqual(materialize(root, run=True)['updated'], 1)

    def test_receipt_changed_during_publication_is_not_overwritten(self):
        from source import replace
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'source.json'
            target.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed during'):
                replace(target, b'original', b'updated')
            self.assertEqual(target.read_bytes(), b'changed')
            self.assertEqual(list(target.parent.glob('.evocond-*.tmp')), [])

    def test_source_receipt_accepts_only_exact_or_crlf_equivalent_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'value.py').write_bytes(b'number = 7\r\n')
            receipt = dict(format='evocond_vendored_source_v1', upstream_commit='pin',
                files={'value.py': sha256(b'number = 7\n').hexdigest()})
            (root/'source.json').write_text(json.dumps(receipt), encoding='utf-8')
            self.assertEqual(reviewed(root, 'pin'), 'pin')
            (root/'value.py').write_bytes(b'number = 8\r\n')
            with self.assertRaises(ValueError):
                reviewed(root, 'pin')

    def test_eight_gpus_and_all_four_seeds_preserve_partitions(self):
        value = dict(datasets=dict(train='assets/train.parquet'), budget=dict(gpus=4))
        before = deepcopy(value)
        plans = recipes(value)
        self.assertEqual(tuple(seed for seed, _ in plans), (42, 52, 3047, 3407))
        self.assertEqual(SEEDS, (42, 52, 3047, 3407))
        self.assertEqual(value, before)
        for seed, plan in plans:
            self.assertEqual(plan['budget'], dict(gpus=8, seed=seed))
            self.assertEqual(plan['contextual']['seed'], seed)
            self.assertEqual(plan['datasets'], before['datasets'])

    def test_initializer_preserves_exports_and_relative_imports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'fixture').mkdir()
            (root/'fixture/initialize.py').write_text('from .value import number\n', encoding='utf-8')
            (root/'fixture/value.py').write_text('number = 7\n', encoding='utf-8')
            finder = Initializers(root, [dict(module='fixture', path='fixture/initialize.py', package=True)])
            with patch.object(sys, 'path', [str(root), *sys.path]), patch.object(sys, 'meta_path', [finder, *sys.meta_path]):
                try:
                    self.assertEqual(importlib.import_module('fixture').number, 7)
                finally:
                    sys.modules.pop('fixture.value', None)
                    sys.modules.pop('fixture', None)

    def test_initializer_rejects_escape_and_missing_source(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                Initializers(folder, [dict(module='fixture', path='../missing.py', package=True)])

    def test_vendored_source_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payload = b'number = 7\n'
            (root/'value.py').write_bytes(payload)
            receipt = dict(format='evocond_vendored_source_v1', upstream_commit='pin',
                files={'value.py': sha256(payload).hexdigest()})
            (root/'source.json').write_text(json.dumps(receipt), encoding='utf-8')
            self.assertEqual(reviewed(root, 'pin'), 'pin')
            (root/'value.py').write_bytes(b'number = 8\n')
            with self.assertRaises(ValueError):
                reviewed(root, 'pin')

    def test_vendored_source_pin_must_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'source.json').write_text(json.dumps(dict(format='evocond_vendored_source_v1',
                upstream_commit='other', files={'value.py': 'x'})), encoding='utf-8')
            with self.assertRaises(ValueError):
                reviewed(root, 'pin')

    def test_package_manifest_is_frozen_with_method_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'configs').mkdir()
            manifest = root/'configs/packages.json'
            manifest.write_text('{}', encoding='utf-8')
            before = directory_digest(root)
            manifest.write_text('{"entries":[]}', encoding='utf-8')
            self.assertNotEqual(directory_digest(root), before)

    def test_vendored_release_receipt_matches_source(self):
        baseline = Path(__file__).resolve().parents[1]/'baselines/SEARL-official'
        if not (baseline/'source.json').is_file():
            self.skipTest('Source receipt is generated in the public release')
        self.assertEqual(reviewed(baseline, 'b535d6199602bd9d747fc260bd29c77764799358'),
            'b535d6199602bd9d747fc260bd29c77764799358')

def model(dimension=2):
    return dict(format=bandit.TEACHER, dimension=dimension, weights=[[0.0]*dimension for _ in range(3)],
        mean=[0.0]*dimension, scale=[1.0]*dimension, fit_ids=['fit'], fitted_snapshot='fixed')

def heldout():
    rows = records(2)
    for index, row in enumerate(rows):
        row['base_id'] = 'heldout'+str(index)
    return rows

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def results(self):
        return study_fixtures.StudyTests().result_manifest(self.root)

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def test_failed_json_publication_leaves_no_partial_artifact(self):
        target = self.root/'broken.json'
        with self.assertRaises(TypeError):
            experiment.write_new(target, {'bad': object()})
        self.assertFalse(target.exists())

    def test_disk_failure_cleans_temporary_publication_files(self):
        target = self.root/'result.json'
        for operation in ('fsync', 'link'):
            with patch('control.contracts.os.'+operation, side_effect=OSError('disk failure')):
                with self.subTest(operation=operation), self.assertRaises(OSError):
                    experiment.write_new(target, {'complete': True})
            self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_lock_identity_cannot_leave_a_stale_lock(self):
        with self.assertRaises(TypeError):
            with experiment.allocation(self.root/'job', self.root, {'bad': object()}, shared=True):
                self.fail('Invalid identity acquired the allocation')
        self.assertFalse((self.root/'runs/.gpu.lock').exists())
        self.assertFalse((self.root/'job/.launch.lock').exists())

    def test_lock_cleanup_uses_the_published_identity_snapshot(self):
        owner = {'attempt': 'original'}
        with experiment.allocation(self.root/'job', self.root, owner, shared=True):
            owner['attempt'] = 'changed'
        self.assertFalse((self.root/'runs/.gpu.lock').exists())
        self.assertFalse((self.root/'job/.launch.lock').exists())

    def test_study_labels_cannot_escape_the_matrix_or_change_the_seed(self):
        values = study.matrix(study_fixtures.StudyTests().config(), seeds=[42], conditions=['evocond'])
        config = values['evocond/seed42']
        for name in ('../escaped', 'evocond/seed52'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                study.write(self.root/'matrix', {name: deepcopy(config)})
        self.assertFalse((self.root/'escaped.json').exists())
        for changes in ({'seed': True}, {'gpus': 4}):
            altered = deepcopy(values)
            altered['evocond/seed42']['budget'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                study.inventory(altered)

    def test_study_resume_reads_bom_but_rejects_duplicate_keys(self):
        values = study.matrix(study_fixtures.StudyTests().config(), seeds=[42], conditions=['evocond'])
        study.write(self.root, values)
        target = self.root/'evocond/seed42.json'
        original = target.read_text(encoding='utf-8')
        target.write_text('\ufeff'+original, encoding='utf-8')
        study.write(self.root, values)
        target.write_text('{"run_id":"tampered",'+original.lstrip()[1:], encoding='utf-8')
        with self.assertRaises(ValueError):
            study.write(self.root, values)

    def test_invalid_study_numbers_are_rejected_before_any_file_is_created(self):
        values = study.matrix(study_fixtures.StudyTests().config(), seeds=[42], conditions=['evocond'])
        values['evocond/seed42']['contextual']['penalty'] = float('nan')
        target = self.root/'matrix'
        with self.assertRaises(ValueError):
            study.write(target, values)
        self.assertFalse(target.exists())

    def test_interrupted_matrix_publication_resumes_without_partial_json(self):
        values = study.matrix(study_fixtures.StudyTests().config(), seeds=[42], conditions=['searl', 'pace'])
        target = self.root/'matrix'
        with patch('control.contracts.os.fsync', side_effect=[None, OSError('disk failure')]):
            with self.assertRaises(OSError):
                study.write(target, values)
        self.assertEqual(json.loads((target/'searl/seed42.json').read_text()), values['searl/seed42'])
        self.assertFalse((target/'pace/seed42.json').exists())
        self.assertFalse((target/'matrix.json').exists())
        self.assertEqual(list(target.rglob('*.tmp')), [])
        self.assertEqual(set(study.write(target, values)['jobs']), set(values))
        self.assertEqual(json.loads((target/'matrix.json').read_text())['format'], study.FORMAT)

    def normalized(self, value):
        result = deepcopy(value)
        result['budget'] = experiment.settings(result['budget'])
        for key in ('official_root', 'model', 'encoder', 'isolation_config', 'isolation_receipt'):
            result[key] = str((self.root/result[key]).resolve())
        result['datasets'] = {key: str((self.root/name).resolve()) for key, name in result['datasets'].items()}
        if 'contextual' in result:
            result['contextual'] = stage.settings(result['contextual'])
        return result

    def test_batch_launch_checks_every_frozen_configuration_before_starting(self):
        config = study_fixtures.StudyTests().config()
        self.write(self.root/'config.json', config)
        values = study.matrix(config, seeds=[42])
        for name, value in values.items():
            target = self.root/'runs/main'/name/'protocol.json'
            target.parent.mkdir(parents=True)
            self.write(target, dict(config=self.normalized(value)))
        target = self.root/'runs/main/pace/seed42/protocol.json'
        frozen = json.loads(target.read_text(encoding='utf-8'))
        frozen['config']['budget']['learning_rate'] *= 10
        self.write(target, frozen)
        with patch.object(run, 'HERE', self.root), patch.object(run.subprocess, 'run') as launch, \
                redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            run.main(['train', '--config', 'config.json', '--root', 'runs', '--seeds', '42', '--run'])
        launch.assert_not_called()

    def test_batch_launch_accepts_matching_normalized_configuration(self):
        config = study_fixtures.StudyTests().config()
        (self.root/'config.json').write_text('\ufeff'+json.dumps(config), encoding='utf-8')
        values = study.matrix(config, seeds=[42], conditions=['evocond'])
        target = self.root/'runs/main/evocond/seed42/protocol.json'
        target.parent.mkdir(parents=True)
        self.write(target, dict(config=self.normalized(values['evocond/seed42'])))
        with patch.object(run, 'HERE', self.root), patch.object(run.subprocess, 'run') as launch, redirect_stdout(io.StringIO()):
            run.main(['train', '--config', 'config.json', '--root', 'runs', '--seeds', '42', '--conditions', 'evocond', '--run'])
        launch.assert_called_once()

    def test_prepare_checks_all_settings_before_starting_any_worker(self):
        config = study_fixtures.StudyTests().config()
        self.write(self.root/'config.json', config)
        values = study.matrix(config, seeds=[42])
        study.write(self.root/'runs/main', values)
        target = self.root/'runs/main/pace/seed42.json'
        altered = deepcopy(values['pace/seed42'])
        altered['budget']['learning_rate'] *= 10
        self.write(target, altered)
        with patch.object(run, 'HERE', self.root), patch.object(run.subprocess, 'run') as launch, \
                redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            run.main(['prepare', '--config', 'config.json', '--root', 'runs', '--seeds', '42', '--run'])
        launch.assert_not_called()

    def test_checkpoint_preflight_rejects_wrong_file_kinds(self):
        from tests.experiments import ExperimentTests
        for name in ('actor', 'data.pt', 'method_runtime/runtime.json'):
            fixture = ExperimentTests()
            fixture.setUp()
            self.addCleanup(fixture.doCleanups)
            checkpoint = fixture.checkpoint()
            target = checkpoint/name
            if target.is_dir():
                for shard in target.iterdir():
                    shard.unlink()
                target.rmdir()
                target.write_bytes(b'not a directory')
            else:
                target.unlink()
                target.mkdir()
            with self.subTest(name=name), self.assertRaises(ValueError):
                experiment.checkpoint(fixture.root, 'evocond', checkpoint)

    def test_concatenation_failure_restores_the_original_batches(self):
        batches = [batch([1], [2]), batch([1, 2], [3, 4])]
        before = [deepcopy(row.__dict__) for row in batches]
        with patch.object(windows, 'validate_batches', wraps=windows.validate_batches), self.assertRaises(RuntimeError):
            windows.join(batches, lambda rows: (_ for _ in ()).throw(RuntimeError('concatenation failed')), 0)
        for row, saved in zip(batches, before):
            for key, value in saved['batch'].items():
                self.assertTrue(np.array_equal(row.batch[key], value, equal_nan=True), key)
            for key, value in saved['non_tensor_batch'].items():
                self.assertEqual(row.non_tensor_batch[key].tolist(), value.tolist())

    def test_contextual_probabilities_and_prices_reject_booleans_and_nonfinite_values(self):
        for key in ('epsilon', 'fixed_alpha', 'identity_shrinkage', 'actor_kl_threshold', 'penalty',
                'reference', 'actor_input_price', 'actor_output_price', 'tool_price', 'retrieval_price'):
            for value in (True, float('nan'), float('inf'), '0.2'):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    stage.settings({key: value})

    def test_portable_paths_reject_device_names_and_alternate_streams(self):
        for name in ('outputs/report.json:extra', 'NUL', 'outputs/part.', 'outputs/part '):
            with self.subTest(name=name), self.assertRaises(ValueError):
                relative(self.root, name)

    def test_summary_rejects_malformed_protocol_metadata(self):
        path, manifest = self.results()
        sources = []
        for entry in manifest['reports']:
            target = self.root/entry['path']
            protocol_path = target.parent/'judge_protocol.json'
            sources.append((entry, target, protocol_path, json.loads(protocol_path.read_text(encoding='utf-8')),
                json.loads(target.read_text(encoding='utf-8'))))
        for changes in ({'arms': manifest['conditions']}, {'dataset': {'sha256': False}},
                {'dataset': []}, {'judge_model': {}}, {'parser': ''}, {'generation': []}):
            # Consistent corruption across all seeds must still be rejected.
            for entry, target, protocol_path, original, report in sources:
                protocol = {**deepcopy(original), **changes}
                report['protocol_sha256'] = digest(protocol)
                self.write(protocol_path, protocol)
                self.write(target, report)
                entry['sha256'] = sha256(target.read_bytes()).hexdigest()
            self.write(path, manifest)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                summary.aggregate(path)

    def test_b01_absorbing_cost_receipts_are_checked(self):
        row = zero('q')
        row['cost_receipts'] = [dict(kind='actor', cost=1000.0)]
        with self.assertRaises(ValueError):
            bandit.validate(row)

    def test_b02_absorbing_evaluation_keeps_the_question_denominator(self):
        row = zero('absorbed')
        observed = heldout()[0]
        row['snapshot'] = observed['snapshot']
        report = bandit.evaluate([row, observed], model())
        self.assertEqual(report['n'], 2)
        self.assertIsNotNone(report['ips'])

    def test_b03_evaluation_cannot_mix_frozen_snapshots(self):
        rows = heldout()
        rows[1]['snapshot'] = 'different'
        with self.assertRaises(ValueError):
            bandit.evaluate(rows, model())

    def test_b04_evaluation_cannot_count_one_question_twice(self):
        rows = heldout()
        rows[1]['base_id'] = rows[0]['base_id']
        with self.assertRaises(ValueError):
            bandit.evaluate(rows, model())

    def test_b05_mixture_branches_require_one_feature_space(self):
        value = dict(format=bandit.MIXTURE, delta=.2, previous=model(2), proposal=model(3))
        with self.assertRaises(ValueError):
            bandit.validate_model(value)

    def test_b06_failed_padding_preserves_all_input_tensors(self):
        left, right = batch([1], [2]), batch([1, 2], [3, 4])
        left.batch['unknown'], right.batch['unknown'] = tensor([[1]]), tensor([[1, 2]])
        before = [deepcopy(row.batch) for row in (left, right)]
        with self.assertRaises(ValueError):
            windows.join([left, right], concatenate, 0)
        for row, original in zip((left, right), before):
            for key, value in original.items():
                self.assertTrue(np.array_equal(row.batch[key], value, equal_nan=True), key)

    def test_b07_rejected_schedule_does_not_poison_pending_batches(self):
        pending = []
        with self.assertRaises(ValueError):
            windows.accumulate(pending, 'new', False, 1, tuple)
        self.assertEqual(pending, [])

    def test_b08_bad_checkpoint_cannot_replace_the_live_registry(self):
        fixture = stage_fixtures.StageTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        runtime = fixture.runtime
        before = deepcopy(runtime.snapshot())
        broken = deepcopy(runtime.snapshot())
        broken['registry'] = dynamic.empty_registry()
        with self.assertRaises(ValueError):
            runtime.restore(broken)
        self.assertEqual(runtime.snapshot(), before)

    def test_b09_scoring_source_changes_cannot_be_pooled(self):
        path, manifest = self.results()
        entry = manifest['reports'][0]
        folder = (path.parent/entry['path']).parent
        protocol = json.loads((folder/'judge_protocol.json').read_text())
        protocol['author_source_sha256'] = {'evaluation.py': 'different'}
        self.write(folder/'judge_protocol.json', protocol)
        result = json.loads((folder/'judge_report.json').read_text())
        result['protocol_sha256'] = digest(protocol)
        self.write(folder/'judge_report.json', result)
        entry['sha256'] = sha256((folder/'judge_report.json').read_bytes()).hexdigest()
        self.write(path, manifest)
        with self.assertRaises(ValueError):
            summary.aggregate(path)

    def test_b10_summary_hash_identifies_the_manifest_actually_read(self):
        path, _ = self.results()
        original_bytes = path.read_bytes()
        original_mean = summary.statistics.mean
        def replace(values):
            self.write(path, {'replaced': True})
            return original_mean(values)
        with patch.object(summary.statistics, 'mean', side_effect=replace):
            result = summary.aggregate(path)
        self.assertEqual(result['source_sha256'], sha256(original_bytes).hexdigest())

    def test_b11_relative_report_paths_cannot_escape_the_manifest_folder(self):
        root = self.root/'study'
        root.mkdir()
        path, manifest = study_fixtures.StudyTests().result_manifest(root)
        entry = manifest['reports'][0]
        outside = self.root/'outside'
        outside.mkdir()
        folder = (root/entry['path']).parent
        for name in ('judge_report.json', 'judge_protocol.json'):
            shutil.copyfile(folder/name, outside/name)
        entry['path'] = '../outside/judge_report.json'
        self.write(path, manifest)
        with self.assertRaises(ValueError):
            summary.aggregate(path)

    def test_b12_empty_explicit_checkpoint_map_cannot_become_fresh_training(self):
        path = self.root/'checkpoints.json'
        self.write(path, dict(format='evocond_checkpoints_v1', checkpoints={}))
        with self.assertRaises(ValueError):
            run.checkpoints(path)

    def test_b13_restore_cannot_discard_live_or_unpublished_windows(self):
        for state in ('live', 'completed', 'pending'):
            with self.subTest(state=state):
                fixture = window_fixtures.WindowTests()
                self.addCleanup(fixture.doCleanups)
                group = fixture.runtime()
                runtime = group.runtime
                checkpoint = runtime.snapshot()
                episodes = group.episodes()
                if state != 'live':
                    for episode in episodes:
                        group.execute(episode)
                if state == 'pending':
                    runtime.release_completed()
                contexts = set(runtime.contexts)
                pending = dynamic.digest(runtime.engine.pending_registry)
                with self.assertRaises(ValueError):
                    runtime.snapshot()
                with patch.object(runtime.dispatcher, 'replace_registry', wraps=runtime.dispatcher.replace_registry) as replace:
                    with self.assertRaises(ValueError):
                        runtime.restore(checkpoint)
                    replace.assert_not_called()
                self.assertEqual(set(runtime.contexts), contexts)
                self.assertEqual(dynamic.digest(runtime.engine.pending_registry), pending)

    def test_e01_boolean_costs_are_not_measured_quantities(self):
        with self.assertRaises(ValueError):
            bandit.reward(True, dict(actor=True, tools=0., retrieval=0.), penalty=.1, reference=10.)

    def test_e02_overflowed_total_cost_is_not_silently_clipped(self):
        with self.assertRaises(ValueError):
            bandit.reward(True, dict(actor=1e308, tools=1e308, retrieval=1e308), penalty=.1, reference=10.)

    def test_e03_empty_actual_action_classes_fail_explicitly(self):
        row = records(1)[0]
        row.update(actions=[], probabilities=[], chosen=None, payload_hashes=[])
        with self.assertRaises(ValueError):
            bandit.validate(row)

    def test_e04_absorbing_rows_cannot_hide_payloads_or_features(self):
        row = zero('q')
        row.update(features=[1.0], payload_hashes=['hidden'])
        with self.assertRaises(ValueError):
            bandit.validate(row)

    def test_e05_reward_heads_require_the_declared_constant_intercept(self):
        row = records(1)[0]
        row['features'][0] = 2.0
        with self.assertRaises(ValueError):
            bandit.validate(row)

    def test_e06_nonfinite_optimizer_settings_cannot_produce_nan_models(self):
        with self.assertRaises(ValueError):
            bandit.fit(records(6), steps=1, rate=float('nan'))

    def test_e07_cyclic_teacher_models_fail_before_recursive_execution(self):
        value = dict(format=bandit.MIXTURE, delta=.2, proposal=model())
        value['previous'] = value
        with self.assertRaises(ValueError):
            bandit.validate_model(value)

    def test_e08_response_rewards_must_match_their_own_generated_tokens(self):
        left, right = batch([1], [2, 3]), batch([1], [2, 3, 4])
        left.batch['old_log_probs'] = tensor([[.1]])
        with self.assertRaises(ValueError):
            windows.join([left, right], concatenate, 0)

    def test_e09_summary_rejects_boolean_accuracy_and_fractional_seed_types(self):
        for field in ('accuracy', 'manifest_seed', 'entry_seed', 'protocol_seed'):
            with self.subTest(field=field):
                folder = self.root/field
                folder.mkdir()
                path, manifest = study_fixtures.StudyTests().result_manifest(folder)
                entry = manifest['reports'][0]
                result_path = folder/entry['path']
                value = json.loads(result_path.read_text())
                if field == 'accuracy':
                    value['arms']['evocond']['acc'] = True
                elif field == 'manifest_seed':
                    manifest['seeds'][0] = 42.0
                elif field == 'entry_seed':
                    entry['seed'] = 42.0
                else:
                    protocol_path = result_path.parent/'judge_protocol.json'
                    protocol = json.loads(protocol_path.read_text())
                    protocol['training_seed'] = 42.0
                    self.write(protocol_path, protocol)
                    value['protocol_sha256'] = digest(protocol)
                self.write(result_path, value)
                entry['sha256'] = sha256(result_path.read_bytes()).hexdigest()
                self.write(path, manifest)
                with self.assertRaises(ValueError):
                    summary.aggregate(path)

    def test_e10_restore_flags_are_not_silently_ignored_by_config_generation(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run.main(['configs', '--resume', 'unused', '--seeds', '42'])

    def test_e11_windows_drive_paths_are_rejected_on_posix_too(self):
        path = self.root/'checkpoints.json'
        self.write(path, dict(format='evocond_checkpoints_v1', checkpoints={'job': 'C:/outside/model'}))
        with patch.object(run, 'Path', PurePosixPath), self.assertRaises(ValueError):
            run.checkpoints(path)

if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
