# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Reproduce evaluation, checkpoint and batching failures before release.'
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import run
import summary
from judge import digest
from control import bandit, dynamic, windows
from tests.bandits import records
from tests.fatigues import zero
from tests import stages as stage_fixtures, studies as study_fixtures, windows as window_fixtures
from tests.windows import batch, concatenate, tensor

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

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
