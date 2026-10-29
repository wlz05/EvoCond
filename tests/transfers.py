# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Separate historical support, semantic reference and current deployment value.'
from copy import deepcopy
import json
import unittest
from unittest.mock import patch
import diagnostic
from control import bandit, dynamic, mixing as fusion, relations
from tests import stages, windows
from tests.bandits import records
class TransferTests(unittest.TestCase):
    def runtime(self):
        fixture = stages.StageTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture.runtime
    def authorize(self, engine):
        engine.teacher = dict(format=bandit.TEACHER, dimension=2, weights=[[-1000, 0], [0, 0], [1000, 0]],
            mean=[0, 0], scale=[1, 1], adoption_protocol=bandit.PROTOCOL, target_reference=bandit.REFERENCE)
        engine.teacher_bottom = engine.bottom
        engine.freeze()
    def select(self, runtime):
        episode = runtime.begin('sum and format', stages.PLAN, split='validation')
        context = runtime.context(episode)
        ranked = fusion.rank(context['dispatcher'].registry, runtime.encoder, runtime.engine.alignment,
            runtime.engine.view(context), recall=12, top_k=3, neighbors=8, metric=runtime.engine.metric)
        semantic = ranked['rankings'][0.0]
        ranked['rankings'] = {0.0: semantic, .5: semantic[1:]+semantic[:1], 1.0: list(reversed(semantic))}
        ranked.update(edge_support=True, features=[1.0, 0.0])
        ranked['abstention']['active'] = False
        with patch.object(fusion, 'rank', return_value=ranked):
            return runtime.select(episode)
    def test_validated_teacher_selects_actual_structural_payload(self):
        runtime = self.runtime()
        self.authorize(runtime.engine)
        result = self.select(runtime)
        self.assertEqual(result['alpha'], 1.0)
        self.assertTrue(result['adoption']['active'])
        self.assertNotEqual(result['payload_hash'], result['semantic_payload_hash'])
    def test_small_actor_change_keeps_parameters_but_sends_semantic_bytes(self):
        runtime = self.runtime()
        self.authorize(runtime.engine)
        before = deepcopy(runtime.engine.teacher)
        engine = runtime.engine
        engine.actor = 'newactor'
        engine.freeze()
        result = self.select(runtime)
        self.assertEqual(engine.teacher, before)
        self.assertFalse(result['adoption']['active'])
        self.assertEqual(result['alpha'], 0.0)
        self.assertEqual(result['payload_hash'], result['semantic_payload_hash'])
        log = json.loads((runtime.root/'retrievals.jsonl').read_text().splitlines()[-1])
        self.assertEqual(log['teacher_depth'], 0)
        self.assertFalse(log['adoption']['active'])
    def test_bank_geometry_and_protocol_change_revoke_deployment(self):
        for field in ('bank', 'geometry', 'protocol'):
            runtime = self.runtime()
            self.authorize(runtime.engine)
            engine = runtime.engine
            if field == 'bank':
                registry = runtime.dispatcher.registry
                registry['relations']['clock'] += 1
                runtime.dispatcher.replace_registry(registry, expected_sha256=dynamic.digest(runtime.dispatcher.registry))
            elif field == 'geometry':
                engine.metric['scale_generation'] += 1
            else:
                engine.options['reference'] += 1
            engine.freeze()
            self.assertIsNotNone(engine.teacher)
            self.assertIsNone(engine.deployed_teacher())
    def test_phase_change_alone_preserves_fixed_bottom_deployment(self):
        runtime = self.runtime()
        self.authorize(runtime.engine)
        bottom = runtime.engine.bottom
        runtime.engine.phase += 1
        runtime.engine.freeze()
        self.assertEqual(runtime.engine.bottom, bottom)
        self.assertIsNotNone(runtime.engine.deployed_teacher())
    def test_binding_checkpoint_roundtrip_and_corruption_are_atomic(self):
        runtime = self.runtime()
        self.authorize(runtime.engine)
        saved = runtime.engine.state()
        runtime.engine.restore(saved)
        self.assertEqual(runtime.engine.state(), saved)
        for value in ('invalid', '0'*64):
            changed = deepcopy(saved)
            changed['teacher_bottom'] = value
            with self.assertRaises(ValueError):
                runtime.engine.restore(changed)
            self.assertEqual(runtime.engine.state(), saved)
    def test_fixed_bottom_update_binds_then_bank_publication_revokes(self):
        fixture = windows.WindowTests()
        self.addCleanup(fixture.doCleanups)
        group = fixture.runtime()
        runtime = group.runtime
        self.authorize(runtime.engine)
        candidate = deepcopy(runtime.engine.teacher)
        runtime.engine.teacher = runtime.engine.teacher_bottom = None
        runtime.engine.freeze()
        for episode in group.episodes():
            group.execute(episode)
        receipt = dict(deployed=True, holdout_ids=[], reason='synthetic_validation_receipt')
        with patch.object(bandit, 'conservative', return_value=(candidate, receipt)):
            runtime.release_completed()
        self.assertIsNotNone(runtime.engine.deployed_teacher())
        for episode in group.episodes():
            group.execute(episode)
        runtime.release_completed()
        self.assertIsNotNone(runtime.engine.teacher)
        self.assertIsNone(runtime.engine.deployed_teacher())
    def test_semantic_reference_is_not_an_arbitrary_first_row(self):
        row = records(1)[0]
        row['actions'] = [1, 0, 2]
        with self.assertRaisesRegex(ValueError, 'semantic action'):
            bandit.validate(row)
        row = records(1)[0]
        row['semantic_payload_hash'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'Semantic reference'):
            bandit.validate(row)
    def test_relative_targets_match_semantic_dr_differences(self):
        for row in bandit.crossfit(records()):
            self.assertEqual(row['reference_action'], 0)
            self.assertEqual(row['reference_payload_hash'], 's')
            self.assertEqual(row['relative_values'][0], 0)
            for value, relative in zip(row['values'], row['relative_values']):
                self.assertAlmostEqual(relative, value-row['values'][0])
    def test_same_declared_snapshot_cannot_hide_mixed_actor_or_bank(self):
        for field in ('actor', 'bank', 'bottom', 'continuation_teacher_sha256'):
            rows = records()
            for row in rows:
                row[field] = 'same'
            rows[0][field] = 'other'
            with self.assertRaisesRegex(ValueError, 'frozen '+field):
                bandit.crossfit(rows)
    def test_holdout_actor_binding_cannot_differ_from_fit_binding(self):
        rows = records()
        for row in rows:
            row.update(actor='same', old_probabilities=[1, 0, 0], probabilities=[.8, .1, .1])
        rows[0]['actor'] = 'other'
        with self.assertRaisesRegex(ValueError, 'frozen actor'):
            bandit.conservative(rows, None, epsilon=.3)
    def test_cost_target_cannot_change_within_one_window(self):
        rows = records()
        rows[0]['reference'] = 20
        rows[0]['reward'] = bandit.reward(rows[0]['success'], rows[0]['costs'], penalty=.1, reference=20)
        with self.assertRaisesRegex(ValueError, 'frozen reference'):
            bandit.crossfit(rows)
    def test_relation_store_rejects_adoption_credit_as_correctness(self):
        event = dict(source='a', target='b', source_version='av', target_version='bv', features=[1],
            context={}, feedback=.3, granularity='terminal', feedback_kind='adoption_advantage')
        with self.assertRaisesRegex(ValueError, 'adoption advantage'):
            relations.observe(relations.new(), [event], base_id='q', actor='a')
    def test_actor_value_reversal_does_not_require_negative_history(self):
        row = diagnostic.transfers()
        self.assertGreater(row['historical_support'], 0)
        self.assertGreater(row['actor_adoption_margins']['oldactor'], 0)
        self.assertLess(row['actor_adoption_margins']['newactor'], 0)
        self.assertGreater(row['decision']['mean_regret'], 0)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
