# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Native-runtime phase barriers, private banks and paired checkpoints.'
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from control import dynamic, mixing as fusion, identity, relations, stage
from control.runtime import Runtime
from tests import runtimes as fixtures
Backend, PLAN, dispatcher_class = fixtures.Backend, fixtures.PLAN, fixtures.dispatcher_class
class Encoder:
    def encode(self, text):
        return [1.0, .8 if 'Add' in text or 'sum' in text else -.8, .2]
    encode_query = encode
class StageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.config = dict(format='evocond_searl_runtime_v1', arm='teacher', run_id='conditional', root=str(root),
            registry_path=str(root/'registry.json'), policy_hash='a'*64, value_contracts=False, encoder_id='frozen',
            encoder_path='assets/encoder', contextual=dict(rollouts=2, alignment_steps=2, teacher_steps=2, minimum_labels=2, frozen_batches=1))
        self.dispatcher = dispatcher_class()(self.config['registry_path'], Backend())
        for name in ('add', 'format', 'report', 'verify'):
            spec = fixtures.RuntimeTests.tool(None, name)
            if name != 'add':
                spec['description'] = name+' a number'
            self.dispatcher.create(spec, {'x': 1, 'y': 2}, task_id='seed')
        self.runtime = Runtime(self.config, self.dispatcher, encoder=Encoder())
    def episodes(self):
        return [self.runtime.begin('sum and format', PLAN, split='train') for _ in range(2)]
    def execute(self, episode, success=True):
        context = self.runtime.context(episode)
        versions = list(context['dispatcher'].registry['semantic'])
        self.runtime.select(episode)
        self.runtime.charge(episode, 'actor', dict(input_tokens=100, output_tokens=10))
        for version in versions[:2]:
            spec = context['dispatcher'].registry['semantic'][version]
            self.runtime.observe(episode, spec['name'], {'value': 2}, success=True, version=version)
        self.runtime.finish(episode, task_success=success)
    def test_shared_bank_is_frozen_until_group_barrier(self):
        episodes = self.episodes()
        before = dynamic.digest(self.dispatcher.registry)
        for episode in episodes:
            self.execute(episode)
        self.assertEqual(dynamic.digest(self.dispatcher.registry), before)
        self.runtime.release_completed()
        self.assertNotEqual(dynamic.digest(self.dispatcher.registry), before)
        self.assertEqual(self.runtime.engine.phase, 1)
    def test_incomplete_group_cannot_publish(self):
        episode = self.runtime.begin('sum and format', PLAN, split='train')
        self.execute(episode)
        with self.assertRaises(ValueError):
            self.runtime.release_completed()
    def test_evaluation_never_mutates_training_bank(self):
        before = dynamic.digest(self.dispatcher.registry)
        episode = self.runtime.begin('unseen', PLAN, split='test')
        self.execute(episode)
        self.runtime.release_completed()
        self.assertEqual(dynamic.digest(self.dispatcher.registry), before)
        self.assertEqual(self.runtime.engine.phase, 0)
    def test_no_extra_rollouts_or_execution_truth_labels(self):
        episodes = self.episodes()
        for episode in episodes:
            self.execute(episode, success=False)
        self.runtime.release_completed()
        memory = self.dispatcher.registry['relations']
        sample = next(iter(memory['edges'].values()))['events'][0]['samples'][0]
        self.assertEqual(sample['feedback'], -1)
        self.assertEqual(sample['granularity'], 'terminal')
        report = json.loads((self.runtime.root/'phases/phase0.json').read_text())
        self.assertEqual(report['extra_rollouts'], 0)
        self.assertEqual(report['original_rollouts'], 2)
    def test_checkpoint_restores_all_conditional_artifacts(self):
        for episode in self.episodes():
            self.execute(episode)
        self.runtime.release_completed()
        self.runtime.set_policy_step(1)
        checkpoint = self.runtime.snapshot()
        self.runtime.engine.phase = 100
        self.runtime.restore(checkpoint)
        self.assertEqual(self.runtime.snapshot(), checkpoint)
    def test_actor_cannot_change_in_flight(self):
        self.episodes()
        with self.assertRaises(ValueError):
            self.runtime.set_policy_step(1)
    def test_contextual_path_has_no_rebinding_or_recovery_schema(self):
        episode = self.runtime.begin('new', PLAN, split='validation')
        result = self.runtime.select(episode)
        self.assertTrue(result['contextual'])
        self.assertNotIn('recovery', self.runtime.context(episode))
        self.assertEqual(result['alpha'], 0)
        self.assertNotIn('alpha', result['note'])
    def test_cached_payload_does_not_intervene_twice(self):
        episode = self.runtime.begin('new', PLAN, split='validation')
        first = self.runtime.select(episode)
        second = self.runtime.select(episode)
        self.assertEqual(first, second)
        self.assertEqual(len(self.runtime.context(episode)['cost_receipts']), 1)
    def test_evaluation_records_actual_outcome_without_training_publication(self):
        episode = self.runtime.begin('unseen', PLAN, split='test')
        self.execute(episode, success=False)
        self.runtime.finish(episode, task_success=False)
        rows = [json.loads(line) for line in (self.runtime.root/'outcomes.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]['success'])
        self.assertEqual(rows[0]['metrics']['semantic_fallbacks'], 1)
        self.assertEqual(rows[0]['metrics']['fallback_payload_matches'], 1)
        self.assertEqual(rows[0]['metrics']['payload_changes'], 0)
        self.assertEqual(rows[0]['metrics']['candidate_set_changes'], 0)
        self.assertGreater(rows[0]['metrics']['selected_payload_bytes'], 0)
        self.assertEqual(rows[0]['runtime_sha256'], dynamic.digest(self.runtime.config))
        retrieval = json.loads((self.runtime.root/'retrievals.jsonl').read_text())
        self.assertEqual(retrieval['exposure']['selected']['versions'],
            [schema['tool_version'] for schema in self.runtime.context(episode)['selection_cache'][0]['schemas']])
        self.assertEqual(retrieval['exposure']['selected']['sha256'], retrieval['payload_hash'])
    def test_blind_zone_forces_semantic_payload_and_logs_only_once(self):
        episode = self.runtime.begin('unseen', PLAN, split='test')
        first = self.runtime.select(episode)
        self.assertTrue(first['abstention']['active'])
        self.assertEqual(first['alpha'], 0)
        self.assertFalse(first['edge_support'])
        path = self.runtime.root/'abstentions.jsonl'
        before = path.read_bytes()
        self.runtime.select(episode)
        self.assertEqual(path.read_bytes(), before)
        row = json.loads(before)
        self.assertEqual(row['reason'], 'no_eligible_history')
        self.assertNotIn('success', row)
    def test_signed_residual_norm_and_semantic_candidate_boundary(self):
        registry = deepcopy(self.dispatcher.registry)
        store = identity.catalogue(registry)
        source, target = list(registry['semantic'])[:2]
        episode = self.runtime.begin('new', PLAN, split='validation')
        view = self.runtime.engine.view(self.runtime.context(episode))
        from control.context import features
        event = dict(source=store['versions'][source], target=store['versions'][target], source_version=source,
            target_version=target, features=features(view, Encoder()), requirements=None, feedback=-1, granularity='terminal')
        registry['relations'] = relations.observe(registry['relations'], [event], base_id='old', actor='p')
        result = fusion.rank(registry, Encoder(), self.runtime.engine.alignment, view, completed=[source], recall=4)
        candidates = {row['version'] for row in result['candidates']}
        for ranking in result['rankings'].values():
            self.assertTrue({row['version'] for row in ranking} <= candidates)
        for row in result['candidates']:
            for direction in ('incoming', 'outgoing'):
                self.assertLessEqual(sum(value*value for value in row[direction]), 1)
    def test_absolute_boundary_zero_support_withdraws_structural_rank(self):
        registry = deepcopy(self.dispatcher.registry)
        store = identity.catalogue(registry)
        versions = list(registry['semantic'])[:2]
        source, target = (store['versions'][version] for version in versions)
        episode = self.runtime.begin('new', PLAN, split='validation')
        view = self.runtime.engine.view(self.runtime.context(episode))
        registry['relations']['outgoing'][source] = {target: 1}
        store['usage'][target] = ['a', 'b', 'c']
        cutoff = self.runtime.engine.metric['options']['absolute']
        receipt = dict(weight=0, mass=0, effective_count=0, support_by_base={}, minimum_distance=cutoff)
        with patch('control.relations.estimate', return_value=receipt):
            result = fusion.rank(registry, Encoder(), self.runtime.engine.alignment, view,
                completed=versions[:1], recall=4, metric=self.runtime.engine.metric)
        self.assertTrue(result['abstention']['semantic_fallback'])
        self.assertFalse(result['abstention']['active'])
        self.assertEqual(result['abstention']['reason'], 'zero_effective_support')
        self.assertFalse(result['edge_support'])
        self.assertEqual(result['rankings'][0], result['rankings'][1])
    def test_bad_protocol_and_unmetered_costs_are_rejected(self):
        with self.assertRaises(ValueError):
            stage.settings(dict(limit=200, horizon=100))
        episode = self.runtime.begin('new', PLAN, split='validation')
        with self.assertRaises(ValueError):
            self.runtime.charge(episode, 'actor', {'estimated_tokens': 5})
    def test_visible_availability_comes_from_real_observed_results(self):
        episode = self.runtime.begin('new', PLAN, split='validation')
        context = self.runtime.context(episode)
        self.assertIsNone(self.runtime.engine.view(context)['available'])
        self.runtime.observe(episode, 'number', 3, success=True)
        available = self.runtime.engine.view(context)['available']
        self.assertTrue(available['number'])
        self.assertFalse(available['string'])
    def test_subtask_correctness_is_not_overridden_by_terminal_failure(self):
        episodes = self.episodes()
        for episode in episodes:
            context = self.runtime.context(episode)
            versions = list(context['dispatcher'].registry['semantic'])
            self.runtime.select(episode)
            self.runtime.charge(episode, 'actor', dict(input_tokens=100, output_tokens=10))
            for i, version in enumerate(versions[:2]):
                self.runtime.assistant(episode, '<subtask>ST'+str(i+1)+': step</subtask>')
                spec = context['dispatcher'].registry['semantic'][version]
                self.runtime.observe(episode, spec['name'], 3, success=True, version=version)
            self.runtime.subtask_feedback(episode, 'ST2', 1, source='unit_explicit_rule_scorer')
            self.runtime.finish(episode, task_success=False)
        self.runtime.release_completed()
        event = next(iter(self.dispatcher.registry['relations']['edges'].values()))['events'][0]['samples'][0]
        self.assertEqual(event['granularity'], 'subtask')
        self.assertEqual(event['feedback'], 1)
    def test_unknown_terminal_feedback_cannot_train_teacher(self):
        episode = self.runtime.begin('new', PLAN, split='train')
        with self.assertRaises(ValueError):
            self.runtime.finish(episode)
    def test_whole_ablation_is_fitted_and_checkpointed_only_at_training_barrier(self):
        config = deepcopy(self.config)
        config['contextual']['variant'] = 'whole'
        config['run_id'], config['root'] = 'whole', config['root']+'/whole'
        config['registry_path'] = config['root']+'/registry.json'
        dispatcher = dispatcher_class()(config['registry_path'], Backend())
        dispatcher.replace_registry(self.dispatcher.registry, expected_sha256=dynamic.digest(dispatcher.registry))
        self.runtime = Runtime(config, dispatcher, encoder=Encoder())
        original = deepcopy(self.runtime.engine.whole)
        for episode in self.episodes():
            self.execute(episode)
        self.assertEqual(self.runtime.engine.whole, original)
        self.runtime.release_completed()
        self.assertTrue(self.runtime.engine.whole['trained'])
        self.runtime.set_policy_step(1)
        snapshot = self.runtime.snapshot()
        self.runtime.restore(snapshot)
        self.assertEqual(self.runtime.snapshot(), snapshot)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
