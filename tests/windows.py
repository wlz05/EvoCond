# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Fixed-bottom native batching, ragged padding and paired checkpoint barriers.'
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
from unittest.mock import patch
from control import bandit, dynamic, stage, windows
from control.runtime import Runtime
from tests import stages as fixtures
class Tensor(np.ndarray):
    def new_full(self, shape, fill):
        return np.full(shape, fill, dtype=self.dtype).view(Tensor)
def tensor(values):
    return np.asarray(values).view(Tensor)
def batch(prompt, response):
    p, r = len(prompt), len(response)
    return SimpleNamespace(batch={'prompts': tensor([prompt]), 'responses': tensor([response]),
        'input_ids': tensor([prompt+response]), 'attention_mask': tensor([[1]*(p+r)]),
        'position_ids': tensor([list(range(p+r))]), 'loss_mask': tensor([[0]*p+[1]*r]),
        'old_log_probs': tensor([[.2]*r]), 'advantages': tensor([[.5]*r]),
        'response_mask': tensor([[1]*r]), 'step_rewards': tensor([[1.]]),
        'tool_use_scores': tensor([[1.]]), 'token_level_rewards': tensor([[1.]*r])},
        non_tensor_batch={'uid': np.array(['q'], dtype=object), 'anchor_obs': np.ones((1, r, 1))}, meta_info={})
def concatenate(rows):
    return SimpleNamespace(batch={key: np.concatenate([row.batch[key] for row in rows]) for key in rows[0].batch},
        non_tensor_batch={key: np.concatenate([row.non_tensor_batch[key] for row in rows]) for key in rows[0].non_tensor_batch},
        meta_info=rows[0].meta_info)
class WindowTests(unittest.TestCase):
    def test_accumulation_preserves_all_existing_batches(self):
        pending = []
        self.assertIsNone(windows.accumulate(pending, 1, False, 2, tuple))
        self.assertEqual(windows.accumulate(pending, 2, True, 2, tuple), (1, 2))
        self.assertEqual(pending, [])
    def test_incomplete_or_exceeded_window_fails_closed(self):
        with self.assertRaises(ValueError):
            windows.accumulate([], 1, True, 2, tuple)
        with self.assertRaises(ValueError):
            windows.accumulate([1], 2, False, 2, tuple)
    def test_native_ragged_sequences_keep_response_offsets_and_masks(self):
        combined = windows.join([batch([4, 5], [6]), batch([7], [8, 9, 10])], concatenate, 0)
        self.assertEqual(combined.batch['input_ids'].tolist(), [[4, 5, 6, 0, 0], [0, 7, 8, 9, 10]])
        self.assertEqual(combined.batch['prompts'].tolist(), [[4, 5], [0, 7]])
        self.assertEqual(combined.batch['loss_mask'].tolist(), [[0, 0, 1, 0, 0], [0, 0, 1, 1, 1]])
        self.assertEqual(combined.batch['advantages'].tolist(), [[.5, 0, 0], [.5, .5, .5]])
        self.assertEqual(combined.meta_info['global_token_num'], [3, 4])
        self.assertTrue(np.isnan(combined.non_tensor_batch['anchor_obs'][0, 1:]).all())
    def test_unknown_ragged_fields_are_not_silently_corrupted(self):
        left, right = batch([1], [2]), batch([1], [2, 3])
        left.batch['unknown'], right.batch['unknown'] = tensor([[1]]), tensor([[1, 2]])
        with self.assertRaises(ValueError):
            windows.join([left, right], concatenate, 0)
    def runtime(self):
        fixture = fixtures.StageTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.runtime = None
        config = deepcopy(fixture.config)
        config['run_id'] = 'fixed-window'
        config['root'] += '/window'
        config['registry_path'] = config['root']+'/registry.json'
        config['contextual']['frozen_batches'] = 2
        dispatcher = fixtures.dispatcher_class()(config['registry_path'], fixtures.Backend())
        dispatcher.replace_registry(fixture.dispatcher.registry, expected_sha256=dynamic.digest(dispatcher.registry))
        fixture.runtime = Runtime(config, dispatcher, encoder=fixtures.Encoder())
        return fixture
    def test_fixed_bottom_is_deployed_before_native_actor_update(self):
        fixture = self.runtime()
        runtime = fixture.runtime
        bank, bottom, metric = dynamic.digest(runtime.dispatcher.registry), runtime.engine.bottom, deepcopy(runtime.engine.metric)
        for episode in fixture.episodes():
            fixture.execute(episode)
        runtime.release_completed()
        self.assertFalse(runtime.engine.schedule()['actor_update_ready'])
        self.assertEqual(dynamic.digest(runtime.dispatcher.registry), bank)
        self.assertEqual(runtime.engine.bottom, bottom)
        self.assertEqual(runtime.engine.metric, metric)
        with self.assertRaises(ValueError):
            runtime.set_policy_step(1)
        with self.assertRaises(ValueError):
            runtime.snapshot()
        for episode in fixture.episodes():
            fixture.execute(episode)
        runtime.release_completed()
        self.assertTrue(runtime.engine.schedule()['actor_update_ready'])
        self.assertNotEqual(dynamic.digest(runtime.dispatcher.registry), bank)
        runtime.set_policy_step(2)
        saved = runtime.snapshot()
        runtime.restore(saved)
        self.assertEqual(saved, runtime.snapshot())
    def test_checkpoint_inside_window_cannot_be_restored(self):
        fixture = self.runtime()
        saved = fixture.runtime.engine.state()
        saved['phase'] = 1
        with self.assertRaisesRegex(ValueError, 'barrier'):
            fixture.runtime.engine.restore(saved)
    def test_adoption_protocol_is_part_of_the_frozen_bottom_and_snapshot(self):
        fixture = self.runtime()
        engine = fixture.runtime.engine
        bottom, snapshot = engine.bottom, engine.snapshot
        self.assertEqual(engine.state()['adoption_protocol'], bandit.PROTOCOL)
        with patch.object(bandit, 'PROTOCOL', 'different_adoption_protocol'):
            engine.freeze()
            self.assertNotEqual(engine.bottom, bottom)
            self.assertNotEqual(engine.snapshot, snapshot)
        engine.freeze()
        self.assertEqual(engine.bottom, bottom)
        self.assertEqual(engine.snapshot, snapshot)
    def test_old_adoption_protocol_is_rejected_before_engine_state_mutation(self):
        fixture = self.runtime()
        engine = fixture.runtime.engine
        original = engine.state()
        for protocol in (None, 'previous_adoption_protocol'):
            state = deepcopy(original)
            if protocol is None:
                state.pop('adoption_protocol')
            else:
                state['adoption_protocol'] = protocol
            with self.subTest(protocol=protocol), self.assertRaisesRegex(ValueError, 'new frozen window'):
                engine.restore(state)
            self.assertEqual(engine.state(), original)
    def test_fixed_alpha_does_not_write_an_observed_intervention(self):
        fixture = self.runtime()
        fixture.runtime.engine.options['fixed_alpha'] = .5
        for episode in fixture.episodes():
            context = fixture.runtime.context(episode)
            context['marked_point'] = 0
            fixture.execute(episode)
        fixture.runtime.release_completed()
    def test_single_batch_legacy_schedule_is_explicit(self):
        self.assertEqual(stage.settings({})['frozen_batches'], 2)
        self.assertEqual(stage.settings({'frozen_batches': 1})['frozen_batches'], 1)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
