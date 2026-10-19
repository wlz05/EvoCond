# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Sampled actor drift and conservative teacher reset checks.'
from copy import deepcopy
import math
from pathlib import Path
import unittest
from control import bandit, drift
from tests import stages as fixtures
class DriftTests(unittest.TestCase):
    def receipt(self, runtime, value):
        return dict(format=drift.FORMAT, valid=True, actor=runtime.policy_hash, estimator=drift.ESTIMATOR,
            policy_kl=value, token_count=10, elapsed_seconds=.01, diagnostics=drift.options(),
            tail_value=value, trimmed_divergence=value, retained_tokens=9, discarded_tokens=1)
    def fixture(self):
        fixture = fixtures.StageTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        runtime = fixture.runtime
        runtime.engine.teacher = dict(format=bandit.MIXTURE, previous=None, proposal=None, delta=.1)
        runtime.engine.teacher_bottom = runtime.engine.bottom
        runtime.engine.freeze()
        return runtime
    def test_identical_actor_and_padding_give_zero(self):
        row = drift.estimate([[-1, float('nan')]], [[-1, float('nan')]], [[1, 0]], actor='a')
        self.assertEqual(row['policy_kl'], 0)
        self.assertEqual(row['token_count'], 1)
    def test_full_old_action_distribution_recovers_forward_kl(self):
        old = [[math.log(.5), math.log(.5)]]
        new = [[math.log(.9), math.log(.1)]]
        row = drift.estimate(old, new, [[1, 1]], actor='a')
        expected = .5*math.log(.5/.9)+.5*math.log(.5/.1)
        self.assertAlmostEqual(row['policy_kl'], expected)
    def test_empty_nonfinite_and_overflow_are_not_silently_zeroed(self):
        for old, new, mask in (([[-1]], [[-1]], [[0]]), ([[-1]], [[float('nan')]], [[1]]), ([[-1000]], [[-1]], [[1]])):
            with self.subTest(old=old), self.assertRaises(ValueError):
                drift.estimate(old, new, mask, actor='a')
    def test_large_actor_drift_resets_chain_and_keeps_bank(self):
        runtime = self.fixture()
        before = deepcopy(runtime.dispatcher.registry)
        report = runtime.set_policy_step(1, self.receipt(runtime, .1))
        self.assertTrue(report['soft_reset'])
        self.assertEqual(report['reason'], 'actor_drift_rule_exceeded')
        self.assertIsNone(runtime.engine.teacher)
        self.assertEqual(report['chain_depth'], 0)
        self.assertEqual(runtime.dispatcher.registry, before)
        self.assertFalse(report['guarantee_transferred'])
    def test_small_drift_retains_parameters_without_deployment(self):
        runtime = self.fixture()
        before = deepcopy(runtime.engine.teacher)
        report = runtime.set_policy_step(1, self.receipt(runtime, .001))
        self.assertFalse(report['soft_reset'])
        self.assertEqual(runtime.engine.teacher, before)
        self.assertTrue(report['requires_new_actor_window'])
        self.assertFalse(report['deployment_authorized'])
        self.assertIsNone(runtime.engine.deployed_teacher())
        self.assertEqual(runtime.engine.adoption_status()['reason'], 'bottom_requires_validation')
    def test_missing_and_wrong_actor_receipts_reset_conservatively(self):
        runtime = self.fixture()
        row = self.receipt(runtime, 0)
        row['actor'] = 'wrong'
        self.assertTrue(runtime.set_policy_step(1, row)['soft_reset'])
        self.assertTrue(runtime.set_policy_step(2)['soft_reset'])
    def test_invalid_nan_receipt_has_a_serializable_reset_log(self):
        runtime = self.fixture()
        row = self.receipt(runtime, float('nan'))
        report = runtime.set_policy_step(1, row)
        self.assertTrue(report['soft_reset'])
        self.assertNotIn('policy_kl', report['drift'])
    def test_initialization_same_step_and_restore_do_not_reset_again(self):
        runtime = self.fixture()
        self.assertIsNone(runtime.set_policy_step(0))
        runtime.set_policy_step(1, self.receipt(runtime, .1))
        saved = runtime.snapshot()
        path = runtime.root/'actorupdates.jsonl'
        before = path.read_bytes()
        self.assertIsNone(runtime.set_policy_step(1))
        runtime.restore(saved)
        self.assertEqual(runtime.snapshot(), saved)
        self.assertEqual(path.read_bytes(), before)
    def test_native_hook_recomputes_after_update_not_reference_kl(self):
        source = (Path(__file__).resolve().parents[1]/'baselines/SEARL-official/verl/trainer/ppo/trainer.py').read_text(encoding='utf-8')
        update = source.index('actor_output = self.actor_rollout_wg.update_actor(update_batch)')
        probe = source.index('current = self.actor_rollout_wg.compute_log_prob(update_batch)', update)
        publish = source.index('runtime_policy_step.remote(self.global_steps, policy_drift)', probe)
        self.assertLess(update, probe)
        self.assertLess(probe, publish)
        self.assertNotIn('actor/kl_loss', source[probe:publish])
    def test_isolated_tail_keeps_raw_kl_but_does_not_reset(self):
        old = [[math.log(.5)]*100]
        new = [[math.log(.5)]*99+[math.log(1e-8)]]
        row = drift.estimate(old, new, [[1]*100], actor='a')
        self.assertGreater(row['policy_kl'], .02)
        self.assertEqual(row['trimmed_divergence'], 0)
        diagnosis = drift.decision(row, 'a', .02)
        self.assertFalse(diagnosis['reset'])
        self.assertTrue(diagnosis['isolated_tail_suppressed'])
    def test_broad_probability_shift_still_resets(self):
        row = drift.estimate([[-1]*100], [[-1.3]*100], [[1]*100], actor='a')
        self.assertTrue(drift.decision(row, 'a', .02)['reset'])
        self.assertGreater(row['trimmed_divergence'], .02)
    def test_raw_mean_control_is_an_explicit_ablation(self):
        recipe = drift.options({'rule': 'mean'})
        row = drift.estimate([[-1]*100], [[-1]*99+[-20]], [[1]*100], actor='a', diagnostics=recipe)
        self.assertTrue(drift.decision(row, 'a', .02, recipe)['reset'])
        with self.assertRaises(ValueError):
            drift.decision(row, 'a', .02)
    def test_masked_out_tail_and_one_token_batches_are_explicit(self):
        row = drift.estimate([[-1, -1]], [[-1, -20]], [[1, 0]], actor='a')
        self.assertEqual(row['discarded_tokens'], 0)
        self.assertEqual(row['retained_tokens'], 1)
        self.assertFalse(drift.decision(row, 'a', .02)['reset'])
    def test_diagnostic_counts_and_nonfinite_quantiles_fail_closed(self):
        row = drift.estimate([[-1]*10], [[-1]*10], [[1]*10], actor='a')
        for key, value in (('discarded_tokens', 0), ('tail_value', float('nan'))):
            changed = deepcopy(row)
            changed[key] = value
            with self.assertRaises(ValueError):
                drift.validate(changed, 'a')

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
