# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Uniform timing, held-out advantages and sampled conservative deployment.'
from copy import deepcopy
from itertools import product
import unittest
from unittest.mock import patch
from control import bandit
from tests.bandits import records
class MixtureTests(unittest.TestCase):
    def test_point_preselection_is_repeatable_and_covers_fixed_horizon(self):
        values = [bandit.marked_point(42, 0, str(i), 7) for i in range(200)]
        self.assertEqual(set(values), set(range(7)))
        self.assertEqual(values, [bandit.marked_point(42, 0, str(i), 7) for i in range(200)])
    def test_small_mixture_is_sampled_not_argmax(self):
        proposal = dict(format=bandit.TEACHER, dimension=1, weights=[[-100], [0], [100]], mean=[0], scale=[1])
        model = dict(format=bandit.MIXTURE, previous=None, proposal=proposal, delta=.2)
        classes = bandit.payloads({a: [a] for a in bandit.ACTIONS}, str)
        p = bandit.probabilities(model, [1], classes)
        self.assertAlmostEqual(p[0], .8)
        picks = [bandit.choose(model, [1], classes, explore=False, epsilon=.2, rng=bandit.randomizer(i))[0] for i in range(200)]
        self.assertIn(2, picks)
        self.assertIn(0, picks)
    def test_absorbing_observation_is_kept_with_no_fake_action(self):
        row = records(1)[0]
        row.update(status='unreached', features=[], actions=[], probabilities=[], chosen=None, payload_hashes=[], interventions=0)
        bandit.validate(row)
        changed = deepcopy(row)
        changed['interventions'] = 1
        with self.assertRaises(ValueError):
            bandit.validate(changed)
    def test_whole_episode_cost_receipts_are_checked(self):
        row = records(1)[0]
        row['cost_receipts'] = [dict(kind=k, cost=v) for k, v in row['costs'].items()]
        bandit.validate(row)
        row['costs']['actor'] += 1
        row['reward'] = bandit.reward(row['success'], row['costs'], penalty=.1, reference=10)
        with self.assertRaises(ValueError):
            bandit.validate(row)
    def test_finite_sample_rule_can_and_should_reject_small_windows(self):
        report = bandit.confidence([.5, .5], horizon=7, penalty=.1, epsilon=.2)
        self.assertLess(report['advantage_lower'], 0)
        self.assertEqual(report['delta'], 0)
    def test_large_positive_fixed_bottom_advantage_gives_bounded_step(self):
        report = bandit.confidence([.5]*10000, horizon=7, penalty=.1, epsilon=.2)
        self.assertGreater(report['delta'], 0)
        self.assertLessEqual(report['delta'], .2)
        self.assertGreaterEqual(report['improvement_lower'], 0)
    def test_unreached_questions_are_in_heldout_denominator(self):
        rows = records(40)
        for row in rows:
            row['old_probabilities'] = [1, 0, 0]
            row['probabilities'] = [1-.2+.2/3, .2/3, .2/3]
            if int(row['base_id']) % 2 == 0:
                row.update(status='unreached', features=[], actions=[], probabilities=[], chosen=None, payload_hashes=[], interventions=0)
        model, report = bandit.conservative(rows, None, minimum=2, steps=2)
        self.assertEqual(report['n'], len(report['holdout_ids']))
        self.assertGreater(report['zeros'], 0)
        self.assertFalse(set(report['fit_ids']) & set(report['holdout_ids']))
        self.assertFalse(set(report['reward_model_fit_ids']) & set(report['holdout_ids']))
        self.assertIsNone(model)
    def test_old_teacher_cannot_change_after_intervention(self):
        rows = records(40)
        for row in rows:
            row['old_probabilities'] = [0, 1, 0]
        with self.assertRaises(ValueError):
            bandit.conservative(rows, None, minimum=2, steps=2)
    def test_reward_head_is_frozen_without_own_question(self):
        fitting = records(4)
        head = bandit.reward_model(fitting)
        with self.assertRaises(ValueError):
            bandit.heldout_targets(fitting[0], head)
        holding = records(1)[0]
        holding['base_id'] = 'new'
        values = bandit.heldout_targets(holding, head)
        self.assertEqual(len(values), 3)
    def test_relative_teacher_is_invariant_to_per_state_target_shifts(self):
        rows = records()
        targets = bandit.crossfit(rows)
        shifted = deepcopy(targets)
        for i, row in enumerate(shifted):
            row['values'] = [v+13*(i-4) for v in row['values']]
            row['relative_values'] = [v-row['values'][0] for v in row['values']]
        with patch.object(bandit, 'crossfit', return_value=targets):
            original, _ = bandit.fit(rows, steps=7)
        with patch.object(bandit, 'crossfit', return_value=shifted):
            changed, _ = bandit.fit(rows, steps=7)
        for first, second in zip(original['weights'], changed['weights']):
            for a, b in zip(first, second):
                self.assertAlmostEqual(a, b, places=12)
        for a, b in zip(original['objective'], changed['objective']):
            self.assertAlmostEqual(a, b, places=12)
        self.assertEqual(original['target_reference'], 'semantic_actual_payload_class')
    def test_zero_sum_envelope_is_attained_on_interval_corners(self):
        policies = ([1, 0, 0], [0, 1, 0], [.2, .3, .5])
        p = [.1, .2, .7]
        for old, new in product(policies, repeat=2):
            receipt = bandit.contrast(old, new, p, width=1.1)
            d = receipt['difference']
            for action in range(3):
                maximum = 0.0
                for values in product((-.1, 1.0), repeat=4):
                    reward, heads = values[0], values[1:]
                    value = sum(a*b for a, b in zip(d, heads))+d[action]/p[action]*(reward-heads[action])
                    maximum = max(maximum, abs(value))
                    self.assertLessEqual(abs(value), receipt['action_bounds'][action]+1e-12)
                self.assertAlmostEqual(maximum, receipt['action_bounds'][action])
            self.assertLessEqual(receipt['bound'], receipt['global_bound']+1e-12)
    def test_agreement_has_zero_samplewise_advantage_and_mixing_scales_bound(self):
        old, new, p = [.2, .3, .5], [.5, .5, 0], [.1, .2, .7]
        agreed = bandit.contrast(old, old, p, width=1.1)
        self.assertEqual(agreed['bound'], 0)
        self.assertEqual(agreed['difference'], [0, 0, 0])
        base = bandit.contrast(old, new, p, width=1.1)
        mixed = bandit.contrast(old, [.8*a+.2*b for a, b in zip(old, new)], p, width=1.1)
        self.assertAlmostEqual(mixed['bound'], .2*base['bound'])
        self.assertAlmostEqual(mixed['total_variation'], .2*base['total_variation'])
    def test_confidence_uses_the_declared_global_not_empirical_envelope(self):
        receipt = bandit.confidence([0, 0], horizon=7, penalty=.1, epsilon=.2)
        self.assertAlmostEqual(receipt['bound'], 1.1/(.2/3))
        self.assertEqual(receipt['bound_kind'], 'zero_sum_dr_global_envelope_v1')
        self.assertEqual(receipt['delta'], 0)
    def test_empty_old_probabilities_cannot_bypass_the_frozen_policy_check(self):
        rows = records(40)
        for row in rows:
            row.update(old_probabilities=[], probabilities=[1-.2+.2/3, .2/3, .2/3])
        with self.assertRaises(ValueError):
            bandit.conservative(rows, None, minimum=2, steps=2)
    def test_logged_mixture_must_match_before_any_fitting(self):
        rows = records(40)
        for row in rows:
            row['old_probabilities'] = [1, 0, 0]
        with patch.object(bandit, 'fit', side_effect=AssertionError('Must reject before fitting')):
            with self.assertRaises(ValueError):
                bandit.conservative(rows, None, minimum=2, steps=2)
    def test_nonfinite_or_boolean_protocol_and_probability_values_are_rejected(self):
        for vector in ([], [True, 0, 0], [float('nan'), .5, .5], [.2, .3, .4], [[1], [0], [0]]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                bandit.contrast(vector, [1, 0, 0], [.2, .3, .5], width=1.1)
        for field in ('penalty', 'epsilon', 'beta', 'delta_max', 'overhead'):
            for value in (True, float('nan'), float('inf')):
                options = dict(horizon=7, penalty=.1, epsilon=.2)
                options[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    bandit.confidence([0, 0], **options)
    def test_nonfinite_logged_reward_is_rejected_including_absorbing_states(self):
        for absorbing in (False, True):
            for value in (float('nan'), float('inf'), True):
                row = records(1)[0]
                row['reward'] = value
                if absorbing:
                    row.update(status='unreached', features=[], actions=[], probabilities=[], chosen=None, payload_hashes=[], interventions=0)
                with self.subTest(absorbing=absorbing, value=value), self.assertRaises(ValueError):
                    bandit.validate(row)
    def test_nonfinite_cost_receipt_is_not_masked_by_absolute_comparison(self):
        row = records(1)[0]
        row['cost_receipts'] = [dict(kind='actor', cost=float('nan'))]
        with self.assertRaises(ValueError):
            bandit.validate(row)
    def test_numerically_unrepresentable_envelopes_do_not_become_certificates(self):
        with self.assertRaises(ValueError):
            bandit.contrast([1, 0], [0, 1], [5e-324, 1], width=1.1)
        for options in (dict(epsilon=5e-324, penalty=.1), dict(epsilon=.2, penalty=1e308)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                bandit.confidence([0, 0], horizon=7, **options)
    def test_semantic_difference_variance_follows_the_exact_zero_sum_formula(self):
        rewards, p, d = [.4, .8], [.5, .5], [-1, 1]
        samples = [d[a]*rewards[a]/p[a] for a in range(2)]
        mean = sum(prob*value for prob, value in zip(p, samples))
        variance = sum(prob*(value-mean)**2 for prob, value in zip(p, samples))
        formula = sum(d[a]**2*rewards[a]**2/p[a] for a in range(2))-sum(d[a]*rewards[a] for a in range(2))**2
        self.assertAlmostEqual(variance, formula)
        self.assertAlmostEqual(variance, 1.44)
    def test_boolean_action_receipt_is_not_an_integer_class_identity(self):
        row = records(1)[0]
        row['chosen'] = True
        with self.assertRaises(ValueError):
            bandit.validate(row)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
