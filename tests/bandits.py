# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Actual payload classes, one-point exploration and grouped DR checks.'
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
from control import bandit
def records(n=9):
    result = []
    for i in range(n):
        costs = dict(actor=1.0, tools=2.0, retrieval=1.0)
        row = dict(format=bandit.FORMAT, split='train', base_id=str(i), snapshot='frozen', features=[1.0, float(i)],
            actions=[0, 1, 2], probabilities=[.2, .3, .5], chosen=i % 3, payload_hashes=['s', 'm', 'r'],
            success=i % 2 == 0, costs=costs, penalty=.1, reference=10.0, interventions=1)
        row['reward'] = bandit.reward(row['success'], costs, penalty=.1, reference=10)
        result.append(row)
    return result
class BanditTests(unittest.TestCase):
    def test_deduplicate_payload_bytes_not_alpha_labels(self):
        rankings = {0.0: ['a'], .5: ['a'], 1.0: ['b']}
        classes = bandit.payloads(rankings, lambda ranking: '|'.join(ranking))
        self.assertEqual(len(classes), 2)
        self.assertEqual(classes[0]['alphas'], [0.0, .5])
        self.assertEqual(classes[0]['alpha'], 0)
    def test_one_existing_rollout_is_preselected_without_rewards(self):
        self.assertEqual(bandit.chosen_rollout(42, 0, 'q', 8), bandit.chosen_rollout(42, 0, 'q', 8))
        self.assertIn(bandit.chosen_rollout(42, 0, 'q', 8), range(8))
    def test_mixture_has_full_class_support(self):
        classes = bandit.payloads({a: [a] for a in bandit.ACTIONS}, str)
        _, probabilities = bandit.choose(None, [1], classes, explore=True, epsilon=.3, rng=bandit.randomizer(42))
        self.assertAlmostEqual(sum(probabilities), 1)
        self.assertEqual(probabilities, [.7999999999999999, .09999999999999999, .09999999999999999])
    def test_continuation_reward_counts_all_costs(self):
        self.assertAlmostEqual(bandit.reward(True, dict(actor=2, tools=2, retrieval=2), penalty=.5, reference=10), .7)
        with self.assertRaises(ValueError):
            bandit.reward(True, dict(tools=2), penalty=.1, reference=10)
    def test_crossfit_never_uses_own_base_reward_or_scales(self):
        rows = records()
        targets = bandit.crossfit(rows)
        changed = deepcopy(rows)
        changed[0]['success'] = not changed[0]['success']
        changed[0]['reward'] = bandit.reward(changed[0]['success'], changed[0]['costs'], penalty=.1, reference=10)
        other = bandit.crossfit(changed)
        self.assertEqual(targets[0]['predictions'], other[0]['predictions'])
        self.assertNotIn('0', targets[0]['model_fit_ids'])
        self.assertEqual(targets[0]['model_mean'], other[0]['model_mean'])

    def test_crossfit_reuses_fold_heads_without_changing_question_alignment(self):
        rows = records(12)
        with patch.object(bandit, 'reward_model', wraps=bandit.reward_model) as fitting:
            targets = bandit.crossfit(rows)
        self.assertEqual(fitting.call_count, 3)
        reversed_targets = {row['base_id']: row for row in bandit.crossfit(list(reversed(rows)))}
        for row in targets:
            self.assertNotIn(row['base_id'], row['model_fit_ids'])
            other = reversed_targets[row['base_id']]
            self.assertEqual(row['model_fit_ids'], other['model_fit_ids'])
            np.testing.assert_allclose(row['predictions'], other['predictions'], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(row['values'], other['values'], rtol=1e-12, atol=1e-12)
    def test_dr_correction_is_unclipped_and_class_specific(self):
        row = records()[0]
        target = bandit.crossfit(records())[0]
        for i, action in enumerate(row['actions']):
            expected = target['predictions'][i]
            if action == row['chosen']:
                expected += (row['reward']-expected)/row['probabilities'][i]
            self.assertAlmostEqual(target['values'][i], expected)
    def test_do_not_pool_different_frozen_snapshots(self):
        rows = records()
        rows[0]['snapshot'] = 'other'
        with self.assertRaises(ValueError):
            bandit.fit(rows)
    def test_repeated_base_questions_are_rejected(self):
        rows = records()
        rows[0]['base_id'] = rows[1]['base_id']
        with self.assertRaises(ValueError):
            bandit.fit(rows)
    def test_fitting_and_ope_scope_are_separate(self):
        model, targets = bandit.fit(records(), steps=3)
        self.assertEqual(len(targets), 9)
        with self.assertRaises(ValueError):
            bandit.evaluate(records(), model)
        independent = records(3)
        for row in independent:
            row['base_id'] = 'heldout'+row['base_id']
        report = bandit.evaluate(independent, model)
        self.assertIn('not_end_to_end', report['scope'])

    def test_finite_feature_overflow_is_rejected_before_reward_fitting(self):
        rows = records()
        for row in rows:
            row['features'][1] = 1e308
        original = deepcopy(rows)
        for operation in (bandit.crossfit, bandit.reward_model, lambda value: bandit.fit(value, steps=1),
                          lambda value: bandit.distill(None, value, steps=1)):
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, 'normalization'):
                operation(rows)
        self.assertEqual(rows, original)

    def test_nonfinite_regression_is_not_hidden_by_prediction_clipping(self):
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value), patch.object(bandit.np.linalg, 'solve', return_value=np.full(2, value)):
                for operation in (bandit.crossfit, bandit.reward_model):
                    with self.assertRaisesRegex(ValueError, 'coefficients'):
                        operation(records())

    def test_singular_reward_regression_has_an_explicit_error(self):
        with patch.object(bandit.np.linalg, 'solve', side_effect=np.linalg.LinAlgError('singular')):
            with self.assertRaisesRegex(ValueError, 'numerically solvable'):
                bandit.crossfit(records())

    def test_unrepresentable_propensity_corrections_do_not_return_infinity(self):
        rows = records()
        rows[0].update(chosen=0, probabilities=[1e-320, .5, .5])
        with patch.object(bandit.np.linalg, 'solve', return_value=np.zeros(2)):
            with self.assertRaisesRegex(ValueError, 'finite numeric range'):
                bandit.crossfit(rows)
        head = bandit.reward_model(records())
        head['coefficients'] = [[0.0, 0.0] for _ in range(3)]
        holding = deepcopy(rows[0])
        holding['base_id'] = 'independent'
        with self.assertRaisesRegex(ValueError, 'finite numeric range'):
            bandit.heldout_targets(holding, head)
        with self.assertRaisesRegex(ValueError, 'finite numeric range'):
            bandit.evaluate([holding], None)

    def test_heldout_head_must_match_snapshot_dimensions_and_finite_scales(self):
        head = bandit.reward_model(records())
        row = records(1)[0]
        row['base_id'] = 'independent'
        for changes in ({'fitted_snapshot': 'other'}, {'frozen': False}, {'scale': [1.0, 0.0]},
                        {'coefficients': [[0.0]]*3}, {'mean': [0.0, float('nan')]}, {'fit_ids': None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                bandit.heldout_targets(row, {**head, **changes})
        changed = deepcopy(row)
        changed['actor'] = 'different-executor'
        with self.assertRaisesRegex(ValueError, 'frozen reward head'):
            bandit.heldout_targets(changed, head)
        changed = deepcopy(row)
        changed['reference'] = 20.0
        changed['reward'] = bandit.reward(changed['success'], changed['costs'], penalty=changed['penalty'], reference=20.0)
        with self.assertRaisesRegex(ValueError, 'frozen reward head'):
            bandit.heldout_targets(changed, head)

    def test_nonfinite_objective_history_is_not_a_valid_checkpoint(self):
        model, _ = bandit.fit(records(), steps=1)
        for value in (float('nan'), float('inf'), True, '1.0'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'objective history'):
                bandit.validate_model({**model, 'objective': [value]})

    def test_sparse_window_still_validates_all_optimizer_and_deployment_budgets(self):
        rows = records(2)
        for row in rows:
            row['old_probabilities'] = [1.0, 0.0, 0.0]
            row['probabilities'] = [1-.2+.2/3, .2/3, .2/3]
        invalid = ({'minimum': 1}, {'minimum': True}, {'chain_limit': 1.5}, {'chain_limit': True},
            {'steps': 0}, {'steps': True}, {'seed': -1}, {'seed': True}, {'holdout': float('nan')},
            {'beta': 0}, {'horizon': 0}, {'delta_max': 0}, {'overhead': float('inf')})
        with patch.object(bandit, 'fit', side_effect=AssertionError('Validate before fitting')):
            for settings in invalid:
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    bandit.conservative(rows, None, **settings)

    def test_policy_compression_rejects_empty_repeated_and_absorbing_inputs(self):
        duplicate = records(2)
        duplicate[1]['base_id'] = duplicate[0]['base_id']
        absorbing = records(1)
        absorbing[0].update(status='unreached', features=[], actions=[], probabilities=[], chosen=None,
            payload_hashes=[], interventions=0)
        for rows, steps in (([], 1), (duplicate, 1), (absorbing, 1), (records(2), 0)):
            with self.subTest(rows=rows, steps=steps), self.assertRaises(ValueError):
                bandit.distill(None, rows, steps=steps)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
