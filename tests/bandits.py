# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Actual payload classes, one-point exploration and grouped DR checks.'
from copy import deepcopy
import unittest
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

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
