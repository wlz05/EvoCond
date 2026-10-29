# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Synthetic attribution identities; not observed policy or tool improvement.'
from copy import deepcopy
import unittest
from joint import FORMAT, CELLS, SHARED, summarize
def receipt(task='base'):
    policies = {'p0': 'a'*64, 'p1': 'b'*64}
    banks = {'k0': 'c'*64, 'k1': 'd'*64}
    shared = {k: 'e'*64 for k in SHARED}
    cells = {}
    for label in CELLS:
        p, k, h = label.split('.')
        cells[label] = {'executed': True, 'success': k == 'k1', 'policy_hash': policies[p],
                        'bank_hash': banks[k], 'guidance_mode': h, 'shared': deepcopy(shared),
                        'initial_candidate_hash': ('f' if k == 'k0' else '0')*64}
    return {'format': FORMAT, 'task_id': task, 'variant': 'original', 'seed': 42,
            'policies': policies, 'banks': banks, 'shared': shared, 'cells': cells}
class JointAuditTests(unittest.TestCase):
    def test_better_tools_are_not_reported_as_policy_improvement(self):
        report = summarize([receipt()], samples=30)
        estimate = report['estimates']
        self.assertEqual(estimate['candidate_only.total_change']['mean'], 1)
        self.assertEqual(estimate['candidate_only.policy_at_initial_bank']['mean'], 0)
        self.assertEqual(estimate['candidate_only.bank_at_initial_policy']['mean'], 1)
    def test_interaction_separated_from_main_effects(self):
        row = receipt()
        for label, cell in row['cells'].items():
            cell['success'] = label.startswith('p1.k1')
        result = summarize([row], samples=30)['estimates']
        self.assertEqual(result['ranked.joint_interaction']['mean'], 1)
        self.assertEqual(result['ranked.policy_at_initial_bank']['mean'], 0)
    def test_ordering_effect_is_not_called_native_skill_learning(self):
        row = receipt()
        row['cells']['p1.k1.candidate_only']['success'] = False
        result = summarize([row], samples=30)['estimates']
        self.assertEqual(result['ordering_effect.p1.k1']['mean'], 1)
    def test_variants_and_seeds_are_clustered_by_base_task(self):
        a, b = receipt(), receipt()
        b['seed'] = 43
        result = summarize([a, b], samples=30)
        self.assertEqual(result['independent_tasks'], 1)
        self.assertEqual(result['paired_receipts'], 2)
    def test_missing_failed_cell_or_mismatched_controls_refused(self):
        for change in ('missing', 'unexecuted', 'policy', 'bank', 'shared', 'pool', 'unknown'):
            row = receipt()
            cell = row['cells']['p0.k0.ranked']
            if change == 'missing':
                del row['cells']['p0.k0.ranked']
            elif change == 'unexecuted':
                cell['executed'] = False
            elif change == 'policy':
                cell['policy_hash'] = row['policies']['p1']
            elif change == 'bank':
                cell['bank_hash'] = row['banks']['k1']
            elif change == 'shared':
                cell['shared']['decoder_hash'] = '0'*64
            elif change == 'pool':
                cell['initial_candidate_hash'] = '0'*64
            else:
                cell['success'] = None
            with self.subTest(change=change), self.assertRaises(ValueError):
                summarize([row], samples=30)
    def test_duplicate_and_mixed_experiments_refused(self):
        with self.assertRaises(ValueError):
            summarize([receipt(), receipt()], samples=30)
        a, b = receipt(), receipt('second')
        b['policies']['p1'] = '0'*64
        with self.assertRaises(ValueError):
            summarize([a, b], samples=30)
        with self.assertRaises(ValueError):
            summarize([], samples=30)
if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
