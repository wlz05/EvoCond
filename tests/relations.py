# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Conditional reversal, shrinkage, deduplication and finite retention checks.'
import unittest
from control import relations
def event(phi=(0.0,), feedback=1.0, **kwargs):
    return dict(source='u', target='v', source_version='uv', target_version='vv', features=list(phi),
                feedback=feedback, granularity='terminal', requirements=None, **kwargs)
class RelationTests(unittest.TestCase):
    def test_risk_identity_and_opposite_condition_signs(self):
        left, right = relations.reversal()
        self.assertAlmostEqual(left['mean'], right['mean'])
        self.assertEqual(left['conditional'][0], -right['conditional'][0])
        for row in (left, right):
            self.assertAlmostEqual(row['gap'], row['variance'])
    def test_kernel_locality_overcomes_same_marginal_mean(self):
        memory = relations.new(bandwidth=.1)
        for i in range(8):
            memory = relations.observe(memory, [event()], base_id=str(i), actor='p')
        for i in range(2):
            memory = relations.observe(memory, [event((1,), -1)], base_id='n'+str(i), actor='p')
        cold = relations.estimate(memory, 'u', 'v', [1], {})
        global_weight = relations.estimate(memory, 'u', 'v', [1], {}, unconditional=True)
        self.assertLess(cold['weight'], 0)
        self.assertGreater(global_weight['weight'], 0)
    def test_retries_and_rollouts_do_not_inflate_clock_or_mass(self):
        memory = relations.observe(relations.new(), [event(), event(feedback=-1)], base_id='q', actor='p')
        repeated = relations.observe(memory, [event()], base_id='q', actor='p2')
        self.assertEqual(memory, repeated)
        receipt = relations.estimate(memory, 'u', 'v', [0], {})
        self.assertEqual(receipt['mass'], 1)
        self.assertEqual(receipt['weight'], 0)
        self.assertEqual(receipt['effective_count'], 1)
    def test_horizon_and_query_bound(self):
        memory = relations.new(horizon=3, limit=2)
        for i in range(5):
            memory = relations.observe(memory, [event()], base_id=str(i), actor='p')
        receipt = relations.estimate(memory, 'u', 'v', [0], {}, audit=True)
        self.assertEqual(receipt['independent_events'], 2)
        self.assertEqual(receipt['omitted_mass'], 1)
        self.assertEqual(len(next(iter(memory['edges'].values()))['events']), 3)
        self.assertAlmostEqual(receipt['truncation_bound'], 5/12)
    def test_shrinkage_and_effective_count_bounds(self):
        memory = relations.observe(relations.new(), [event()], base_id='q', actor='p')
        receipt = relations.estimate(memory, 'u', 'v', [0], {})
        self.assertLess(abs(receipt['weight']), 1)
        self.assertLessEqual(abs(receipt['weight']), receipt['mass']/(1+receipt['mass']))
    def test_known_versions_and_explicit_conflicts_gate_mass(self):
        row = event()
        row['requirements'] = {'network': True}
        memory = relations.observe(relations.new(), [row], base_id='q', actor='p')
        self.assertEqual(relations.estimate(memory, 'u', 'v', [0], {}, versions=lambda *args: False)['mass'], 0)
        self.assertEqual(relations.estimate(memory, 'u', 'v', [0], {'requirements': {'network': False}})['mass'], 0)
        self.assertGreater(relations.estimate(memory, 'u', 'v', [0], {})['mass'], 0)
    def test_merge_preserves_observed_versions_and_contexts(self):
        memory = relations.observe(relations.new(), [event()], base_id='q', actor='p')
        remapped = relations.remap(memory, {'u': 'a'})
        row = next(iter(remapped['edges'].values()))['events'][0]['samples'][0]
        self.assertEqual(row['source_version'], 'uv')
        self.assertEqual(row['features'], [0.0])
        self.assertEqual(relations.neighbors(remapped, 'a', limit=1), ['v'])
    def test_unknown_feedback_does_not_become_correct_execution(self):
        row = event()
        row['granularity'] = 'execution'
        with self.assertRaises(ValueError):
            relations.observe(relations.new(), [row], base_id='q', actor='p')
    def test_tight_truncation_bound_is_attained_by_opposite_feedback(self):
        memory = relations.new(horizon=3, limit=2, shrinkage=3)
        for base, feedback in (('a', 1), ('b', -1), ('c', -1)):
            memory = relations.observe(memory, [event(feedback=feedback)], base_id=base, actor='p')
        used = relations.estimate(memory, 'u', 'v', [0], {}, audit=True)
        full = dict(memory, limit=3)
        actual = abs(relations.estimate(full, 'u', 'v', [0], {})['weight']-used['weight'])
        self.assertAlmostEqual(used['truncation_bound'], actual)
        self.assertLess(used['truncation_bound'], 2*used['omitted_mass']/(3+used['mass']+used['omitted_mass']))

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
