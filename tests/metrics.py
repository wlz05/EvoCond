# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Low-rank context, local density and absolute-distance boundary checks.'
from copy import deepcopy
import unittest
import numpy as np
from control import metric, relations
def descriptor(semantic=(1.0, 0.0), available=(), budget=10.0, progress=.5):
    return dict(semantic=list(semantic), available=list(available) if available is not None else None,
                budget=budget, progress=progress)
def sample(base, value, feedback=1, actor='p'):
    view = dict(format='evocond_visible_context_v1', available={}, progress=dict(budget=10, completed=1, total=2), requirements=None)
    event = dict(source='a', target='b', source_version='av', target_version='bv', features=[*value, 1, 0, 0, 0, 0, 0],
        context=view, feedback=feedback, granularity='terminal', requirements=None)
    return event, base, actor
class MetricTests(unittest.TestCase):
    def test_distance_is_bounded_and_metadata_not_swamped(self):
        artifact = metric.initialize(2, 'e')
        a, b = descriptor(), descriptor(available=('string',), budget=100, progress=0)
        self.assertAlmostEqual(metric.distance(artifact, a, b), .1+.09+.05)
        self.assertTrue(0 <= metric.distance(artifact, a, descriptor((-1, 0))) <= 1)
    def test_missing_fields_use_neutral_not_total_exclusion(self):
        artifact = metric.initialize(2, 'e')
        a, b = descriptor(available=None, budget=None, progress=None), descriptor()
        squared = metric.distance(artifact, a, b)
        self.assertAlmostEqual(squared, .15)
        self.assertGreater(metric.kernel(artifact, a, b, .5, .5), 0)
    def test_absolute_gate_blocks_far_neighbors_despite_local_scale(self):
        artifact = metric.initialize(2, 'e', {'absolute': .5})
        a, b = descriptor(), descriptor((-1, 0))
        self.assertEqual(metric.kernel(artifact, a, b, 100, 100), 0)
        self.assertGreater(metric.kernel(artifact, a, b, 100, 100, absolute=False), .99)
    def test_zero_projection_uses_fixed_initial_projection(self):
        artifact = metric.initialize(2, 'e')
        artifact['projection'] = [[0, 0], [0, 0]]
        self.assertTrue(np.allclose(metric.projected(artifact, descriptor()), [1, 0]))
    def test_empty_pool_does_not_create_support(self):
        artifact = metric.initialize(2, 'e')
        self.assertIsNone(metric.scale(artifact, descriptor(), []))
        self.assertEqual(metric.kernel(artifact, descriptor(), descriptor(), None, .1), 0)
    def memory(self, actors=('p', 'p', 'p')):
        memory = relations.new()
        for i, value in enumerate(((1, 0), (.99, .1), (-1, 0))):
            event, base, actor = sample(str(i), value, 1 if i < 2 else -1, actors[i])
            memory = relations.observe(memory, [event], base_id=base, actor=actor)
        return memory
    def test_actor_strata_and_leave_question_out_are_enforced(self):
        artifact = metric.initialize(2, 'e', {'steps': 2})
        fitted = metric.fit(artifact, self.memory(('p', 'q', 'r')))
        self.assertEqual(fitted['usable_neighborhoods'], 0)
        self.assertEqual(fitted['losses'], [])
        self.assertEqual(fitted['projection'], artifact['projection'])
        self.assertEqual(fitted['initial_projection'], artifact['initial_projection'])
    def test_feedback_granularity_and_scorers_cannot_form_fit_neighbors(self):
        for field, values in (('granularity', ('terminal', 'subtask')), ('scorer', ('judge', 'verifier'))):
            memory = relations.new()
            for i, value in enumerate(values):
                event, base, actor = sample(str(i), (1, 0))
                event[field] = value
                memory = relations.observe(memory, [event], base_id=base, actor=actor)
            fitted = metric.fit(metric.initialize(2, 'e'), memory)
            self.assertEqual(fitted['usable_neighborhoods'], 0)
            self.assertEqual(fitted['target_kind'], 'observed_correctness_not_adoption_value')
    def test_metric_geometry_binds_feedback_strata(self):
        artifact = metric.fit(metric.initialize(2, 'e'), self.memory())
        artifact['pool'][0]['granularity'] = 'subtask'
        with self.assertRaisesRegex(ValueError, 'cached scales'):
            metric.validate(artifact)
    def test_fitting_rebuilds_bounded_current_projection_cache(self):
        artifact = metric.initialize(2, 'e', {'steps': 2, 'pool': 2})
        fitted = metric.fit(artifact, self.memory())
        self.assertEqual(len(fitted['pool']), 2)
        self.assertEqual(fitted['scale_generation'], 1)
        self.assertTrue(fitted['fitted'])
        self.assertEqual(artifact['pool'], [])
        for row in fitted['pool']:
            self.assertIn('projection_sha256', row['descriptor'])
        metric.validate(fitted, 'e')
    def test_relation_query_uses_new_kernel_and_far_fallback(self):
        memory = self.memory()
        artifact = metric.fit(metric.initialize(2, 'e', {'steps': 1, 'weights': [1, 0, 0, 0], 'absolute': .2}), memory)
        event, _, _ = sample('query', (0, 1))
        receipt = relations.estimate(memory, 'a', 'b', event['features'], event['context'], metric=artifact)
        self.assertEqual(receipt['mass'], 0)
        self.assertEqual(receipt['kernel_kind'], 'low_rank_local_absolute')
    def test_dimension_and_protocol_corruption_fail_closed(self):
        artifact = metric.initialize(2, 'e')
        with self.assertRaises(ValueError):
            metric.projected(artifact, descriptor((1, 0, 0)))
        invalid = deepcopy(artifact)
        invalid['projection'][0][0] = float('nan')
        with self.assertRaises(ValueError):
            metric.validate(invalid)
    def test_collapsed_projection_keeps_original_semantic_separation(self):
        artifact = metric.initialize(2, 'e', {'rank': 1, 'weights': [1, 0, 0, 0]})
        a, b = descriptor((1, 1)), descriptor((1, -1))
        value = metric.distance(artifact, a, b, matrix=np.array([[1.0, 0.0]]))
        self.assertAlmostEqual(value, .15*.5)
    def test_untrained_and_empty_edge_neighborhood_use_full_identity(self):
        artifact = metric.initialize(2, 'e', {'rank': 1, 'weights': [1, 0, 0, 0]})
        a, b = descriptor((1, 1)), descriptor((1, -1))
        self.assertAlmostEqual(metric.distance(artifact, a, b), .5)
        self.assertTrue(metric.identity_fallback(artifact, []))
        self.assertEqual(artifact['fallback'], 'full_dimension_identity')
    def test_novel_context_increases_residual_without_new_labels(self):
        artifact = metric.fit(metric.initialize(2, 'e', {'steps': 1}), self.memory())
        near, far = descriptor((1, 0)), descriptor((0, 1))
        floor = metric.residual_weight(artifact, near, near)
        novel = metric.residual_weight(artifact, far, near)
        self.assertEqual(floor, .15)
        self.assertGreater(novel, floor)
        self.assertLessEqual(novel, 1)
    def test_modified_recipe_rejects_stale_local_scales(self):
        artifact = metric.fit(metric.initialize(2, 'e', {'steps': 1}), self.memory())
        artifact['options']['residual'] = .2
        with self.assertRaisesRegex(ValueError, 'cached scales'):
            metric.validate(artifact)
    def test_zero_residual_is_not_a_supported_protocol(self):
        with self.assertRaises(ValueError):
            metric.options({'residual': 0})
    def test_prepared_query_matches_uncached_distance(self):
        artifact = metric.fit(metric.initialize(2, 'e', {'steps': 1}), self.memory())
        left, right = descriptor((0, 1)), artifact['pool'][0]['descriptor']
        cached = metric.prepare(artifact, left)
        self.assertAlmostEqual(metric.distance(artifact, left, right), metric.distance(artifact, cached, right))
    def test_corrupt_original_semantic_cache_is_rejected(self):
        artifact = metric.fit(metric.initialize(2, 'e', {'steps': 1}), self.memory())
        artifact['pool'][0]['descriptor']['original'] = [0, 0]
        with self.assertRaisesRegex(ValueError, 'original semantic cache'):
            metric.validate(artifact)
    def test_far_context_emits_explicit_blind_receipt(self):
        memory = self.memory()
        artifact = metric.fit(metric.initialize(2, 'e', {'steps': 1, 'weights': [1, 0, 0, 0], 'absolute': .2}), memory)
        event, _, _ = sample('query', (0, 1))
        receipt = relations.estimate(memory, 'a', 'b', event['features'], event['context'], metric=artifact)
        self.assertTrue(receipt['cold_start_blind'])
        self.assertEqual(receipt['effective_count'], 0)
        self.assertGreater(receipt['minimum_distance'], .2)
        self.assertEqual(receipt['weight'], 0)
    def test_auditing_omitted_events_does_not_change_accessed_support_receipt(self):
        memory = self.memory()
        memory['limit'] = 1
        artifact = metric.fit(metric.initialize(2, 'e', {'steps': 1, 'absolute': .2}), memory)
        event, _, _ = sample('query', (1, 0))
        normal = relations.estimate(memory, 'a', 'b', event['features'], event['context'], metric=artifact)
        audited = relations.estimate(memory, 'a', 'b', event['features'], event['context'], metric=artifact, audit=True)
        for key in ('weight', 'mass', 'cold_start_blind', 'eligible_contexts', 'minimum_distance'):
            self.assertEqual(normal[key], audited[key])

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
