# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Confidence arithmetic, independent hold-out counting and protocol provenance.'
import math
from copy import deepcopy
import json
import unittest
from unittest.mock import patch
from control import bandit, stage, metric, identity, dynamic
from control.runtime import Runtime
from calibrate import audit, simulate
from tests.bandits import records
from tests import stages, metrics
class CalibrationTests(unittest.TestCase):
    def test_actual_default_window_penalty_exceeds_reward_width(self):
        result = audit({'budget': {'batch_size': 256}})
        self.assertEqual(result['n'], 128)
        self.assertAlmostEqual(result['protocol']['beta'], .00005)
        self.assertAlmostEqual(result['global_envelope'], 16.5)
        self.assertGreater(result['allocated_window_range_penalty'], 6)
        self.assertAlmostEqual(result['single_window_range_penalty'], 14*16.5*math.log(40)/(3*127))
        self.assertEqual(result['kind'], 'configured_arithmetic_not_observed_training')
    def test_variance_rule_can_update_same_data_rejected_by_global_bound(self):
        values = [.05, .15]*64
        kwargs = dict(horizon=7, penalty=.1, epsilon=.2, beta=.00005)
        strict = bandit.confidence(values, **kwargs)
        normal = bandit.confidence(values, **kwargs, method='normal')
        self.assertEqual(strict['delta'], 0)
        self.assertGreater(normal['delta'], 0)
        self.assertGreater(normal['improvement_lower'], 0)
        self.assertEqual(normal['bound_kind'], bandit.NORMAL)
        self.assertFalse(normal['finite_sample_certificate'])
        self.assertEqual(normal['range_penalty'], 0)
        self.assertGreater(normal['bernstein_range_penalty'], 6)
    def test_small_degenerate_and_null_samples_do_not_force_updates(self):
        for values, reason in (([.1]*128, 'degenerate_variance'), ([.05, .15]*15, 'insufficient_holdout'),
            ([0.0]*128, 'degenerate_variance'), ([], 'insufficient_holdout')):
            result = bandit.confidence(values, horizon=7, penalty=.1, epsilon=.2, method='normal')
            self.assertEqual(result['delta'], 0)
            self.assertEqual(result['confidence_reason'], reason)
            self.assertIsNone(result['advantage_lower'])
        result = bandit.confidence([-.1, .1]*64, horizon=7, penalty=.1, epsilon=.2, method='normal')
        self.assertEqual(result['delta'], 0)
    def test_extreme_beta_uses_stable_tail_quantile(self):
        result = bandit.confidence([.05, .15]*64, horizon=7, penalty=.1, epsilon=.2, beta=1e-300, method='normal')
        self.assertTrue(math.isfinite(result['critical_value']))
        self.assertTrue(math.isfinite(result['advantage_lower']))
    def test_unknown_and_negative_critical_value_methods_rejected(self):
        for options in ({'method': 'bootstrap'}, {'method': 'normal', 'beta': .5}, {'method': 'normal', 'beta': .9}):
            with self.assertRaises(ValueError):
                bandit.confidence([.05, .15]*64, horizon=7, penalty=.1, epsilon=.2, **options)
        with self.assertRaises(ValueError):
            stage.settings({'confidence_method': 'unknown'})
    def test_conservative_stores_actual_zero_inclusive_advantages(self):
        rows = records(80)
        for row in rows:
            row.update(old_probabilities=[1, 0, 0], probabilities=[1-.2+.2/3, .2/3, .2/3])
            if int(row['base_id']) % 2 == 0:
                row.update(status='unreached', features=[], actions=[], probabilities=[], chosen=None, payload_hashes=[], interventions=0)
        _, report = bandit.conservative(rows, None, minimum=2, steps=2, method='normal')
        self.assertEqual(report['n'], len(report['heldout_advantages']))
        self.assertEqual(report['n'], len(report['holdout_ids']))
        self.assertGreater(report['zeros'], 0)
        self.assertEqual(report['confidence_method'], 'normal')
        self.assertFalse(set(report['holdout_ids']) & set(report['reward_model_fit_ids']))
    def test_audit_requires_matching_actual_phase_protocol(self):
        result = audit({})
        values = [.05, .15]*64
        phase = dict(confidence_protocol=result['protocol'], teacher_update=dict(heldout_advantages=values,
            holdout_ids=list(range(128)), n=128))
        self.assertEqual(audit({}, phase=phase)['kind'], 'observed_phase')
        with self.assertRaises(ValueError):
            audit({'contextual': {'epsilon': .3}}, phase=phase)
    def test_uniform_shrink_cannot_change_strict_bound_sign(self):
        values = [.05, .15]*64
        first = bandit.confidence(values, horizon=7, penalty=.1, epsilon=.2)
        n, factor = len(values), .1
        width, pmin = 1.1, .2/3
        mean = sum(values)/n
        variance = sum((z-mean)**2 for z in values)/(n-1)
        radius = math.sqrt(2*factor**2*variance*math.log(40)/n)+14*factor*width/pmin*math.log(40)/(3*(n-1))
        self.assertAlmostEqual(7*(factor*mean-radius), factor*first['advantage_lower'])
    def test_raw_metric_updates_evidence_without_projection_training(self):
        artifact = metric.initialize(2, 'e', {'rank': 1, 'protection': 'identity', 'residual': 1.0})
        with patch('numpy.linalg.svd', side_effect=AssertionError('No learned projection')):
            fitted = metric.fit(artifact, metrics.MetricTests().memory())
        self.assertEqual(fitted['projection'], artifact['projection'])
        self.assertEqual(fitted['losses'], [])
        self.assertFalse(fitted['fitted'])
        self.assertGreater(len(fitted['pool']), 0)
        self.assertEqual(fitted['objective'], 'fixed_raw_context_no_projection_fit')
    def test_sparse_optimistic_heads_expose_normal_undercoverage(self):
        result = simulate(stage.settings({}), trials=1000)
        stress = result['scenarios'][-1]
        self.assertEqual(stress['true_total_advantage'], 0.0)
        self.assertGreater(stress['methods']['normal']['false_positive_rate'], result['protocol']['beta'])
        self.assertLess(stress['methods']['normal']['coverage_among_computed'], result['nominal_coverage'])
        self.assertEqual(stress['methods']['bernstein']['false_positive_rate'], 0)
        self.assertFalse(result['finite_sample_guarantee_for_normal'])
    def test_minimal_control_skips_alignment_and_keeps_single_point_feedback(self):
        fixture = stages.StageTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        config = deepcopy(fixture.config)
        config['root'] += '/simple'
        config['registry_path'] = config['root']+'/registry.json'
        config['contextual']['variant'] = 'simple'
        dispatcher = stages.dispatcher_class()(config['registry_path'], stages.Backend())
        dispatcher.replace_registry(fixture.dispatcher.registry, expected_sha256=dynamic.digest(dispatcher.registry))
        with patch.object(identity, 'fit', side_effect=AssertionError('No learned identity alignment')):
            fixture.runtime = Runtime(config, dispatcher, encoder=stages.Encoder())
            episodes = fixture.episodes()
            for episode in episodes:
                fixture.runtime.context(episode)['marked_point'] = 6
                fixture.execute(episode)
            fixture.runtime.release_completed()
        engine = fixture.runtime.engine
        self.assertEqual(engine.options['metric']['protection'], 'identity')
        self.assertIsNone(engine.options['fixed_alpha'])
        self.assertEqual(engine.alignment['loss'], [])
        report = json.loads((fixture.runtime.root/'phases/phase0.json').read_text())
        self.assertEqual(report['base_observations'], 1)
        self.assertEqual(len(report['teacher_update']['fit_ids'])+len(report['teacher_update']['holdout_ids']), 1)
        rows = [json.loads(line) for line in (fixture.runtime.root/'interventions.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['interventions'], 0)
        self.assertEqual(rows[0]['status'], 'unreached')

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
