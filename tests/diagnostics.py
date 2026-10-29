# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Check information loss without loading actors or fabricating benchmark scores.'
from copy import deepcopy
from io import StringIO
import json
import unittest
from unittest.mock import patch
import diagnostic
from control import relations
class InformationTests(unittest.TestCase):
    def test_shared_randomization_cannot_remove_opposite_optimal_decisions(self):
        for p, first, second in ((.5, .3, .3), (.2, .1, .7), (1e-9, .3, .4)):
            for selection in (0, .1, .5, .9, 1):
                row = diagnostic.shared(p, first, second, selection)
                self.assertGreater(row['minimum_regret'], 0)
                self.assertGreaterEqual(row['regret']+1e-15, row['minimum_regret'])
            best = diagnostic.shared(p, first, second, row['optimal_selection'])
            self.assertAlmostEqual(best['regret'], best['minimum_regret'])
        self.assertAlmostEqual(diagnostic.shared(.5, .3, .3)['minimum_regret'], .15)
    def test_shared_bound_requires_its_stated_support_and_margin_assumptions(self):
        for values in ((0, .3, .3), (1, .3, .3), (.5, 0, .3), (.5, .3, -1),
                       (True, .3, .3), (.5, float('nan'), .3), (.5, 3, .3), (.5, .3, .3, 2),
                       (.1, 5e-324, .3)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                diagnostic.shared(*values)
    def test_tight_access_bound_is_not_monotone_in_used_mass(self):
        bound = lambda used: (10-used)*(1+2*used)/(11*(1+used))
        self.assertAlmostEqual(bound(0), 10/11)
        self.assertAlmostEqual(bound(1), 27/22)
        self.assertGreater(bound(1), bound(0))
        self.assertEqual(bound(10), 0)
    def test_semantic_reference_subtraction_can_increase_dr_variance(self):
        vectors = ((.8, 0), (0, 1.6))
        differences = [second-first for first, second in vectors]
        mean = sum(differences)/2
        self.assertAlmostEqual(mean, .4)
        self.assertAlmostEqual(sum((value-mean)**2 for value in differences)/2, 1.44)
    def table(self, pairs, representations=('shared', 'shared')):
        return [dict(context=str(i), representation=representations[i], action=a, utility=u)
            for i, pair in enumerate(pairs) for a, u in zip(('a', 'b'), pair)]
    def test_legal_positive_actions_can_reverse_the_best_decision(self):
        rows = self.table(((.9, .6), (.6, .9)))
        original = deepcopy(rows)
        receipt = diagnostic.decisions(rows)
        self.assertAlmostEqual(receipt['mean_regret'], .15)
        self.assertEqual(receipt['positive_regret_contexts'], 1)
        self.assertEqual(receipt['records'][1]['best'], 'b')
        self.assertEqual(rows, original)
    def test_binary_contrast_matches_jensen_gap_with_unequal_support(self):
        receipt = diagnostic.decisions(self.table(((.9, .6), (.6, .9))), {'0': 1, '1': 3})
        self.assertAlmostEqual(receipt['mean_regret'], .075)
        self.assertAlmostEqual(receipt['jensen_gap'], .075)
        self.assertAlmostEqual(receipt['jensen_error'], 0)
        self.assertAlmostEqual(receipt['binary_contrast'][0]['weighted_loss'], .075)
        self.assertEqual(receipt['common_optimal_actions']['shared'], [])
    def test_sharing_boundary_separates_value_variation_from_decision_conflict(self):
        for pairs, share, conflicts in ((((.9, .6), (.6, .9)), False, 1),
                                       (((.95, .85), (.15, .05)), True, 0)):
            receipt = diagnostic.decisions(self.table(pairs))
            boundary = receipt['sharing_boundary'][0]
            self.assertEqual(boundary['can_share_decision'], share)
            self.assertEqual(receipt['conflicting_groups'], conflicts)
            self.assertAlmostEqual(boundary['weighted_loss'], receipt['mean_regret'])
            self.assertEqual(boundary['contexts'], ['0', '1'])
    def test_refinement_keeps_utilities_actions_and_masses_fixed(self):
        rows, masses = self.table(((.9, .6), (.6, .9))), {'0': 1, '1': 3}
        original = deepcopy((rows, masses))
        receipt = diagnostic.refinement(rows, masses)
        self.assertAlmostEqual(receipt['recoverable_decision_loss'], .075)
        self.assertEqual(receipt['context_preserved']['mean_regret'], 0)
        self.assertEqual(receipt['context_preserved']['conflicting_groups'], 0)
        self.assertEqual([r['mass'] for r in receipt['compressed']['records']],
                         [r['mass'] for r in receipt['context_preserved']['records']])
        self.assertEqual((rows, masses), original)
    def test_more_mass_without_new_conditions_cannot_repair_compression(self):
        result = diagnostic.report([1, 2, 32])
        for row in result['decision_sharing_growth']:
            self.assertAlmostEqual(row['boundary']['compressed']['mean_regret'], .15)
            self.assertEqual(row['boundary']['context_preserved']['mean_regret'], 0)
        self.assertEqual([r['mass_multiplier'] for r in result['decision_sharing_growth']], [1, 2, 32])
    def test_controlled_input_is_explicit_and_does_not_infer_missing_actions(self):
        rows = self.table(((.9, .6), (.6, .9)))
        data = dict(format='evocond_controlled_utilities_v1', rows=rows)
        original = deepcopy(data)
        result = diagnostic.controlled(data)
        self.assertAlmostEqual(result['decision_boundary']['recoverable_decision_loss'], .15)
        self.assertFalse(result['gpu_started'])
        self.assertFalse(result['files_written'])
        self.assertEqual(data, original)
        invalid = (None, [], {}, dict(data, format='rollouts'), dict(data, reward=1),
                   dict(data, rows=rows[:-1]), dict(data, rows=rows+[rows[0]]))
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                diagnostic.controlled(value)
    def test_controlled_cli_reads_relative_utf8_input_without_writing(self):
        data = dict(format='evocond_controlled_utilities_v1', rows=self.table(((.9, .6), (.6, .9))))
        output = StringIO()
        with patch('sys.argv', ['diagnostic.py', '--table', 'utilities.json']), \
             patch('diagnostic.Path.read_text', return_value=json.dumps(data)) as read, \
             patch('sys.stdout', output):
            diagnostic.main()
        read.assert_called_once_with(encoding='utf-8-sig')
        self.assertFalse(json.loads(output.getvalue())['files_written'])
        self.assertEqual(json.loads(output.getvalue())['format'], 'evocond_controlled_decision_diagnostic_v1')
    def test_controlled_cli_rejects_invalid_or_partial_tables(self):
        data = dict(format='evocond_controlled_utilities_v1', rows=self.table(((.9, .6), (.6, .9)))[:-1])
        for payload in ('{', json.dumps(data)):
            with patch('sys.argv', ['diagnostic.py', '--table', 'utilities.json']), \
                 patch('diagnostic.Path.read_text', return_value=payload), patch('sys.stderr', StringIO()), \
                 self.assertRaises(SystemExit) as error:
                diagnostic.main()
            self.assertEqual(error.exception.code, 2)
    def test_group_sharing_losses_sum_with_mixed_decision_boundaries(self):
        rows = self.table(((.9, .6), (.6, .9)))
        rows += [dict(context='2', representation='separate', action=a, utility=u)
                 for a, u in zip(('a', 'b'), (.7, .4))]
        receipt = diagnostic.decisions(rows, {'0': 1, '1': 3, '2': 2})
        self.assertEqual(receipt['conflicting_groups'], 1)
        self.assertAlmostEqual(sum(r['weighted_loss'] for r in receipt['sharing_boundary']), receipt['mean_regret'])
        self.assertAlmostEqual(receipt['mean_regret'], .05)
    def test_common_best_and_nonoptimal_crossings_need_no_decision_refinement(self):
        rows = self.table(((.3, .2), (.2, .3)))
        rows += [dict(context=str(i), representation='shared', action='c', utility=.8) for i in range(2)]
        result = diagnostic.refinement(rows)
        self.assertEqual(result['recoverable_decision_loss'], 0)
        self.assertTrue(result['compressed']['sharing_boundary'][0]['can_share_decision'])
        self.assertEqual(result['compressed']['sharing_boundary'][0]['shared_action'], 'c')
    def test_common_value_shift_changes_prediction_error_not_decision_loss(self):
        rows = self.table(((.7, .4), (.4, .7)))
        shifted = [dict(row, utility=row['utility']+(.1 if row['context'] == '0' else -.2)) for row in rows]
        first, second = diagnostic.decisions(rows), diagnostic.decisions(shifted)
        self.assertAlmostEqual(first['mean_regret'], second['mean_regret'])
        self.assertNotAlmostEqual(first['utility_prediction_gap'], second['utility_prediction_gap'])
        self.assertEqual(first['common_optimal_actions'], second['common_optimal_actions'])
    def test_large_value_variation_can_matter_less_than_small_margin_reversal(self):
        large = diagnostic.decisions(self.table(((.95, .85), (.15, .05))))
        small = diagnostic.decisions(self.table(((.55, .45), (.45, .55))))
        self.assertGreater(large['utility_prediction_gap'], small['utility_prediction_gap'])
        self.assertEqual(large['mean_regret'], 0)
        self.assertAlmostEqual(small['mean_regret'], .05)
    def test_nonoptimal_pair_reversal_does_not_imply_multiaction_decision_loss(self):
        rows = self.table(((.3, .2), (.2, .3)))
        rows += [dict(context=str(i), representation='shared', action='c', utility=.8) for i in range(2)]
        receipt = diagnostic.decisions(rows)
        self.assertEqual(receipt['common_optimal_actions']['shared'], ['c'])
        self.assertEqual(receipt['mean_regret'], 0)
        self.assertEqual(receipt['binary_contrast'], [])
    def test_ties_and_multiple_representation_cells_obey_zero_loss_condition(self):
        receipt = diagnostic.decisions(self.table(((.5, .5), (.3, .5))))
        self.assertEqual(receipt['common_optimal_actions']['shared'], ['b'])
        self.assertEqual(receipt['mean_regret'], 0)
        separate = diagnostic.decisions(self.table(((.9, .6), (.6, .9)), ('first', 'second')), {'0': 1, '1': 4})
        self.assertAlmostEqual(sum(row['weighted_loss'] for row in separate['binary_contrast']), 0)
        self.assertAlmostEqual(separate['jensen_error'], 0)
    def test_invalid_masses_or_underflow_are_not_silent_changes_of_support(self):
        rows = self.table(((.9, .6), (.6, .9)))
        for masses in ({'0': 1}, {'0': True, '1': 1}, {'0': 0, '1': 1}, {'0': float('nan'), '1': 1}, {'0': 5e-324, '1': 1e308}):
            with self.subTest(masses=masses), self.assertRaises(ValueError):
                diagnostic.decisions(rows, masses)
    def test_conditional_prediction_gap_need_not_change_the_best_action(self):
        receipt = diagnostic.decisions(self.table(((.9, .5), (.6, .2))))
        self.assertGreater(receipt['utility_prediction_gap'], 0)
        self.assertEqual(receipt['mean_regret'], 0)
        self.assertEqual(receipt['positive_regret_contexts'], 0)
    def test_separate_representations_preserve_each_controlled_decision(self):
        receipt = diagnostic.decisions(self.table(((.9, .6), (.6, .9)), ('first', 'second')))
        self.assertEqual(receipt['utility_prediction_gap'], 0)
        self.assertEqual(receipt['mean_regret'], 0)
        self.assertAlmostEqual(diagnostic.decisions(self.table(((1, 1), (.5, .5))))['mean_regret'], 0)
    def test_incomplete_or_duplicate_utilities_are_not_counterfactual_evidence(self):
        rows = self.table(((.9, .6), (.6, .9)))
        for current in (rows[:-1], rows+[rows[0]], [dict(rows[0], representation='other'), *rows[1:]]):
            with self.subTest(current=current), self.assertRaises(ValueError):
                diagnostic.decisions(current)
    def test_controlled_utility_values_must_be_finite_and_bounded(self):
        rows = self.table(((.9, .6), (.6, .9)))
        for value in (True, float('nan'), float('inf'), 2, '1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                diagnostic.decisions([dict(rows[0], utility=value), *rows[1:]])
    def test_controlled_regret_is_not_reported_as_agent_performance(self):
        receipt = diagnostic.report([1])
        self.assertGreater(receipt['decision_reversal']['mean_regret'], 0)
        self.assertEqual(receipt['common_best_action_control']['mean_regret'], 0)
        self.assertIn('not_observed_counterfactuals_or_agent_results', receipt['decision_reversal']['interpretation'])
    def test_fiber_holds_the_entire_feedback_sequence_fixed(self):
        row = diagnostic.fiber()
        self.assertEqual(row['assignments'], 1022)
        self.assertEqual(row['feedback'], [1]*8+[-1]*2)
        self.assertEqual((row['minimum'], row['maximum'], row['diameter']), (-1, 1, 2))
    def test_identical_ordered_feedback_allows_opposite_local_signs(self):
        row = diagnostic.replay()
        self.assertTrue(row['same_ordered_aggregate'])
        self.assertTrue(row['same_unconditional_weight'])
        self.assertTrue(row['same_erased_weight'])
        self.assertTrue(row['conditional_sign_reversal'])
        for history in row['histories']:
            self.assertEqual(history['local']['omitted_mass'], 0)
            self.assertAlmostEqual(history['risk']['heterogeneity'], .64)
    def test_time_decay_does_not_distinguish_the_paired_aggregate(self):
        row = diagnostic.replay(decay=.8)
        self.assertTrue(row['same_unconditional_weight'])
        self.assertTrue(row['conditional_sign_reversal'])
    def test_retaining_events_cannot_restore_collapsed_features(self):
        row = diagnostic.replay(collapse=True)
        self.assertFalse(row['conditional_sign_reversal'])
        for history in row['histories']:
            self.assertEqual(history['local']['weight'], history['unconditional']['weight'])
            self.assertAlmostEqual(history['risk']['compression_gap'], .64)
    def test_retention_growth_does_not_resolve_erased_condition(self):
        result = diagnostic.report([1, 2, 4])
        self.assertEqual([r['synthetic_bases'] for r in result['growth']], [10, 20, 40])
        self.assertTrue(all(r['conditional_sign_reversal'] for r in result['growth']))
        self.assertFalse(result['gpu_started'])
        self.assertFalse(result['files_written'])
        self.assertAlmostEqual(result['invariant_control']['compression_gap'], 0)
        self.assertGreater(result['invariant_control']['conditional_risk'], 0)
    def test_same_tools_change_order_without_an_illegal_dependency(self):
        first, second = diagnostic.filters(1, 4, .1, .5), diagnostic.filters(1, 4, .95, .01)
        self.assertEqual(first['best_order'], 'ab')
        self.assertEqual(second['best_order'], 'ba')
        self.assertAlmostEqual(first['cost_difference'], -3.1)
        self.assertAlmostEqual(second['cost_difference'], .79)
        self.assertEqual(diagnostic.filters(1, 1, .5, .5)['best_order'], 'tie')
    def test_representation_risk_identity_and_zero_loss_for_identity(self):
        _, statistics = diagnostic.histories()
        self.assertTrue(all(abs(row['identity_error']) < 1e-12 for row in statistics))
        self.assertTrue(all(row['compression_gap'] == 0 for row in statistics))
        rows = [dict(context=str(i), representation='all', feedback=(-1)**i) for i in range(4)]
        original = deepcopy(rows)
        self.assertAlmostEqual(diagnostic.risk(rows)['compression_gap'], 1)
        self.assertEqual(rows, original)
    def test_stochastic_context_representation_is_not_the_stated_identity(self):
        rows = [dict(context='a', representation=str(i), feedback=1) for i in range(2)]
        with self.assertRaises(ValueError):
            diagnostic.risk(rows)
    def test_invalid_values_and_unbounded_synthetic_work_are_rejected(self):
        for value in (True, float('nan'), float('inf'), 2, '1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                diagnostic.risk([dict(context='a', representation='b', feedback=value)])
        for value in (0, 65, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                diagnostic.replay(value)
        with self.assertRaises(ValueError):
            diagnostic.report([1, 1])
        for value in ([[]], [True], [0], [65], [1.5]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                diagnostic.report(value)
        for values in ((0, 1, .5, .5), (1, 1, -1, .5), (1, 1, .5, 2), (True, 1, .5, .5)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                diagnostic.filters(*values)
    def test_existing_reversal_helper_preserves_feedback_chronology(self):
        left, right = relations.reversal()
        self.assertEqual(left['conditional'][0], -right['conditional'][0])
        self.assertAlmostEqual(left['gap'], right['gap'])
if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
