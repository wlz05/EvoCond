# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Known-propensity memory diagnostics and paired checkpoint attribution.'
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import fatigue as command
from control import bandit, contracts, dynamic, fatigue
from tests.bandits import records

def observed(base, chosen, *, success, snapshot='fixed'):
    row = records(1)[0]
    row.update(base_id=str(base), snapshot=snapshot, actions=[0, 2], probabilities=[.8, .2],
        chosen=chosen, payload_hashes=['semantic', 'structural'], success=success, phase=0)
    row['reward'] = bandit.reward(success, row['costs'], penalty=row['penalty'], reference=row['reference'])
    return row

def zero(base, status='unreached'):
    row = observed(base, 0, success=True)
    row.update(features=[], actions=[], probabilities=[], chosen=None, payload_hashes=[], interventions=0, status=status)
    return row

class FatigueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_known_propensity_estimator_recovers_difference_in_expectation(self):
        rows = [observed(i, 0 if i < 80 else 2, success=i >= 80) for i in range(100)]
        report = fatigue.window(rows)
        self.assertAlmostEqual(report['terminal_success_lift']['mean'], 1.0)
        self.assertAlmostEqual(report['clipped_cost_lift']['mean'], 0.0)
        self.assertAlmostEqual(report['net_utility_lift']['mean'], 1.0)
        self.assertEqual(report['minimum_propensity'], .2)
        self.assertFalse(report['deployment_decision'])

    def test_absorbing_questions_remain_in_denominator(self):
        row = observed('active', 2, success=True)
        active = fatigue.window([row])['net_utility_lift']['mean']
        report = fatigue.window([row, zero('a'), zero('b', 'unsupported'), zero('c', 'single_class')])
        self.assertEqual(report['net_utility_lift']['n'], 4)
        self.assertAlmostEqual(report['net_utility_lift']['mean'], active/4)
        self.assertEqual(report['observed_marks'], 1)

    def test_no_logging_is_unavailable_not_a_zero_gain(self):
        self.assertIsNone(fatigue.window([])['net_utility_lift']['mean'])
        report = fatigue.window([zero('a'), zero('b')])
        self.assertEqual(report['net_utility_lift']['mean'], 0)
        self.assertEqual(report['net_utility_lift']['sample_std'], 0)

    def test_utility_components_match_the_logged_reward_scale(self):
        rows = [observed('a', 0, success=False), observed('b', 2, success=True)]
        rows[1]['costs']['actor'] = 15.0
        rows[1]['reward'] = bandit.reward(True, rows[1]['costs'], penalty=.1, reference=10.0)
        report = fatigue.window(rows)
        self.assertAlmostEqual(report['net_utility_lift']['mean'], report['terminal_success_lift']['mean']-.1*report['clipped_cost_lift']['mean'])
        self.assertGreater(report['actual_mean_cost']['mean'], 4)

    def test_uniform_target_uses_real_classes_not_nominal_ratios(self):
        row = records(1)[0]
        row.update(chosen=1, success=True)
        row['reward'] = bandit.reward(True, row['costs'], penalty=.1, reference=10)
        expected = (.5/.3)*(row['reward']-.45)
        self.assertAlmostEqual(fatigue.window([row])['net_utility_lift']['mean'], expected)

    def test_reject_duplicate_question_and_mixed_bottom(self):
        for rows in ([zero('same'), zero('same')], [observed('a', 0, success=True), observed('b', 2, success=True, snapshot='other')]):
            with self.assertRaises(ValueError):
                fatigue.window(rows)

    def test_support_counts_are_separate_and_shared_edges_count_once(self):
        edge = dict(source='a', target='b', mass=3.0, weight=.7)
        result = dict(candidates=[dict(edges=[edge]), dict(edges=[deepcopy(edge)])], abstention=dict(effective_count=2))
        support = fatigue.support(result)
        self.assertEqual(support['conditional_mass'], 3)
        self.assertEqual(support['queried_edges'], 1)
        self.assertEqual(support['positive_edges'], 1)
        row = observed('q', 2, success=False)
        row['relation_support'] = support
        report = fatigue.window([row, zero('unseen')])
        self.assertEqual(report['support_at_randomized_points']['conditional_mass']['n'], 1)
        self.assertEqual(report['net_utility_lift']['n'], 2)
        self.assertLess(report['net_utility_lift']['mean'], 0)
        result['candidates'][1]['edges'][0]['mass'] = 4
        with self.assertRaises(ValueError):
            fatigue.support(result)

    def test_trace_uses_completed_windows_and_exposes_incomplete_tail(self):
        (self.root/'phases').mkdir()
        row = zero('q')
        tail = zero('next')
        tail.update(phase=1, snapshot='next')
        (self.root/'interventions.jsonl').write_text(json.dumps(row)+'\n'+json.dumps(tail)+'\n', encoding='utf-8')
        report = dict(phase=0, snapshot='fixed', base_observations=1, memory_sizes={'read_snapshot': {'relations': 1}})
        (self.root/'phases/phase0.json').write_text(json.dumps(report), encoding='utf-8')
        output = fatigue.trace(self.root)
        self.assertEqual(len(output['phases']), 1)
        self.assertEqual(output['incomplete_observations'], 1)
        report['base_observations'] = 2
        (self.root/'phases/phase0.json').write_text(json.dumps(report), encoding='utf-8')
        with self.assertRaises(ValueError):
            fatigue.trace(self.root)

    def trace_fixture(self, payload=None, count=1):
        (self.root/'phases').mkdir(exist_ok=True)
        payload = (json.dumps(zero('q'))+'\n').encode() if payload is None else payload
        (self.root/'interventions.jsonl').write_bytes(payload)
        report = dict(phase=0, snapshot='fixed', base_observations=count)
        (self.root/'phases/phase0.json').write_text(json.dumps(report), encoding='utf-8')
        return payload

    def test_trace_streams_and_hashes_the_captured_bytes_including_bom(self):
        payload = self.trace_fixture(b'\xef\xbb\xbf'+(json.dumps(zero('q'))+'\n').encode())
        with patch.object(Path, 'read_text', side_effect=AssertionError('Use streamed log reads')):
            report = fatigue.trace(self.root)
        self.assertEqual(report['intervention_sha256'], sha256(payload).hexdigest())
        self.assertEqual(report['intervention_bytes'], len(payload))

    def test_later_appends_do_not_change_the_analyzed_prefix_checksum(self):
        payload = self.trace_fixture()
        original = fatigue.window
        def append(rows):
            tail = zero('future')
            tail.update(phase=1, snapshot='future')
            with (self.root/'interventions.jsonl').open('ab') as stream:
                stream.write((json.dumps(tail)+'\n').encode())
            return original(rows)
        with patch.object(fatigue, 'window', side_effect=append):
            report = fatigue.trace(self.root)
        self.assertEqual(report['intervention_sha256'], sha256(payload).hexdigest())
        self.assertEqual(report['incomplete_observations'], 0)
        self.assertGreater((self.root/'interventions.jsonl').stat().st_size, report['intervention_bytes'])

    def test_uncommitted_partial_tail_is_counted_but_completed_data_is_required(self):
        raw = (json.dumps(zero('q'))+'\n').encode()
        fragment = b'{"phase":1,"label":"\xe4'
        self.trace_fixture(raw+fragment)
        report = fatigue.trace(self.root)
        self.assertEqual(report['incomplete_tail_bytes'], len(fragment))
        self.trace_fixture(raw+fragment, count=2)
        with self.assertRaises(ValueError):
            fatigue.trace(self.root)
        self.trace_fixture(raw+b'{invalid}\n')
        with self.assertRaises(ValueError):
            fatigue.trace(self.root)

    def test_trace_rejects_wrong_phase_filename_and_boolean_denominator(self):
        self.trace_fixture()
        path = self.root/'phases/phase0.json'
        report = json.loads(path.read_text(encoding='utf-8'))
        for change in (dict(phase=1), dict(base_observations=True)):
            path.write_text(json.dumps({**report, **change}), encoding='utf-8')
            with self.assertRaises(ValueError):
                fatigue.trace(self.root)

    def test_publish_is_complete_exclusive_and_cleans_failed_staging(self):
        path = self.root/'reports/result.json'
        command.publish(path, '{"complete":true}')
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')), {'complete': True})
        with self.assertRaises(FileExistsError):
            command.publish(path, '{"complete":false}')
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')), {'complete': True})
        self.assertEqual(list(path.parent.glob('.evocond-*.tmp')), [])
        with patch.object(contracts.os, 'fsync', side_effect=OSError('Interrupted write')):
            with self.assertRaises(OSError):
                command.publish(path.parent/'failed.json', '{}')
        self.assertFalse((path.parent/'failed.json').exists())
        self.assertEqual(list(path.parent.glob('.evocond-*.tmp')), [])

    def fixture(self, *, adopted=False):
        actors, banks = ['a'*64, 'b'*64], ['c'*64, 'd'*64]
        protocol = dict(split='validation', dataset_sha256='e'*64, questions=2, base_ids_sha256=dynamic.digest(['q0', 'q1']), capability_pool_sha256='f'*64,
            execution_sha256='1'*64, presentation='fixed_structure', penalty=.1, reference=10.0)
        manifest = dict(format=fatigue.PANEL, seed=42, actors=actors, banks=banks, protocol=protocol, cells=[])
        costs = dict(actor=1.0, tools=1.0, retrieval=1.0)
        for i, actor in enumerate(actors):
            for j, bank in enumerate(banks):
                rows = []
                for k in range(2):
                    row = dict(base_id='q'+str(k), semantic=dict(success=i == 1, costs=deepcopy(costs)),
                        structural=dict(success=i == 1 or j == 0 or k == 0, costs=deepcopy(costs)))
                    if adopted:
                        row['adopted'] = deepcopy(row['structural'])
                    rows.append(row)
                cell = dict(format=fatigue.CELL, actor=actor, bank=bank, seed=42,
                    protocol_sha256=dynamic.digest(protocol), records=rows)
                if adopted:
                    cell['teacher_sha256'] = '2'*64
                name = f'cell{i}{j}.json'
                payload = json.dumps(cell).encode()
                (self.root/name).write_bytes(payload)
                manifest['cells'].append(dict(actor=actor, bank=bank, path=name, sha256=sha256(payload).hexdigest()))
        return manifest

    def save(self, manifest):
        path = self.root/'panel.json'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        return path

    def change(self, manifest, function):
        entry = manifest['cells'][-1]
        path = self.root/entry['path']
        value = json.loads(path.read_text(encoding='utf-8'))
        function(value)
        payload = json.dumps(value).encode()
        path.write_bytes(payload)
        entry['sha256'] = sha256(payload).hexdigest()

    def test_factorial_panel_separates_actor_bank_and_interaction(self):
        report = fatigue.panel(self.save(self.fixture()))
        terms = {name: value['mean'] for name, value in report['decomposition'].items()}
        self.assertAlmostEqual(terms['actor_at_old_bank'], -1)
        self.assertAlmostEqual(terms['bank_at_old_actor'], -.5)
        self.assertAlmostEqual(terms['interaction'], .5)
        self.assertAlmostEqual(sum(terms[name] for name in ('actor_at_old_bank', 'bank_at_old_actor', 'interaction')), terms['total_change'])
        self.assertEqual(report['task_count'], 2)
        self.assertEqual(report['cells']['actor1_bank1']['outcomes']['structural']['success']['mean'], 1)
        self.assertEqual(report['cells']['actor1_bank1']['net_utility_lift']['mean'], 0)

    def test_benefit_components_keep_one_denominator_and_the_signed_cost(self):
        manifest = self.fixture(adopted=True)
        def change(cell):
            first, second = cell['records']
            first['semantic'].update(success=False, costs=dict(actor=1., tools=1., retrieval=1.))
            first['structural'].update(success=True, costs=dict(actor=4., tools=4., retrieval=4.))
            second['semantic'].update(success=True, costs=dict(actor=4., tools=4., retrieval=4.))
            second['structural'].update(success=False, costs=dict(actor=0., tools=0., retrieval=0.))
        self.change(manifest, change)
        cell = fatigue.panel(self.save(manifest))['cells']['actor1_bank1']
        values = cell['benefit_components']['structural_vs_semantic']
        self.assertEqual(values['rescue_share']['mean'], .5)
        self.assertEqual(values['harm_share']['mean'], .5)
        self.assertAlmostEqual(values['cost_penalty']['mean'], -.015)
        self.assertAlmostEqual(values['net_utility_lift']['mean'], .015)
        self.assertTrue(all(value['n'] == 2 for value in values.values()))
        for comparison, components in cell['benefit_components'].items():
            self.assertAlmostEqual(components['rescue_share']['mean']-components['harm_share']['mean']-
                components['cost_penalty']['mean'], components['net_utility_lift']['mean'], msg=comparison)
        self.assertAlmostEqual(values['net_utility_lift']['mean'], cell['net_utility_lift']['mean'])

    def test_benefit_accounting_rejects_a_different_reward_scale(self):
        rows = [dict(semantic=dict(success=0., clipped_cost=.3, utility=-.03),
            structural=dict(success=1., clipped_cost=1., utility=.9))]
        self.assertAlmostEqual(fatigue.components(rows, .1)['net_utility_lift']['mean'], .93)
        with self.assertRaisesRegex(ValueError, 'components'):
            fatigue.components(rows, .2)
        for value in (True, -1., float('nan')):
            with self.assertRaises(ValueError):
                fatigue.components(rows, value)

    def test_zero_average_success_lift_can_hide_rescues_and_harm(self):
        rows = [dict(semantic=dict(success=left), structural=dict(success=right))
            for left, right in ((False, True), (True, False), (False, False), (True, True))]
        report = fatigue.success_pairs(rows)
        self.assertEqual(report['counts'], dict(both_failed=1, rescued=1, harmed=1, both_succeeded=1))
        self.assertEqual(report['rates']['success_lift'], 0)
        self.assertEqual(report['rates']['rescue_rate'], .5)
        self.assertEqual(report['rates']['harm_rate'], .5)
        self.assertEqual(report['rates']['rescue_share']-report['rates']['harm_share'], 0)

    def test_actor_progress_can_remove_rescue_opportunities_without_harm(self):
        report = fatigue.panel(self.save(self.fixture()))
        old = report['cells']['actor0_bank0']['paired_success']['structural_vs_semantic']
        new = report['cells']['actor1_bank0']['paired_success']['structural_vs_semantic']
        self.assertEqual(old['counts']['rescued'], 2)
        self.assertEqual(old['rates']['rescue_rate'], 1)
        self.assertEqual(new['reference_failures'], 0)
        self.assertIsNone(new['rates']['rescue_rate'])
        self.assertEqual(new['counts']['harmed'], 0)
        self.assertEqual(new['rates']['success_lift'], 0)

    def test_unchanged_conditional_rates_can_reverse_the_observed_value(self):
        old = [(0, 1)]*3+[(0, 0)]*3+[(1, 0), (1, 1)]
        new = [(0, 1), (0, 0)]+[(1, 0)]*3+[(1, 1)]*3
        def report(pairs):
            return fatigue.success_pairs([dict(semantic=dict(success=a), structural=dict(success=b)) for a, b in pairs])
        previous, current = report(old), report(new)
        for name in ('rescue_rate', 'harm_rate'):
            self.assertEqual(previous['rates'][name], .5)
            self.assertEqual(current['rates'][name], .5)
        self.assertEqual(previous['rates']['success_lift'], .25)
        self.assertEqual(current['rates']['success_lift'], -.25)
        self.assertEqual(previous['rates']['reference_failure_rate'], .75)
        self.assertEqual(current['rates']['reference_failure_rate'], .25)

    def transition_cell(self, pairs, extra_cost=0.):
        rows = [dict(semantic=dict(success=float(a), clipped_cost=0., utility=float(a)),
            structural=dict(success=float(b), clipped_cost=extra_cost, utility=b-.1*extra_cost)) for a, b in pairs]
        parts = fatigue.components(rows, .1)
        return dict(paired_success=dict(structural_vs_semantic=fatigue.success_pairs(rows)),
            benefit_components=dict(structural_vs_semantic=parts), net_utility_lift=parts['net_utility_lift'])

    def test_opportunity_composition_alone_explains_constant_rate_value_reversal(self):
        old = self.transition_cell([(0, 1)]*3+[(0, 0)]*3+[(1, 0), (1, 1)])
        new = self.transition_cell([(0, 1), (0, 0)]+[(1, 0)]*3+[(1, 1)]*3)
        change = fatigue.opportunity(old, new)
        self.assertTrue(change['defined'])
        self.assertEqual(change['terms'], dict(opportunity_mix=-.5, rescue_rate=0., harm_rate=0., cost=0., total=-.5))

    def test_opportunity_accounting_separates_response_and_cost_with_symmetric_signs(self):
        old = self.transition_cell([(0, 1)]*3+[(0, 0)]*3+[(1, 0), (1, 1)])
        new = self.transition_cell([(0, 1)]*2+[(1, 1)]*6, extra_cost=.2)
        forward, reverse = fatigue.opportunity(old, new), fatigue.opportunity(new, old)
        expected = dict(opportunity_mix=-.5, rescue_rate=.25, harm_rate=.25, cost=-.02, total=-.02)
        for name, value in expected.items():
            self.assertAlmostEqual(forward['terms'][name], value)
            self.assertAlmostEqual(reverse['terms'][name], -value)
        self.assertAlmostEqual(sum(value for name, value in forward['terms'].items() if name != 'total'),
            forward['terms']['total'])

    def test_opportunity_accounting_preserves_empty_groups_without_zero_imputation(self):
        report = fatigue.panel(self.save(self.fixture()))
        change = report['opportunity_changes']['actor_at_old_bank']
        self.assertFalse(change['defined'])
        self.assertIn('previous:harm_rate', change['undefined'])
        self.assertIn('current:rescue_rate', change['undefined'])
        for name in ('opportunity_mix', 'rescue_rate', 'harm_rate'):
            self.assertIsNone(change['terms'][name])
        self.assertEqual(change['terms']['total'], -1.)
        self.assertEqual(change['terms']['cost'], 0.)
        bad = deepcopy(report['cells']['actor1_bank0'])
        bad['paired_success']['structural_vs_semantic']['n'] = 1
        with self.assertRaises(ValueError):
            fatigue.opportunity(report['cells']['actor0_bank0'], bad)

    def test_partial_opportunity_support_cannot_supply_a_four_seed_mean(self):
        old = self.transition_cell([(0, 1)]*3+[(0, 0)]*3+[(1, 0), (1, 1)])
        new = self.transition_cell([(0, 1), (0, 0)]+[(1, 0)]*3+[(1, 1)]*3)
        defined = fatigue.opportunity(old, new)
        missing = fatigue.opportunity(self.transition_cell([(0, 1)]*8), self.transition_cell([(1, 1)]*8))
        result = fatigue.summarize_changes([(42, defined), (52, missing), (3047, missing), (3407, missing)])
        self.assertEqual(result['defined_seeds']['opportunity_mix'], [42])
        self.assertEqual(result['terms']['opportunity_mix']['n'], 1)
        self.assertIsNone(result['terms']['opportunity_mix']['mean'])
        self.assertIsNone(result['terms']['opportunity_mix']['standard_error'])
        self.assertEqual(result['terms']['total']['n'], 4)
        self.assertEqual(result['terms']['cost']['n'], 4)

    def test_receiver_credit_reverses_while_both_absolute_scores_improve(self):
        old, new = (.4, .6), (.9, .8)
        self.assertTrue(all(right > left for left, right in zip(old, new)))
        report = fatigue.transport([old[1]-old[0]], [new[1]-new[0]])
        metric = report['metrics']
        self.assertAlmostEqual(metric['advantage_change']['mean'], -.3)
        self.assertAlmostEqual(metric['carried_preference_loss']['mean'], .1)
        self.assertAlmostEqual(metric['receiver_blind_loss']['mean'], .05)
        self.assertEqual(metric['positive_to_negative_share']['mean'], 1.)
        self.assertFalse(report['deployment_decision'])
        self.assertIn('hindsight', report['scope'])

    def test_receiver_conflicts_can_exist_without_a_mean_advantage_change(self):
        values = fatigue.transport([1., -1.], [-1., 1.])['metrics']
        self.assertEqual(values['advantage_change']['mean'], 0.)
        self.assertEqual(values['carried_preference_loss']['mean'], 1.)
        self.assertEqual(values['receiver_blind_loss']['mean'], .5)
        self.assertEqual(values['positive_to_negative_share']['mean'], .5)
        self.assertEqual(values['negative_to_positive_share']['mean'], .5)

    def test_receiver_ties_use_semantic_without_becoming_strict_sign_flips(self):
        values = fatigue.transport([0., 1., -1., 0.], [1., 0., 0., 0.])['metrics']
        self.assertEqual(values['source_tie_share']['mean'], .5)
        self.assertEqual(values['target_tie_share']['mean'], .75)
        self.assertEqual(values['carried_preference_loss']['mean'], .25)
        self.assertEqual(values['receiver_blind_loss']['mean'], 0.)
        self.assertEqual(values['negative_to_positive_share']['mean'], 0.)

    def test_receiver_blind_loss_is_symmetric_but_carry_loss_is_directional(self):
        forward = fatigue.transport([.8], [-.2])['metrics']
        reverse = fatigue.transport([-.2], [.8])['metrics']
        self.assertEqual(forward['receiver_blind_loss']['mean'], reverse['receiver_blind_loss']['mean'])
        self.assertEqual(forward['carried_preference_loss']['mean'], .2)
        self.assertEqual(reverse['carried_preference_loss']['mean'], .8)
        for row in (forward, reverse):
            self.assertAlmostEqual(row['current_preference_gain']['mean']-row['carried_preference_gain']['mean'],
                row['carried_preference_loss']['mean'])

    def test_receiver_decline_without_preference_conflict_has_no_selection_loss(self):
        values = fatigue.transport([.8, -.2], [.1, -.9])['metrics']
        self.assertLess(values['advantage_change']['mean'], 0.)
        self.assertEqual(values['carried_preference_loss']['mean'], 0.)
        self.assertEqual(values['receiver_blind_loss']['mean'], 0.)

    def test_receiver_comparisons_reject_missing_or_nonfinite_advantages(self):
        for left, right in (([], []), ([0.], []), ([True], [0.]), (['1'], [0.]),
                ([float('inf')], [0.]), ([10**1000], [0.]), ([1e308], [-1e308])):
            with self.assertRaises(ValueError):
                fatigue.transport(left, right)

    def test_receiver_comparison_retains_task_denominators_and_panel_direction(self):
        report = fatigue.panel(self.save(self.fixture()))
        transfer = report['receiver_transfer']['bank0']
        self.assertEqual(transfer['n'], 2)
        self.assertEqual(transfer['source_actor'], 'a'*64)
        self.assertEqual(transfer['target_actor'], 'b'*64)
        self.assertEqual(transfer['bank'], 'c'*64)
        self.assertEqual(transfer['metrics']['previous_advantage']['mean'], 1.)
        self.assertEqual(transfer['metrics']['current_advantage']['mean'], 0.)
        self.assertTrue(all(row['n'] == 2 for row in transfer['metrics'].values()))

    def test_empty_transition_groups_stay_undefined_and_nonbinary_outcomes_fail(self):
        report = fatigue.success_pairs([])
        self.assertEqual(report['n'], 0)
        self.assertTrue(all(value is None for value in report['rates'].values()))
        with self.assertRaises(ValueError):
            fatigue.success_pairs([dict(semantic=dict(success=0), structural=dict(success=.5))])

    def test_extra_cost_can_lower_utility_without_success_interference(self):
        manifest = self.fixture()
        self.change(manifest, lambda cell: [row['structural']['costs'].update(actor=20., tools=20., retrieval=20.) for row in cell['records']])
        report = fatigue.panel(self.save(manifest))
        cell = report['cells']['actor1_bank1']
        self.assertEqual(cell['success_lift']['mean'], 0)
        self.assertEqual(cell['paired_success']['structural_vs_semantic']['counts']['harmed'], 0)
        self.assertEqual(cell['cost_lift']['mean'], 57)
        self.assertAlmostEqual(cell['clipped_cost_lift']['mean'], .7)
        self.assertAlmostEqual(cell['net_utility_lift']['mean'], -.07)
        self.assertAlmostEqual(cell['net_utility_lift']['mean'], cell['success_lift']['mean']-.1*cell['clipped_cost_lift']['mean'])

    def test_randomized_single_branch_logs_do_not_supply_paired_outcomes(self):
        self.assertNotIn('paired_success', fatigue.window([observed('q', 2, success=True)]))

    def test_optional_adoption_recovery_is_separate_from_fixed_target(self):
        report = fatigue.panel(self.save(self.fixture(adopted=True)))
        self.assertEqual(report['cells']['actor1_bank1']['adopted_vs_structural']['mean'], 0)
        self.assertEqual(report['decomposition']['total_change']['mean'], -1)
        manifest = self.fixture(adopted=True)
        self.change(manifest, lambda cell: cell.pop('teacher_sha256'))
        with self.assertRaises(ValueError):
            fatigue.panel(self.save(manifest))

    def test_adoption_reports_which_structural_failures_it_recovers(self):
        manifest = self.fixture(adopted=True)
        self.change(manifest, lambda cell: cell['records'][0]['structural'].update(success=False))
        cell = fatigue.panel(self.save(manifest))['cells']['actor1_bank1']
        report = cell['paired_success']['adopted_vs_structural']
        self.assertEqual(report['reference'], 'structural')
        self.assertEqual(report['alternative'], 'adopted')
        self.assertEqual(report['counts']['rescued'], 1)
        self.assertEqual(report['counts']['harmed'], 0)
        self.assertEqual(cell['paired_success']['adopted_vs_semantic']['rates']['success_lift'], 0)

    def test_panel_rejects_missing_cell_and_repeated_checkpoint(self):
        manifest = self.fixture()
        for name in ('missing', 'repeated'):
            altered = deepcopy(manifest)
            if name == 'missing':
                altered['cells'].pop()
            else:
                altered['actors'][1] = altered['actors'][0]
            with self.assertRaises(ValueError):
                fatigue.panel(self.save(altered))

    def test_panel_rejects_unpaired_questions_and_duplicate_outcomes(self):
        for duplicate in (False, True):
            manifest = self.fixture()
            self.change(manifest, lambda cell: cell['records'][1].update(base_id='q0' if duplicate else 'unmatched'))
            with self.assertRaises(ValueError):
                fatigue.panel(self.save(manifest))

    def test_matching_partial_cells_cannot_shrink_the_reference_denominator(self):
        manifest = self.fixture()
        for entry in manifest['cells']:
            path = self.root/entry['path']
            cell = json.loads(path.read_text(encoding='utf-8'))
            cell['records'].pop()
            payload = json.dumps(cell).encode()
            path.write_bytes(payload)
            entry['sha256'] = sha256(payload).hexdigest()
        with self.assertRaises(ValueError):
            fatigue.panel(self.save(manifest))

    def test_panel_rejects_wrong_binding_and_corrupt_source(self):
        manifest = self.fixture()
        self.change(manifest, lambda cell: cell.update(protocol_sha256='0'*64))
        with self.assertRaises(ValueError):
            fatigue.panel(self.save(manifest))
        manifest = self.fixture()
        manifest['cells'][0]['sha256'] = '0'*64
        with self.assertRaises(ValueError):
            fatigue.panel(self.save(manifest))

    def test_panel_source_hash_identifies_the_manifest_actually_parsed(self):
        path = self.save(self.fixture())
        payload = path.read_bytes()
        original = fatigue.outcome
        def replace(row, protocol):
            path.write_text('{"replaced":true}', encoding='utf-8')
            return original(row, protocol)
        with patch.object(fatigue, 'outcome', side_effect=replace):
            report = fatigue.panel(path)
        self.assertEqual(report['source_sha256'], sha256(payload).hexdigest())
        self.assertNotEqual(report['source_sha256'], sha256(path.read_bytes()).hexdigest())

    def test_panel_rejects_nonfinite_cost_and_partial_adoption(self):
        for mode in ('cost', 'adoption'):
            manifest = self.fixture(adopted=True)
            if mode == 'cost':
                self.change(manifest, lambda cell: cell['records'][0]['semantic']['costs'].update(actor=float('nan')))
            else:
                self.change(manifest, lambda cell: cell['records'][0].pop('adopted'))
            with self.assertRaises(ValueError):
                fatigue.panel(self.save(manifest))

    def test_panel_rejects_training_split_and_path_escape(self):
        manifest = self.fixture()
        for mode in ('split', 'path'):
            altered = deepcopy(manifest)
            if mode == 'split':
                altered['protocol']['split'] = 'train'
            else:
                altered['cells'][0]['path'] = '../outside.json'
            with self.assertRaises(ValueError):
                fatigue.panel(self.save(altered))

    def test_panel_rejects_nonportable_cell_paths(self):
        manifest = self.fixture()
        for name in ('C:outside.json', 'cells\\outside.json', 'cells/NUL.json'):
            altered = deepcopy(manifest)
            altered['cells'][0]['path'] = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                fatigue.panel(self.save(altered))

    def test_panel_rejects_cell_symlink_outside_manifest_root(self):
        manifest = self.fixture()
        entry = manifest['cells'][0]
        path = self.root/entry['path']
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'cell.json'
            target.write_bytes(path.read_bytes())
            link = self.root/'linked.json'
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest('File symlinks are unavailable')
            entry['path'] = link.name
            try:
                with self.assertRaisesRegex(ValueError, 'declared root'):
                    fatigue.panel(self.save(manifest))
            finally:
                link.unlink()

    def panels(self):
        original, paths = self.root, []
        try:
            for seed in (42, 52, 3047, 3407):
                self.root = original/str(seed)
                self.root.mkdir()
                manifest = self.fixture()
                manifest['seed'] = seed
                for entry in manifest['cells']:
                    path = self.root/entry['path']
                    cell = json.loads(path.read_text(encoding='utf-8'))
                    cell['seed'] = seed
                    payload = json.dumps(cell).encode()
                    path.write_bytes(payload)
                    entry['sha256'] = sha256(payload).hexdigest()
                paths.append(self.save(manifest))
        finally:
            self.root = original
        return paths

    def test_four_seed_summary_uses_runs_as_the_uncertainty_unit(self):
        paths = self.panels()
        report = fatigue.aggregate(paths)
        self.assertEqual(report['decomposition']['total_change']['n'], 4)
        self.assertEqual(report['decomposition']['total_change']['mean'], -1)
        self.assertEqual(report['decomposition']['total_change']['sample_std'], 0)
        changes = report['opportunity_changes']['actor_at_old_bank']
        self.assertEqual(changes['terms']['total']['mean'], -1)
        self.assertEqual(changes['terms']['total']['n'], 4)
        self.assertEqual(changes['terms']['opportunity_mix']['n'], 0)
        self.assertIsNone(changes['terms']['opportunity_mix']['mean'])
        outcomes = report['cells']['actor1_bank1']['outcomes']
        self.assertEqual(outcomes['structural']['success']['n'], 4)
        self.assertEqual(outcomes['structural']['success']['mean'], 1)
        self.assertEqual(outcomes['semantic']['success']['mean'], 1)
        components = report['cells']['actor1_bank1']['benefit_components']['structural_vs_semantic']
        for metric, values in components.items():
            self.assertEqual(values['n'], 4, msg=metric)
            self.assertEqual(set(values['per_seed']), {'42', '52', '3047', '3407'})
        self.assertAlmostEqual(components['rescue_share']['mean']-components['harm_share']['mean']-
            components['cost_penalty']['mean'], components['net_utility_lift']['mean'])
        transfers = report['receiver_transfer']['bank0']
        self.assertFalse(transfers['deployment_decision'])
        for values in transfers['metrics'].values():
            self.assertEqual(values['n'], 4)
            self.assertEqual(set(values['per_seed']), {'42', '52', '3047', '3407'})
        with self.assertRaises(ValueError):
            fatigue.aggregate(paths[:3])
        with self.assertRaises(ValueError):
            fatigue.aggregate([paths[0]]*4)

    def test_undefined_conditional_rates_cannot_be_averaged_over_selected_seeds(self):
        paths = self.panels()
        manifest = json.loads(paths[0].read_text(encoding='utf-8'))
        original = self.root
        try:
            self.root = paths[0].parent
            self.change(manifest, lambda cell: cell['records'][0]['semantic'].update(success=False))
            self.save(manifest)
        finally:
            self.root = original
        paired = fatigue.aggregate(paths)['cells']['actor1_bank1']['paired_success']['structural_vs_semantic']
        self.assertEqual(paired['defined_seeds']['rescue_rate'], [42])
        self.assertEqual(paired['rates']['rescue_rate']['n'], 1)
        self.assertIsNone(paired['rates']['rescue_rate']['mean'])
        self.assertEqual(paired['rates']['success_lift']['n'], 4)
        self.assertEqual(paired['rates']['success_lift']['mean'], .125)
        self.assertEqual(paired['per_seed']['42']['reference_failures'], 1)
        self.assertEqual(paired['per_seed']['52']['reference_failures'], 0)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
