# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Paired evidence must reject leakage, missing cases and invented measurements.'
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import assessment
import design
from control.dynamic import digest
from control.outcomes import FORMAT, measured
from experiment import read
class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = {'contextual': {'seed': 42}}
        self.manifest = dict(format='evocond_paired_assessment_v1', split='test', comparison='fixed_actor_bank',
            bootstrap=dict(seed=42, samples=100), protocol={key: 'a'*64 for key in ('split', 'judge', 'executor', 'budget')},
            tasks=[dict(task_id=str(i), family='first' if i < 2 else 'second', stratum='reversal' if i < 2 else 'invariant') for i in range(4)],
            baseline='semantic', runs=[])
        self.rows = {}
        for method in ('semantic', 'conditional'):
            (self.root/(method+'.json')).write_text(json.dumps(self.runtime), encoding='utf-8')
            self.manifest['runs'].append(dict(method=method, seed=42, outcomes=method+'.jsonl', runtime=method+'.json', policy_step=1))
            self.rows[method] = [dict(format=FORMAT, task_id=str(i), episode_id=method+str(i), split='test',
                policy_step=1, collection_branch=False, success=i > 1 if method == 'semantic' else i > 0,
                actor='a'*64, bank='b'*64, runtime_sha256=digest(self.runtime), metrics=dict(tool_calls=3, elapsed_seconds=2.0)) for i in range(4)]
        self.save()
    def save(self):
        for method, rows in self.rows.items():
            (self.root/(method+'.jsonl')).write_text('\ufeff'+'\r\n'.join(json.dumps(row) for row in rows)+'\r\n', encoding='utf-8')
    def test_paired_rates_and_unknown_costs(self):
        result = assessment.assess(self.manifest, self.root)
        self.assertEqual(result['base_questions'], 4)
        self.assertEqual(result['paired']['conditional']['percentage_points'], 25)
        self.assertEqual(result['paired']['conditional']['relative_percent'], 50)
        self.assertEqual(result['paired']['conditional']['gain'], 1)
        self.assertFalse(result['paired']['conditional']['confidence']['informative'])
        self.assertIsNone(result['methods']['conditional']['metrics']['accounted_cost']['all'])
        self.assertFalse(result['SOTA_verified'])
        self.assertIsNone(result['methods']['conditional']['metrics']['payload_changes']['all'])
        self.assertIsNone(result['methods']['conditional']['metrics']['selected_payload_bytes']['all'])
    def test_missing_and_duplicate_draws_are_not_silently_removed(self):
        original = deepcopy(self.rows)
        for mode in ('missing', 'duplicate'):
            self.rows = deepcopy(original)
            if mode == 'missing':
                self.rows['conditional'].pop()
            else:
                self.rows['conditional'].append(self.rows['conditional'][0])
            self.save()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                assessment.assess(self.manifest, self.root)
    def test_fixed_actor_and_bank_attribution_is_checked(self):
        self.rows['conditional'][0]['actor'] = 'c'*64
        self.save()
        with self.assertRaises(ValueError):
            assessment.assess(self.manifest, self.root)
        self.manifest['comparison'] = 'end_to_end'
        self.assertEqual(assessment.assess(self.manifest, self.root)['comparison'], 'end_to_end')
    def test_unknown_correctness_and_runtime_provenance_are_refused(self):
        original = deepcopy(self.rows)
        for field, value in (('success', None), ('success', 1), ('collection_branch', True), ('runtime_sha256', 'c'*64)):
            self.rows = deepcopy(original)
            self.rows['conditional'][0][field] = value
            self.save()
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                assessment.assess(self.manifest, self.root)
    def test_nonfinite_and_negative_metrics_are_refused(self):
        for value in (-1, float('nan'), True):
            self.rows['conditional'][0]['metrics']['elapsed_seconds'] = value
            self.save()
            with self.subTest(value=value), self.assertRaises(ValueError):
                assessment.assess(self.manifest, self.root)
    def test_seed_panels_and_task_families_are_explicit(self):
        current = deepcopy(self.manifest)
        current['runs'][0]['seed'] = 7
        with self.assertRaises(ValueError):
            assessment.validate(current)
        current = deepcopy(self.manifest)
        current['tasks'].append(current['tasks'][0])
        with self.assertRaises(ValueError):
            assessment.validate(current)
    def test_no_successful_failure_cost_illusion(self):
        for method in self.rows:
            for row in self.rows[method]:
                row['metrics']['accounted_cost'] = 10 if row['success'] else 0
        self.save()
        result = assessment.assess(self.manifest, self.root)
        metrics = result['methods']['semantic']['metrics']['accounted_cost']
        self.assertEqual(metrics['all']['mean'], 5)
        self.assertEqual(metrics['successful_only']['mean'], 10)
class DesignTests(unittest.TestCase):
    def test_method_positioning_has_one_object_and_explicit_lineage(self):
        root = Path(__file__).resolve().parents[1]
        result = design.generate(read(root/'configs/contextual.json'), root/'configs/unused')
        contract = result['method_contract']
        self.assertEqual(contract['object'], 'conditional_relation_reuse_under_coevolution')
        self.assertEqual(contract['unit'], 'conditioned_proposal_on_shared_function_identity')
        self.assertEqual(contract['decision_rule'], 'conditional_support_proposes_current_actor_relative_utility_selects')
        self.assertEqual(contract['update_schedule']['relation'], 'query_local_reconstruction')
        self.assertEqual(contract['update_schedule']['adoption'], 'fixed_execution_window_reestimation')
        self.assertEqual(contract['contributions'], ['identity_sharing_with_condition_preserving_role_reconstruction',
            'receiver_relative_actual_payload_adoption_learning', 'execution_bound_coevolution_updates'])
        self.assertFalse(contract['exclusive_priority_claim'])
        self.assertIn('DeMem', result['decision_contract']['lineage'])
        self.assertIn('doubly robust estimation', contract['inherited_tools'])
    def test_related_method_targets_are_not_official_recipe_implementations(self):
        root = Path(__file__).resolve().parents[1]
        result = design.generate(read(root/'configs/contextual.json'), root/'configs/unused')
        papers = {row['name']: row for row in result['literature']}
        self.assertEqual(papers['DeMem']['version'], '2605.10870v1')
        self.assertEqual(papers['MemCalib']['version'], '2609.24259v3')
        self.assertEqual(papers['DeMem']['venue'], 'preprint')
        self.assertEqual(papers['MemCalib']['venue'], 'preprint')
        self.assertIn('independent ideal-use labels', papers['MemCalib']['control'])
        self.assertNotIn('DeMem', {row['name'] for row in result['recipes']})
        self.assertNotIn('MemCalib', {row['name'] for row in result['recipes']})
        self.assertEqual(result['method_contract']['comparison_status'], 'literature_targets_not_implemented_official_baselines')
    def test_contrasts_preserve_common_budget_and_data(self):
        root = Path(__file__).resolve().parents[1]
        config = read(root/'configs/contextual.json')
        original = deepcopy(config)
        rows = design.recipes(config)
        self.assertEqual(len(rows), 12)
        self.assertEqual(config, original)
        for row in rows:
            self.assertEqual(row['config']['budget'], config['budget'])
            self.assertEqual(row['config']['datasets'], config['datasets'])
            self.assertEqual(row['config']['contextual']['recall'], config['contextual']['recall'])
        self.assertEqual(next(row for row in rows if row['name'] == 'whole')['config']['contextual']['variant'], 'whole')
        simple = next(row for row in rows if row['name'] == 'simple')['config']['contextual']
        self.assertEqual(simple['metric']['protection'], 'identity')
        self.assertIsNone(simple['fixed_alpha'])
        self.assertEqual(next(row for row in rows if row['name'] == 'bernstein')['config']['contextual']['confidence_method'], 'bernstein')
    def test_dry_generation_does_not_launch_or_write(self):
        root = Path(__file__).resolve().parents[1]
        config = read(root/'configs/contextual.json')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'new'
            result = design.generate(config, output)
            self.assertFalse(output.exists())
            self.assertEqual(result['target_arm'], 'teacher')
            self.assertIn('old/new actor crossed with old/new bank', result['missing_experiments'])
            self.assertTrue(result['diagnostic']['synthetic'])
            self.assertFalse(result['diagnostic']['actor_loaded'])
            self.assertEqual(result['diagnostic']['command'], 'python -B diagnostic.py')
            self.assertEqual(result['exposure']['format'], 'evocond_retrieval_exposure_v1')
            self.assertIn('target recall without independent annotation', result['exposure']['unknown'])
            self.assertEqual(result['decision_contract']['invariance'], 'per_state_common_value_shift')
            self.assertEqual(result['teacher_contract']['receipt'], 'studentized_dr_normal_v1')
            self.assertEqual(result['teacher_contract']['strict_receipt'], 'zero_sum_dr_global_envelope_v1')
            self.assertIn('nonoptimal pair reversal', result['decision_contract']['negative_controls'])
            self.assertIn('common-value-shift versus small-margin-reversal with matched support', result['missing_experiments'])
class OutcomeTests(unittest.TestCase):
    def test_unknown_usage_is_not_zero(self):
        context = dict(task_id='task', split='test', policy_step=1, terminal_success=False, invocations=[])
        row = measured(context, episode='ep', arm='teacher', actor='a'*64, bank='b'*64, elapsed=2)
        self.assertIsNone(row['metrics']['actor_input_tokens'])
        self.assertIsNone(row['metrics']['accounted_cost'])
        self.assertIsNone(row['metrics']['retrievals'])
        self.assertFalse(row['success'])
    def test_fallback_requires_the_actual_semantic_payload(self):
        context = dict(task_id='task', split='test', policy_step=1, terminal_success=True, invocations=[],
            costs=dict(actor=2, tools=3, retrieval=1), actor_metered=True,
            cost_receipts=[dict(kind='actor', quantities=dict(input_tokens=5, output_tokens=2)), dict(kind='retrieval', quantities=dict(seconds=.2))],
            selection_cache={0: dict(alpha=0, abstention=dict(semantic_fallback=True), payload_hash='p', semantic_payload_hash='p')})
        row = measured(context, episode='ep', arm='teacher', actor='a'*64, bank='b'*64, elapsed=2)
        self.assertEqual(row['metrics']['fallback_payload_matches'], 1)
        self.assertEqual(row['metrics']['semantic_fallbacks'], 1)
        context['selection_cache'][0]['payload_hash'] = 'other'
        self.assertEqual(measured(context, episode='ep', arm='teacher', actor='a'*64, bank='b'*64, elapsed=2)['metrics']['fallback_payload_matches'], 0)
if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
