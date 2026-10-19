# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from copy import deepcopy
import unittest
from control import dynamic, edges, boundary as boundary
from control.ranking import build_alignment_pairs, describe, prepare_alignment, rank
from abstraction import audit
from tests.generation import FakeSandbox, tool
from tests.budgets import Encoder
OLD = 'first.value = 1; second.value = 2'
NEW = 'first.value = 1; second.value = 3'
def memory():
    registry = dynamic.empty_registry()
    for name in ('first', 'second', 'third'):
        spec = tool(name)
        spec['arguments'] = {'type': 'object', 'properties': {'value': {'type': 'integer'}}}
        registry, _ = dynamic.admit(registry, spec, FakeSandbox(), probe_inputs={'value': 1}, task_id='admit')
    return registry
def footprint(registry, text=OLD):
    return boundary.capture(registry, registry['aliases']['first'], registry['aliases']['second'], text)
def observed(registry, *, text=OLD, task='source', with_boundary=True, policy_step=None, rollout='r0'):
    arguments = dict(source_name='first', target_name='second', task_id=task, split='train',
        context_vector=[1., 0.], encoder_id=Encoder.encoder_id, task_success=True)
    if with_boundary:
        arguments['reuse_boundary'] = footprint(registry, text)
    if policy_step is None:
        return edges.observe_transition(registry, direct_feedback='verified', **arguments)
    return edges.observe_policy_transition(registry, policy_step=policy_step, rollout_id=rollout,
                                                feedback=['verified'], **arguments)
def view(registry, text=OLD, mode='explicit'):
    return edges.edge_view(registry, registry['aliases']['first'], registry['aliases']['second'],
        context_vector=[1., 0.], encoder_id=Encoder.encoder_id, reuse_boundary_mode=mode, current_user_text=text)
class ReuseBoundaryTests(unittest.TestCase):
    def test_boundary_is_source_grounded_versioned_and_read_only(self):
        registry = memory()
        original = deepcopy(registry)
        value = footprint(registry)
        self.assertEqual(boundary.compare(value, OLD)['state'], 'matched')
        check = boundary.compare(value, NEW)
        self.assertEqual(check['state'], 'conflict')
        self.assertFalse(check['historical_signal_allowed'])
        self.assertFalse(check['capability_blocked'])
        self.assertFalse(check['reuse_authorized'])
        self.assertEqual(registry, original)
    def test_missing_or_uncovered_natural_language_abstains(self):
        registry = memory()
        self.assertEqual(boundary.compare(None, OLD)['state'], 'unknown')
        self.assertEqual(boundary.compare(footprint(registry), 'Solve a new problem')['state'], 'unknown')
        self.assertEqual(boundary.compare(footprint(registry, 'Solve a problem'), OLD)['state'], 'unknown')
        self.assertEqual(boundary.compare(footprint(registry), 'first.value = 1')['state'], 'unknown')
    def test_schema_and_source_tampering_rejected_even_after_rehash(self):
        registry = memory()
        value = footprint(registry)
        for field, replacement in (('source_text', NEW), ('tools', [])):
            changed = deepcopy(value)
            changed[field] = replacement
            changed['sha256'] = dynamic.digest({k: v for k, v in changed.items() if k != 'sha256'})
            with self.subTest(field=field), self.assertRaises(ValueError):
                boundary.validate(changed, registry=registry, source_version=registry['aliases']['first'],
                                  target_version=registry['aliases']['second'])
    def test_different_endpoint_cannot_borrow_another_boundary(self):
        registry = memory()
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            edges.observe_transition(registry, source_name='first', target_name='third',
                task_id='source', split='train', context_vector=[1., 0.], encoder_id=Encoder.encoder_id,
                task_success=True, direct_feedback='verified', reuse_boundary=footprint(registry))
    def test_gate_precedes_all_association_support_and_utility(self):
        registry = observed(memory())
        original = deepcopy(registry)
        self.assertGreater(view(registry)['score'], 0)
        for text, state in ((NEW, 'conflict'), ('Solve a new problem', 'unknown')):
            blocked = view(registry, text)
            for key in ('association', 'support', 'utility', 'score', 'matching_tasks'):
                self.assertEqual(blocked[key], 0)
            self.assertEqual(blocked['reuse_boundary'][state], 1)
        self.assertGreater(view(registry, NEW, 'off')['score'], 0)
        self.assertEqual(registry, original)
    def test_legacy_evidence_unknown_not_negative(self):
        registry = observed(memory(), with_boundary=False)
        self.assertGreater(view(registry, mode='off')['score'], 0)
        gated = view(registry)
        self.assertEqual(gated['score'], 0)
        self.assertEqual(gated['reuse_boundary']['unknown'], 1)
        self.assertEqual(gated['utility'], 0)
        with self.assertRaisesRegex(ValueError, 'trusted user text'):
            view(registry, None)
    def test_only_matched_question_mass_survives_not_conflicting_rewards(self):
        registry = observed(memory())
        registry = observed(registry, text=NEW, task='other')
        result = view(registry)
        self.assertEqual(result['matching_tasks'], 1)
        self.assertEqual(result['independent_tasks'], 2)
        self.assertEqual(result['reuse_boundary']['matched'], 1)
        self.assertEqual(result['reuse_boundary']['conflict'], 1)
        self.assertAlmostEqual(result['association'], .02*.995)
    def test_same_policy_cannot_change_scope_new_step_refreshes_without_duplicate_support(self):
        registry = observed(memory(), policy_step=0)
        with self.assertRaisesRegex(ValueError, 'reuse boundaries'):
            observed(registry, text=NEW, policy_step=0, rollout='r1')
        registry = observed(registry, text=NEW, policy_step=1)
        self.assertEqual(registry['edge_memory']['clock'], 1)
        self.assertEqual(view(registry)['reuse_boundary']['conflict'], 1)
        self.assertEqual(view(registry, NEW)['matching_tasks'], 1)
    def test_fit_serve_share_gate_and_high_alpha_does_not_resurrect_neighbors(self):
        registry = observed(memory())
        encoder = Encoder()
        pairs = build_alignment_pairs(registry, encoder, query_id='fit', query='generated unrelated goal',
            reuse_boundary_mode='explicit', current_user_text=NEW)
        self.assertTrue(all('Supported successors: second' not in row['structural_text'] for row in pairs))
        alignment = prepare_alignment(pairs, encoder_id=Encoder.encoder_id, train_ids={'fit'},
            calibration_ids={'cal'}, validation_ids={'val'}, test_ids={'test'})
        encoder.texts.clear()
        result = rank(registry, encoder, alignment, None, query='generated unrelated goal',
            previous_tool='first', reuse_boundary_mode='explicit', current_user_text=NEW, fixed_alpha=1., top_k=3)
        self.assertEqual(len(result['tools']), 3)
        self.assertTrue(all(row['edge']['score'] == 0 for row in result['tools']))
        self.assertCountEqual([row['structural_text'] for row in pairs],
                             [text for text in encoder.texts if text.startswith('Input contract:')])
        self.assertEqual(result['structure_receipt']['reuse_boundary']['edge_task_observations']['conflict'], 1)
        with self.assertRaisesRegex(ValueError, 'boundary differs'):
            describe(registry, encoder, alignment, query=NEW)
        with self.assertRaisesRegex(ValueError, 'trusted user text'):
            describe(registry, encoder, alignment, query=OLD, reuse_boundary_mode='explicit')
    def test_alignment_rejects_mixed_boundary_policies(self):
        registry = memory()
        old = build_alignment_pairs(registry, Encoder(), query_id='fit', query=OLD)
        new = build_alignment_pairs(registry, Encoder(), query_id='other', query=OLD,
                                    reuse_boundary_mode='explicit', current_user_text=OLD)
        with self.assertRaisesRegex(ValueError, 'mix reuse boundary'):
            prepare_alignment(old+new, encoder_id=Encoder.encoder_id, train_ids={'fit', 'other'},
                calibration_ids={'cal'}, validation_ids={'val'}, test_ids={'test'})
    def test_audit_reports_coverage_without_loading_a_model_or_removing_tools(self):
        registry = observed(memory())
        original = deepcopy(registry)
        report = audit(registry, reuse_boundary_mode='explicit', current_user_text=NEW)
        self.assertEqual(report['reuse_boundary']['edge_task_observations']['conflict'], 1)
        self.assertEqual(report['reuse_boundary']['capabilities_removed'], 0)
        self.assertFalse(report['accuracy_measured'])
        self.assertEqual(registry, original)
if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
