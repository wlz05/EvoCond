# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Contract/mechanism fixtures only; not observed benchmark improvements.'
import ast
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from control.binding import prepare, apply_rewrites, response_loss_mask
from control.runtime import Runtime, execution_context_fingerprint, schemas
from control.evidence import REFERENCE
from tests import runtimes as fixture
from tests.checkout import official_root
import binding as audit_script
import ablation as ablations
import experiment as experiment
import bootstrap as bootstrap
TOOLS = [{'name': 'choose', 'parameters': {'type': 'object', 'required': ['tier'],
          'properties': {'tier': {'type': 'string', 'enum': ['budget', 'premium']}, 'limit': {'type': 'integer'}}}}]
class BindingTests(unittest.TestCase):
    def bind(self, text, args, **kwargs):
        return prepare([{'role': 'user', 'content': text}], TOOLS, 'choose', args, strategy='rebind', **kwargs)
    def test_current_literal_rebinds_wrong_value_without_mutation(self):
        raw = {'tier': 'premium', 'limit': 3}
        bound, audit = self.bind('choose.tier = "budget"', raw)
        self.assertEqual(bound, {'tier': 'budget', 'limit': 3})
        self.assertEqual(raw['tier'], 'premium')
        self.assertTrue(audit['changed'])
        self.assertEqual(audit['before']['decision'], 'conflict')
        self.assertEqual(audit['after']['decision'], 'accept')
        self.assertFalse(audit['semantic_correctness_certified'])
        self.assertEqual(audit['unverified_fields'], ['limit'])
    def test_missing_required_value_can_be_supplied_by_exact_evidence(self):
        bound, audit = self.bind('tier = "budget"', {})
        self.assertEqual(bound['tier'], 'budget')
        self.assertFalse(audit['rewrites'][0]['was_present'])
    def test_reject_control_retains_proposal_and_blocks(self):
        bound, audit = prepare([{'role': 'user', 'content': 'tier = "budget"'}], TOOLS,
                               'choose', {'tier': 'premium'}, strategy='reject')
        self.assertEqual(bound['tier'], 'premium')
        self.assertFalse(audit['changed'])
        self.assertTrue(audit['after']['blocked'])
    def test_unknown_natural_language_is_not_silently_verified_or_corrected(self):
        bound, audit = self.bind('Please make it inexpensive', {'tier': 'premium'})
        self.assertEqual(bound['tier'], 'premium')
        self.assertEqual(audit['after']['decision'], 'unknown')
        self.assertEqual(audit['covered_fields'], [])
    def test_latest_user_only_and_no_old_result_as_goal(self):
        messages = [{'role': 'user', 'content': 'tier = "premium"'},
                    {'role': 'tool', 'content': 'tier = "premium"'},
                    {'role': 'user', 'content': 'tier = "budget"'}]
        bound, audit = prepare(messages, TOOLS, 'choose', {'tier': 'premium'}, strategy='rebind')
        self.assertEqual(bound['tier'], 'budget')
        self.assertEqual(audit['rewrites'][0]['evidence']['quote'], 'tier = "budget"')
    def test_negation_history_quote_and_hypothesis_do_not_supply_values(self):
        for text in ('Previously tier = "budget"', 'Not tier = "budget"',
                     'Example tier = "budget"', 'If tier = "budget"', '`tier = "budget"`'):
            with self.subTest(text=text):
                _, audit = self.bind(text, {'tier': 'premium'})
                self.assertFalse(audit['changed'])
    def test_current_contradiction_does_not_pick_last_mention(self):
        _, audit = self.bind('tier = "budget"; tier = "premium"', {'tier': 'premium'})
        self.assertFalse(audit['changed'])
        self.assertTrue(audit['after']['blocked'])
        self.assertEqual(audit['after']['reason'], 'ambiguous_current_binding')
    def test_inequality_blocks_but_does_not_invent_a_replacement(self):
        bound, audit = self.bind('tier != "premium"', {'tier': 'premium'})
        self.assertEqual(bound['tier'], 'premium')
        self.assertFalse(audit['changed'])
        self.assertTrue(audit['after']['blocked'])
    def test_invalid_schema_value_and_unknown_field_do_not_project(self):
        for text in ('tier = "invalid"', 'absent = 5', 'limit = true', 'limit = 5kg'):
            with self.subTest(text=text):
                _, audit = self.bind(text, {'tier': 'premium'})
                self.assertFalse(audit['changed'])
    def test_scope_ambiguous_without_explicit_tool(self):
        tools = TOOLS+[{'name': 'other', 'parameters': TOOLS[0]['parameters']}]
        bound, audit = prepare([{'role': 'user', 'content': 'tier = "budget"'}], tools,
                               'choose', {'tier': 'premium'}, strategy='rebind')
        self.assertEqual(bound['tier'], 'premium')
        self.assertFalse(audit['changed'])
    def test_untrusted_alias_is_not_a_policy_authority(self):
        tools = deepcopy(TOOLS)
        tools[0]['parameters']['properties']['tier']['x-task-contract-aliases'] = [
            {'phrase': 'a low budget', 'value': 'budget'}]
        messages = [{'role': 'user', 'content': 'I need a low budget'}]
        _, unknown = prepare(messages, tools, 'choose', {'tier': 'premium'}, strategy='rebind')
        bound, trusted = prepare(messages, tools, 'choose', {'tier': 'premium'}, strategy='rebind', trusted_alias_tools=('choose',))
        self.assertFalse(unknown['changed'])
        self.assertTrue(trusted['changed'])
        self.assertEqual(bound['tier'], 'budget')
    def test_rebinding_equivariance_on_exact_constraints_and_unchanged_fields(self):
        # Engine invariant under verified goal substitutions, not a Qwen result.
        for current in ('budget', 'premium'):
            bound, _ = self.bind(f'tier = "{current}"', {'tier': 'premium', 'limit': 7})
            self.assertEqual(bound, {'tier': current, 'limit': 7})
    def test_result_refs_survive_only_on_unchanged_fields(self):
        raw = {'tier': {REFERENCE: {'call_id': 'old', 'path': ''}},
               'limit': {REFERENCE: {'call_id': 'current', 'path': ''}}}
        _, audit = self.bind('tier = "budget"', {'tier': 'premium', 'limit': 7})
        effective = apply_rewrites(raw, audit)
        self.assertEqual(effective['tier'], 'budget')
        self.assertEqual(effective['limit'], raw['limit'])
    def test_nonfinite_duplicate_and_nonobject_arguments_rejected(self):
        for args in ('{"tier":"budget","tier":"premium"}', {'limit': float('nan')}, []):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.bind('tier = "budget"', args)
    def test_credit_mask_keeps_plan_and_later_native_action(self):
        self.assertEqual(response_loss_mask([[1], [2], [3, 4], [5], [6]], [2]), [1, 0, 0, 0, 0, 1])
        for turns in ([0], [1], [6], [True], [2, 2]):
            with self.subTest(turns=turns), self.assertRaises(ValueError):
                response_loss_mask([[1], [2], [3]], turns)
class BindingRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.RuntimeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
    def runtime(self):
        config = {**self.fixture.config, 'binding_strategy': 'rebind', 'root': str(self.fixture.root/'rebind'),
                  'registry_path': str(self.fixture.root/'rebind/registry.json')}
        dispatcher = fixture.dispatcher_class()(config['registry_path'], fixture.Backend())
        runtime = Runtime(config, dispatcher)
        dispatcher.create(self.fixture.tool(), {'x': 1, 'y': 2}, task_id='setup')
        actor = SimpleNamespace(method_runtime=runtime, tool_manager=SimpleNamespace(functions=[]),
                                graphrag_engine=None, mcps={})
        return runtime, actor
    def test_actual_dispatch_uses_rebound_arguments_and_mask(self):
        runtime, actor = self.runtime()
        identity = runtime.begin('add.x = 8', fixture.PLAN, split='validation')
        runtime.select(identity)
        runtime.assistant(identity, 'call add with x=1, y=2')
        result = json.loads(asyncio.run(fixture.actor_method('_execute_runtime_tool')(actor,
                    {'name': 'add', 'args': {'x': 1, 'y': 2}, '_runtime_context': identity})))
        self.assertEqual(result['execution_result'], 10)
        self.assertEqual(result['binding_correction']['effective_arguments'], {'x': 8, 'y': 2})
        self.assertEqual(runtime.context(identity)['calls'][0]['arguments'], {'x': 8, 'y': 2})
        self.assertEqual(runtime.binding_mask(identity), [2])
        runtime.finish(identity)
        events = list(audit_script.rows(runtime.root/'binding_events.jsonl'))
        self.assertEqual(audit_script.events(events)['counts']['executed_rewrites'], 1)
        runtime.release_completed()
        self.assertTrue((runtime.root/'binding_events.jsonl').is_file())
    def test_overwritten_reference_is_not_falsely_credited_as_edge(self):
        runtime, actor = self.runtime()
        identity = runtime.begin('add.x = 8', fixture.PLAN, split='validation')
        source = runtime.observe(identity, 'source', '3', success=True, version='a'*64, execution_result=3)
        runtime.select(identity)
        runtime.assistant(identity, 'call')
        args = {'x': {REFERENCE: {'call_id': source['call_id'], 'path': ''}}, 'y': 2}
        result = json.loads(asyncio.run(fixture.actor_method('_execute_runtime_tool')(actor,
                          {'name': 'add', 'args': args, '_runtime_context': identity})))
        self.assertEqual(result['execution_result'], 10)
        self.assertEqual(runtime.context(identity)['calls'][-1]['bindings'], [])
    def test_blocked_call_is_not_an_executed_correction_or_masked_action(self):
        runtime, actor = self.runtime()
        identity = runtime.begin('add.x = 8; add.y != 2', fixture.PLAN, split='validation')
        runtime.select(identity)
        runtime.assistant(identity, 'call')
        result = json.loads(asyncio.run(fixture.actor_method('_execute_runtime_tool')(actor,
                    {'name': 'add', 'args': {'x': 1, 'y': 2}, '_runtime_context': identity})))
        self.assertTrue(result['binding_correction'])
        self.assertEqual(runtime.context(identity)['calls'], [])
        self.assertEqual(runtime.binding_mask(identity), [])
    def test_missing_sampled_turn_cannot_receive_correction_credit(self):
        runtime, _ = self.runtime()
        identity = runtime.begin('add.x = 8', fixture.PLAN, split='validation')
        catalog = schemas(runtime.context(identity)['dispatcher'])
        # The actor checks resolved values against canonical, not reference-union schemas.
        catalog = [dict(row, parameters=runtime.dispatcher.registry['semantic'][row['tool_version']]['arguments'])
                   for row in catalog]
        with self.assertRaisesRegex(ValueError, 'sampled assistant turn'):
            runtime.prepare_arguments(identity, 'add', {'x': 1, 'y': 2}, catalog)
    def test_changed_strategy_invalidates_labels_and_disabled_rebinding_refused(self):
        self.assertNotEqual(execution_context_fingerprint({'value_contracts': True}),
            execution_context_fingerprint({'value_contracts': True, 'binding_strategy': 'rebind'}))
        with self.assertRaisesRegex(ValueError, 'requires value_contracts'):
            Runtime({**self.fixture.config, 'value_contracts': False, 'binding_strategy': 'rebind'}, self.fixture.dispatcher)
    def test_native_and_composite_credit_are_distinct_objectives(self):
        runtime, actor = self.runtime()
        identity = runtime.begin('add.x = 8', fixture.PLAN, split='validation')
        runtime.select(identity)
        runtime.assistant(identity, 'call')
        asyncio.run(fixture.actor_method('_execute_runtime_tool')(actor,
                    {'name': 'add', 'args': {'x': 1, 'y': 2}, '_runtime_context': identity}))
        self.assertEqual(runtime.binding_mask(identity), [2])
        runtime.config['binding_credit'] = 'composite'
        self.assertEqual(runtime.binding_mask(identity), [])
        self.assertNotEqual(execution_context_fingerprint(self.fixture.config),
            execution_context_fingerprint({**self.fixture.config, 'binding_credit': 'composite'}))
        with self.assertRaisesRegex(ValueError, 'binding_credit'):
            Runtime({**self.fixture.config, 'binding_credit': 'invented'}, self.fixture.dispatcher)
    def test_bootstrap_binding_mode_matches_deployment(self):
        protocol = {'experiment': {'config': {'run_id': 'test', 'budget': experiment.settings({}, collection=True), 'encoder': 'fake'},
                                  'fingerprints': {'model': {'sha256': 'a'*64}, 'official_source': {'sha256': 'b'*64}}},
                    'input_config': {'encoder_id': 'fixture', 'binding_strategy': 'rebind'}}
        self.assertEqual(bootstrap.runtime_config(protocol, self.fixture.root, 'collect')['binding_strategy'], 'rebind')
        self.assertNotIn('binding_strategy', bootstrap.runtime_config(protocol, self.fixture.root, 'warmup'))
    def test_official_loss_composer_queries_identified_runtime_not_tool_prose(self):
        path = official_root()/'envs/utils/graph.py'
        cls = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == '_process_single_response_async')
        import itertools
        scope = {'itertools': itertools}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
        async def mask(identity):
            self.assertEqual(identity, 'actual')
            return [2]
        actor = SimpleNamespace(runtime_binding_mask=SimpleNamespace(remote=mask))
        utils = SimpleNamespace(strict_runtime=True, is_validate=True,
                                env_object=SimpleNamespace(plan_list=[{'_runtime_context': 'actual'}]))
        result = asyncio.run(scope[method.name](utils, 0, [[1], [2], [3, 4], [5], [6]], None, actor))
        self.assertEqual(result['loss_mask'], [1, 0, 0, 0, 0, 1])
class DiagnosticTests(unittest.TestCase):
    def receipt(self, task='base'):
        controls = {k: 'a'*64 for k in ('policy_hash', 'prefix_hash', 'candidate_hash', 'environment_hash', 'budget_hash')}
        cells = {k: {'executed': True, 'success': k.endswith('rebind'), 'controls': deepcopy(controls),
                     'first_action': {'name': 'choose', 'arguments': {'tier': 'budget'}}} for k in audit_script.CELLS}
        return {'format': 'evocond_binding_factorial_v1', 'task_id': task, 'variant': 'changed_goal',
                'seed': 42, 'controls': controls, 'cells': cells}
    def test_factorial_clusters_variants_and_seeds_by_base_task(self):
        first = self.receipt()
        second = {**deepcopy(first), 'seed': 43}
        report = audit_script.paired([first, second], samples=40)
        self.assertEqual(report['independent_tasks'], 1)
        self.assertEqual(report['estimates']['binding_at_semantic']['mean'], 1)
        self.assertEqual(report['estimates']['interaction']['mean'], 0)
        self.assertEqual(report['first_action_comparisons']['semantic.reject__structural.reject.identical_proposed_action'], 2)
    def test_same_tool_different_arguments_is_not_called_wrong_without_labels(self):
        row = self.receipt()
        row['cells']['structural.reject']['first_action']['arguments']['tier'] = 'premium'
        report = audit_script.paired([row], samples=40)
        self.assertEqual(report['first_action_comparisons']['semantic.reject__structural.reject.same_tool_arguments_differ'], 1)
    def test_invalid_bootstrap_parameters_rejected(self):
        for kwargs in ({'samples': 0}, {'samples': True}, {'seed': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                audit_script.paired([self.receipt()], **kwargs)
    def test_unpaired_missing_failed_cells_and_changed_controls_rejected(self):
        for change in ('missing', 'unexecuted', 'controls', 'duplicate'):
            row = self.receipt()
            if change == 'missing':
                del row['cells'][audit_script.CELLS[0]]
            elif change == 'unexecuted':
                row['cells'][audit_script.CELLS[0]]['executed'] = False
            elif change == 'controls':
                row['cells'][audit_script.CELLS[0]]['controls']['prefix_hash'] = 'b'*64
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit_script.paired([row, row] if change == 'duplicate' else [row])
        with self.assertRaises(ValueError):
            audit_script.paired([])
    def test_full_run_configs_need_two_matching_teacher_paths_and_new_identities(self):
        config = {'format': experiment.FORMAT, 'run_id': 'study', 'teacher': {'value_contracts': True}}
        values = ablations.variants(config, {'reject': 'reject.json', 'rebind': 'rebind.json'})
        self.assertEqual(len(values), 6)
        self.assertEqual(values['semantic.reject']['teacher']['fixed_alpha'], 0)
        self.assertEqual(values['structural.rebind']['teacher']['fixed_alpha'], 1)
        self.assertEqual(config['run_id'], 'study')
        with self.assertRaises(ValueError):
            ablations.variants(config, {'reject': 'same.json', 'rebind': 'same.json'})
if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
