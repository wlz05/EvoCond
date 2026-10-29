# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from copy import deepcopy
import ast
import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock
from types import SimpleNamespace
from control import dynamic, edges
from control.runtime import Runtime, load_config, plan_from_searl, recovery, reward_outcomes, schemas, task_identity
from tests.checkout import official_root
PLAN = "## ST1: Compute sum\n1. Invoke addition\n## ST2: Format result\n1. Format observed sum\n##DAG_LIST\n[(ST1, ST2)]"
class Backend:
    isolation = "container"
    isolation_verified = True
    def run_function(self, source, name, inputs, timeout):
        # A transport stub, never executes generated source in the test process.
        return {"run_success": True, "result": sum(inputs.values())}
def dispatcher_class():
    path = official_root()/"envs/tool_manager/centralized/dispatch.py"
    spec = importlib.util.spec_from_file_location("tested_isolated_dispatch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IsolatedToolDispatcher
def actor_method(name):
    path = official_root()/"envs/tool_manager/centralized/central.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CentralizedToolActor")
    method = next(n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    scope = {"asyncio": asyncio, "json": json, "time": time}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]
class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = {"format": "evocond_searl_runtime_v1", "arm": "evocond", "run_id": "unit-run",
                       "root": str(self.root), "registry_path": str(self.root/"registry.json"),
                       "policy_hash": "a"*64, "value_contracts": True}
        self.dispatcher = dispatcher_class()(self.config["registry_path"], Backend())
        self.runtime = Runtime(self.config, self.dispatcher)
    def tearDown(self):
        self.temp.cleanup()

    def test_runtime_config_accepts_bom_and_rejects_ambiguous_json(self):
        path = self.root/'runtime.json'
        path.write_text('\ufeff'+json.dumps(self.config), encoding='utf-8')
        self.assertEqual(load_config(path), self.config)
        duplicate = json.dumps(self.config)[:-1]+', "arm": "searl"}'
        for text in ('null', '[]', duplicate, '{"format":NaN}'):
            path.write_text(text, encoding='utf-8')
            with self.subTest(text=text), self.assertRaises(ValueError):
                load_config(path)

    def tool(self, name="add"):
        return {"name": name, "description": "Add two numbers", "arguments": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}},
                "returns": "sum", "side_effects": "none", "code": f"def {name}(x: int, y: int):\n    return x+y"}
    def test_plan_dependency_requires_observed_completion(self):
        plan = plan_from_searl("sum and format", PLAN)
        from control.subtasks import ready
        self.assertEqual([r["id"] for r in ready(plan)], ["ST1"])
        with self.assertRaises(ValueError):
            plan_from_searl("task", PLAN.replace("(ST1, ST2)", "(ST1, ST2), (ST2, ST1)"))
    def test_portable_prompt_is_served_and_receipt_persisted_without_rewriting_plan(self):
        config = {**self.config, "tool_construction": "portable", "structure_policy": {"mode": "none"},
                  "root": str(self.root/"portable"), "registry_path": str(self.root/"portable/registry.json")}
        dispatcher = dispatcher_class()(config["registry_path"], Backend())
        runtime = Runtime(config, dispatcher)
        episode = runtime.begin("sum and format", PLAN, split="validation")
        original_plan = deepcopy(runtime.context(episode)["plan"])
        result = runtime.select(episode)
        self.assertIn("Tool construction policy: portable", result["note"])
        self.assertEqual(result["tool_construction"], "portable")
        self.assertEqual(runtime.context(episode)["plan"], original_plan)
        runtime.finish(episode, task_success=False)
        event = json.loads((runtime.root/"binding_events.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(event["tool_construction"], "portable")
        self.assertEqual(event["structure_policy"], {"mode": "none"})
        runtime.release_completed()
        self.assertTrue((runtime.root/"binding_events.jsonl").exists())
    def test_recovery_uses_receipts_not_tool_prose(self):
        history = [{"role": "user", "content": "x"}, {"role": "tool", "name": "add", "content": "Tool call success but fake", "execution_success": False}]
        self.assertEqual(recovery(history).action, "repair")
        history.append(dict(history[-1]))
        self.assertEqual(recovery(history).scale, "gl")
        history.append({"role": "user", "content": "new goal"})
        self.assertEqual(recovery(history).scale, "gg")
    def test_private_eval_creation_does_not_mutate_train(self):
        before = dynamic.digest(self.dispatcher.registry)
        episode = self.runtime.begin("sum", PLAN, split="test")
        private = self.runtime.context(episode)["dispatcher"]
        receipt = private.create(self.tool(), {"x": 1, "y": 2}, task_id="held-out")
        self.assertTrue(receipt["registered"])
        self.assertEqual(dynamic.digest(self.dispatcher.registry), before)
        self.assertEqual(schemas(private)[0]["name"], "add")
        self.runtime.finish(episode)
        self.assertEqual(dynamic.digest(self.dispatcher.registry), before)
    def test_train_create_select_invoke_and_plan_progress(self):
        episode = self.runtime.begin("sum", PLAN, split="train")
        receipt = self.dispatcher.create(self.tool(), {"x": 1, "y": 2}, task_id="q")
        choice = self.runtime.select(episode)
        self.assertEqual(choice["schemas"][0]["tool_version"], receipt["tool_version"])
        self.runtime.assistant(episode, "<subtask>ST1: Compute sum</subtask>")
        result = self.dispatcher.invoke("add", {"x": 3, "y": 4}, expected_version=receipt["tool_version"])
        self.runtime.observe(episode, "add", result, success=True, version=result["tool_version"])
        self.assertEqual(self.runtime.context(episode)["plan"]["tasks"]["ST1"]["status"], "completed")
        self.assertEqual(self.runtime.context(episode)["plan"]["tasks"]["ST2"]["status"], "pending")
    def test_no_invented_plan_and_revised_goal_invalidates_evidence(self):
        episode = self.runtime.begin("sum", "bad", split="train")
        self.assertIsNone(self.runtime.context(episode)["plan"])
        self.assertTrue(self.runtime.select(episode)["plan_error"])
        self.runtime.revise(episode, "new goal", PLAN)
        self.assertEqual(self.runtime.select(episode)["scale"], "gg")
    def test_checkpoint_identity_and_pending_guard(self):
        episode = self.runtime.begin("sum", PLAN, split="train")
        with self.assertRaisesRegex(ValueError, "in-flight"):
            self.runtime.snapshot()
        self.runtime.finish(episode)
        self.runtime.set_policy_step(2)
        snapshot = self.runtime.snapshot()
        self.runtime.set_policy_step(3)
        self.runtime.restore(snapshot)
        self.assertEqual(self.runtime.policy_step, 2)
        snapshot["config_hash"] = "x"
        with self.assertRaisesRegex(ValueError, "identity"):
            self.runtime.restore(snapshot)
    def test_root_reuse_and_cross_arm_config_rejected(self):
        config = {**self.config, "arm": "searl"}
        with self.assertRaisesRegex(ValueError, "another experiment"):
            Runtime(config, self.dispatcher)
    def test_teacher_artifacts_are_not_optional(self):
        path = self.root/"config.json"
        path.write_text(json.dumps({**self.config, "arm": "teacher"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fitted artifact"):
            load_config(path)
        path.write_text(json.dumps({**self.config, "registry_path": str(self.root.parent/"escape.json")}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "private registry"):
            load_config(path)
    def test_unknown_episodes_and_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            self.runtime.select("not-real")
        self.runtime.begin("sum", PLAN, split="train", episode_id="one")
        with self.assertRaises(ValueError):
            self.runtime.begin("sum", PLAN, split="train", episode_id="one")
    def test_task_identity_matches_normalized_dataset_question(self):
        self.assertEqual(task_identity('\uff21\uff22\uff23\n  How?'), task_identity('abc how?'))
    def test_frozen_rollout_contract_and_teacher_context_identity(self):
        from control.runtime import execution_context_fingerprint, verify_rollout_contract
        budget = dict(max_turns=6, max_context=32768, max_prompt=4096, max_response=2048)
        config = dict(rollout_contract=budget, official_source_sha256='a'*64)
        verify_rollout_contract(config, budget)
        with self.assertRaisesRegex(ValueError, 'Actual rollout'):
            verify_rollout_contract(config, {**budget, 'max_turns': 7})
        self.assertNotEqual(execution_context_fingerprint(config),
            execution_context_fingerprint({**config, 'official_source_sha256': 'b'*64}))
        self.assertNotEqual(execution_context_fingerprint(config),
            execution_context_fingerprint({**config, 'rollout_contract': {**budget, 'max_response': 1024}}))
    def test_final_reward_not_shaped_total_is_used(self):
        self.assertEqual(reward_outcomes({'acc': [1.7, -0.1]}, 2), [None, None])
        self.assertEqual(reward_outcomes({'answer_correctness': [1, 0.4, -1, 0, None]}, 5),
                         [True, None, False, False, None])
        with self.assertRaises(ValueError):
            reward_outcomes({'answer_correctness': [1]}, 2)
        with self.assertRaises(ValueError):
            reward_outcomes({'answer_correctness': [float('nan')]}, 1)
    def test_coordinator_barrier_and_split_checked_before_finalization(self):
        train = self.runtime.begin('training task', PLAN, split='train')
        heldout = self.runtime.begin('test task', PLAN, split='test')
        actor = SimpleNamespace(method_runtime=self.runtime, _runtime_sync_graph=lambda: None,
                                runtime_finish=lambda identity, task_success=None: self.runtime.finish(identity, task_success=task_success))
        with self.assertRaisesRegex(ValueError, 'Cross-split'):
            actor_method('runtime_complete_batch')(actor, [train, heldout], [True, True], 'train')
        self.assertFalse(self.runtime.context(train)['completed'])
        with self.assertRaisesRegex(ValueError, 'in flight'):
            self.runtime.release_completed()
        self.runtime.finish(heldout)
        actor_method('runtime_complete_batch')(actor, [train], [True], 'train')
        self.assertEqual(self.runtime.contexts, {})
    def teacher_runtime(self, mode="frozen_controller_transfer", **options):
        from tests.policies import records, train
        from control.ranking import VIEW_SCHEMA
        alignment = {"format": "evocond_searl_alignment_v1", "dimension": 3, "encoder_id": "test-encoder-v1",
                     "view_schema": VIEW_SCHEMA, "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
        if 'reuse_boundary_mode' in options:
            alignment['reuse_boundary_mode'] = options['reuse_boundary_mode']
        rows = records()
        for row in rows:
            row["alignment_hash"], row["policy_hash"] = dynamic.digest(alignment), self.config["policy_hash"]
            for action in row["actions"]:
                action["alignment_hash"], action["policy_hash"] = row["alignment_hash"], row["policy_hash"]
        teacher, _ = train(rows)
        teacher_path, alignment_path = self.root/"teacher.json", self.root/"alignment.json"
        teacher_path.write_text(json.dumps(teacher), encoding="utf-8")
        alignment_path.write_text(json.dumps(alignment), encoding="utf-8")
        class Encoder:
            encoder_id = "test-encoder-v1"
            def encode(self, text):
                return [1., 0., 0.]
        root = self.root/"teacher-runtime"
        config = {**self.config, "arm": "teacher", "root": str(root), "registry_path": str(root/"registry.json"),
                  "teacher_path": str(teacher_path), "alignment_path": str(alignment_path),
                  "encoder_path": "unused-in-unit-test", "encoder_id": Encoder.encoder_id, "teacher_policy_mode": mode, **options}
        from control.runtime import execution_context_fingerprint
        teacher['execution_context_hash'] = execution_context_fingerprint(config)
        teacher_path.write_text(json.dumps(teacher), encoding='utf-8')
        dispatcher = dispatcher_class()(config["registry_path"], Backend())
        return Runtime(config, dispatcher, encoder=Encoder())
    def test_teacher_real_fitted_artifact_and_explicit_policy_transfer(self):
        runtime = self.teacher_runtime()
        runtime.dispatcher.create(self.tool(), {"x": 1, "y": 2}, task_id="q")
        episode = runtime.begin("sum", PLAN, split="train")
        self.assertFalse(runtime.select(episode)["teacher_transfer"])
        runtime.set_policy_step(1)
        self.assertTrue(runtime.select(episode)["teacher_transfer"])
        self.assertNotEqual(runtime.policy_hash, runtime.teacher["policy_hash"])
    def test_teacher_exact_policy_mode_rejects_stale_teacher(self):
        runtime = self.teacher_runtime("exact_policy")
        with self.assertRaisesRegex(ValueError, "Qwen changed"):
            runtime.set_policy_step(1)
    def test_teacher_policy_change_fallback_overrides_fixed_structural_weight_and_persists(self):
        runtime = self.teacher_runtime('semantic_fallback_on_policy_change', fixed_alpha=1.)
        runtime.dispatcher.create(self.tool(), {'x': 1, 'y': 2}, task_id='q')
        episode = runtime.begin('sum', PLAN, split='train')
        first = runtime.select(episode)
        self.assertEqual(first['teacher']['structure_weight'], 1.)
        self.assertFalse(first['teacher_authority']['semantic_fallback'])
        runtime.set_policy_step(1)
        second = runtime.select(episode)
        self.assertEqual(second['teacher']['structure_weight'], 0.)
        self.assertTrue(second['teacher_authority']['semantic_fallback'])
        self.assertEqual(second['teacher_authority']['reason'], 'policy_changed_unrecalibrated')
        self.assertTrue(second['teacher_transfer'])
        self.assertEqual([row['name'] for row in first['schemas']], [row['name'] for row in second['schemas']])
        runtime.finish(episode)
        report = json.loads((runtime.root/'binding_events.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(report['teacher_authority'][-1]['effective_alpha'], 0.)
    def test_runtime_captures_current_user_boundary_not_generated_subtask_goal(self):
        runtime = self.teacher_runtime(reuse_boundary_mode='explicit')
        for name in ('first', 'second'):
            runtime.dispatcher.create(self.tool(name), {'x': 1, 'y': 2}, task_id='q')
        text = 'first.x = 1; second.x = 2'
        episode = runtime.begin(text, PLAN, split='train')
        runtime.select(episode)
        versions = runtime.dispatcher.registry['aliases']
        for name in ('first', 'second'):
            runtime.observe(episode, name, '3', success=True, version=versions[name])
        runtime.finish(episode)
        edge = next(iter(runtime.dispatcher.registry['edge_memory']['edges'].values()))
        event = next(iter(edge['observations'].values()))
        self.assertEqual(event['reuse_boundary']['source_text'], text)
        report = json.loads((runtime.root/'binding_events.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(report['reuse_boundary_mode'], 'explicit')
        runtime.release_completed()
        other = runtime.begin('first.x = 1; second.x = 3', PLAN, split='test')
        selected = runtime.select(other)
        self.assertEqual(selected['structure_receipt']['reuse_boundary']['edge_task_observations']['conflict'], 1)
    def test_new_boundary_mode_changes_teacher_execution_context(self):
        from control.runtime import execution_context_fingerprint
        self.assertNotEqual(execution_context_fingerprint(self.config),
            execution_context_fingerprint({**self.config, 'reuse_boundary_mode': 'explicit'}))
    def test_teacher_train_only_edge_updates_unknown_not_fake_success(self):
        runtime = self.teacher_runtime()
        for name in ("first", "second"):
            runtime.dispatcher.create(self.tool(name), {"x": 1, "y": 2}, task_id="q")
        episode = runtime.begin("sum", PLAN, split="train")
        versions = runtime.dispatcher.registry["aliases"]
        for name in ("first", "second"):
            runtime.observe(episode, name, "3", success=True, version=versions[name])
        runtime.finish(episode)
        edge = next(iter(runtime.dispatcher.registry["edge_memory"]["edges"].values()))
        self.assertEqual(next(iter(edge["observations"].values()))["direct_feedback"], "unknown")
        before = dynamic.digest(runtime.dispatcher.registry)
        episode = runtime.begin("held-out", PLAN, split="validation")
        for name in ("first", "second"):
            runtime.observe(episode, name, "error", success=False, version=versions[name])
        runtime.finish(episode, task_success=False)
        self.assertEqual(dynamic.digest(runtime.dispatcher.registry), before)
    def test_official_actor_dynamic_dispatch_without_host_tool_registration(self):
        execute = actor_method('_execute_runtime_tool')
        actor = SimpleNamespace(method_runtime=self.runtime, tool_manager=SimpleNamespace(functions=[{
            'name': 'create_and_execute_mcp', 'parameters': {'type': 'object', 'properties': {}}}]),
                                graphrag_engine=None, mcps={}, _runtime_sync_graph=lambda: None)
        episode = self.runtime.begin("sum", PLAN, split="train")
        call = {"name": "create_and_execute_mcp", "args": {**self.tool(), "inputs": {"x": 1, "y": 2}}, "_runtime_context": episode}
        result = asyncio.run(execute(actor, call))
        self.assertTrue(json.loads(result)["registered"])
        self.runtime.select(episode)
        result = asyncio.run(execute(actor, {"name": "add", "args": {"x": 4, "y": 5}, "_runtime_context": episode}))
        self.assertEqual(json.loads(result)["result"], 9)
        result = asyncio.run(execute(actor, {"name": "add", "args": {"x": "bad", "y": 5}, "_runtime_context": episode}))
        self.assertTrue(json.loads(result)['audit']['blocked'])

    def test_retrieval_service_errors_never_become_successful_runtime_evidence(self):
        from control.isolation import SandboxInfrastructureError
        execute = actor_method('_execute_runtime_tool')
        self.runtime.config['value_contracts'] = False
        for text in ('Search error: retrieval HTTP status 503', 'An error occurred', '', None):
            call = Mock(return_value=text)
            actor = SimpleNamespace(method_runtime=self.runtime,
                tool_manager=SimpleNamespace(functions=[], _call_tool=call), graphrag_engine=None, mcps={})
            episode = self.runtime.begin('sum', PLAN, split='train')
            with self.subTest(text=text), self.assertRaises(SandboxInfrastructureError):
                asyncio.run(execute(actor, {'name': 'search', 'args': {'query': 'question'},
                    '_runtime_context': episode}))
            self.assertEqual(self.runtime.context(episode)['invocations'], [])
            call.assert_called_once()

    def test_search_input_errors_are_recoverable_and_empty_results_are_valid(self):
        execute = actor_method('_execute_runtime_tool')
        self.runtime.config['value_contracts'] = False
        call = Mock(return_value='No relevant documents found')
        actor = SimpleNamespace(method_runtime=self.runtime,
            tool_manager=SimpleNamespace(functions=[], _call_tool=call), graphrag_engine=None, mcps={})
        for arguments in ({'query': ''}, {'query': 'question', 'topk': True}, {'query': 'q', 'unexpected': 1}):
            episode = self.runtime.begin('sum', PLAN, split='train')
            result = json.loads(asyncio.run(execute(actor, {'name': 'search', 'args': arguments,
                '_runtime_context': episode})))
            self.assertEqual(result['error'], 'invalid_tool_input')
        call.assert_not_called()
        episode = self.runtime.begin('sum', PLAN, split='train')
        asyncio.run(execute(actor, {'name': 'search', 'args': {'query': 'question'}, '_runtime_context': episode}))
        self.assertTrue(self.runtime.context(episode)['invocations'][-1]['success'])
    def test_quarantined_tool_is_not_recorded_as_observed_version(self):
        actor = SimpleNamespace(method_runtime=self.runtime, tool_manager=SimpleNamespace(functions=[{
            'name': 'create_and_execute_mcp', 'parameters': {'type': 'object', 'properties': {}}}]),
                                graphrag_engine=None, mcps={}, _runtime_sync_graph=lambda: None)
        captured = {}
        def reject(*args, **kwargs):
            captured.update(kwargs)
            return {'run_success': True, 'registered': False, 'tool_version': 'unadmitted-version'}
        self.dispatcher.create = reject
        self.runtime.encoder = object()
        episode = self.runtime.begin('sum', PLAN, split='train')
        call = {'name': 'create_and_execute_mcp', 'args': {**self.tool(), 'inputs': {'x': 1, 'y': 2}}, '_runtime_context': episode}
        asyncio.run(actor_method('_execute_runtime_tool')(actor, call))
        self.assertIs(captured['encoder'], self.runtime.encoder)
        self.assertIsNone(self.runtime.context(episode)['invocations'][-1]['tool_version'])
        self.assertFalse(self.runtime.context(episode)['invocations'][-1]['success'])
    def test_executed_ll_gl_and_explicit_policy_replanning(self):
        actor = SimpleNamespace(method_runtime=self.runtime, tool_manager=SimpleNamespace(functions=[]),
                                graphrag_engine=None, mcps={})
        self.dispatcher.create(self.tool(), {'x':1,'y':2}, task_id='train')
        episode = self.runtime.begin('sum', PLAN, split='train')
        self.runtime.select(episode)
        self.runtime.assistant(episode,'<subtask>ST1: Compute sum</subtask>')
        calls = []
        def fail(source,name,inputs,timeout):
            calls.append(inputs)
            return {'run_success':False,'error':'observed execution failed'}
        self.dispatcher.backend.run_function = fail
        execute = actor_method('_execute_runtime_tool')
        def invoke(x):
            return json.loads(asyncio.run(execute(actor, {'name':'add','args':{'x':x,'y':2},'_runtime_context':episode})))
        self.assertFalse(invoke(1)['run_success'])
        self.assertEqual(invoke(1)['audit']['reason'],'identical_failed_call_requires_repair')
        self.assertEqual(len(calls),1)
        self.assertFalse(invoke(2)['run_success'])
        state = self.runtime.context(episode)['recovery']
        self.assertEqual(state['pending']['scale'],'gl')
        self.assertEqual(invoke(3)['audit']['reason'],'explicit_revised_plan_required')
        payload = {'goal_epoch':state['goal_epoch'],'base_revision':state['revision'],
                   'question_sha256':state['question_sha256'],'searl_plan':PLAN.replace('Compute sum','Compute sum differently')}
        result = asyncio.run(execute(actor, {'name':'submit_revised_plan','args':payload,'_runtime_context':episode}))
        self.assertTrue(json.loads(result)['accepted'])
        self.assertIsNone(self.runtime.context(episode)['recovery']['pending'])
        self.assertEqual(len(calls),2)
    def test_gg_trusted_goal_change_requires_fresh_plan_and_clears_ledger(self):
        episode = self.runtime.begin('sum', PLAN, split='train')
        self.runtime.observe(episode,'old','old observation',success=True)
        self.runtime.revise(episode,'new question')
        context = self.runtime.context(episode)
        self.assertIsNone(context['plan'])
        self.assertEqual(context['calls'],[])
        self.assertEqual(context['recovery']['pending']['scale'],'gg')
        audit = self.runtime.check(episode,'some_tool',{},[])
        self.assertTrue(audit['blocked'])
    def test_teacher_verified_edge_requires_actual_explicit_result_binding(self):
        runtime = self.teacher_runtime()
        for name in ('first','second'):
            runtime.dispatcher.create(self.tool(name),{'x':1,'y':2},task_id='train')
        actor = SimpleNamespace(method_runtime=runtime,tool_manager=SimpleNamespace(functions=[]),
                                graphrag_engine=None,mcps={})
        episode = runtime.begin('sum',PLAN,split='train')
        runtime.select(episode)
        execute = actor_method('_execute_runtime_tool')
        first = json.loads(asyncio.run(execute(actor, {'name':'first','args':{'x':2,'y':3},'_runtime_context':episode})))
        second = json.loads(asyncio.run(execute(actor, {'name':'second','args':{
            'x':{'__searl_result_ref__':{'call_id':first['call_id'],'path':''}},'y':2},'_runtime_context':episode})))
        self.assertEqual(second['execution_result'],7)
        runtime.finish(episode,task_success=True)
        edge = next(iter(runtime.dispatcher.registry['edge_memory']['edges'].values()))
        observation = next(iter(edge['observations'].values()))
        self.assertEqual(observation['direct_feedback'],'verified')
        self.assertIs(observation['task_success'],True)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
