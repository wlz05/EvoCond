# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from copy import deepcopy
import ast
import asyncio
import itertools
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from control.runtime import Runtime, task_identity
from control.runtime import load_config
from control.pairing import collect_teacher_policy_pairs, snapshot_hash
from control.continuation import observed_outcome, validate_snapshot
from control import dynamic, ranking
from tests.checkout import official_root
from tests.runtimes import Backend, dispatcher_class, actor_method, PLAN
class TensorDict(dict):
    def __init__(self, data, batch_size):
        super().__init__(data)
        self.batch_size = [batch_size]
class Proto:
    def __init__(self, batch=None, non_tensor_batch=None):
        self.batch = batch or {}
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = {}
    @classmethod
    def from_single_dict(cls, data):
        return cls(batch={key: value for key, value in data.items() if isinstance(value, torch.Tensor)},
                   non_tensor_batch={key: value for key, value in data.items() if not isinstance(value, torch.Tensor)})
    def pop(self, batch_keys, non_tensor_batch_keys):
        return Proto({key: self.batch.pop(key) for key in batch_keys},
                     {key: self.non_tensor_batch.pop(key) for key in non_tensor_batch_keys})
    def union(self, other):
        self.batch.update(other.batch)
        self.non_tensor_batch.update(other.non_tensor_batch)
        return self
    def __getitem__(self, selected):
        result = Proto({key: value[selected] for key, value in self.batch.items()},
                       {key: value[selected] for key, value in self.non_tensor_batch.items()})
        result.meta_info = deepcopy(self.meta_info)
        return result
class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)
class Remote:
    def __init__(self, fn):
        self.remote = fn
def helper_functions():
    path = official_root()/'envs/utils/continuation.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    scope = dict(deepcopy=deepcopy, asyncio=asyncio, itertools=itertools, random=random, np=np,
                 ray=SimpleNamespace(get=lambda value: value), torch=torch, TensorDict=TensorDict,
                 DataProto=Proto, snapshot_hash=snapshot_hash)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(tree, str(path), 'exec'), scope)
    return scope
class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.encoder = SimpleNamespace(encoder_id='unit-v1', encode=lambda text: [1., 0., 0.])
        alignment = {'format': 'evocond_searl_alignment_v1', 'dimension': 3, 'encoder_id': 'unit-v1',
                     'view_schema': ranking.VIEW_SCHEMA, 'matrix': [[1,0,0],[0,1,0],[0,0,1]]}
        (self.root/'alignment.json').write_text(json.dumps(alignment), encoding='utf-8')
        self.config = {'format': 'evocond_searl_runtime_v1', 'arm': 'searl', 'run_id': 'paired',
                       'root': str(self.root), 'registry_path': str(self.root/'registry.json'),
                       'policy_hash': 'a'*64, 'collection': True, 'alignment_path': str(self.root/'alignment.json'),
                       'encoder_path': 'fake', 'encoder_id': 'unit-v1'}
        self.dispatcher = dispatcher_class()(self.config['registry_path'], Backend())
        self.spec = {'name': 'add', 'description': 'sum', 'arguments': {'type':'object','properties': {'x': {'type':'integer'}}},
                     'returns': 'sum', 'side_effects': 'none', 'code': 'def add(x: int):\n    return x'}
        self.dispatcher.create(self.spec, {'x': 1}, task_id='fit')
        self.runtime = Runtime(self.config, self.dispatcher, encoder=self.encoder)
        actor = SimpleNamespace(method_runtime=self.runtime)
        self.actor = actor
        actor.runtime_capture = Remote(lambda ids: actor_method('runtime_capture')(actor, ids))
        actor.runtime_resume = Remote(lambda states, alpha, seed: actor_method('runtime_resume')(actor, states, alpha, seed))
        actor.runtime_branch_receipts = Remote(lambda ids: actor_method('runtime_branch_receipts')(actor, ids))
        self.helpers = helper_functions()
        self.executions = 0
    def tearDown(self):
        self.temp.cleanup()
    def su(self):
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                return SimpleNamespace(input_ids=[[ord(c) for c in text] for text in texts])
        env = SimpleNamespace(tool_manager=SimpleNamespace(centralized_actor_handle=self.actor))
        async def render(observation, tokenizer, mode, plan, done, valid, is_tool, names):
            decision = self.runtime.select(plan['_runtime_context'])
            return ('observation:'+str(observation)+';tools:'+','.join(r['name'] for r in decision['schemas']),
                    done, valid, is_tool, names)
        env._process_single_prompt = render
        return SimpleNamespace(env_object=env, tokenizer=Tokenizer(), meta_info={}, batch_size=1,
            loop_cnt=1, loop_responses_token=[[[10,11], [20,21], [30], [40,41]]],
            init_prompt_token=torch.tensor([[10,11]]), init_attention_mask=torch.tensor([[1,1]]),
            tool_use=[[0]], mcp_ids=[[['plan']]], max_turns=5, config_prompt_length=4,
            config_response_length=2, max_tool_response_length=1024, context_budget=2046,
            sampling_config={'temperature':.7,'top_p':.95,'top_k':-1},
            strict_runtime=True, pad_token_id=0)
    def prefix(self):
        su = self.su()
        identity = self.runtime.begin('sum', PLAN, split='train')
        context = self.runtime.context(identity)
        receipt = context['dispatcher'].invoke('add', {'x': 2})
        self.executions += 1
        self.runtime.observe(identity, 'add', receipt, success=True, version=receipt['tool_version'])
        su.env_object.plan_list = [{'_runtime_context': identity, '_policy_plan': 'actual original plan'}]
        su.env_object.plan_success_list = [True]
        self.helpers['capture'](su, 1, ['2'], [False], [1], [1], [['add']], [0], [0])
        return su.continuation_snapshots[0]
    def test_five_actual_helper_branches_do_not_reexecute_prefix_or_write_shared_registry(self):
        snapshot = self.prefix()
        before = dynamic.digest(self.dispatcher.registry)
        frozen = snapshot_hash(snapshot)
        self.assertEqual(self.runtime.contexts, {})
        calls, rewards = [], []
        def evaluator(request):
            su = self.su()
            next_prompt = self.helpers['restore'](su, [request.state], request.alpha, request.seed)
            identity = su.env_object.plan_list[0]['_runtime_context']
            self.assertEqual(su.loop_cnt, 2)
            self.assertEqual(next_prompt.batch['input_ids'][0, :7].tolist(), [10,11,20,21,30,40,41])
            self.assertEqual(next_prompt.meta_info['continuation_seed'], request.seed)
            self.assertIs(next_prompt.meta_info['do_sample'], True)
            first = self.runtime.context(identity)
            self.assertEqual(first['interventions'], 1)
            self.assertNotIn('next_alpha', first)
            second = self.runtime.select(identity)
            self.assertEqual(second['teacher']['structure_weight'], 0.0)
            # Fake transport stands in for the next actual worker/tool execution, not a reward estimate.
            result = first['dispatcher'].invoke('add', {'x': 5})
            self.executions += 1
            self.runtime.observe(identity, 'add', result, success=True, version=result['tool_version'])
            calls.append(request.alpha)
            self.helpers['attach_receipts'](su, output := Proto())
            rewards.append(result['result'])
            return observed_outcome(request, output.non_tensor_batch['continuation_receipt'][0],
                                    int(result['result'] == 5), shared_registry_hash=before)
        runtime = validate_snapshot(snapshot, question_id=task_identity('sum'))
        view = runtime['view']
        records = collect_teacher_policy_pairs(evaluator=evaluator, query_id=task_identity('sum'), decision_id='turn1',
            split='train', state=snapshot, registry=runtime['registry'], candidates=view['candidates'],
            alignment=runtime['alignment'], features=view['features'], query_embedding=view['query_embedding'],
            encoder_id=view['encoder_id'], policy_hash=runtime['policy_hash'], seeds=[12])
        self.assertEqual(calls, [0,.25,.5,.75,1])
        self.assertEqual(rewards, [5]*5)
        self.assertEqual(self.executions, 6)
        self.assertEqual(len(records[0]['actions']), 5)
        self.assertEqual(self.runtime.contexts, {})
        self.assertEqual(dynamic.digest(self.dispatcher.registry), before)
        self.assertEqual(snapshot_hash(snapshot), frozen)
    def test_history_budget_raises_without_silent_last_n_truncation(self):
        su = self.su()
        su.context_budget = 5
        with self.assertRaisesRegex(ValueError, 'context budget'):
            self.helpers['append_observations'](su, ['new observation'], [False], [['tool']], [0], [0])
    def test_heldout_and_changed_policy_cannot_supply_training_pairs(self):
        snapshot = self.prefix()
        snapshot['runtime']['context']['split'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'training question'):
            validate_snapshot(snapshot, question_id=task_identity('sum'))
        snapshot['runtime']['context']['split'] = 'train'
        self.runtime.set_policy_step(1)
        with self.assertRaisesRegex(ValueError, 'policy'):
            self.helpers['restore'](self.su(), [snapshot], .5, 12)
    def test_rng_restores_same_python_numpy_torch_draws_for_common_seed(self):
        state = self.helpers['capture_rng']()
        draws = []
        for _ in range(2):
            self.helpers['restore_rng'](state, 17)
            draws.append((random.random(), np.random.random(), torch.rand(1).item()))
        self.assertEqual(draws[0], draws[1])
    def test_no_estimated_or_shaped_reward_accepted(self):
        request = SimpleNamespace(seed=1, alpha=.5, action_scope='next_retrieval_only')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            observed_outcome(request, {'executed':False}, .9, shared_registry_hash='x')
    def driver(self, missing_correctness=False):
        path = official_root()/'verl/trainer/collection.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        scope = dict(deepcopy=deepcopy, Path=Path, json=json, np=np, ray=SimpleNamespace(get=lambda x:x), DataProto=Proto,
            pad_dataproto_to_divisor=lambda x,n:(deepcopy(x),0), unpad_dataproto=lambda x,n:x,
            task_identity=task_identity, load_config=load_config, collect_teacher_policy_pairs=collect_teacher_policy_pairs,
            observed_outcome=observed_outcome, validate_snapshot=validate_snapshot)
        exec(compile(tree, str(path), 'exec'), scope)
        config_path = self.root/'config.json'
        config_path.write_text(json.dumps(self.config), encoding='utf-8')
        configuration = Config(trainer=Config(teacher_collect_mode='pairs', teacher_collect_path=str(self.root/'out'),
             teacher_collect_seeds=[13], teacher_capture_turn=1), actor_rollout_ref=Config(
             rollout=Config(n=1,name='vllm'), env=Config(runtime_config_path=str(config_path))))
        self.actor.runtime_registry_hash = Remote(lambda: dynamic.digest(self.dispatcher.registry))
        calls = []
        def generate(prompts):
            mode = prompts.meta_info['teacher_continuation_mode']
            calls.append(mode)
            if mode == 'capture':
                array = np.empty(1,dtype=object)
                array[0] = self.prefix()
                return Proto(non_tensor_batch={'continuation_snapshot':array})
            su = self.su()
            self.helpers['restore'](su,list(prompts.non_tensor_batch['continuation_snapshot']),
                                    prompts.meta_info['continuation_alpha'],prompts.meta_info['continuation_seed'])
            identity = su.env_object.plan_list[0]['_runtime_context']
            self.runtime.assistant(identity,'final_answer(5)')
            return self.helpers['attach_receipts'](su,Proto())
        reward_calls = []
        def reward(batch, return_dict):
            self.assertEqual(batch.non_tensor_batch['reward_model'][0]['ground_truth'],'5')
            reward_calls.append(1)
            return {'reward_tensor':torch.tensor([[1.7]]), 'reward_extra_info':
                    {'acc':[1.7]} if missing_correctness else {'answer_correctness':[1.0],'acc':[1.7]}}
        data = {'input_ids':torch.tensor([[10,11]]),'attention_mask':torch.tensor([[1,1]]),
                'position_ids':torch.tensor([[0,1]]),'question':np.asarray(['sum'],dtype=object),
                'raw_prompt_ids':np.asarray([[10,11]],dtype=object),
                'reward_model':np.asarray([{'ground_truth':'5'}],dtype=object)}
        trainer = Config(config=configuration, global_steps=0, async_rollout_mode=False,
            _method_actor=lambda:self.actor, actor_rollout_wg=Config(world_size=1, generate_sequences_loop=generate),
            train_dataloader=[data], train_dataset=[0], reward_fn=reward)
        return lambda:scope['run'](trainer), calls, reward_calls
    def test_official_driver_calls_worker_and_real_reward_for_each_branch_no_optimizer(self):
        run, calls, rewards = self.driver()
        run()
        self.assertEqual(calls,['capture']+['resume']*5)
        self.assertEqual(len(rewards),5)
        audit = json.loads((self.root/'out/audit.json').read_text(encoding='utf-8'))
        self.assertTrue(audit['complete'])
        self.assertFalse(audit['qwen_updated'])
        row = json.loads((self.root/'out/pairs.jsonl').read_text(encoding='utf-8'))
        self.assertTrue(all(action['reward']==1 for action in row['actions']))
    def test_driver_aborts_without_complete_audit_when_correctness_missing(self):
        run, calls, rewards = self.driver(missing_correctness=True)
        with self.assertRaisesRegex(Exception,'answer_correctness'):
            run()
        self.assertFalse((self.root/'out/audit.json').exists())
        self.assertEqual((self.root/'out/pairs.jsonl').read_text(encoding='utf-8'),'')
if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
