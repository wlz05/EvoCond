# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""JSON-only snapshots at the official action/retrieval boundary."""
from copy import deepcopy
import asyncio
import itertools
import random
import numpy as np
import ray
import torch
from tensordict import TensorDict
from verl import DataProto
from control.pairing import snapshot_hash

def capture_rng():
    np_state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [np_state[0], np_state[1].tolist(), *np_state[2:]],
            'torch': torch.get_rng_state().tolist(),
            'cuda': [state.tolist() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}

def restore_rng(state, seed):
    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, (tuple, list)) else value
    random.setstate(tuples(state['python']))
    ns = state['numpy']
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(torch.tensor(state['torch'], dtype=torch.uint8))
    if torch.cuda.is_available() and state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Continuation CUDA topology changed')
        torch.cuda.set_rng_state_all([torch.tensor(row, dtype=torch.uint8) for row in state['cuda']])
    # Each matched action starts from the same explicit continuation seed.
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def append_observations(su, infos, dones, names, scores, indices):
    """The official full-history concatenation, truncation and loss alternation."""
    strict = getattr(su, 'strict_runtime', False)
    tokens = su.tokenizer(infos, truncation=not strict, max_length=su.max_tool_response_length).input_ids
    if strict and any(len(row) > su.max_tool_response_length for row in tokens):
        raise ValueError('Visible tool observation exceeds the frozen tool-response budget; refusing hidden truncation')
    sequences, active = [], []
    for offset, index in enumerate(indices):
        if dones[offset]:
            continue
        su.loop_responses_token[index].append(tokens[offset])
        sequences.append(list(itertools.chain.from_iterable(su.loop_responses_token[index])))
        active.append(index)
        su.tool_use[index].append(scores[offset])
        su.mcp_ids[index].append(names[offset])
    if not sequences:
        return None
    limit = su.context_budget if strict else su.config_prompt_length
    if strict and any(len(row) > limit for row in sequences):
        raise ValueError('Official full visible history exceeds context budget; no reward emitted, increase common budget in a new run')
    width = max(map(len, sequences)) if strict else max(max(map(len, sequences)), su.config_prompt_length)
    ids = torch.tensor([[su.pad_token_id]*(width-len(seq))+seq for seq in sequences], dtype=torch.int64)
    attention = ids != su.pad_token_id
    positions = torch.clip(torch.cumsum(attention, dim=-1)-1, min=0)*attention
    batch = TensorDict({'input_ids': ids[:, -limit:], 'position_ids': positions[:, -limit:],
                        'attention_mask': attention[:, -limit:]}, batch_size=len(sequences))
    raw = np.empty(len(sequences), dtype=object)
    raw[:] = [seq[-limit:] for seq in sequences]
    result = DataProto(batch=batch, non_tensor_batch={'raw_prompt_ids': raw})
    result.meta_info.update(su.meta_info)
    result.meta_info.update(index=active, do_sample=bool(su.meta_info.get('sample_tool_turns', False)))
    su.loop_cnt += 1
    return result

def capture(su, step, infos, dones, valid, is_tool, names, scores, indices):
    pending = {index: {'observation': infos[i], 'done': bool(dones[i]), 'valid': int(valid[i]),
                      'is_tool': int(is_tool[i]), 'names': names[i], 'score': float(scores[i])}
               for i, index in enumerate(indices)}
    actor = su.env_object.tool_manager.centralized_actor_handle
    identities = [row['_runtime_context'] for row in su.env_object.plan_list]
    contexts = ray.get(actor.runtime_capture.remote(identities))
    rng = capture_rng()
    snapshots = []
    for index, runtime in enumerate(contexts):
        item = pending.get(index)
        eligible = bool(item and not item['done'] and runtime['view']['candidates'])
        snapshot = {'format': 'searl_official_continuation_v1', 'runtime': runtime,
                    'eligible': eligible, 'reason': None if eligible else 'terminated_or_no_candidates',
                    'step': step, 'pending': item, 'rng': rng,
                    'loop_responses_token': deepcopy(su.loop_responses_token[index]),
                    'init_prompt_token': su.init_prompt_token[index].tolist(),
                    'init_attention_mask': su.init_attention_mask[index].tolist(),
                    'tool_use': deepcopy(su.tool_use[index]), 'mcp_ids': deepcopy(su.mcp_ids[index]),
                    'policy_plan': deepcopy(su.env_object.plan_list[index]['_policy_plan']),
                    'plan_success': bool(su.env_object.plan_success_list[index]),
                    'max_turns': su.max_turns, 'prompt_length': su.config_prompt_length,
                    'response_length': su.config_response_length}
        snapshot['context_budget'] = su.context_budget
        snapshot['sampling_config'] = deepcopy(su.sampling_config)
        snapshot['sample_tool_turns'] = bool(su.meta_info.get('sample_tool_turns', True))
        snapshot_hash(snapshot)
        snapshots.append(snapshot)
    su.continuation_snapshots = snapshots
    return None

def capture_output(su):
    values = np.empty(len(su.continuation_snapshots), dtype=object)
    values[:] = su.continuation_snapshots
    return DataProto(batch=TensorDict({'continuation_row': torch.zeros((len(values), 1), dtype=torch.int64)},
                                     batch_size=len(values)), non_tensor_batch={'continuation_snapshot': values})

def restore(su, snapshots, alpha, seed):
    if not snapshots or any(not row.get('eligible') for row in snapshots):
        raise ValueError('Only eligible captured prefixes can resume')
    first = snapshots[0]
    for row in snapshots:
        snapshot_hash(row)
        if (row['format'] != 'searl_official_continuation_v1' or row['step'] != first['step']
                or row['max_turns'] != su.max_turns or row['prompt_length'] != su.config_prompt_length
                or row['context_budget'] != su.context_budget
                or row['sampling_config'] != su.sampling_config or not row['sample_tool_turns']
                or row['response_length'] != su.config_response_length):
            raise ValueError('Official continuation shape or horizon changed')
    restore_rng(first['rng'], seed)
    actor = su.env_object.tool_manager.centralized_actor_handle
    identities = ray.get(actor.runtime_resume.remote([row['runtime'] for row in snapshots], alpha, seed))
    su.branch_sources = deepcopy(snapshots)
    su.batch_size = len(snapshots)
    su.loop_cnt = first['step']
    su.loop_responses_token = [deepcopy(row['loop_responses_token']) for row in snapshots]
    su.init_prompt_token = torch.tensor([row['init_prompt_token'] for row in snapshots], dtype=torch.int64)
    su.init_attention_mask = torch.tensor([row['init_attention_mask'] for row in snapshots])
    su.tool_use = [deepcopy(row['tool_use']) for row in snapshots]
    su.mcp_ids = [deepcopy(row['mcp_ids']) for row in snapshots]
    su.tool_call_count = None
    su.env_object.plan_list = [{'_runtime_context': identity, '_policy_plan': deepcopy(row['policy_plan'])}
                               for identity, row in zip(identities, snapshots)]
    su.env_object.plan_success_list = [row['plan_success'] for row in snapshots]
    async def render():
        return await asyncio.gather(*[su.env_object._process_single_prompt(
            row['pending']['observation'], su.tokenizer, 'plan_call' if row['step'] == 0 else 'tool_call',
            plan, False, row['pending']['valid'], row['pending']['is_tool'], row['pending']['names'])
            for row, plan in zip(snapshots, su.env_object.plan_list)])
    rendered = asyncio.run(render())
    su.meta_info['continuation_seed'] = seed
    su.meta_info['sample_tool_turns'] = True
    return append_observations(su, [row[0] for row in rendered], [False]*len(snapshots),
                               [row['pending']['names'] for row in snapshots],
                               [row['pending']['score'] for row in snapshots], list(range(len(snapshots))))

def attach_receipts(su, output):
    actor = su.env_object.tool_manager.centralized_actor_handle
    ids = [row['_runtime_context'] for row in su.env_object.plan_list]
    receipts = ray.get(actor.runtime_branch_receipts.remote(ids))
    for receipt, snapshot in zip(receipts, su.branch_sources):
        runtime = snapshot['runtime']
        receipt.update(context_hash=snapshot_hash(snapshot), candidate_hash=snapshot_hash(runtime['view']['candidates']),
                       registry_hash=snapshot_hash(runtime['registry']), alignment_hash=snapshot_hash(runtime['alignment']),
                       encoder_id=runtime['view']['encoder_id'], action_scope='next_retrieval_only', executed=True)
        receipt['execution_context_hash'] = runtime['execution_context_hash']
    values = np.empty(len(receipts), dtype=object)
    values[:] = receipts
    output.non_tensor_batch['continuation_receipt'] = values
    return output

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
