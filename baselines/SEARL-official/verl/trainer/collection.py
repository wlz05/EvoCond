# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Frozen-policy bootstrap using the official worker and official reward manager."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import ray
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from control.runtime import task_identity, load_config
from control.pairing import collect_teacher_policy_pairs
from control.continuation import observed_outcome, validate_snapshot

def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)

def split_batch(batch):
    keys = ('raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs', 'raw_prompt', 'tools_kwargs',
            'interaction_kwargs', 'index', 'agent_name', 'question', 'mcp_worthiness', 'tool_call_count')
    if 'question' not in batch.non_tensor_batch:
        raise ValueError('Teacher bootstrap requires full visible question metadata')
    generation = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                           non_tensor_batch_keys=[key for key in keys if key in batch.non_tensor_batch])
    generation.meta_info.update(validate=False, global_steps=0, do_sample=True)
    return batch, generation

def run(trainer):
    config = trainer.config
    mode = config.trainer.teacher_collect_mode
    path = Path(config.trainer.teacher_collect_path)
    seeds = list(config.trainer.teacher_collect_seeds)
    turn = config.trainer.teacher_capture_turn
    if (mode not in ('warmup', 'pairs') or trainer.global_steps != 0 or trainer.async_rollout_mode
            or config.actor_rollout_ref.rollout.n != 1 or config.trainer.get('val_only', False)
            or config.actor_rollout_ref.rollout.name != 'vllm'):
        raise ValueError('Bootstrap needs frozen step-zero Qwen, n=1 synchronous official vLLM; no validation/test')
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Distinct explicit common random seeds are required')
    runtime = load_config(config.actor_rollout_ref.env.runtime_config_path)
    if (runtime['arm'] != 'searl' or bool(runtime.get('collection')) != (mode == 'pairs')
            or bool(runtime.get('bootstrap_warmup')) != (mode == 'warmup')):
        raise ValueError('Warmup and private paired collection runtime identities differ')
    actor = trainer._method_actor()
    if actor is None:
        raise ValueError('Teacher collection requires the isolated runtime actor')
    shared_hash = ray.get(actor.runtime_registry_hash.remote())
    path.mkdir(parents=True, exist_ok=True)
    result_path = path/('snapshots.jsonl' if mode == 'warmup' else 'pairs.jsonl')
    if result_path.exists() or (path/'audit.json').exists():
        raise FileExistsError('Bootstrap output is immutable; use a fresh run directory')
    seen, count, skipped = set(), 0, []
    with result_path.open('x', encoding='utf-8') as stream:
        for data in trainer.train_dataloader:
            base, generation = split_batch(DataProto.from_single_dict(data))
            questions = list(generation.non_tensor_batch['question'])
            question_ids = [task_identity(str(question)) for question in questions]
            if len(set(question_ids)) != len(question_ids) or seen.intersection(question_ids):
                raise ValueError('Bootstrap must traverse each training question once')
            seen.update(question_ids)
            if mode == 'warmup':
                generation.meta_info['continuation_seed'] = seeds[0]
                padded, padding = pad_dataproto_to_divisor(generation, trainer.actor_rollout_wg.world_size)
                padded_base, base_padding = pad_dataproto_to_divisor(base, trainer.actor_rollout_wg.world_size)
                if padding != base_padding:
                    raise ValueError('Warmup reward metadata padding changed row alignment')
                output = trainer.actor_rollout_wg.generate_sequences_loop(padded)
                scored = padded_base.union(output)
                reward = trainer.reward_fn(scored, return_dict=True)
                trainer._method_complete(scored, reward.get('reward_extra_info', {}), 'train')
                continue
            generation.meta_info.update(teacher_continuation_mode='capture', teacher_capture_turn=turn,
                                        continuation_seed=seeds[0])
            padded, padding = pad_dataproto_to_divisor(generation, trainer.actor_rollout_wg.world_size)
            output = trainer.actor_rollout_wg.generate_sequences_loop(padded)
            output = unpad_dataproto(output, padding)
            snapshots = list(output.non_tensor_batch['continuation_snapshot'])
            if len(snapshots) != len(questions):
                raise ValueError('Captured prefix row alignment changed')
            for index, snapshot in enumerate(snapshots):
                prefix = validate_snapshot(snapshot, question_id=question_ids[index])
                if not snapshot['eligible']:
                    skipped.append({'query_id': question_ids[index], 'reason': snapshot['reason']})
                    continue
                original_generation, original_base = generation[index:index+1], base[index:index+1]
                def evaluate(request):
                    resumed = deepcopy(original_generation)
                    values = np.empty(1, dtype=object)
                    values[0] = deepcopy(request.state)
                    resumed.non_tensor_batch['continuation_snapshot'] = values
                    resumed.meta_info.update(teacher_continuation_mode='resume', continuation_alpha=request.alpha,
                                             continuation_seed=request.seed)
                    padded, padding = pad_dataproto_to_divisor(resumed, trainer.actor_rollout_wg.world_size)
                    generated = trainer.actor_rollout_wg.generate_sequences_loop(padded)
                    generated = unpad_dataproto(generated, padding)
                    scored = deepcopy(original_base).union(generated)
                    # This is the same configured official reward_fn used by PPO, not a new solver or heuristic.
                    reward = trainer.reward_fn(scored, return_dict=True)
                    correctness = reward.get('reward_extra_info', {}).get('answer_correctness')
                    if correctness is None or len(correctness) != 1:
                        raise ValueError('Official reward manager must export actual answer_correctness')
                    if ray.get(actor.runtime_registry_hash.remote()) != shared_hash:
                        raise ValueError('Paired continuation wrote shared memory')
                    receipt = generated.non_tensor_batch['continuation_receipt'][0]
                    return observed_outcome(request, receipt, float(correctness[0]), shared_registry_hash=shared_hash)
                view = prefix['view']
                records = collect_teacher_policy_pairs(evaluator=evaluate, query_id=question_ids[index],
                    decision_id=f'official_loop_turn_{snapshot["step"]}', split='train', state=snapshot,
                    registry=prefix['registry'], candidates=view['candidates'], alignment=prefix['alignment'],
                    features=view['features'], query_embedding=view['query_embedding'], encoder_id=view['encoder_id'],
                    policy_hash=prefix['policy_hash'], seeds=seeds, execution_context_hash=prefix['execution_context_hash'])
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                    stream.flush()
                    count += 1
        if mode == 'warmup':
            exported = ray.get(actor.runtime_bootstrap_export.remote())
            for row in exported['queries']:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                count += 1
            write_new(path/'registry.json', exported['registry'])
    if len(seen) != len(trainer.train_dataset):
        raise ValueError('Bootstrap did not cover the complete frozen training partition')
    if not count:
        raise ValueError('No observed bootstrap rows; no teacher artifact can be claimed')
    write_new(path/'audit.json', {'complete': True, 'mode': mode, 'queries': len(seen), 'records': count,
        'skipped': skipped, 'qwen_updated': False, 'policy_hash': runtime['policy_hash'],
        'train_only': True, 'source': 'official_generate_sequences_loop_and_reward_fn',
        'continuation_policy': 'next_retrieval_override_then_fixed_semantic',
        'seeds': seeds, 'capture_turn': turn, 'shared_registry_hash_before': shared_hash,
        'shared_registry_hash_after': ray.get(actor.runtime_registry_hash.remote())})

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
