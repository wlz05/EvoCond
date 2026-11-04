# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Validate observed official-worker continuation outcomes, never estimates.'
from control.pairing import snapshot_hash
from control.teacher import PROVENANCE_FIELDS
def observed_outcome(request, receipt, correctness, *, shared_registry_hash):
    # Python Boolean equality must not stand in for an execution count or seed.
    if (not isinstance(receipt, dict) or receipt.get('executed') is not True
            or type(receipt.get('interventions')) is not int or receipt['interventions'] != 1):
        raise ValueError('Official worker did not execute exactly one retrieval intervention')
    if type(receipt.get('seed')) is not int or receipt['seed'] != request.seed:
        raise ValueError('Official continuation receipt changed seed')
    alpha = receipt.get('alpha')
    if type(alpha) not in (int, float) or not 0 <= alpha <= 1 or alpha != request.alpha:
        raise ValueError('Official continuation receipt changed alpha')
    for name in (*PROVENANCE_FIELDS, 'action_scope'):
        if receipt.get(name) != getattr(request, name):
            raise ValueError(f'Official continuation receipt changed {name}')
    if receipt.get('shared_registry_hash') != shared_registry_hash:
        raise ValueError('A private continuation mutated shared tool memory')
    if request.execution_context_hash is not None and receipt.get('execution_context_hash') != request.execution_context_hash:
        raise ValueError('Continuation execution context changed')
    # Bounds also reject NaN, infinity and huge integers without float overflow.
    if type(correctness) not in (int, float) or not -1 <= correctness <= 1:
        raise ValueError('Unsupported official correctness scale; need a finite value in [-1,1]')
    # Official MCPBaseEnv uses -1 for no valid final answer; F1 otherwise.
    return {**receipt, 'reward': max(0., float(correctness)), 'success': correctness == 1.0}
def validate_snapshot(snapshot, *, question_id):
    if not isinstance(snapshot, dict) or snapshot.get('format') != 'searl_official_continuation_v1':
        raise ValueError('Need a restorable official-loop snapshot, not a retrieval summary')
    snapshot_hash(snapshot)
    if not isinstance(question_id, str) or not question_id.strip():
        raise ValueError('Training question identity must be a nonempty string')
    runtime = snapshot.get('runtime')
    if not isinstance(runtime, dict) or not isinstance(runtime.get('context'), dict):
        raise ValueError('Snapshot needs an official runtime context')
    context = runtime['context']
    if context.get('split') != 'train' or context.get('task_id') != question_id:
        raise ValueError('Snapshot was not captured from this training question')
    if context.get('private') is not True:
        raise ValueError('Snapshot needs a private training context')
    step, maximum = snapshot.get('step'), snapshot.get('max_turns')
    if type(step) is not int or step < 0 or type(maximum) is not int or maximum <= 0:
        raise ValueError('Snapshot step and horizon must be nonnegative integer indices')
    if step+1 >= maximum:
        raise ValueError('Snapshot leaves no real continuation step')
    return runtime

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
