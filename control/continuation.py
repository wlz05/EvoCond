# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Validate observed official-worker continuation outcomes, never estimates.'
import math
from control.pairing import snapshot_hash
from control.teacher import PROVENANCE_FIELDS
def observed_outcome(request, receipt, correctness, *, shared_registry_hash):
    if receipt.get('executed') is not True or receipt.get('interventions') != 1:
        raise ValueError('Official worker did not execute exactly one retrieval intervention')
    for name in (*PROVENANCE_FIELDS, 'seed', 'alpha', 'action_scope'):
        if receipt.get(name) != getattr(request, name):
            raise ValueError(f'Official continuation receipt changed {name}')
    if receipt.get('shared_registry_hash') != shared_registry_hash:
        raise ValueError('A private continuation mutated shared tool memory')
    if request.execution_context_hash is not None and receipt.get('execution_context_hash') != request.execution_context_hash:
        raise ValueError('Continuation execution context changed')
    if type(correctness) not in (int, float) or not math.isfinite(correctness):
        raise ValueError('Official answer correctness is missing or nonfinite')
    if not -1 <= correctness <= 1:
        raise ValueError('Unsupported official correctness scale; cannot invent normalization')
    # Official MCPBaseEnv uses -1 for no valid final answer; F1 otherwise.
    return {**receipt, 'reward': max(0., float(correctness)), 'success': correctness == 1.0}
def validate_snapshot(snapshot, *, question_id):
    snapshot_hash(snapshot)
    if snapshot.get('format') != 'searl_official_continuation_v1':
        raise ValueError('Need a restorable official-loop snapshot, not a retrieval summary')
    runtime = snapshot['runtime']
    if runtime['context']['split'] != 'train' or runtime['context']['task_id'] != question_id:
        raise ValueError('Snapshot was not captured from this training question')
    if snapshot['step']+1 >= snapshot['max_turns']:
        raise ValueError('Snapshot leaves no real continuation step')
    return runtime

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
