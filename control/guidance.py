# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Withdraw historical ordering, not callable tools or current observations.'
from copy import deepcopy
from control.dynamic import digest
from control.ranking import apply_action
MODES = ('legacy', 'ranked', 'candidate_only')
def validate(mode):
    if mode not in MODES:
        raise ValueError('Unknown guidance mode')
def present(view, mode, *, top_k):
    validate(mode)
    alpha = view['teacher']['structure_weight']
    if mode == 'legacy':
        tools = deepcopy(view['tools'])
    else:
        pool = apply_action(view, 0., top_k=top_k)
        fixed = {**view, 'candidates': pool}
        tools = apply_action(fixed, alpha, top_k=top_k)
        if mode == 'candidate_only':
            tools.sort(key=lambda row: (row['name'], row['version']))
    identities = sorted((row['name'], row['version']) for row in tools)
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate presented tool versions')
    return tools, {'mode': mode, 'candidate_pool_hash': digest(identities),
        'presented_order': [row['name'] for row in tools],
        'scope': 'same_semantic_candidates_order_only' if mode != 'legacy' else 'legacy_candidate_selection',
        'no_memory': False, 'generated_values_are_current_task_decisions': True}

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
