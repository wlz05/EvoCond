# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Measured episode evidence; absent observations are never replaced by zeros.'
from __future__ import annotations
import math
from control import exposure
FORMAT = 'evocond_episode_outcome_v1'
def measured(context, *, episode, arm, actor, bank, elapsed, variant=None):
    success = context.get('terminal_success')
    if success is not None and type(success) is not bool:
        raise ValueError('Terminal correctness must be boolean or unknown')
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Episode elapsed time must be measured and nonnegative')
    receipts = context.get('cost_receipts', [])
    observed_retrieval = 'selection_cache' in context
    selections = list(context.get('selection_cache', {}).values())
    actor_receipts = [row for row in receipts if row['kind'] == 'actor']
    retrievals = [row for row in receipts if row['kind'] == 'retrieval']
    tools = [row for row in receipts if row['kind'] == 'tools']
    fallbacks = [row for row in selections if row.get('abstention', {}).get('semantic_fallback')]
    tokens = {key: sum(row['quantities'][key] for row in actor_receipts) if actor_receipts else None
              for key in ('input_tokens', 'output_tokens')}
    metrics = dict(elapsed_seconds=elapsed, tool_calls=len(context.get('invocations', [])),
        actor_input_tokens=tokens['input_tokens'], actor_output_tokens=tokens['output_tokens'],
        retrieval_seconds=sum(row['quantities']['seconds'] for row in retrievals) if selections else None,
        tool_seconds=sum(row['quantities']['seconds'] for row in tools) if tools else None,
        accounted_cost=sum(context['costs'].values()) if context.get('actor_metered') else None,
        retrievals=len(selections) if observed_retrieval else None,
        structural_selections=sum(row.get('alpha', 0) > 0 for row in selections) if observed_retrieval else None,
        semantic_fallbacks=len(fallbacks) if observed_retrieval else None,
        fallback_payload_matches=sum(row.get('payload_hash') == row.get('semantic_payload_hash')
            and row.get('semantic_payload_hash') is not None for row in fallbacks) if observed_retrieval else None)
    metrics.update(exposure.summarize(selections))
    return dict(format=FORMAT, episode_id=episode, task_id=context['task_id'], split=context['split'],
        arm=arm, variant=variant, policy_step=context['policy_step'], actor=actor, bank=bank,
        snapshot=context.get('phase_snapshot'), phase=context.get('phase'), ordinal=context.get('ordinal'),
        success=success, metrics=metrics, collection_branch=context.get('collection_branch', False),
        scope='observed_task_outcome_not_causal_edge_credit_or_SOTA')

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
