# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Adjacent-actor sampled token KL receipts, without reference-policy substitution.'
from __future__ import annotations
import math
import numpy as np
FORMAT = 'evocond_actor_drift_v2'
ESTIMATOR = 'old_policy_sampled_expm1_log_ratio_minus_log_ratio'
DEFAULTS = dict(rule='mean_tail_or_trimmed', quantile=.95, trim=.05, tail_threshold=.02)
def options(value=None):
    if value is not None and (not isinstance(value, dict) or set(value)-set(DEFAULTS)):
        raise ValueError('Unknown actor drift diagnostic fields')
    result = {**DEFAULTS, **(value or {})}
    if result['rule'] not in ('mean', 'mean_tail_or_trimmed'):
        raise ValueError('Unknown declared actor drift decision rule')
    if any(type(result[key]) not in (int, float) or not math.isfinite(result[key]) for key in ('quantile', 'trim', 'tail_threshold')):
        raise ValueError('Actor drift diagnostics must be finite')
    if not .5 <= result['quantile'] < 1 or not 0 <= result['trim'] < .5 or result['tail_threshold'] <= 0:
        raise ValueError('Invalid actor drift tail or trimming protocol')
    return result
def estimate(old, new, mask, *, actor, seconds=0.0, diagnostics=None):
    recipe = options(diagnostics)
    before, after, active = (np.asarray(value, dtype=float) for value in (old, new, mask))
    if before.ndim != 2 or before.shape != after.shape or before.shape != active.shape:
        raise ValueError('Actor drift requires matched response tokens and masks')
    if not np.isfinite(active).all() or not np.isin(active, (0, 1)).all():
        raise ValueError('Actor drift requires a binary generated-token mask')
    selected = active.astype(bool)
    left, right = before[selected], after[selected]
    if not left.size or not np.isfinite(left).all() or not np.isfinite(right).all() or np.any(left > 1e-6) or np.any(right > 1e-6):
        raise ValueError('Actor drift has no valid sampled token log probabilities')
    ratio = right-left
    with np.errstate(over='ignore', invalid='ignore'):
        values = np.expm1(ratio)-ratio
    if not np.isfinite(values).all():
        raise ValueError('Actor drift estimate overflowed; do not clip silently')
    discarded = min(int(math.ceil(recipe['trim']*values.size)), values.size-1)
    retained = np.argsort(np.abs(ratio), kind='stable')[:values.size-discarded]
    result = dict(format=FORMAT, valid=True, actor=actor, estimator=ESTIMATOR,
        policy_kl=float(values.mean()), token_count=int(values.size), elapsed_seconds=float(seconds),
        diagnostics=recipe, tail_value=float(np.quantile(values, recipe['quantile'])),
        trimmed_divergence=float(values[retained].mean()), retained_tokens=int(retained.size),
        discarded_tokens=discarded, diagnostic_scope='tail_and_trimmed_statistics_are_not_KL_estimates',
        scope='sampled_old_occupancy_token_average_not_global_or_reference_KL')
    validate(result, actor)
    return result
def validate(receipt, actor):
    if not isinstance(receipt, dict) or receipt.get('format') != FORMAT or receipt.get('actor') != actor or receipt.get('valid') is not True or receipt.get('estimator') != ESTIMATOR:
        raise ValueError('Missing or mismatched adjacent-actor drift receipt')
    value, seconds, count = receipt.get('policy_kl'), receipt.get('elapsed_seconds'), receipt.get('token_count')
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or type(count) is not int or count < 1:
        raise ValueError('Invalid sampled actor KL estimate')
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Actor drift accounting must be finite')
    recipe = options(receipt.get('diagnostics'))
    if receipt.get('diagnostics') != recipe:
        raise ValueError('Actor drift diagnostics require their frozen recipe')
    for key in ('tail_value', 'trimmed_divergence'):
        if type(receipt.get(key)) not in (int, float) or not math.isfinite(receipt[key]) or receipt[key] < 0:
            raise ValueError('Invalid actor drift tail diagnostic')
    retained, discarded = receipt.get('retained_tokens'), receipt.get('discarded_tokens')
    expected = min(int(math.ceil(recipe['trim']*count)), count-1)
    if type(retained) is not int or type(discarded) is not int or retained < 1 or retained+discarded != count or discarded != expected:
        raise ValueError('Actor drift trimming counts do not match the recipe')
    return receipt
def decision(receipt, actor, threshold, diagnostics=None):
    validate(receipt, actor)
    recipe = options(diagnostics)
    if recipe != receipt['diagnostics'] or type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('Actor drift decision differs from its frozen protocol')
    mean = receipt['policy_kl'] > threshold
    tail = receipt['tail_value'] > recipe['tail_threshold']
    trimmed = receipt['trimmed_divergence'] > threshold
    reset = mean if recipe['rule'] == 'mean' else (mean and tail) or trimmed
    return dict(reset=reset, rule=recipe['rule'], mean_exceeded=mean, tail_exceeded=tail,
        trimmed_exceeded=trimmed, isolated_tail_suppressed=mean and not reset,
        threshold=threshold, tail_threshold=recipe['tail_threshold'], quantile=recipe['quantile'], trim=recipe['trim'])

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
