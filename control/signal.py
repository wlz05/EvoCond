# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Count usable learning signal, not optimizer steps or causal action quality.'
from collections import defaultdict
import math
def metrics(rewards, groups, advantages, loss_masks, *, tolerance=1e-8, answer_correctness=None):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Signal tolerance must be finite and nonnegative')
    if not rewards or not (len(rewards) == len(groups) == len(advantages) == len(loss_masks)):
        raise ValueError('Training signal rows must be nonempty and aligned')
    reward_groups = defaultdict(list)
    tokens = active = empty = 0
    for reward, group, adv, mask in zip(rewards, groups, advantages, loss_masks):
        if type(reward) not in (int, float) or not math.isfinite(reward):
            raise ValueError('Nonfinite episode reward')
        if not isinstance(group, (str, int)) or isinstance(group, bool) or group == '':
            raise ValueError('Missing batch group identity')
        if len(adv) != len(mask) or any(type(m) not in (int, float, bool) or m not in (0, 1) for m in mask):
            raise ValueError('Native loss mask must be binary and match advantages')
        if any(type(a) not in (int, float) or not math.isfinite(a) for a in adv):
            raise ValueError('Nonfinite policy advantage')
        reward_groups[group].append(reward)
        tokens += sum(mask)
        active += sum(bool(m) and abs(a) > tolerance for a, m in zip(adv, mask))
        empty += not any(mask)
    informative = sum(max(values)-min(values) > tolerance for values in reward_groups.values())
    result = {'signal/batch_reward_groups': len(reward_groups),
            'signal/groups_with_reward_variation': informative,
            'signal/reward_variation_fraction': informative/len(reward_groups),
            'signal/native_optimized_tokens': tokens,
            'signal/active_advantage_tokens': active,
            'signal/active_advantage_fraction': active/tokens if tokens else 0.,
            'signal/fully_masked_trajectories': empty}
    # Shaped reward variation can be cost/format/KL variation, not better answers.
    if answer_correctness is not None:
        if len(answer_correctness) != len(groups):
            raise ValueError('Answer correctness must align with actual batch groups')
        correct_groups = defaultdict(list)
        for group, value in zip(groups, answer_correctness):
            if value is not None and (type(value) not in (int, float, bool) or
                                      not math.isfinite(value) or not -1 <= value <= 1):
                raise ValueError('Invalid answer correctness')
            known = value is not None and value in (-1, 0, 1)
            correct_groups[group].append(bool(value == 1) if known else None)
        known = [values for values in correct_groups.values() if all(v is not None for v in values)]
        result.update({'signal/groups_with_known_correctness': len(known),
                       'signal/groups_with_correctness_variation': sum(len(set(v)) > 1 for v in known)})
    return result

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
