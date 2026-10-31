# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Read-only protocol arithmetic and synthetic fixed-proposal coverage checks.'
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from control import bandit, stage, windows

def configured(config, *, questions=None):
    if not isinstance(config, dict) or 'config' in config and not isinstance(config['config'], dict):
        raise ValueError('Calibration needs an experiment configuration or frozen protocol')
    base = config.get('config', config)
    options = stage.settings(base.get('contextual', {}))
    if 'config' in config:
        if questions is not None:
            raise ValueError('A frozen protocol already fixes the training schedule')
        schedule = config.get('schedule', {})
        if not isinstance(schedule, dict):
            raise ValueError('Frozen calibration schedule must be an object')
        if 'total_teacher_evaluations' in schedule and windows.confidence(options, schedule) != options:
            raise ValueError('Frozen confidence allocation differs from its scheduled evaluations')
    elif questions is not None:
        budget = base.get('budget', {})
        if not isinstance(budget, dict):
            raise ValueError('Calibration budget must be an object')
        schedule = windows.layout(questions, budget.get('batch_size', 256), options['frozen_batches'],
            tail=budget.get('tail_policy', 'error'), epochs=budget.get('epochs', 1))
        options = windows.confidence(options, schedule)
    return base, options

def audit(config, *, size=128, phase=None, questions=None):
    if type(size) is not int or size < 2:
        raise ValueError('Declare at least two reference held-out base tasks')
    base, options = configured(config, questions=questions)
    protocol = dict(horizon=options['horizon_steps'], penalty=options['penalty'], epsilon=options['epsilon'],
        beta=options['beta']/options['maximum_windows'], delta_max=options['delta_max'])
    if phase is not None and not isinstance(phase, dict):
        raise ValueError('Calibration phase must be an object')
    receipt = phase.get('teacher_update') if phase is not None else None
    if phase is not None and (not isinstance(receipt, dict) or 'heldout_advantages' not in receipt):
        raise ValueError('This phase has no recorded independent hold-out advantages')
    if receipt is not None:
        saved = phase.get('confidence_protocol')
        if saved != protocol:
            raise ValueError('Audit config differs from the recorded phase confidence protocol')
        values = receipt['heldout_advantages']
        if len(values) != len(receipt['holdout_ids']) or receipt['n'] != len(values):
            raise ValueError('Recorded hold-out count and advantages differ')
        size = len(values)
    else:
        values = [0.0]*size
    strict = bandit.confidence(values, **protocol, method='bernstein')
    normal = bandit.confidence(values, **protocol, method='normal')
    width = 1+options['penalty']
    single = bandit.confidence([0.0]*size, **{**protocol, 'beta': options['beta']}, method='bernstein')
    return dict(kind='observed_phase' if receipt is not None else 'configured_arithmetic_not_observed_training',
        n=size, expected_holdout=base.get('budget', {}).get('batch_size', 256)*options['holdout'],
        maximum_evaluations=options['maximum_windows'], total_beta=options['beta'],
        protocol=protocol, reward_width=width, global_envelope=width/(options['epsilon']/3),
        single_window_range_penalty=single['range_penalty'], allocated_window_range_penalty=strict['range_penalty'],
        sample_size_floor_for_range_below_reward_width=math.floor(14*math.log(2/protocol['beta'])/(options['epsilon']))+2,
        methods=dict(bernstein=strict, normal=normal), gpu_used=False, model_loaded=False)
def simulate(options, *, size=128, trials=1000, seed=42):
    if type(size) is not int or size < 2 or type(trials) is not int or trials < 1 or type(seed) is not int or seed < 0:
        raise ValueError('Simulation needs positive sizes and an identified seed')
    protocol = dict(horizon=options['horizon_steps'], penalty=options['penalty'], epsilon=options['epsilon'],
        beta=options['beta']/options['maximum_windows'], delta_max=options['delta_max'])
    rng = np.random.default_rng(seed)
    old, new = np.array([1.0, 0.0, 0.0]), np.array([.8, .1, .1])
    d = new-old
    p = (1-options['epsilon'])*old+options['epsilon']/3
    result = []
    for name, means, support, oracle in (
        ('positive_oracle', [.4, .6, .8], 1.0, True),
        ('null_oracle', [.5, .5, .5], 1.0, True),
        ('negative_oracle', [.8, .6, .4], 1.0, True),
        ('positive_sparse', [.4, .6, .8], .2, True),
        ('null_misspecified', [.5, .5, .5], 1.0, False),
        ('negative_sparse_misspecified', [.8, .6, .4], .2, False),
        ('null_sparse_optimistic', [0, 0, 0], .2, [0, 1, 1])):
        mu = np.array(means)
        head = np.asarray(oracle, dtype=float) if isinstance(oracle, list) else mu if oracle else np.zeros(3)
        truth = float(support*(d @ mu)*protocol['horizon'])
        truth = 0.0 if abs(truth) < 1e-12 else truth
        counts = {method: dict(computed=0, covered=0, positive=0, false_positive=0, width=0.0) for method in bandit.CONFIDENCE}
        for _ in range(trials):
            chosen = rng.choice(3, size=size, p=p)
            rewards = rng.binomial(1, mu[chosen])
            active = rng.random(size) < support
            values = (d @ head+d[chosen]*(rewards-head[chosen])/p[chosen])*active
            for method in bandit.CONFIDENCE:
                receipt = bandit.confidence(values.tolist(), **protocol, method=method)
                count = counts[method]
                lower = receipt['advantage_lower']
                if lower is not None:
                    count['computed'] += 1
                    count['covered'] += int(lower <= truth+1e-12)
                    count['width'] += protocol['horizon']*receipt['confidence_radius']
                count['positive'] += int(receipt['delta'] > 0)
                count['false_positive'] += int(receipt['delta'] > 0 and truth <= 0)
        for count in counts.values():
            computed = count['computed']
            count.update(coverage_among_computed=count.pop('covered')/computed if computed else None,
                mean_one_sided_radius=count.pop('width')/computed if computed else None,
                positive_rate=count.pop('positive')/trials, false_positive_rate=count.pop('false_positive')/trials)
        result.append(dict(scenario=name, true_total_advantage=truth, methods=counts))
    return dict(kind='synthetic_not_benchmark', n=size, trials=trials, seed=seed, protocol=protocol,
        nominal_coverage=1-protocol['beta'], scenarios=result, finite_sample_guarantee_for_normal=False,
        note='A small Monte Carlo run cannot resolve an extreme nominal tail probability')
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/contextual.json'))
    parser.add_argument('--phase', type=Path)
    parser.add_argument('--questions', type=int)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--trials', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    phase = json.loads(args.phase.read_text(encoding='utf-8')) if args.phase else None
    result = audit(config, size=args.size, phase=phase, questions=args.questions)
    if args.simulate:
        _, options = configured(config, questions=args.questions)
        result['simulation'] = simulate(options, size=args.size, trials=args.trials, seed=args.seed)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
if __name__ == '__main__':
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
