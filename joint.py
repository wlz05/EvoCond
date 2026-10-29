# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Separate policy, tool-bank and ordering changes in an executed factorial.'
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
from bootstrap import artifact_rows
FORMAT = 'evocond_joint_factorial_v1'
GUIDANCE = ('ranked', 'candidate_only')
CELLS = tuple(f'p{p}.k{k}.{h}' for p in (0, 1) for k in (0, 1) for h in GUIDANCE)
SHARED = {'task_hash', 'environment_hash', 'budget_hash', 'method_hash', 'decoder_hash'}
def fingerprint(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Expected an explicit SHA256 identity')
    return value
def summarize(records, *, samples=1000, seed=42):
    if type(samples) is not int or samples < 2 or type(seed) is not int or seed < 0:
        raise ValueError('Invalid cluster-bootstrap configuration')
    clusters, seen = defaultdict(list), set()
    experiment = None
    for row in records:
        if row.get('format') != FORMAT or not isinstance(row.get('task_id'), str) or not row['task_id']:
            raise ValueError('Expected an identified joint factorial receipt')
        if not isinstance(row.get('variant'), str) or type(row.get('seed')) is not int or row['seed'] < 0:
            raise ValueError('Explicit variant and nonnegative sampling seed required')
        key = row['task_id'], row['variant'], row['seed']
        if key in seen:
            raise ValueError('Duplicate base task/variant/seed')
        seen.add(key)
        policies, banks, shared = row.get('policies', {}), row.get('banks', {}), row.get('shared', {})
        if set(policies) != {'p0', 'p1'} or set(banks) != {'k0', 'k1'} or set(shared) != SHARED:
            raise ValueError('Two policy checkpoints, two versioned banks and shared controls required')
        for value in (*policies.values(), *banks.values(), *shared.values()):
            fingerprint(value)
        identity = {**policies, **banks, **{k: shared[k] for k in SHARED-{'task_hash'}}}
        if experiment is not None and experiment != identity:
            raise ValueError('Different checkpoints, banks or evaluation settings cannot be pooled')
        experiment = identity
        cells = row.get('cells', {})
        if set(cells) != set(CELLS):
            raise ValueError('All eight executed cells, including failures, are required')
        values = {}
        for label, cell in cells.items():
            p, k, guidance = label.split('.')
            if (cell.get('executed') is not True or type(cell.get('success')) is not bool or
                    cell.get('policy_hash') != policies[p] or cell.get('bank_hash') != banks[k] or
                    cell.get('shared') != shared or cell.get('guidance_mode') != guidance):
                raise ValueError('Cell execution/outcome/control identity mismatch')
            fingerprint(cell.get('initial_candidate_hash'))
            values[label] = int(cell['success'])
        for p in ('p0', 'p1'):
            for k in ('k0', 'k1'):
                if cells[f'{p}.{k}.ranked']['initial_candidate_hash'] != cells[f'{p}.{k}.candidate_only']['initial_candidate_hash']:
                    raise ValueError('Ordering comparison changed initial callable candidates')
        effects = {}
        for h in GUIDANCE:
            v00, v10, v01, v11 = (values[f'p{p}.k{k}.{h}'] for p, k in ((0, 0), (1, 0), (0, 1), (1, 1)))
            effects.update({f'{h}.policy_at_initial_bank': v10-v00,
                            f'{h}.bank_at_initial_policy': v01-v00,
                            f'{h}.joint_interaction': v11-v10-v01+v00,
                            f'{h}.total_change': v11-v00})
        for p in ('p0', 'p1'):
            for k in ('k0', 'k1'):
                effects[f'ordering_effect.{p}.{k}'] = values[f'{p}.{k}.ranked']-values[f'{p}.{k}.candidate_only']
        effects.update({'success.'+k: v for k, v in values.items()})
        clusters[row['task_id']].append(effects)
    if not clusters:
        raise ValueError('No executed data; cannot fabricate joint-evolution evidence')
    means = [{k: sum(v[k] for v in group)/len(group) for k in group[0]} for group in clusters.values()]
    rng = random.Random(seed)
    boot = {k: [] for k in means[0]}
    for _ in range(samples):
        draw = [rng.choice(means) for _ in means]
        for k in boot:
            boot[k].append(sum(v[k] for v in draw)/len(draw))
    estimates = {k: {'mean': sum(v[k] for v in means)/len(means),
                     'cluster_bootstrap_interval': [sorted(v)[int(.025*(samples-1))], sorted(v)[int(.975*(samples-1))]]}
                 for k, v in boot.items()}
    return {'independent_tasks': len(clusters), 'paired_receipts': len(seen), 'controls': experiment,
            'estimates': estimates,
            'decomposition': 'total_change = policy_at_initial_bank + bank_at_initial_policy + joint_interaction',
            'warning': 'Descriptive factorial of caller-recorded executions. Not a causal proof, a no-tools test, or a guarantee of policy internalization. All cells require a live collector not provided by this script.'}
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    report = summarize(artifact_rows(args.input))
    if args.run:
        if args.output is None:
            parser.error('--run requires a new output file')
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
if __name__ == '__main__':
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
