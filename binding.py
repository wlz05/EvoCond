# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Read real binding events or externally collected paired diagnostic receipts.'
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
from control.binding import FORMAT
from bootstrap import artifact_rows
CELLS = ('semantic.reject', 'semantic.rebind', 'structural.reject', 'structural.rebind')
def rows(path):
    yield from artifact_rows(path)
def events(records):
    counts = Counter()
    seen = set()
    for row in records:
        key = (row.get('arm'), row.get('episode_id'))
        if row.get('format') != FORMAT or key in seen or not row.get('episode_id') or not row.get('task_id'):
            raise ValueError('Missing or duplicate identified binding episode')
        seen.add(key)
        if row.get('collection_branch'):
            raise ValueError('Branch and deployment events must be analyzed separately')
        counts['episodes'] += 1
        if type(row.get('task_success')) is bool:
            counts['known_final_outcomes'] += 1
            counts['correct'] += int(row['task_success'])
        for audit in row['audits']:
            counts['proposed_calls'] += 1
            counts['disabled_calls'] += audit['strategy'] == 'disabled'
            counts['covered_calls'] += bool(audit.get('covered_fields'))
            counts['before_conflicts'] += audit['before']['decision'] == 'conflict'
            counts['after_unknown'] += audit['after']['decision'] == 'unknown'
            counts['rewritten_calls'] += bool(audit['changed'])
            counts['executed_rewrites'] += bool(audit['changed'] and audit['executed'])
    return {'counts': dict(counts), 'claim': 'Observed explicit-contract coverage/corrections only; not general semantic error rate.'}
def paired(records, *, samples=1000, seed=42):
    if type(samples) is not int or samples < 2 or type(seed) is not int or seed < 0:
        raise ValueError('Bootstrap needs at least two samples and a nonnegative integer seed')
    groups, seen = defaultdict(list), set()
    divergences = Counter()
    for row in records:
        if row.get('format') != 'evocond_binding_factorial_v1' or not row.get('task_id'):
            raise ValueError('Expected identified factorial receipts')
        if type(row.get('seed')) is not int or row['seed'] < 0 or not isinstance(row.get('variant'), str):
            raise ValueError('Expected an explicit nonnegative seed and variant')
        key = (row['task_id'], row['variant'], row['seed'])
        if key in seen:
            raise ValueError('Repeated task/variant/seed receipt')
        seen.add(key)
        controls = row.get('controls', {})
        if set(controls) != {'policy_hash', 'prefix_hash', 'candidate_hash', 'environment_hash', 'budget_hash'} or any(
                not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v) for v in controls.values()):
            raise ValueError('Explicit shared control fingerprints are required')
        cells = row.get('cells', {})
        if set(cells) != set(CELLS):
            raise ValueError('All four paired cells are required; no dropping failed cells')
        for cell in cells.values():
            if cell.get('executed') is not True or type(cell.get('success')) is not bool or cell.get('controls') != controls:
                raise ValueError('All cells must declare real execution, final outcomes and matched controls')
            action = cell.get('first_action')
            if not isinstance(action, dict) or not isinstance(action.get('name'), str) or not isinstance(action.get('arguments'), dict):
                raise ValueError('First proposed action is required for divergence diagnostics')
        for left, right in (('semantic.reject', 'structural.reject'), ('semantic.rebind', 'structural.rebind')):
            a, b = cells[left]['first_action'], cells[right]['first_action']
            if a['name'] != b['name'] or a.get('tool_version') != b.get('tool_version'):
                kind = 'tool_or_version_differs'
            elif a['arguments'] != b['arguments']:
                kind = 'same_tool_arguments_differ'
            else:
                kind = 'identical_proposed_action'
            divergences[left+'__'+right+'.'+kind] += 1
        values = {k: int(cells[k]['success']) for k in CELLS}
        sr, sb, tr, tb = (values[k] for k in CELLS)
        groups[row['task_id']].append({'retrieval_at_reject': tr-sr, 'retrieval_at_rebind': tb-sb,
            'binding_at_semantic': sb-sr, 'binding_at_structural': tb-tr,
            'interaction': (tb-tr)-(sb-sr), **{'success.'+k: v for k, v in values.items()}})
    if not groups:
        raise ValueError('No paired data; cannot manufacture evidence')
    task_means = [{k: sum(r[k] for r in data)/len(data) for k in data[0]} for data in groups.values()]
    rng = random.Random(seed)
    boot = {k: [] for k in task_means[0]}
    for _ in range(samples):
        selected = [rng.choice(task_means) for _ in task_means]
        for k in boot:
            boot[k].append(sum(r[k] for r in selected)/len(selected))
    estimates = {k: {'mean': sum(r[k] for r in task_means)/len(task_means),
                     'cluster_bootstrap_interval': [sorted(v)[int(.025*(samples-1))], sorted(v)[int(.975*(samples-1))]]}
                 for k, v in boot.items()}
    return {'independent_tasks': len(groups), 'paired_receipts': len(seen), 'estimates': estimates,
            'first_action_comparisons': dict(divergences),
            'claim': 'Caller-recorded paired execution; cluster by base task. Not validated causal identification or a safety bound.'}
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('events', 'paired'))
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    report = (events if args.kind == 'events' else paired)(rows(args.input))
    if args.run:
        if args.output is None:
            parser.error('--run requires a new output file')
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
