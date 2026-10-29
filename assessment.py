# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Strict paired pass-at-one evidence with seed and template-family uncertainty.'
from __future__ import annotations
import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path, PureWindowsPath
import random
from control.contracts import strict_json
from control.dynamic import digest
from control.outcomes import FORMAT
from control.exposure import METRICS as EXPOSURE
from experiment import read, write_new
METRICS = ('tool_calls', 'elapsed_seconds', 'actor_input_tokens', 'actor_output_tokens',
    'tool_seconds', 'retrieval_seconds', 'accounted_cost', 'order_conflicts', 'old_route_misuse',
    'retrievals', 'structural_selections', 'semantic_fallbacks', 'fallback_payload_matches', *EXPOSURE)
def relative(base, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError('Evidence paths must be relative to the manifest')
    return (base/value).resolve(strict=True)
def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0
def samples(path):
    before = sha256(path.read_bytes()).hexdigest()
    rows = []
    with path.open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = strict_json(line)
                    if not isinstance(row, dict):
                        raise ValueError('Outcome must be a JSON object')
                    rows.append(row)
                except ValueError as error:
                    raise ValueError(f'Invalid outcome line {number}: {error}') from error
    if sha256(path.read_bytes()).hexdigest() != before:
        raise ValueError('Outcome evidence changed during analysis')
    return rows, before
def validate(manifest):
    if manifest.get('format') != 'evocond_paired_assessment_v1' or manifest.get('split') not in ('validation', 'test'):
        raise ValueError('Declare a development or test paired assessment')
    if manifest.get('comparison') not in ('end_to_end', 'fixed_actor_bank'):
        raise ValueError('Separate total-method comparisons from fixed-bottom attribution')
    bootstrap = manifest.get('bootstrap', {})
    if set(bootstrap) != {'seed', 'samples'} or type(bootstrap['seed']) is not int or bootstrap['seed'] < 0 or type(bootstrap['samples']) is not int or not 100 <= bootstrap['samples'] <= 10000:
        raise ValueError('Declare the bounded bootstrap before inspecting outcomes')
    protocol = manifest.get('protocol', {})
    if set(protocol) != {'split', 'judge', 'executor', 'budget'} or any(not isinstance(value, str) or len(value) != 64 or any(x not in '0123456789abcdef' for x in value) for value in protocol.values()):
        raise ValueError('Declare matching split, judge, executor and budget fingerprints')
    tasks = manifest.get('tasks', [])
    if not tasks or any(not isinstance(row, dict) or set(row) != {'task_id', 'family', 'stratum'} or any(not isinstance(value, str) or not value for value in row.values()) for row in tasks):
        raise ValueError('Explicit task identities, template families and analysis strata are required')
    if len({row['task_id'] for row in tasks}) != len(tasks):
        raise ValueError('Repeated base questions cannot count as independent test tasks')
    runs = manifest.get('runs', [])
    if not runs or any(not isinstance(row, dict) or set(row) != {'method', 'seed', 'outcomes', 'runtime', 'policy_step'} for row in runs):
        raise ValueError('Each run must identify outcomes, runtime configuration and checkpoint step')
    pairs = []
    for row in runs:
        if not isinstance(row['method'], str) or not row['method'] or type(row['seed']) is not int or row['seed'] < 0 or type(row['policy_step']) is not int or row['policy_step'] < 0:
            raise ValueError('Invalid method, independent seed or checkpoint step')
        pairs.append((row['method'], row['seed']))
    methods = sorted({method for method, seed in pairs})
    seeds = sorted({seed for method, seed in pairs})
    if len(methods) < 2 or len(set(pairs)) != len(pairs) or set(pairs) != {(method, seed) for method in methods for seed in seeds}:
        raise ValueError('Methods require the same complete set of independent seeds')
    if manifest.get('baseline') not in methods:
        raise ValueError('Declare a baseline included in the paired panel')
    return tasks, methods, seeds
def mean(values):
    return sum(values)/len(values) if values else None
def distribution(values):
    ordered = sorted(values)
    def quantile(q):
        index = (len(ordered)-1)*q
        left = int(index)
        return ordered[left]+(ordered[min(left+1, len(ordered)-1)]-ordered[left])*(index-left)
    return dict(mean=mean(ordered), p50=quantile(.5), p95=quantile(.95)) if ordered else None
def summarize(rows):
    result = dict(n=len(rows), correct=sum(row['success'] for row in rows), pass_at_one=mean([int(row['success']) for row in rows]), metrics={})
    for key in METRICS:
        present = [row['metrics'][key] for row in rows if row['metrics'].get(key) is not None]
        successful = [row['metrics'][key] for row in rows if row['success'] and row['metrics'].get(key) is not None]
        result['metrics'][key] = dict(observed=len(present), missing=len(rows)-len(present),
            all=distribution(present) if len(present) == len(rows) else None,
            successful_only=distribution(successful), successful_observed=len(successful))
    return result
def confidence(data, baseline, method, tasks, seeds, recipe):
    groups = defaultdict(list)
    for row in tasks:
        groups[row['family']].append(row['task_id'])
    families = sorted(groups)
    rng = random.Random(recipe['seed'])
    values = []
    for _ in range(recipe['samples']):
        selected_seeds = rng.choices(seeds, k=len(seeds))
        selected_tasks = [task for family in rng.choices(families, k=len(families)) for task in groups[family]]
        values.append(mean([int(data[method, seed, task]['success'])-int(data[baseline, seed, task]['success'])
            for seed in selected_seeds for task in selected_tasks]))
    values.sort()
    return dict(lower=values[int(.025*(len(values)-1))], upper=values[int(.975*(len(values)-1))],
        level=.95, unit='paired_seed_and_template_family', informative=len(seeds) >= 3 and len(families) >= 2)
def assess(manifest, base):
    tasks, methods, seeds = validate(manifest)
    identities = {row['task_id'] for row in tasks}
    data, hashes = {}, []
    for run in manifest['runs']:
        path = relative(base, run['outcomes'])
        runtime = read(relative(base, run['runtime']))
        if runtime.get('contextual', {}).get('seed', run['seed']) != run['seed']:
            raise ValueError('Declared independent seed differs from the runtime configuration')
        rows, checksum = samples(path)
        selected = [row for row in rows if row.get('split') == manifest['split'] and row.get('policy_step') == run['policy_step']]
        if not selected:
            raise ValueError('No observations for the declared split and checkpoint')
        for row in selected:
            if row.get('format') != FORMAT or row.get('runtime_sha256') != digest(runtime) or row.get('collection_branch') is not False or type(row.get('success')) is not bool:
                raise ValueError('Unknown correctness, runtime provenance or collection branch')
            if row.get('task_id') not in identities or not isinstance(row.get('metrics'), dict):
                raise ValueError('Outcome does not belong to the declared paired task set')
            for value in row['metrics'].values():
                if value is not None and not finite(value):
                    raise ValueError('Observed metrics must be finite and nonnegative')
            key = run['method'], run['seed'], row['task_id']
            if key in data:
                raise ValueError('Duplicate draw: choose one sealed pass-at-one evaluation attempt')
            if any(not isinstance(row.get(field), str) or len(row[field]) != 64 for field in ('actor', 'bank')):
                raise ValueError('Actor and public-bank snapshots must be identified')
            data[key] = row
        if {row['task_id'] for row in selected} != identities:
            raise ValueError('Incomplete paired tasks; do not silently take their intersection')
        hashes.append(dict(method=run['method'], seed=run['seed'], outcomes_sha256=checksum, runtime_sha256=digest(runtime)))
    if manifest['comparison'] == 'fixed_actor_bank':
        for seed in seeds:
            snapshots = {(data[method, seed, task]['actor'], data[method, seed, task]['bank']) for method in methods for task in identities}
            if len(snapshots) != 1:
                raise ValueError('Fixed-bottom attribution requires the same actor and public bank')
    baseline = manifest['baseline']
    summaries, pairs = {}, {}
    for method in sorted(methods, key=lambda name: (name != baseline, name)):
        ordered = [data[method, seed, row['task_id']] for seed in seeds for row in tasks]
        summaries[method] = summarize(ordered)
        summaries[method]['strata'] = {name: summarize([data[method, seed, row['task_id']] for seed in seeds for row in tasks if row['stratum'] == name])
            for name in sorted({row['stratum'] for row in tasks})}
        if method == baseline:
            continue
        comparisons = [(data[method, seed, row['task_id']], data[baseline, seed, row['task_id']]) for seed in seeds for row in tasks]
        gain = sum(left['success'] and not right['success'] for left, right in comparisons)
        loss = sum(right['success'] and not left['success'] for left, right in comparisons)
        delta = (gain-loss)/len(comparisons)
        rate = summaries[baseline]['pass_at_one']
        pairs[method] = dict(gain=gain, loss=loss, success_delta=delta, percentage_points=100*delta,
            relative_percent=100*delta/rate if rate else None,
            confidence=confidence(data, baseline, method, tasks, seeds, manifest['bootstrap']), metric_deltas={})
        for metric in METRICS:
            known = [(left['metrics'].get(metric), right['metrics'].get(metric)) for left, right in comparisons]
            pairs[method]['metric_deltas'][metric] = mean([left-right for left, right in known]) if all(left is not None and right is not None for left, right in known) else None
    missing = sorted({key for summary in summaries.values() for key, row in summary['metrics'].items() if row['missing']})
    return dict(format='evocond_paired_report_v1', manifest_sha256=digest(manifest), comparison=manifest['comparison'],
        split=manifest['split'], base_questions=len(tasks), independent_seeds=len(seeds), template_families=len({row['family'] for row in tasks}),
        methods=summaries, paired=pairs, inputs=hashes, missing_metrics=missing, protocol=manifest['protocol'],
        protocol_scope='declared_matching_protocol_and_verified_runtime_link_not_independent_execution_attestation',
        evidence_complete=False, SOTA_verified=False, warning='Rankings do not prove conditional causality; costs exclude unrecorded setup and training overhead.')
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = assess(read(args.manifest), args.manifest.resolve().parent)
    if args.output:
        write_new(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))
if __name__ == '__main__':
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
