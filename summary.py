# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import statistics
from control.contracts import publish, relative, strict_json
from judge import digest
from study import SEEDS

def effects(conditions, cells, weights):
    # Form each matched-seed contrast before aggregating training runs.
    benchmarks = {}
    for benchmark in conditions[cells[-1]]['benchmarks']:
        estimates = {}
        for name, coefficients in weights.items():
            scores = [sum(weight*conditions[cell]['benchmarks'][benchmark]['per_seed'][str(seed)]
                for cell, weight in zip(cells, coefficients)) for seed in SEEDS]
            estimates[name] = dict(mean=statistics.mean(scores), sample_std=statistics.stdev(scores),
                per_seed=dict(zip(map(str, SEEDS), scores)))
        benchmarks[benchmark] = estimates
    return benchmarks

def contrasts(conditions):
    cells = ('pooled', 'global', 'fixed', 'evocond')
    weights = dict(condition_at_fixed=(-1, 0, 1, 0), condition_at_learned=(0, -1, 0, 1),
        adoption_at_pooled=(-1, 1, 0, 0), adoption_at_conditional=(0, 0, -1, 1),
        condition_average=(-.5, -.5, .5, .5), adoption_average=(-.5, .5, -.5, .5),
        interaction=(1, -1, -1, 1))
    return dict(metric='pass_at_one_accuracy_difference', unit='fraction', cells=list(cells),
        seed_unit='independent_training_run', comparison='independently_trained_conditions',
        coefficients={name: dict(zip(cells, values)) for name, values in weights.items()},
        benchmarks=effects(conditions, cells, weights))

def systems(conditions):
    cells = ('searl', 'pace', 'evocond')
    weights = dict(total=(-1, 0, 1), cadence=(-1, 1, 0), matched_cadence=(0, -1, 1))
    return dict(metric='pass_at_one_accuracy_difference', unit='fraction', cells=list(cells),
        seed_unit='independent_training_run', comparison='independently_trained_systems',
        coefficients={name: dict(zip(cells, values)) for name, values in weights.items()},
        benchmarks=effects(conditions, cells, weights))

def aggregate(path):
    path = Path(path).resolve()
    manifest_bytes = path.read_bytes()
    value = strict_json(manifest_bytes.decode('utf-8-sig'))
    fields = {'format', 'benchmarks', 'conditions', 'seeds', 'reports'}
    if not isinstance(value, dict) or set(value) != fields or value['format'] != 'evocond_results_v1':
        raise ValueError('Expected a complete four-seed result manifest')
    if not isinstance(value['seeds'], list) or any(type(seed) is not int for seed in value['seeds']) or sorted(value['seeds']) != sorted(SEEDS):
        raise ValueError('Report all four declared training seeds')
    for key in ('benchmarks', 'conditions'):
        items = value[key]
        if not isinstance(items, list) or not items or any(not isinstance(item, str) or not item for item in items) or len(set(items)) != len(items):
            raise ValueError('Declare unique benchmarks and conditions')
    expected = {(name, seed) for name in value['benchmarks'] for seed in SEEDS}
    if not isinstance(value['reports'], list):
        raise ValueError('Result reports must be an explicit list')
    reports, identities, judge_identity = {}, {}, None
    seen_reports = set()
    for item in value['reports']:
        if not isinstance(item, dict) or set(item) != {'benchmark', 'seed', 'path', 'sha256'}:
            raise ValueError('Every result needs benchmark, seed, path and SHA256')
        if not isinstance(item['benchmark'], str) or type(item['seed']) is not int or item['seed'] not in SEEDS:
            raise ValueError('Result benchmark and training seed must have declared identities')
        key = (item['benchmark'], item['seed'])
        if key not in expected or key in reports:
            raise ValueError('Unknown, duplicate or absolute result entry')
        report_path = relative(path.parent, item['path'])
        payload = report_path.read_bytes()
        if sha256(payload).hexdigest() != item['sha256'] or item['sha256'] in seen_reports:
            raise ValueError('Result checksum mismatch')
        seen_reports.add(item['sha256'])
        report = strict_json(payload.decode('utf-8-sig'), max_text=128_000_000)
        protocol_path = relative(path.parent, (report_path.parent/'judge_protocol.json').relative_to(path.parent).as_posix())
        protocol = strict_json(protocol_path.read_text(encoding='utf-8-sig'), max_text=128_000_000)
        if not isinstance(report, dict) or not isinstance(protocol, dict) or report.get('complete') is not True or digest(protocol) != report.get('protocol_sha256'):
            raise ValueError('Incomplete result or unmatched judge protocol')
        if protocol.get('split') != 'test' or type(protocol.get('samples_per_question')) is not int or protocol['samples_per_question'] != 1 or protocol.get('dataset_name') != key[0] or type(protocol.get('training_seed')) is not int or protocol['training_seed'] != key[1]:
            raise ValueError('Aggregate only the declared test benchmark with one answer per question')
        if not isinstance(protocol.get('arms'), dict) or set(protocol['arms']) != set(value['conditions']):
            raise ValueError('Judge protocol and result condition labels differ')
        dataset = protocol.get('dataset')
        if (not isinstance(dataset, dict) or not isinstance(dataset.get('sha256'), str)
                or not re.fullmatch(r'[a-f0-9]{64}', dataset['sha256'])
                or dataset.get('format') not in ('jsonl', 'parquet')
                or any(not isinstance(dataset.get(name), str) or not dataset[name].startswith('/')
                    or re.search(r'~(?![01])', dataset[name]) for name in ('question_pointer', 'answer_pointer'))):
            raise ValueError('Dataset protocol needs its content SHA256, format and explicit field mappings')
        generation = protocol.get('generation')
        if (not isinstance(protocol.get('judge_model'), dict) or not protocol['judge_model']
                or not isinstance(protocol.get('parser'), str) or not protocol['parser'].strip()
                or not isinstance(generation, dict) or type(generation.get('temperature')) not in (int, float)
                or generation['temperature'] != 0):
            raise ValueError('Declare the judge identity, parser and deterministic generation settings')
        current_judge = digest({name: protocol.get(name) for name in ('judge_model', 'parser', 'generation',
            'author_source_sha256', 'wrapper_sha256', 'metric', 'deviations')})
        if judge_identity is not None and judge_identity != current_judge:
            raise ValueError('Judge model or evaluation settings differ between reports')
        judge_identity = current_judge
        if type(protocol.get('reference_count')) is not int or protocol['reference_count'] < 1:
            raise ValueError('Reference denominators must be positive integers')
        dataset = (digest({name: dataset[name] for name in
            ('sha256', 'format', 'question_pointer', 'answer_pointer')}), protocol['reference_count'])
        if key[0] in identities and identities[key[0]] != dataset:
            raise ValueError('Test inputs, field mappings or denominators differ across seeds')
        identities[key[0]] = dataset
        if not isinstance(report.get('arms'), dict) or set(report['arms']) != set(value['conditions']):
            raise ValueError('Result conditions do not match the declared comparison')
        for row in report['arms'].values():
            if not isinstance(row, dict) or type(row.get('n')) is not int or row['n'] <= 0 or row['n'] != protocol['reference_count'] or type(row.get('correct')) is not int or not 0 <= row['correct'] <= row['n']:
                raise ValueError('Invalid complete-test denominator or correctness count')
            if type(row.get('acc')) not in (int, float) or not 0 <= row['acc'] <= 1 or not math.isfinite(row['acc']) or not math.isclose(row['acc'], row['correct']/row['n'], rel_tol=0., abs_tol=1e-12):
                raise ValueError('Reported accuracy differs from its counts')
        reports[key] = report['arms']
    if set(reports) != expected:
        raise ValueError('Missing benchmark/seed reports; no partial-seed averages')
    output = {}
    for condition in value['conditions']:
        benchmarks = {}
        for benchmark in value['benchmarks']:
            scores = [reports[benchmark, seed][condition]['correct']/reports[benchmark, seed][condition]['n'] for seed in SEEDS]
            benchmarks[benchmark] = dict(mean=statistics.mean(scores), sample_std=statistics.stdev(scores),
                per_seed=dict(zip(map(str, SEEDS), scores)), questions=identities[benchmark][1])
        macro = [statistics.mean(benchmarks[name]['per_seed'][str(seed)] for name in value['benchmarks']) for seed in SEEDS]
        weighted = [sum(reports[name, seed][condition]['correct'] for name in value['benchmarks'])/
            sum(reports[name, seed][condition]['n'] for name in value['benchmarks']) for seed in SEEDS]
        output[condition] = dict(benchmarks=benchmarks,
            macro=dict(mean=statistics.mean(macro), sample_std=statistics.stdev(macro)),
            question_weighted=dict(mean=statistics.mean(weighted), sample_std=statistics.stdev(weighted)))
    result = dict(format='evocond_results_summary_v1', seeds=list(SEEDS), seed_unit='independent_training_run',
        question_unit='scheduled_base_question', source_sha256=sha256(manifest_bytes).hexdigest(), conditions=output)
    # Emit crossed effects only when all four registered cells are present.
    if {'pooled', 'global', 'fixed', 'evocond'} <= set(output):
        result['conditional_adoption'] = contrasts(output)
    if {'searl', 'pace', 'evocond'} <= set(output):
        result['system_comparison'] = systems(output)
    return result

def main():
    parser = argparse.ArgumentParser(description='Summarize complete four-seed test reports.')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = aggregate(args.manifest)
    if args.output:
        publish(args.output, json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, allow_nan=False))

if __name__ == '__main__':
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
