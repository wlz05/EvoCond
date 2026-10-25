# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from control.contracts import strict_json
from judge import digest
from study import SEEDS

def aggregate(path):
    path = Path(path).resolve()
    value = strict_json(path.read_text(encoding='utf-8'))
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
    reports, identities, judge_identity = {}, {}, None
    seen_reports = set()
    for item in value['reports']:
        if not isinstance(item, dict) or set(item) != {'benchmark', 'seed', 'path', 'sha256'}:
            raise ValueError('Every result needs benchmark, seed, path and SHA256')
        key = (item['benchmark'], item['seed'])
        if key not in expected or key in reports or not isinstance(item['path'], str) or Path(item['path']).is_absolute():
            raise ValueError('Unknown, duplicate or absolute result entry')
        report_path = path.parent/item['path']
        payload = report_path.read_bytes()
        if sha256(payload).hexdigest() != item['sha256'] or item['sha256'] in seen_reports:
            raise ValueError('Result checksum mismatch')
        seen_reports.add(item['sha256'])
        report = strict_json(payload.decode('utf-8'))
        protocol = strict_json((report_path.parent/'judge_protocol.json').read_text(encoding='utf-8'))
        if report.get('complete') is not True or digest(protocol) != report.get('protocol_sha256'):
            raise ValueError('Incomplete result or unmatched judge protocol')
        if protocol.get('split') != 'test' or protocol.get('samples_per_question') != 1 or protocol.get('dataset_name') != key[0] or protocol.get('training_seed') != key[1]:
            raise ValueError('Aggregate only the declared test benchmark with one answer per question')
        if set(protocol.get('arms', {})) != set(value['conditions']):
            raise ValueError('Judge protocol and result condition labels differ')
        current_judge = digest(dict(model=protocol['judge_model'], parser=protocol['parser'], generation=protocol['generation']))
        if judge_identity is not None and judge_identity != current_judge:
            raise ValueError('Judge model or evaluation settings differ between reports')
        judge_identity = current_judge
        dataset = (protocol['dataset']['sha256'], protocol['reference_count'])
        if key[0] in identities and identities[key[0]] != dataset:
            raise ValueError('Test questions or denominators differ across seeds')
        identities[key[0]] = dataset
        if set(report.get('arms', {})) != set(value['conditions']):
            raise ValueError('Result conditions do not match the declared comparison')
        for row in report['arms'].values():
            if type(row.get('n')) is not int or row['n'] <= 0 or row['n'] != protocol['reference_count'] or type(row.get('correct')) is not int or not 0 <= row['correct'] <= row['n']:
                raise ValueError('Invalid complete-test denominator or correctness count')
            if not math.isclose(row.get('acc', -1), row['correct']/row['n'], abs_tol=1e-12):
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
    return dict(format='evocond_results_summary_v1', seeds=list(SEEDS), seed_unit='independent_training_run',
        question_unit='scheduled_base_question', source_sha256=sha256(path.read_bytes()).hexdigest(), conditions=output)

def main():
    parser = argparse.ArgumentParser(description='Summarize complete four-seed test reports.')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = aggregate(args.manifest)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(result, allow_nan=False))

if __name__ == '__main__':
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
