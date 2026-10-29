# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Measure marginal memory value without changing the adoption policy.'
from collections import Counter
from hashlib import sha256
import math
import os
from pathlib import Path
import re
import statistics
from control import bandit, dynamic
from control.contracts import relative, strict_json

FORMAT = 'evocond_memory_value_v1'
PANEL = 'evocond_fatigue_panel_v1'
CELL = 'evocond_fatigue_cell_v1'
TARGET = 'uniform_distinct_nonsemantic_actual_classes_vs_semantic'

def moments(values):
    values = list(values)
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError('Memory value observations must be finite numbers')
    count = len(values)
    deviation = statistics.stdev(values) if count > 1 else None
    result = dict(n=count, mean=statistics.mean(values) if count else None, sample_std=deviation,
        standard_error=deviation/math.sqrt(count) if deviation is not None else None)
    if any(value is not None and not math.isfinite(value) for value in result.values()):
        raise ValueError('Memory value moments overflowed')
    return result

def inventory(registry):
    return dict(capabilities=len(registry.get('capabilities', {}).get('nodes', {})),
        versions=len(registry['semantic']), relations=len(registry.get('relations', {}).get('edges', {})),
        retained_edge_events=sum(len(edge['events']) for edge in registry.get('relations', {}).get('edges', {}).values()))

def support(result):
    edges = {}
    for row in result['candidates']:
        for edge in row['edges']:
            key = (edge['source'], edge['target'])
            if key in edges and any(edges[key][field] != edge[field] for field in ('mass', 'weight')):
                raise ValueError('Shared relation receipts disagree within one query')
            edges[key] = edge
    return dict(queried_edges=len(edges), conditional_mass=sum(edge['mass'] for edge in edges.values()),
        query_effective_count=result['abstention']['effective_count'],
        positive_edges=sum(edge['weight'] > 0 for edge in edges.values()),
        negative_edges=sum(edge['weight'] < 0 for edge in edges.values()),
        mean_absolute_weight=statistics.mean(abs(edge['weight']) for edge in edges.values()) if edges else 0.0)

def window(records):
    records = list(records)
    if records:
        for row in records:
            if not isinstance(row, dict):
                raise ValueError('Memory value records must be observation objects')
            bandit.validate(row)
        bandit.frozen_window(records)
        ids = [row['base_id'] for row in records]
        if any(not isinstance(base, str) or not base for base in ids) or len(set(ids)) != len(ids):
            raise ValueError('Memory value needs one observation per independent base question')
    success, cost, utility, weights, propensities = [], [], [], [], []
    for row in records:
        # A fixed midpoint centers the estimator without fitting on these outcomes.
        coefficient = 0.0
        if row['interventions'] == 1:
            if len(row['actions']) < 2:
                raise ValueError('Randomized memory comparisons need distinct actual lists')
            index = row['actions'].index(row['chosen'])
            target = -1.0 if row['chosen'] == 0 else 1.0/(len(row['actions'])-1)
            coefficient = target/row['probabilities'][index]
            propensities.extend(row['probabilities'])
        clipped = min(sum(row['costs'].values())/row['reference'], 1.0)
        success.append(coefficient*(float(row['success'])-.5))
        cost.append(coefficient*(clipped-.5))
        utility.append(success[-1]-row['penalty']*cost[-1])
        weights.append(abs(coefficient))
    supported = [row['relation_support'] for row in records if row['interventions'] == 1 and 'relation_support' in row]
    return dict(format=FORMAT, target=TARGET, estimator='fixed_midpoint_known_propensity_ips',
        denominator='all_preselected_base_questions_in_this_frozen_window',
        scope='single_randomized_point_under_logged_continuation_not_full_policy_gain',
        terminal_success_lift=moments(success), clipped_cost_lift=moments(cost), net_utility_lift=moments(utility),
        observed_marks=sum(row['interventions'] for row in records),
        statuses=dict(Counter(row.get('status', 'observed') for row in records)),
        minimum_propensity=min(propensities) if propensities else None, largest_absolute_weight=max(weights, default=0.0),
        actual_mean_cost=moments(sum(row['costs'].values()) for row in records),
        actual_success=moments(float(row['success']) for row in records), actual_utility=moments(row['reward'] for row in records),
        support_at_randomized_points={key: moments(row[key] for row in supported)
            for key in ('queried_edges', 'conditional_mass', 'query_effective_count', 'positive_edges', 'negative_edges', 'mean_absolute_weight')},
        binding={key: records[0].get(key) if records else None for key in ('snapshot', 'actor', 'bank', 'bottom')},
        cross_window_causal_attribution=False, deployment_decision=False)

def trace(root):
    root = Path(root)
    observations = root/'interventions.jsonl'
    if not observations.is_file():
        raise ValueError('The run has no randomized intervention log')
    metadata, reports = {}, {}
    for path in (root/'phases').glob('phase*.json'):
        payload = path.read_bytes()
        report = strict_json(payload.decode('utf-8-sig'), max_text=128_000_000)
        if not isinstance(report, dict):
            raise ValueError('Completed phases must be identified report objects')
        phase = report.get('phase')
        if type(phase) is not int or phase < 0 or phase in metadata or not isinstance(report.get('snapshot'), str) or not report['snapshot']:
            raise ValueError('Duplicate or unidentified completed phase')
        if path.name != f'phase{phase}.json' or type(report.get('base_observations')) is not int or report['base_observations'] < 0:
            raise ValueError('Phase filename or observation denominator differs')
        metadata[phase] = (report, sha256(payload).hexdigest())
    if not metadata:
        raise ValueError('No completed phase reports are available')
    groups = {phase: [] for phase in metadata}
    def complete(phase):
        report, identity = metadata[phase]
        reports[phase] = dict(phase=phase, snapshot=report['snapshot'], memory=report.get('memory_sizes'),
            phase_report_sha256=identity, value=window(groups[phase]))
        groups[phase] = None
    for phase, (report, _) in metadata.items():
        if report['base_observations'] == 0:
            complete(phase)
    digest = sha256()
    incomplete, trailing = 0, 0
    # Read a fixed byte prefix; later appends belong to the next inspection.
    with observations.open('rb') as stream:
        captured = remaining = os.fstat(stream.fileno()).st_size
        first = True
        while remaining:
            raw = stream.readline(min(remaining, 1_000_001))
            if not raw:
                raise ValueError('The intervention log was truncated during inspection')
            remaining -= len(raw)
            digest.update(raw)
            if len(raw) > 1_000_000:
                raise ValueError('An intervention record exceeds the supported line size')
            encoding = 'utf-8-sig' if first else 'utf-8'
            first = False
            try:
                text = raw.decode(encoding)
                if not text.strip():
                    continue
                row = strict_json(text)
            except ValueError:
                if not remaining and not raw.endswith(b'\n'):
                    trailing = len(raw)
                    break
                raise
            if not isinstance(row, dict) or type(row.get('phase')) is not int or row['phase'] < 0:
                raise ValueError('Intervention records need a nonnegative phase')
            phase = row['phase']
            if phase not in metadata:
                incomplete += 1
                continue
            report = metadata[phase][0]
            if row.get('snapshot') != report['snapshot']:
                raise ValueError('Phase report and randomized observations differ')
            if groups[phase] is None:
                raise ValueError('Completed phase has extra randomized observations')
            groups[phase].append(row)
            if len(groups[phase]) == report['base_observations']:
                complete(phase)
    if any(rows is not None for rows in groups.values()):
        raise ValueError('Completed phase has missing randomized observations')
    return dict(format='evocond_memory_value_trace_v1', phases=[reports[key] for key in sorted(reports)],
        intervention_sha256=digest.hexdigest(), intervention_bytes=captured,
        incomplete_observations=incomplete, incomplete_tail_bytes=trailing,
        scope='training_stream_diagnostic_compare_checkpoints_with_a_matched_panel', deployment_decision=False)

def fingerprint(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Expected an immutable SHA256 identity')
    return value

def outcome(row, protocol):
    if not isinstance(row, dict) or set(row) != {'success', 'costs'} or type(row['success']) is not bool:
        raise ValueError('Each outcome needs explicit correctness and all measured costs')
    if not isinstance(row['costs'], dict) or set(row['costs']) != {'actor', 'tools', 'retrieval'}:
        raise ValueError('Panel costs must include actor, tools and retrieval')
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in row['costs'].values()):
        raise ValueError('Panel costs must be finite nonnegative numbers')
    total = sum(row['costs'].values())
    if not math.isfinite(total):
        raise ValueError('Panel cost total overflowed')
    utility = bandit.reward(row['success'], row['costs'], penalty=protocol['penalty'], reference=protocol['reference'])
    return dict(success=float(row['success']), cost=total, clipped_cost=min(total/protocol['reference'], 1.0), utility=utility)

def success_pairs(rows, alternative='structural', reference='semantic'):
    pairs = [(row[reference]['success'], row[alternative]['success']) for row in rows]
    if any(type(value) not in (bool, int, float) or value not in (0, 1) for pair in pairs for value in pair):
        raise ValueError('Paired success transitions require binary observed outcomes')
    matrix, n = Counter(pairs), len(pairs)
    counts = dict(both_failed=matrix[0, 0], rescued=matrix[0, 1], harmed=matrix[1, 0], both_succeeded=matrix[1, 1])
    failed = counts['both_failed']+counts['rescued']
    succeeded = counts['harmed']+counts['both_succeeded']
    return dict(n=n, reference=reference, alternative=alternative, counts=counts,
        reference_failures=failed, reference_successes=succeeded,
        rates=dict(rescue_share=counts['rescued']/n if n else None, harm_share=counts['harmed']/n if n else None,
            success_lift=(counts['rescued']-counts['harmed'])/n if n else None,
            reference_failure_rate=failed/n if n else None, reference_success_rate=succeeded/n if n else None,
            rescue_rate=counts['rescued']/failed if failed else None,
            harm_rate=counts['harmed']/succeeded if succeeded else None),
        scope='paired_observed_success_transitions')

def components(rows, penalty, alternative='structural', reference='semantic'):
    # Account for rescues, harms and signed cost on the same task denominator.
    if type(penalty) not in (int, float) or not math.isfinite(penalty) or penalty < 0:
        raise ValueError('Use the frozen nonnegative utility penalty')
    rescued, harmed, costs, utility = [], [], [], []
    for row in rows:
        left, right = row[reference], row[alternative]
        difference = right['success']-left['success']
        rescued.append(max(difference, 0.))
        harmed.append(max(-difference, 0.))
        costs.append(penalty*(right['clipped_cost']-left['clipped_cost']))
        utility.append(right['utility']-left['utility'])
    if any(not math.isclose(gain-loss-cost, value, rel_tol=1e-10, abs_tol=1e-12)
            for gain, loss, cost, value in zip(rescued, harmed, costs, utility)):
        raise ValueError('Paired utility differs from its success and cost components')
    return dict(rescue_share=moments(rescued), harm_share=moments(harmed),
        cost_penalty=moments(costs), net_utility_lift=moments(utility))

def summarize_pairs(entries):
    result = dict(per_seed={str(seed): row for seed, row in entries}, rates={}, defined_seeds={})
    for name in entries[0][1]['rates']:
        values = [(seed, row['rates'][name]) for seed, row in entries if row['rates'][name] is not None]
        result['defined_seeds'][name] = [seed for seed, _ in values]
        metric = moments(value for _, value in values)
        # A subset of defined conditional rates is not a complete seed average.
        if len(values) != len(entries):
            metric.update(mean=None, sample_std=None, standard_error=None)
        result['rates'][name] = metric
    return result

def panel(path):
    path = Path(path).resolve()
    manifest_bytes = path.read_bytes()
    manifest = strict_json(manifest_bytes.decode('utf-8-sig'))
    fields = {'format', 'seed', 'actors', 'banks', 'protocol', 'cells'}
    if not isinstance(manifest, dict) or set(manifest) != fields or manifest['format'] != PANEL:
        raise ValueError('Expected a declared actor-by-memory fatigue panel')
    if type(manifest['seed']) is not int or manifest['seed'] not in (42, 52, 3047, 3407):
        raise ValueError('Use one declared training seed per panel')
    for name in ('actors', 'banks'):
        values = manifest[name]
        if not isinstance(values, list) or len(values) != 2 or len(set(map(fingerprint, values))) != 2:
            raise ValueError('Declare distinct old and new actor and memory identities')
    protocol = manifest['protocol']
    fields = {'split', 'dataset_sha256', 'base_ids_sha256', 'questions', 'capability_pool_sha256',
        'execution_sha256', 'presentation', 'penalty', 'reference'}
    if not isinstance(protocol, dict) or set(protocol) != fields or protocol['split'] not in ('validation', 'test') or protocol['presentation'] != 'fixed_structure':
        raise ValueError('Use held-out tasks and one fixed structural presentation rule')
    for name in ('dataset_sha256', 'base_ids_sha256', 'capability_pool_sha256', 'execution_sha256'):
        fingerprint(protocol[name])
    if type(protocol['questions']) is not int or protocol['questions'] < 1:
        raise ValueError('Declare the complete scheduled probe denominator')
    for name in ('penalty', 'reference'):
        if type(protocol[name]) not in (int, float) or not math.isfinite(protocol[name]) or protocol[name] < 0 or name == 'reference' and protocol[name] == 0:
            raise ValueError('Declare a common finite utility and cost scale')
    if not isinstance(manifest['cells'], list) or len(manifest['cells']) != 4:
        raise ValueError('All four actor-by-memory cells are required')
    expected = {(actor, bank) for actor in manifest['actors'] for bank in manifest['banks']}
    cells, sources, ids, arms = {}, {}, None, None
    for entry in manifest['cells']:
        if not isinstance(entry, dict) or set(entry) != {'actor', 'bank', 'path', 'sha256'}:
            raise ValueError('Each cell needs its identities, relative path and checksum')
        fingerprint(entry['actor'])
        fingerprint(entry['bank'])
        key = (entry['actor'], entry['bank'])
        name = entry['path']
        if key not in expected or key in cells:
            raise ValueError('Unknown, duplicate or nonportable panel cell')
        payload = relative(path.parent, name).read_bytes()
        if sha256(payload).hexdigest() != fingerprint(entry['sha256']):
            raise ValueError('Panel cell checksum mismatch')
        cell = strict_json(payload.decode('utf-8-sig'), max_text=128_000_000)
        fields = {'format', 'actor', 'bank', 'seed', 'protocol_sha256', 'records'}
        if not isinstance(cell, dict) or set(cell) not in (fields, fields | {'teacher_sha256'}) or cell['format'] != CELL or (cell['actor'], cell['bank']) != key or type(cell['seed']) is not int or cell['seed'] != manifest['seed'] or cell['protocol_sha256'] != dynamic.digest(protocol):
            raise ValueError('Cell provenance differs from the frozen panel')
        if not isinstance(cell['records'], list) or not cell['records']:
            raise ValueError('Panel cells need complete paired task outcomes')
        indexed = {}
        for row in cell['records']:
            if not isinstance(row, dict) or set(row) not in ({'base_id', 'semantic', 'structural'}, {'base_id', 'semantic', 'structural', 'adopted'}):
                raise ValueError('Keep semantic and structural outcomes paired by base question')
            base = row['base_id']
            if not isinstance(base, str) or not base or base in indexed:
                raise ValueError('Panel task identities must be nonempty and unique')
            current_arms = set(row)-{'base_id'}
            if arms is not None and arms != current_arms:
                raise ValueError('Optional adopted outcomes must cover the whole panel')
            arms = current_arms
            indexed[base] = {arm: outcome(row[arm], protocol) for arm in current_arms}
        if ('adopted' in arms) != ('teacher_sha256' in cell):
            raise ValueError('Adopted outcomes require their frozen teacher identity')
        if 'adopted' in arms:
            fingerprint(cell['teacher_sha256'])
        if ids is not None and ids != set(indexed):
            raise ValueError('All cells must cover exactly the same scheduled base questions')
        if len(indexed) != protocol['questions'] or dynamic.digest(sorted(indexed)) != protocol['base_ids_sha256']:
            raise ValueError('Cell tasks differ from the predeclared probe reference')
        ids = set(indexed)
        cells[key], sources[key] = indexed, dict(sha256=entry['sha256'], teacher_sha256=cell.get('teacher_sha256'))
    if set(cells) != expected:
        raise ValueError('Missing actor-by-memory cell')
    order = sorted(ids)
    gains, summaries = {}, {}
    for i, actor in enumerate(manifest['actors']):
        for j, bank in enumerate(manifest['banks']):
            rows = cells[actor, bank]
            gains[i, j] = [rows[base]['structural']['utility']-rows[base]['semantic']['utility'] for base in order]
            summary = dict(actor=actor, bank=bank, **sources[actor, bank], net_utility_lift=moments(gains[i, j]),
                success_lift=moments(rows[base]['structural']['success']-rows[base]['semantic']['success'] for base in order),
                clipped_cost_lift=moments(rows[base]['structural']['clipped_cost']-rows[base]['semantic']['clipped_cost'] for base in order),
                cost_lift=moments(rows[base]['structural']['cost']-rows[base]['semantic']['cost'] for base in order))
            summary['outcomes'] = {arm: {metric: moments(rows[base][arm][metric] for base in order)
                for metric in ('success', 'cost', 'clipped_cost', 'utility')} for arm in sorted(arms)}
            summary['paired_success'] = dict(structural_vs_semantic=success_pairs(rows.values()))
            summary['benefit_components'] = dict(structural_vs_semantic=components(rows.values(), protocol['penalty']))
            if 'adopted' in arms:
                summary['adopted_vs_semantic'] = moments(rows[base]['adopted']['utility']-rows[base]['semantic']['utility'] for base in order)
                summary['adopted_vs_structural'] = moments(rows[base]['adopted']['utility']-rows[base]['structural']['utility'] for base in order)
                summary['paired_success']['adopted_vs_semantic'] = success_pairs(rows.values(), 'adopted')
                summary['paired_success']['adopted_vs_structural'] = success_pairs(rows.values(), 'adopted', 'structural')
                summary['benefit_components']['adopted_vs_semantic'] = components(rows.values(), protocol['penalty'], 'adopted')
                summary['benefit_components']['adopted_vs_structural'] = components(rows.values(), protocol['penalty'], 'adopted', 'structural')
            summaries[f'actor{i}_bank{j}'] = summary
    # Pair each task before averaging the factorial contrasts.
    first, bank, actor, last = gains[0, 0], gains[0, 1], gains[1, 0], gains[1, 1]
    terms = dict(actor_at_old_bank=[a-b for a, b in zip(actor, first)],
        bank_at_old_actor=[a-b for a, b in zip(bank, first)],
        interaction=[d-c-b+a for a, b, c, d in zip(first, bank, actor, last)],
        total_change=[d-a for a, d in zip(first, last)])
    return dict(format='evocond_fatigue_panel_report_v1', seed=manifest['seed'], protocol=protocol,
        task_count=len(order), paired_ids_sha256=dynamic.digest(order), cells=summaries,
        decomposition={name: moments(values) for name, values in terms.items()},
        source_sha256=sha256(manifest_bytes).hexdigest(), inference_unit='paired_base_question_within_one_training_seed',
        scope='fixed_structure_marginal_value_not_raw_accuracy_or_automatic_mechanism_proof', deployment_decision=False)

def aggregate(paths):
    reports = [panel(path) for path in paths]
    if len(reports) != 4 or sorted(row['seed'] for row in reports) != [42, 52, 3047, 3407]:
        raise ValueError('Aggregate all four independent declared training seeds')
    for field in ('protocol', 'task_count', 'paired_ids_sha256'):
        if len({dynamic.digest(row[field]) for row in reports}) != 1:
            raise ValueError('Four-seed panels need matching tasks and evaluation settings')
    result = dict(format='evocond_fatigue_summary_v1', seeds=[42, 52, 3047, 3407],
        protocol=reports[0]['protocol'], task_count=reports[0]['task_count'], seed_unit='independent_training_run',
        sources={str(row['seed']): row['source_sha256'] for row in reports})
    result['decomposition'] = {name: moments(row['decomposition'][name]['mean'] for row in reports)
        for name in reports[0]['decomposition']}
    optional = {'adopted_vs_semantic', 'adopted_vs_structural'}
    fields = {'net_utility_lift', 'success_lift', 'cost_lift', 'clipped_cost_lift'}
    for row in reports:
        for cell in row['cells'].values():
            if {key for key in cell if key in optional} != {key for key in reports[0]['cells']['actor0_bank0'] if key in optional}:
                raise ValueError('Adoption recovery must have the same coverage across seeds')
    fields |= optional & set(reports[0]['cells']['actor0_bank0'])
    result['cells'] = {cell: {name: moments(row['cells'][cell][name]['mean'] for row in reports)
        for name in sorted(fields)} for cell in reports[0]['cells']}
    for name, summary in result['cells'].items():
        summary['outcomes'] = {arm: {metric: moments(row['cells'][name]['outcomes'][arm][metric]['mean'] for row in reports)
            for metric in ('success', 'cost', 'clipped_cost', 'utility')} for arm in reports[0]['cells'][name]['outcomes']}
        summary['paired_success'] = {comparison: summarize_pairs(sorted(
            (row['seed'], row['cells'][name]['paired_success'][comparison]) for row in reports))
            for comparison in reports[0]['cells'][name]['paired_success']}
        summary['benefit_components'] = {comparison: {
            metric: dict(**moments(row['cells'][name]['benefit_components'][comparison][metric]['mean'] for row in reports),
                per_seed={str(row['seed']): row['cells'][name]['benefit_components'][comparison][metric]['mean'] for row in reports})
            for metric in values} for comparison, values in reports[0]['cells'][name]['benefit_components'].items()}
    return result

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
