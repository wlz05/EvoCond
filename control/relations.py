# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Bounded conditional signed events, with independent-question clocks.'
from __future__ import annotations
from copy import deepcopy
import math
from control import dynamic
from control.context import compatible
FORMAT = 'evocond_conditional_relations_v1'
def new(*, horizon=128, limit=64, bandwidth=1.0, decay=1.0, shrinkage=1.0):
    if type(horizon) is not int or type(limit) is not int or not 1 <= limit <= horizon:
        raise ValueError('Require 1 <= event query limit <= retention horizon')
    if not all(math.isfinite(x) for x in (bandwidth, decay, shrinkage)) or bandwidth <= 0 or not 0 < decay <= 1 or shrinkage <= 0:
        raise ValueError('Invalid conditional weight hyperparameters')
    return dict(format=FORMAT, horizon=horizon, limit=limit, bandwidth=bandwidth, decay=decay,
                shrinkage=shrinkage, clock=0, tasks={}, edges={}, outgoing={})
def vector(values):
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError('Events require finite visible features')
    return result
def key(source, target):
    if not source or not target:
        raise ValueError('Relation endpoints require canonical identities')
    return dynamic.digest([source, target])
def observe(memory, events, *, base_id, actor, split='train'):
    if memory.get('format') != FORMAT or split != 'train' or not base_id or not actor:
        raise ValueError('Only identified training feedback may update relations')
    result = deepcopy(memory)
    if base_id not in result['tasks']:
        result['clock'] += 1
        result['tasks'][base_id] = result['clock']
    clock = result['tasks'][base_id]
    groups = {}
    for event in events:
        value = float(event['feedback'])
        if not math.isfinite(value) or not -1 <= value <= 1 or event.get('granularity') not in ('terminal', 'subtask'):
            raise ValueError('Feedback needs an explicit bounded correctness granularity')
        if not event.get('source_version') or not event.get('target_version'):
            raise ValueError('Observed versions are required')
        phi = vector(event['features'])
        group = key(event['source'], event['target'])
        if event.get('feedback_kind', 'observed_correctness') != 'observed_correctness':
            raise ValueError('Relation events do not accept adoption advantage as correctness')
        groups.setdefault(group, []).append({**deepcopy(event), 'features': phi, 'feedback_kind': 'observed_correctness'})
    for group, records in groups.items():
        edge = result['edges'].setdefault(group, dict(source=records[0]['source'], target=records[0]['target'], events=[]))
        if base_id in edge.setdefault('seen', [row['base_id'] for row in edge['events']]):
            continue
        edge['seen'].append(base_id)
        dimension = len(records[0]['features'])
        if any(len(row['features']) != dimension for row in records):
            raise ValueError('Context dimension changed within a question')
        # Each question contributes unit mass; repeated contexts are retained as a mixture.
        entry = dict(base_id=base_id, actor=actor, clock=clock, samples=records,
                     feedback=sum(row['feedback'] for row in records)/len(records), aggregation='uniform_context_mixture')
        edge['events'].append(entry)
        edge['events'] = sorted(edge['events'], key=lambda row: (row['clock'], row['base_id']))[-result['horizon']:]
        result['outgoing'].setdefault(edge['source'], {})[edge['target']] = max(row['clock'] for row in edge['events'])
    return result
def estimate(memory, source, target, phi, context, *, versions=None, unconditional=False, erase=False, audit=False, metric=None, metric_context=None):
    if memory.get('format') != FORMAT:
        raise ValueError('Unknown relation store')
    phi = vector(phi)
    edge = memory['edges'].get(key(source, target), {})
    entries = edge.get('events', [])[-memory['limit']:]
    pool, descriptor = [], None
    if metric is not None and not unconditional and not erase:
        from control import metric as geometry
        descriptor = metric_context if metric_context is not None else geometry.prepare(metric, geometry.describe(phi, context))
        pool = [row for row in metric['pool'] if row['edge'] == key(source, target) and
            (versions is None or versions(row['source_version'], source) and versions(row['target_version'], target)) and
            compatible(row.get('requirements'), context)]
        fallback = geometry.identity_fallback(metric, pool)
        local_scale = geometry.scale(metric, descriptor, pool, identity=fallback)
    weights, numerators, distances = [], [], []
    def terms(entry, collect=False):
        mass = signed = 0.0
        for row in entry['samples']:
            if len(row['features']) != len(phi):
                raise ValueError('Frozen encoder/context dimension mismatch')
            if versions is not None and not (versions(row['source_version'], source) and versions(row['target_version'], target)):
                continue
            if not compatible(row.get('requirements'), context):
                continue
            distance = sum((a-b)**2 for a, b in zip(phi, row['features']))
            if unconditional or erase:
                kernel = 1.0
            elif metric is not None:
                if collect and row.get('context'):
                    distances.append(math.sqrt(geometry.distance(metric, descriptor, geometry.describe(row['features'], row['context']), identity=fallback)))
                kernel = geometry.support(metric, descriptor, row, pool, local_scale) if row.get('context') else 0.0
            else:
                kernel = math.exp(-distance/(2*memory['bandwidth']**2))
            weight = kernel*memory['decay']**(memory['clock']-entry['clock'])/len(entry['samples'])
            mass += weight
            signed += weight*(entry['feedback'] if erase else row['feedback'])
        return mass, signed
    for entry in entries:
        weight, numerator = terms(entry, collect=True)
        weights.append(weight)
        numerators.append(numerator)
    mass, signed = sum(weights), sum(numerators)
    largest = max(weights, default=0.0)
    scaled = [weight/largest for weight in weights] if largest else []
    effective = sum(scaled)**2/sum(weight*weight for weight in scaled) if scaled else 0.0
    cutoff = metric['options']['absolute'] if metric is not None else None
    all_far = bool(distances) and all(value > cutoff for value in distances) if cutoff is not None else False
    blind = metric is not None and not unconditional and not erase and effective < 1.0 and (all_far or not distances)
    omitted = edge.get('events', [])[:-memory['limit']]
    missing = sum(terms(entry)[0] for entry in omitted) if audit else None
    return dict(weight=0.0 if blind else signed/(memory['shrinkage']+mass), mass=mass, signed_mass=signed,
                effective_count=effective, independent_events=len(entries),
                support_by_base={entry['base_id']: weight for entry, weight in zip(entries, weights)},
                cold_start_blind=blind, eligible_contexts=len(distances), minimum_distance=min(distances, default=None),
                all_distances_above_absolute=all_far, absolute_distance=cutoff,
                metric_fallback='full_dimension_identity' if descriptor is not None and fallback else None,
                omitted_mass=missing, truncation_bound=missing*(memory['shrinkage']+2*mass)/((memory['shrinkage']+mass+missing)*(memory['shrinkage']+mass)) if audit else None,
                discarded_history_unknown=True, uncertainty_kind='concentration_not_confidence_interval',
                target_kind='observed_relation_support_not_adoption_value',
                kernel_kind='low_rank_local_absolute' if metric is not None else 'legacy_fixed_gaussian')
def neighbors(memory, node, *, limit, incoming=False):
    if not incoming:
        candidates = memory['outgoing'].get(node, {})
        return sorted(candidates, key=lambda target: (-candidates[target], target))[:limit]
    candidates = [edge for edge in memory['edges'].values() if edge['target' if incoming else 'source'] == node]
    candidates.sort(key=lambda edge: (-max((row['clock'] for row in edge['events']), default=0), edge['source'], edge['target']))
    return [edge['source' if incoming else 'target'] for edge in candidates[:limit]]
def remap(memory, mapping):
    result = deepcopy(memory)
    result['edges'] = {}
    result['outgoing'] = {}
    for edge in memory['edges'].values():
        source, target = mapping.get(edge['source'], edge['source']), mapping.get(edge['target'], edge['target'])
        merged = result['edges'].setdefault(key(source, target), dict(source=source, target=target, events=[], seen=[]))
        merged['seen'] = sorted(set(merged['seen']+edge.get('seen', [row['base_id'] for row in edge['events']])))
        by_id = {row['base_id']: row for row in merged['events']}
        for event in deepcopy(edge['events']):
            for sample in event['samples']:
                sample.update(source=source, target=target)
            if event['base_id'] in by_id:
                old = by_id[event['base_id']]
                old['samples'].extend(event['samples'])
                old['feedback'] = sum(row['feedback'] for row in old['samples'])/len(old['samples'])
            else:
                by_id[event['base_id']] = event
        merged['events'] = sorted(by_id.values(), key=lambda row: (row['clock'], row['base_id']))[-result['horizon']:]
        result['outgoing'].setdefault(source, {})[target] = max(row['clock'] for row in merged['events'])
    return result
def reversal():
    histories = [[(0, 1)]*8+[(1, -1)]*2, [(1, 1)]*8+[(0, -1)]*2]
    result = []
    for history in histories:
        mean = sum(y for _, y in history)/len(history)
        conditional = {c: sum(y for x, y in history if x == c)/sum(x == c for x, _ in history) for c in (0, 1)}
        marginal_risk = sum((y-mean)**2 for _, y in history)/len(history)
        conditional_risk = sum((y-conditional[c])**2 for c, y in history)/len(history)
        variance = sum((conditional[c]-mean)**2 for c, _ in history)/len(history)
        result.append(dict(mean=mean, conditional=conditional, marginal_risk=marginal_risk,
                           conditional_risk=conditional_risk, gap=marginal_risk-conditional_risk, variance=variance,
                           interpretation='prediction_only_not_causal_advantage'))
    return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
