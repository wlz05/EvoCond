# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Semantic recall with identity-only alignment and signed directional residuals.'
from __future__ import annotations
from copy import deepcopy
import math
from control import bandit, identity, relations
from control.context import embedding, features
VARIANTS = ('semantic', 'unconditional', 'conditional', 'aligned', 'full', 'erased', 'whole', 'simple')
def dot(left, right):
    return sum(a*b for a, b in zip(left, right))
def rank(registry, encoder, alignment, context, *, completed=(), recall=12, top_k=3, neighbors=8, variant='full', metric=None, identity_shrinkage=2.0, whole_alignment=None):
    if variant not in VARIANTS or type(recall) is not int or type(top_k) is not int or type(neighbors) is not int or not 1 <= top_k < recall or neighbors < 1:
        raise ValueError('Invalid bounded retrieval recipe')
    if not math.isfinite(identity_shrinkage) or identity_shrinkage <= 0:
        raise ValueError('Identity shrinkage must be finite and positive')
    if variant == 'whole':
        from control import whole
        if whole_alignment is None:
            raise ValueError('Whole-structure ablation requires its declared trained projection path')
        whole.validate(whole_alignment, alignment['dimension'])
    store = identity.catalogue(registry)
    active = sorted(set(registry['aliases'].values()))
    query = embedding(encoder, context['goal'])
    phi = features(context, encoder)
    metric_context = None
    if metric is not None and variant not in ('semantic', 'unconditional', 'erased'):
        from control import metric as geometry
        metric_context = geometry.prepare(metric, geometry.describe(phi, context))
    candidates = []
    for version in active:
        spec = registry['semantic'][version]
        semantic = dot(query, embedding(encoder, spec['description'], query=False))
        candidates.append(dict(version=version, canonical=store['versions'][version], name=spec['name'], semantic=semantic))
    candidates.sort(key=lambda row: (-row['semantic'], row['canonical'], row['version']))
    unique = {}
    for row in candidates:
        unique.setdefault(row['canonical'], row)
    candidates = list(unique.values())[:recall]
    memory = registry.get('relations', relations.new())
    vectors = {}
    aligned = variant in ('aligned', 'full', 'whole')
    def node(canonical):
        if canonical not in vectors:
            vectors[canonical] = identity.node_vector(registry, alignment, encoder, canonical) if aligned else embedding(encoder, store['nodes'][canonical]['description'], query=False)
        return vectors[canonical]
    projected = identity.project(alignment, query, semantic=True) if aligned else query
    if variant == 'whole':
        projected = whole.project(whole_alignment, query, semantic=True)
    remaining = (context.get('progress') or {}).get('remaining_goal', '')
    qrem = embedding(encoder, remaining) if remaining else [0.0]*len(query)
    qrem = identity.project(alignment, qrem, semantic=True) if aligned and remaining else qrem
    active_by_id = {}
    for version in active:
        active_by_id.setdefault(store['versions'][version], []).append(version)
    def eligible(version, canonical):
        return any(identity.eligible(registry, version, current) for current in active_by_id.get(canonical, ()))
    total_mass = total_effective = 0.0
    evidence = []
    for row in candidates:
        canonical = row['canonical']
        if variant == 'semantic':
            row.update(structure=row['semantic'], incoming=[0.0]*len(query), outgoing=[0.0]*len(query), edges=[])
            continue
        recent = [store['versions'][version] for version in reversed(completed) if version in store['versions']]
        sources = list(dict.fromkeys(recent))[:neighbors]
        targets = relations.neighbors(memory, canonical, limit=neighbors)
        receipts = []
        def residual(nodes, incoming):
            signed, mass = [0.0]*len(query), 0.0
            for other in nodes:
                if other not in store['nodes']:
                    continue
                receipt = relations.estimate(memory, other if incoming else canonical, canonical if incoming else other,
                    phi, context, versions=eligible, unconditional=variant == 'unconditional', erase=variant == 'erased', metric=metric, metric_context=metric_context)
                weight = receipt['weight']
                receipts.append(dict(source=other if incoming else canonical, target=canonical if incoming else other, **receipt))
                signed = [a+weight*b for a, b in zip(signed, node(other))]
                mass += abs(weight)
            return [value/(1+mass) for value in signed]
        incoming, outgoing = residual(sources, True), residual(targets, False)
        base = node(canonical)
        if variant == 'whole':
            base = whole.project(whole_alignment, [*base, *incoming, *outgoing])
        count = len(set(store.get('usage', {}).get(canonical, [])))
        zeta = count/(count+identity_shrinkage)
        structural = (dot(projected, base)+math.tanh(dot(projected, incoming)+dot(qrem, outgoing)))/2 if any(base) else row['semantic']
        if variant == 'whole':
            structural = dot(projected, base) if any(base) else row['semantic']
        structure = (1-zeta)*row['semantic']+zeta*structural
        row.update(structure=structure, incoming=incoming, outgoing=outgoing, edges=receipts, verified_uses=count, identity_support=zeta)
        evidence.extend(receipts)
        total_mass += sum(receipt['mass'] for receipt in receipts)
        total_effective += sum(receipt['effective_count'] for receipt in receipts)
    seen, by_base, observed_distances = set(), {}, []
    for receipt in evidence:
        edge = (receipt['source'], receipt['target'])
        if edge in seen:
            continue
        seen.add(edge)
        for base, weight in receipt['support_by_base'].items():
            by_base[base] = by_base.get(base, 0.0)+weight
        if receipt['minimum_distance'] is not None:
            observed_distances.append(receipt['minimum_distance'])
    largest = max(by_base.values(), default=0.0)
    values = [value/largest for value in by_base.values()] if largest else []
    effective = sum(values)**2/sum(value*value for value in values) if values else 0.0
    guarded = metric is not None and variant not in ('semantic', 'unconditional', 'erased')
    far = bool(observed_distances) and min(observed_distances) > metric['options']['absolute'] if metric is not None else False
    blind = guarded and effective < 1.0 and (far or not observed_distances)
    unsupported = guarded and not largest
    if unsupported:
        total_mass = total_effective = 0.0
        for row in candidates:
            row.update(structure=row['semantic'], incoming=[0.0]*len(query), outgoing=[0.0]*len(query))
    abstention = dict(active=blind, semantic_fallback=unsupported,
        reason='outside_absolute_support' if blind and far else 'no_eligible_history' if blind else 'zero_effective_support' if unsupported else 'supported_or_control',
        effective_count=effective, eligible_edges=len(observed_distances), minimum_distance=min(observed_distances, default=None),
        absolute_distance=metric['options']['absolute'] if metric is not None else None,
        message='Structural memory is in a cold-start blind zone; fully reverting to native semantic reasoning.' if blind else None)
    rankings = {}
    for alpha in bandit.ACTIONS:
        rankings[alpha] = sorted(deepcopy(candidates), key=lambda row: (-(1-alpha)*row['semantic']-alpha*row['structure'], row['canonical'], row['version']))[:top_k]
    baseline = [row['version'] for row in rankings[0.0]]
    disagreement = [sum(row['version'] != baseline[i] for i, row in enumerate(rankings[alpha]))/max(1, len(baseline)) for alpha in bandit.ACTIONS[1:]]
    progress = context.get('progress') or {}
    statistics = [*disagreement, total_mass, total_effective, float(progress.get('completed', 0)),
                  float(progress.get('turn', 0)), float(progress.get('budget') or 0), float(context['available'] is not None)]
    return dict(rankings=rankings, candidates=candidates, features=[1.0, *query, *statistics],
                edge_support=total_mass > 0, context_features=phi, variant=variant, abstention=abstention,
                access=dict(active_versions=len(active), active_identities=len(unique),
                    recalled_versions=[row['version'] for row in candidates], recall_truncated=len(unique) > recall,
                    retention_horizon=memory['horizon'], history_query_limit=memory['limit']),
                bounds=dict(recall=recall, neighbors=neighbors, events=memory['limit'], actions=3))

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
