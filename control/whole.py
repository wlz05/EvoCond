# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Whole-structure contrastive alignment for the declared alignment ablation.'
from __future__ import annotations
from copy import deepcopy
import numpy as np
from control import context, dynamic, identity
FORMAT = 'evocond_whole_alignment_v1'
def initialize(alignment):
    size = alignment['dimension']
    return dict(format=FORMAT, dimension=size, semantic_map=deepcopy(alignment['semantic_map']),
        whole_map=np.concatenate([np.eye(size)]*3, axis=1).tolist(), fit_ids=[], loss=[], samples=0,
        objective='canonical_whole_structure_includes_conditional_residuals', trained=False)
def validate(artifact, dimension):
    if artifact.get('format') != FORMAT or artifact.get('dimension') != dimension:
        raise ValueError('Whole-alignment dimension or format mismatch')
    for name, shape in (('semantic_map', (dimension, dimension)), ('whole_map', (dimension, 3*dimension))):
        values = np.asarray(artifact[name], dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError('Whole-alignment projection is invalid')
def project(artifact, values, *, semantic=False):
    validate(artifact, artifact['dimension'])
    name = 'semantic_map' if semantic else 'whole_map'
    return context.unit(np.asarray(artifact[name]) @ np.asarray(values), artifact['dimension'], zero=True)
def fit(registry, encoder, alignment, artifact, metric, *, actor, neighbors=8, recall=12, samples_limit=256, steps=20, rate=.02, temperature=.1):
    from control import mixing
    validate(artifact, alignment['dimension'])
    if type(steps) is not int or steps < 1 or not 0 < rate <= 1 or not 0 < temperature <= 1 or not actor or type(samples_limit) is not int or samples_limit < 2 or recall < 2:
        raise ValueError('Whole alignment requires a bounded identified training phase')
    store = identity.catalogue(registry)
    initial = set(alignment['initial_versions'])
    permitted = {key for key, node in store['nodes'].items() if node['initial_version'] in initial}
    samples, bases, visited, contexts = [], set(), set(), []
    for edge in registry.get('relations', {}).get('edges', {}).values():
        for event in edge['events']:
            if event['actor'] != actor:
                continue
            for row in event['samples']:
                view = row.get('context')
                if not view:
                    continue
                signature = dynamic.digest([event['base_id'], view, row['source_version']])
                if signature in visited:
                    continue
                visited.add(signature)
                contexts.append((event['clock'], event['base_id'], signature, row))
    for _, base, _, row in sorted(contexts, key=lambda entry: entry[:3], reverse=True):
        ranked = mixing.rank(registry, encoder, alignment, row['context'], completed=[row['source_version']],
            recall=recall, top_k=1, neighbors=neighbors, variant='aligned', metric=metric)
        for candidate in ranked['candidates']:
            canonical = candidate['canonical']
            if canonical not in permitted:
                continue
            stable = identity.node_vector(registry, alignment, encoder, canonical)
            semantic = context.embedding(encoder, store['nodes'][canonical]['description'], query=False)
            samples.append((canonical, semantic, [*stable, *candidate['incoming'], *candidate['outgoing']]))
            bases.add(base)
            if len(samples) >= samples_limit:
                break
        if len(samples) >= samples_limit:
            break
    result = deepcopy(artifact)
    if len({row[0] for row in samples}) < 2:
        result['last_update'] = 'insufficient_distinct_initial_capabilities'
        return result
    semantic = np.asarray([row[1] for row in samples])
    structure = np.asarray([row[2] for row in samples])
    ps, pw = np.asarray(result['semantic_map']), np.asarray(result['whole_map'])
    positive = np.asarray([[float(left[0] == right[0]) for right in samples] for left in samples])
    targets = positive/positive.sum(axis=1, keepdims=True)
    losses = []
    for _ in range(steps):
        raw_s, raw_h = semantic @ ps.T, structure @ pw.T
        ns, nh = np.linalg.norm(raw_s, axis=1, keepdims=True), np.linalg.norm(raw_h, axis=1, keepdims=True)
        if np.any(ns < 1e-12) or np.any(nh < 1e-12):
            raise ValueError('Whole-structure projection collapsed')
        s, h = raw_s/ns, raw_h/nh
        scores = s @ h.T/temperature
        def softmax(values):
            exp = np.exp(values-values.max(axis=1, keepdims=True))
            return exp/exp.sum(axis=1, keepdims=True)
        forward, reverse = softmax(scores), softmax(scores.T)
        losses.append(float(-np.sum(targets*(np.log(forward+1e-30)+np.log(reverse+1e-30)))/(2*len(samples))))
        gradient = (forward-targets+(reverse-targets).T)/(2*len(samples)*temperature)
        gs, gh = gradient @ h, gradient.T @ s
        gs = (gs-s*np.sum(gs*s, axis=1, keepdims=True))/ns
        gh = (gh-h*np.sum(gh*h, axis=1, keepdims=True))/nh
        ps -= rate*(gs.T @ semantic)
        pw -= rate*(gh.T @ structure)
    result.update(semantic_map=ps.tolist(), whole_map=pw.tolist(), fit_ids=sorted(bases), loss=losses,
        samples=len(samples), samples_limit=samples_limit, trained=True, actor=actor, last_update='training_contexts_only')
    validate(result, alignment['dimension'])
    return result

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
