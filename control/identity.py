# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Canonical capability identity, compatible versions, and identity-only alignment.'
from __future__ import annotations
from copy import deepcopy
import ast
import json
import numpy as np
from control import dynamic
from control.context import embedding, unit
FORMAT = 'evocond_capability_identity_v1'
ALIGNMENT = 'evocond_content_alignment_v2'
def catalogue(registry):
    result = deepcopy(registry.get('capabilities', dict(format=FORMAT, nodes={}, versions={}, compatibility=[])))
    result.setdefault('usage', {})
    if result.get('format') != FORMAT:
        raise ValueError('Unknown capability identity schema')
    for version, spec in sorted(registry['semantic'].items()):
        if version not in result['versions']:
            canonical = dynamic.digest(['capability', version])
            result['versions'][version] = canonical
            result['nodes'][canonical] = dict(description=spec['description'], versions=[version], initial_version=version)
    return result
def attach(registry):
    result = deepcopy(registry)
    result['capabilities'] = catalogue(result)
    return result
def link(registry, source_version, target_version, receipt):
    result = attach(registry)
    store = result['capabilities']
    if source_version not in store['versions'] or target_version not in store['versions']:
        raise ValueError('Capability link refers to an unknown version')
    left, right = result['semantic'][source_version], result['semantic'][target_version]
    if any(left[field] != right[field] for field in ('arguments', 'returns', 'side_effects')):
        raise ValueError('Cannot merge incompatible interfaces or side effects')
    probes = receipt.get('behavioral_probe_sha256', [])
    if receipt.get('decision') != 'merge_verified' or len(set(probes)) < 2 or receipt.get('behavioral_probes', 0) < 2:
        raise ValueError('Capability merge requires multi-input isolated behavior evidence')
    keep, remove = store['versions'][source_version], store['versions'][target_version]
    if keep != remove:
        versions = store['nodes'].pop(remove)['versions']
        store['nodes'][keep]['versions'] = sorted(set(store['nodes'][keep]['versions']+versions))
        for version in versions:
            store['versions'][version] = keep
        store['usage'][keep] = sorted(set(store['usage'].get(keep, [])+store['usage'].pop(remove, [])))
    pair = dict(source=source_version, target=target_version, receipt=deepcopy(receipt))
    if pair not in store['compatibility']:
        store['compatibility'].append(pair)
    if 'relations' in result and keep != remove:
        from control.relations import remap
        result['relations'] = remap(result['relations'], {remove: keep})
    return result
def eligible(registry, observed, current):
    store = catalogue(registry)
    if observed == current:
        return observed in store['versions']
    graph = {}
    for pair in store['compatibility']:
        graph.setdefault(pair['source'], set()).add(pair['target'])
        graph.setdefault(pair['target'], set()).add(pair['source'])
    pending, visited = [observed], set()
    while pending:
        version = pending.pop()
        if version == current:
            return True
        if version not in visited:
            visited.add(version)
            pending.extend(graph.get(version, ())-visited if version in graph else ())
    return False
def content(spec):
    tree = ast.parse(spec['code'])
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if functions:
        name = functions[0].name
        functions[0].name = 'capability'
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == name:
                node.id = 'capability'
    return json.dumps(dict(arguments=spec['arguments'], returns=spec['returns'], side_effects=spec['side_effects'],
        postconditions=spec.get('postconditions', spec['description']), code=ast.dump(tree, include_attributes=False)), sort_keys=True)
def views(registry, encoder):
    store, rows = catalogue(registry), []
    for canonical, node in sorted(store['nodes'].items()):
        spec = registry['semantic'][node['initial_version']]
        semantic = embedding(encoder, node['description'], query=False)
        stable = embedding(encoder, content(spec), query=False)
        rows.append(dict(canonical=canonical, semantic=semantic, stable=stable, versions=node['versions']))
    return rows
def initialize(registry, encoder, encoder_id):
    rows = views(registry, encoder)
    dimension = len(rows[0]['semantic']) if rows else len(embedding(encoder, 'empty capability', query=False))
    return dict(format=ALIGNMENT, encoder_id=encoder_id, dimension=dimension,
                semantic_map=np.eye(dimension).tolist(), identity_map=np.eye(dimension).tolist(),
                fit_ids=[], loss=[], frozen=False, initial_versions=[registry['capabilities']['nodes'][row['canonical']]['initial_version'] if 'capabilities' in registry else catalogue(registry)['nodes'][row['canonical']]['initial_version'] for row in rows],
                objective='canonical_identity_only_no_context_or_rewards', identity_source='shared_content_no_id_lookup')
def validate(artifact, encoder_id=None):
    if artifact.get('format') != ALIGNMENT or (encoder_id is not None and artifact['encoder_id'] != encoder_id):
        raise ValueError('Frozen identity alignment/encoder mismatch')
    dimension = artifact['dimension']
    for field in ('semantic_map', 'identity_map'):
        matrix = np.asarray(artifact[field], dtype=float)
        if matrix.shape != (dimension, dimension) or not np.isfinite(matrix).all():
            raise ValueError('Invalid identity projection')
    if 'identities' in artifact:
        raise ValueError('Content alignment cannot contain trainable identity lookups')
def project(artifact, values, *, semantic=False):
    return unit(np.asarray(artifact['semantic_map' if semantic else 'identity_map']) @ np.asarray(values), artifact['dimension'], zero=True)
def node_vector(registry, artifact, encoder, canonical):
    node = catalogue(registry)['nodes'][canonical]
    spec = registry['semantic'][node['initial_version']]
    values = project(artifact, embedding(encoder, content(spec), query=False))
    if not any(values):
        values = project(artifact, embedding(encoder, node['description'], query=False), semantic=True)
    return values
def fit(registry, encoder, artifact, *, steps=50, rate=.02, temperature=.1, fit_ids=()):
    validate(artifact)
    if type(steps) is not int or steps < 0 or not 0 < rate <= 1 or not 0 < temperature <= 1:
        raise ValueError('Invalid identity contrastive budget')
    result = deepcopy(artifact)
    if artifact['frozen']:
        return result
    if not result['initial_versions']:
        result['initial_versions'] = sorted(registry['semantic'])
    initial = set(result['initial_versions'])
    rows = [row for row in views(registry, encoder) if catalogue(registry)['nodes'][row['canonical']]['initial_version'] in initial]
    if not rows:
        return result
    semantic = np.asarray([row['semantic'] for row in rows])
    stable = np.asarray([row['stable'] for row in rows])
    ps, pa = np.asarray(result['semantic_map']), np.asarray(result['identity_map'])
    losses = []
    for _ in range(steps):
        inputs = stable
        raw_s, raw_a = semantic @ ps.T, inputs @ pa.T
        ns, na = np.linalg.norm(raw_s, axis=1, keepdims=True), np.linalg.norm(raw_a, axis=1, keepdims=True)
        if np.any(ns < 1e-12) or np.any(na < 1e-12):
            raise ValueError('Contrastive projection collapsed')
        s, a = raw_s/ns, raw_a/na
        scores = s @ a.T/temperature
        def softmax(matrix):
            exp = np.exp(matrix-matrix.max(axis=1, keepdims=True))
            return exp/exp.sum(axis=1, keepdims=True)
        forward, reverse = softmax(scores), softmax(scores.T)
        losses.append(float(-np.mean(np.log(np.diag(forward)+1e-30)+np.log(np.diag(reverse)+1e-30))/2))
        gradient = (forward+reverse.T-2*np.eye(len(rows)))/(2*len(rows)*temperature)
        gs, ga = gradient @ a, gradient.T @ s
        gs = (gs-s*np.sum(gs*s, axis=1, keepdims=True))/ns
        ga = (ga-a*np.sum(ga*a, axis=1, keepdims=True))/na
        ps -= rate*(gs.T @ semantic)
        pa -= rate*(ga.T @ inputs)
    result.update(semantic_map=ps.tolist(), identity_map=pa.tolist(), fit_ids=sorted(set(fit_ids)), loss=losses, frozen=True)
    validate(result)
    return result
def admit(registry, spec, backend, *, probe_inputs, task_id, encoder, ledger=(), equivalence_inputs=None,
          timeout=15.0, recall=4, threshold=.88, probe_budget=8):
    dynamic._check_spec(spec)
    if recall < 1 or probe_budget < 4 or not -1 <= threshold <= 1:
        raise ValueError('Invalid shared capability admission budget')
    version = dynamic._version(spec)
    if version in registry['semantic']:
        return attach(registry), dict(decision='already_admitted', version=version)
    query = embedding(encoder, spec['description'], query=False)
    candidates = sorted(registry['semantic'], key=lambda key: (-sum(a*b for a, b in zip(query,
        embedding(encoder, registry['semantic'][key]['description'], query=False))), key))[:recall]
    budget, match, receipt = probe_budget, None, None
    for candidate in candidates:
        old = registry['semantic'][candidate]
        score = sum(a*b for a, b in zip(query, embedding(encoder, old['description'], query=False)))
        if score < threshold or any(old[field] != spec[field] for field in ('arguments', 'returns', 'side_effects')):
            continue
        observed = [row['arguments'] for row in ledger if row.get('tool_version') == candidate and row.get('success') and isinstance(row.get('arguments'), dict)]
        inputs = [probe_inputs, *(equivalence_inputs or []), *observed]
        unique = {dynamic.digest(value): value for value in inputs}
        probes = sorted(unique.items())[:min(3, budget//2)]
        if len(probes) < 2:
            continue
        passed = True
        for _, value in probes:
            left = dynamic._run(backend, old, value, timeout)
            right = dynamic._run(backend, spec, value, timeout)
            budget -= 2
            if not left['run_success'] or not right['run_success'] or left.get('result') != right.get('result'):
                passed = False
                break
        if passed:
            match = candidate
            receipt = dict(decision='merge_verified', behavioral_probes=len(probes),
                           behavioral_probe_sha256=[key for key, _ in probes], similarity=score,
                           isolation='paired_existing_backend', probe_budget=probe_budget)
            break
        if budget < 4:
            break
    updated, decision = dynamic.admit(registry, spec, backend, probe_inputs=probe_inputs, task_id=task_id, timeout=timeout)
    updated = attach(updated)
    if decision['decision'] == 'new_versioned_node' and match is not None:
        updated = link(updated, match, version, receipt)
        decision.update(capability_decision='compatible_version', canonical=updated['capabilities']['versions'][version], compatibility=receipt)
    else:
        decision['capability_decision'] = 'new_capability' if decision['decision'] == 'new_versioned_node' else decision['decision']
    decision.update(probe_runs=probe_budget-budget+1, semantic_recall=recall)
    return updated, decision

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
