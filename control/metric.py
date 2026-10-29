# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Frozen low-rank relation distance, local scales and absolute support.'
from __future__ import annotations
from copy import deepcopy
import math
import numpy as np
from control import dynamic
from control.context import compatible
FORMAT = 'evocond_relation_metric_v3'
DEFAULTS = dict(rank=32, neighbors=7, pool=128, sigma_min=.05, absolute=.65, budget_scale=100.0,
    neutral=.5, residual=.15, protection='adaptive', weights=[.7, .1, .1, .1], steps=20, rate=.01, regularization=.01, shrinkage=1.0)
def options(value):
    if not isinstance(value, dict) or set(value)-set(DEFAULTS):
        raise ValueError('Unknown relation metric fields')
    result = {**deepcopy(DEFAULTS), **value}
    for field in ('rank', 'neighbors', 'pool', 'steps'):
        if type(result[field]) is not int or result[field] < 1:
            raise ValueError('Metric budgets must be positive integers')
    for field in ('sigma_min', 'absolute', 'budget_scale', 'rate', 'regularization', 'shrinkage'):
        if not math.isfinite(result[field]) or result[field] <= 0:
            raise ValueError('Metric scales must be finite and positive')
    weights = result['weights']
    if result['protection'] not in ('adaptive', 'fixed', 'identity', 'unprotected'):
        raise ValueError('Unknown explicitly declared protection ablation')
    if result['protection'] == 'unprotected' and result['residual'] != 0:
        raise ValueError('Unprotected ablation must explicitly declare zero residual')
    if not math.isfinite(result['residual']) or not (0 <= result['residual'] <= 1 if result['protection'] == 'unprotected' else 0 < result['residual'] <= 1):
        raise ValueError('Original semantic distance needs a positive residual floor')
    if len(weights) != 4 or any(not math.isfinite(x) or x < 0 for x in weights) or abs(sum(weights)-1) > 1e-9 or not 0 <= result['neutral'] <= 1:
        raise ValueError('Distance is a convex mixture of four bounded terms')
    return result
def describe(phi, context):
    semantic = list(phi[:-6]) if context.get('format') == 'evocond_visible_context_v1' else list(phi)
    if not semantic or not all(math.isfinite(x) for x in semantic):
        raise ValueError('Metric requires finite frozen semantic features')
    available = context.get('available')
    types = sorted(key for key, value in available.items() if value) if available is not None else None
    progress = context.get('progress') or {}
    budget = progress.get('budget')
    done, total = progress.get('completed'), progress.get('total')
    fraction = min(1.0, max(0.0, done/total)) if total and done is not None else None
    return dict(semantic=semantic, available=types, budget=budget, progress=fraction,
                missing=dict(available=types is None, budget=budget is None, progress=fraction is None))
def initialize(dimension, encoder_id, settings=None):
    config = options(settings or {})
    rank = min(config['rank'], dimension)
    projection = np.eye(dimension)[:rank]
    result = dict(format=FORMAT, dimension=dimension, encoder_id=encoder_id, options=config,
        projection=projection.tolist(), projection_id=dynamic.digest(projection.tolist()), initial_projection=projection.tolist(), fitted=False, pool=[], losses=[],
        fit_ids=[], scale_generation=0, fallback='full_dimension_identity', initialization='provisional_until_training_contexts', calibration='declared_not_dev_verified')
    result['geometry_id'] = geometry_id(result)
    return result
def geometry_id(artifact):
    return dynamic.digest(dict(options=artifact['options'], projection=artifact['projection_id'], fitted=artifact['fitted'],
        reference=[{key: row[key] for key in ('edge', 'base_id', 'actor', 'granularity', 'scorer', 'neighborhood_fitted')} |
            dict(descriptor={key: row['descriptor'][key] for key in ('semantic', 'available', 'budget', 'progress')}) for row in artifact['pool']]))
def validate(artifact, encoder_id=None):
    if artifact.get('format') != FORMAT or encoder_id is not None and artifact['encoder_id'] != encoder_id:
        raise ValueError('Relation metric and frozen encoder differ')
    config = options(artifact['options'])
    shape = (min(config['rank'], artifact['dimension']), artifact['dimension'])
    for name in ('projection', 'initial_projection'):
        matrix = np.asarray(artifact[name])
        if matrix.shape != shape or not np.isfinite(matrix).all():
            raise ValueError('Malformed relation projection')
    if len(artifact['pool']) > config['pool']:
        raise ValueError('Recent context cache exceeds its fixed budget')
    if artifact['projection_id'] != dynamic.digest(artifact['projection']):
        raise ValueError('Relation projection and scale cache have different versions')
    if artifact.get('geometry_id') != geometry_id(artifact) or artifact.get('fallback') != 'full_dimension_identity':
        raise ValueError('Relation distance recipe and cached scales have different versions')
    for row in artifact['pool']:
        descriptor = row['descriptor']
        if descriptor.get('projection_sha256') != artifact['projection_id']:
            raise ValueError('Stale projected context cache')
        if descriptor.get('geometry_sha256') != artifact['geometry_id']:
            raise ValueError('Stale protected distance cache')
        value = np.asarray(descriptor.get('projected'), dtype=float)
        if value.shape != (shape[0],) or not np.isfinite(value).all() or np.linalg.norm(value) > 1+1e-8:
            raise ValueError('Invalid projected context cache')
        semantic = np.asarray(descriptor['semantic'], dtype=float)
        original = np.asarray(descriptor.get('original'), dtype=float)
        norm = np.linalg.norm(semantic)
        expected = semantic/norm if norm > 1e-12 else np.zeros(artifact['dimension'])
        if original.shape != (artifact['dimension'],) or not np.isfinite(original).all() or not np.allclose(original, expected):
            raise ValueError('Invalid original semantic cache')
        if not np.allclose(value, projected(artifact, descriptor, np.asarray(artifact['projection']))):
            raise ValueError('Projected semantic cache differs from the frozen mapping')
        novelty = descriptor.get('novelty')
        expected_novelty = 0.0 if norm > 1e-12 else config['neutral']
        if not isinstance(novelty, (int, float)) or not math.isfinite(novelty) or abs(novelty-expected_novelty) > 1e-8:
            raise ValueError('Historical reference novelty must match its own cached context')
        if not math.isfinite(row['local_scale']) or row['local_scale'] < config['sigma_min']:
            raise ValueError('Invalid local neighborhood scale')
def projected(artifact, descriptor, matrix=None):
    if matrix is None and descriptor.get('projection_sha256') == artifact['projection_id']:
        return np.asarray(descriptor['projected'], dtype=float)
    vector = np.asarray(descriptor['semantic'], dtype=float)
    if vector.shape != (artifact['dimension'],) or not np.isfinite(vector).all():
        raise ValueError('Frozen context encoder dimension changed')
    mapping = np.asarray(artifact['projection'] if matrix is None else matrix)
    raw = mapping @ vector
    norm = np.linalg.norm(raw)
    if norm <= 1e-12:
        raw = np.asarray(artifact['initial_projection']) @ vector
        norm = np.linalg.norm(raw)
    return raw/norm if norm > 1e-12 else np.zeros(mapping.shape[0])
def fields(left, right, config):
    neutral = config['neutral']
    a, b = left['available'], right['available']
    availability = neutral if a is None or b is None else 1-len(set(a)&set(b))/len(set(a)|set(b)) if set(a)|set(b) else 0.0
    a, b = left['budget'], right['budget']
    budget = neutral if a is None or b is None else abs(min(1, max(0, a/config['budget_scale']))-min(1, max(0, b/config['budget_scale'])))
    a, b = left['progress'], right['progress']
    progress = neutral if a is None or b is None else abs(a-b)
    return availability, budget, progress
def raw_distance(artifact, left, right):
    a, b = (np.asarray(row['original'] if row.get('geometry_sha256') == artifact.get('geometry_id') and 'original' in row else row['semantic'], dtype=float) for row in (left, right))
    if any(value.shape != (artifact['dimension'],) or not np.isfinite(value).all() for value in (a, b)):
        raise ValueError('Original context encoder dimension changed')
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return min(1.0, max(0.0, (1-float((a/na) @ (b/nb)))/2)) if na > 1e-12 and nb > 1e-12 else artifact['options']['neutral']
def prepare(artifact, descriptor):
    result = deepcopy(descriptor)
    vector = np.asarray(result['semantic'], dtype=float)
    if vector.shape != (artifact['dimension'],) or not np.isfinite(vector).all():
        raise ValueError('Protected distance requires the frozen full-dimensional encoder')
    norm = np.linalg.norm(vector)
    result['original'] = (vector/norm if norm > 1e-12 else np.zeros_like(vector)).tolist()
    result['projected'] = projected(artifact, result).tolist()
    result['projection_sha256'] = artifact['projection_id']
    result['novelty'] = min((raw_distance(artifact, result, row['descriptor']) for row in artifact['pool']), default=1.0)
    result['geometry_sha256'] = artifact['geometry_id']
    return result
def residual_weight(artifact, left, right, matrix=None, *, identity=False):
    floor = artifact['options']['residual']
    mode = artifact['options']['protection']
    if mode == 'unprotected':
        return 0.0
    if mode == 'identity':
        return 1.0
    if identity or matrix is None and (not artifact['fitted'] or not artifact['pool']):
        return 1.0
    if matrix is not None or mode == 'fixed':
        return floor
    novelty = max(row['novelty'] if row.get('geometry_sha256') == artifact['geometry_id'] and 'novelty' in row else
        min(raw_distance(artifact, row, old['descriptor']) for old in artifact['pool']) for row in (left, right))
    return floor+(1-floor)*min(1.0, novelty/artifact['options']['absolute']**2)
def identity_fallback(artifact, pool):
    return not artifact['fitted'] or not any(row.get('neighborhood_fitted', False) for row in pool)
def distance(artifact, left, right, matrix=None, *, identity=False):
    config = artifact['options']
    a, b = projected(artifact, left, matrix), projected(artifact, right, matrix)
    residual = residual_weight(artifact, left, right, matrix, identity=identity)
    original = raw_distance(artifact, left, right)
    learned = (1-float(a @ b))/2 if np.any(a) and np.any(b) else original
    semantic = (1-residual)*learned+residual*original
    return min(1.0, max(0.0, float(np.asarray(config['weights']) @ np.asarray([semantic, *fields(left, right, config)]))))
def scale(artifact, query, pool, *, exclude=None, matrix=None, identity=False):
    distances = [math.sqrt(distance(artifact, query, row['descriptor'], matrix, identity=identity)) for row in pool if row['base_id'] != exclude]
    if not distances:
        return None
    distances.sort()
    return max(artifact['options']['sigma_min'], distances[min(artifact['options']['neighbors'], len(distances))-1])
def kernel(artifact, left, right, left_scale, right_scale, matrix=None, *, absolute=True, identity=False):
    if left_scale is None or right_scale is None:
        return 0.0
    squared = distance(artifact, left, right, matrix, identity=identity)
    gate = max(0.0, 1-squared/artifact['options']['absolute']**2) if absolute else 1.0
    return gate*math.exp(-squared/(left_scale*right_scale))
def records(memory):
    result = []
    for edge_id, edge in sorted(memory['edges'].items()):
        for entry in edge['events']:
            for sample in entry['samples']:
                if not sample.get('context'):
                    continue
                result.append(dict(edge=edge_id, base_id=entry['base_id'], actor=entry['actor'], clock=entry['clock'],
                    descriptor=describe(sample['features'], sample['context']), feedback=sample['feedback'],
                    granularity=sample['granularity'], scorer=sample.get('scorer', 'unspecified'),
                    source_version=sample['source_version'], target_version=sample['target_version'], requirements=sample.get('requirements'),
                    mixture_mass=1/len(entry['samples'])))
    return result
def fit(artifact, memory, *, version_compatible=None):
    validate(artifact)
    result = deepcopy(artifact)
    pool = sorted(records(memory), key=lambda row: (row['clock'], row['base_id'], row['edge']))[-artifact['options']['pool']:]
    if not pool:
        return result
    config = artifact['options']
    raw = config['protection'] == 'identity' or config['protection'] == 'fixed' and config['residual'] == 1.0
    matrix = np.asarray(artifact['projection'], dtype=float)
    if not artifact['fitted'] and not raw:
        vectors = np.asarray([row['descriptor']['semantic'] for row in pool])
        _, _, basis = np.linalg.svd(vectors, full_matrices=False)
        rank = matrix.shape[0]
        matrix[:min(rank, len(basis))] = basis[:rank]
        result['initial_projection'] = matrix.tolist()
    previous = matrix.copy()
    def permitted(a, b):
        versions = (a['source_version'] == b['source_version'] and a['target_version'] == b['target_version'])
        if version_compatible is not None:
            versions = version_compatible(a['source_version'], b['source_version']) and version_compatible(a['target_version'], b['target_version'])
        return (a['edge'] == b['edge'] and a['actor'] == b['actor'] and a['base_id'] != b['base_id'] and versions and
            a['granularity'] == b['granularity'] and a['scorer'] == b['scorer'])
    neighborhoods = [[other for other in pool if permitted(row, other)] for row in pool]
    scales = [scale(result, row['descriptor'], neighbors, matrix=previous) for row, neighbors in zip(pool, neighborhoods)]
    scale_by_row = {id(row): value for row, value in zip(pool, scales)}
    result['losses'] = []
    usable = sum(bool(neighbors) for neighbors in neighborhoods)
    if not usable:
        matrix = np.asarray(artifact['projection'], dtype=float)
        result['initial_projection'] = deepcopy(artifact['initial_projection'])
    for _ in range(config['steps'] if usable and not raw else 0):
        gradient = np.zeros_like(matrix)
        loss = 0.0
        for i, (row, neighbors) in enumerate(zip(pool, neighborhoods)):
            if not neighbors:
                continue
            left = row['descriptor']
            weights, derivatives = [], []
            for other in neighbors:
                right = other['descriptor']
                sigma = scale_by_row[id(other)]
                k = kernel(result, left, right, scales[i], sigma, matrix)
                weights.append(k*other['mixture_mass'])
                raw_l, raw_r = matrix @ np.asarray(left['semantic']), matrix @ np.asarray(right['semantic'])
                nl, nr = np.linalg.norm(raw_l), np.linalg.norm(raw_r)
                derivative = np.zeros_like(matrix)
                if nl > 1e-12 and nr > 1e-12 and k > 0:
                    zl, zr = raw_l/nl, raw_r/nr
                    cosine = float(zl @ zr)
                    residual = residual_weight(result, left, right, matrix)
                    dd = -config['weights'][0]*(1-residual)/2*(np.outer((zr-cosine*zl)/nl, left['semantic'])+np.outer((zl-cosine*zr)/nr, right['semantic']))
                    squared = distance(result, left, right, matrix)
                    exponential = math.exp(-squared/(scales[i]*sigma))
                    dk = exponential*(-1/config['absolute']**2-max(0, 1-squared/config['absolute']**2)/(scales[i]*sigma))
                    derivative = dk*dd*other['mixture_mass']
                derivatives.append(derivative)
            mass = sum(weights)
            signed = sum(weight*other['feedback'] for weight, other in zip(weights, neighbors))
            denominator = config['shrinkage']+mass
            error = signed/denominator-row['feedback']
            loss += error**2/usable
            for other, derivative in zip(neighbors, derivatives):
                gradient += 2*error*(other['feedback']*denominator-signed)/denominator**2*derivative/usable
        loss += config['regularization']*float(np.sum((matrix-previous)**2))
        result['losses'].append(loss)
        matrix -= config['rate']*(gradient+2*config['regularization']*(matrix-previous))
    result.update(projection=matrix.tolist(), fitted=artifact['fitted'] or usable > 0 and not raw, pool=pool, fit_ids=sorted({row['base_id'] for row in pool}),
        scale_generation=artifact['scale_generation']+1, initialization='fixed_raw_context' if raw else 'training_context_svd' if usable else artifact['initialization'],
        objective='fixed_raw_context_no_projection_fit' if raw else 'leave_base_question_out_actor_and_feedback_stratified', target_kind='observed_correctness_not_adoption_value',
        usable_neighborhoods=usable)
    result['projection_id'] = dynamic.digest(result['projection'])
    for row in result['pool']:
        row['neighborhood_fitted'] = any(permitted(row, other) for other in result['pool'])
    result['geometry_id'] = geometry_id(result)
    for row in result['pool']:
        row['descriptor'] = prepare(result, row['descriptor'])
    for row in result['pool']:
        neighborhood = [other for other in result['pool'] if permitted(row, other)]
        row['neighborhood_fitted'] = bool(neighborhood)
        row['local_scale'] = scale(result, row['descriptor'], neighborhood or [row], identity=not neighborhood)
        row['descriptor']['projected'] = projected(result, row['descriptor']).tolist()
        row['descriptor']['projection_sha256'] = result['projection_id']
    validate(result)
    return result
def support(artifact, descriptor, sample, pool, left_scale=None):
    fallback = identity_fallback(artifact, pool)
    left_scale = scale(artifact, descriptor, pool, identity=fallback) if left_scale is None else left_scale
    right = describe(sample['features'], sample['context'])
    saved = next((row for row in pool if row['source_version'] == sample['source_version'] and row['target_version'] == sample['target_version'] and
        row['descriptor']['semantic'] == right['semantic'] and all(row['descriptor'][key] == right[key] for key in ('available', 'budget', 'progress'))), None)
    right_scale = scale(artifact, right, pool, identity=True) if fallback else saved['local_scale'] if saved is not None else scale(artifact, right, pool)
    right = saved['descriptor'] if saved is not None else prepare(artifact, right)
    return kernel(artifact, descriptor, right, left_scale, right_scale, identity=fallback)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
