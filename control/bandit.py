# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'One-point logged exploration and grouped cross-fitted teacher learning.'
from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
import json
import math
import random
from statistics import NormalDist
import numpy as np
from control import dynamic
FORMAT = 'evocond_uniform_intervention_v2'
TEACHER = 'evocond_contextual_teacher_v1'
MIXTURE = 'evocond_conservative_mixture_v1'
ACTIONS = (0.0, 0.5, 1.0)
PROTOCOL = 'evocond_contrast_adoption_v2'
REFERENCE = 'semantic_actual_payload_class'
BOUND = 'zero_sum_dr_global_envelope_v1'
NORMAL = 'studentized_dr_normal_v1'
CONFIDENCE = ('normal', 'bernstein')

def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False
def simplex(values, size, *, positive=False):
    if not isinstance(values, (list, tuple, np.ndarray)) or np.asarray(values).ndim != 1 or len(values) != size or not size:
        raise ValueError('Require one probability per actual payload class')
    if any(isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, float, np.integer, np.floating)) for p in values):
        raise ValueError('Payload probabilities must be numeric, not booleans or strings')
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all() or np.any(result > 1) or np.any(result <= 0 if positive else result < 0) or abs(math.fsum(result)-1) > 1e-8:
        raise ValueError('Require a finite normalized payload probability vector')
    return result
def contrast(previous, proposal, logging, *, width):
    if type(width) not in (int, float) or not math.isfinite(width) or width <= 0:
        raise ValueError('Declare a finite positive reward interval width')
    old = simplex(previous, len(previous))
    new = simplex(proposal, len(old))
    p = simplex(logging, len(old), positive=True)
    difference = new-old
    variation = float(np.abs(difference).sum()/2)
    global_bound = width/float(p.min())
    if not math.isfinite(global_bound):
        raise ValueError('The reward and propensity envelope must be numerically representable')
    envelopes = np.asarray([width*variation+(width/float(prob)-width)*abs(float(d)) for prob, d in zip(p, difference)])
    return dict(difference=difference.tolist(), total_variation=variation, bound=float(envelopes.max()),
        action_bounds=envelopes.tolist(), global_bound=global_bound,
        assumptions='same actual payload classes; reward and heads in one fixed interval',
        interpretation='zero_sum_dr_envelope_not_a_task_improvement_certificate')
def randomizer(seed, *parts):
    return random.Random(int(sha256(json.dumps([seed, *parts], sort_keys=True).encode()).hexdigest(), 16))
def chosen_rollout(seed, phase, base_id, count):
    if type(count) is not int or count < 2:
        raise ValueError('Exploration reuses at least two existing grouped rollouts')
    return randomizer(seed, phase, base_id, 'rollout').randrange(count)
def marked_point(seed, phase, base_id, horizon):
    if type(horizon) is not int or horizon < 1:
        raise ValueError('Preselect from a fixed finite retrieval horizon')
    return randomizer(seed, phase, base_id, 'uniform_point').randrange(horizon)
def payloads(rankings, serializer):
    result, seen = [], {}
    for alpha in ACTIONS:
        payload = serializer(rankings[alpha])
        if not isinstance(payload, str):
            raise ValueError('Deduplicate the actual serialized actor payload')
        if payload in seen:
            result[seen[payload]]['alphas'].append(alpha)
        else:
            seen[payload] = len(result)
            result.append(dict(alpha=alpha, action=ACTIONS.index(alpha), alphas=[alpha], payload=payload,
                               payload_hash=sha256(payload.encode()).hexdigest(), ranking=deepcopy(rankings[alpha])))
    return result
def probabilities(model, x, classes):
    if not classes:
        return []
    validate_model(model)
    return policy_probabilities(model, x, classes)

def policy_probabilities(model, x, classes):
    if model is None:
        result = [0.0]*len(classes)
        result[next(i for i, row in enumerate(classes) if 0.0 in row['alphas'])] = 1.0
        return result
    if model.get('format') == MIXTURE:
        old = policy_probabilities(model['previous'], x, classes)
        proposal = policy_probabilities(model['proposal'], x, classes)
        return [(1-model['delta'])*a+model['delta']*b for a, b in zip(old, proposal)]
    values = np.asarray(x, dtype=float)
    if values.shape != (model['dimension'],) or not np.isfinite(values).all():
        raise ValueError('Teacher feature dimension mismatch')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        scaled = (values-np.asarray(model['mean']))/np.asarray(model['scale'])
        logits = np.asarray(model['weights'])[[row['action'] for row in classes]] @ scaled
    if not np.isfinite(scaled).all() or not np.isfinite(logits).all():
        raise ValueError('Teacher logits overflowed on the visible features')
    exp = np.exp(logits-logits.max())
    return (exp/exp.sum()).tolist()
def choose(model, x, classes, *, explore, epsilon, rng):
    if not 0 < epsilon <= 1:
        raise ValueError('Exploration requires positive full-support epsilon')
    normal = probabilities(model, x, classes)
    if not normal:
        return None, []
    probability = [(1-epsilon)*p+epsilon/len(classes) for p in normal] if explore else normal
    index = rng.choices(range(len(classes)), weights=probability, k=1)[0]
    return index, probability
def reward(success, costs, *, penalty, reference):
    if type(success) is not bool or not finite(penalty) or penalty < 0 or not finite(reference) or reference <= 0:
        raise ValueError('Reward needs correctness and fixed positive cost reference')
    if not isinstance(costs, dict) or set(costs) != {'actor', 'tools', 'retrieval'} or any(not finite(x) or x < 0 for x in costs.values()):
        raise ValueError('Record all measured whole-episode cost terms')
    try:
        total = math.fsum(costs.values())
    except OverflowError as error:
        raise ValueError('Whole-episode costs overflowed') from error
    if not math.isfinite(total):
        raise ValueError('Whole-episode costs must have a finite total')
    return float(success)-penalty*min(total/reference, 1.0)
def validate(row):
    required = {'features', 'reward', 'success', 'costs', 'penalty', 'reference', 'actions', 'probabilities', 'chosen', 'payload_hashes', 'interventions'}
    if not isinstance(row, dict) or not required <= set(row) or row.get('format') != FORMAT or row.get('split') != 'train' or any(not isinstance(row.get(key), str) or not row[key] for key in ('base_id', 'snapshot')):
        raise ValueError('Teacher observations require an identified training snapshot')
    try:
        x = np.asarray(row['features'], dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Invalid visible teacher features') from error
    if not finite(row['reward']):
        raise ValueError('Logged rewards must be finite numbers')
    if row.get('status', 'observed') not in ('observed', 'unreached', 'single_class', 'unsupported'):
        raise ValueError('Unknown randomized observation status')
    expected = reward(row['success'], row['costs'], penalty=row['penalty'], reference=row['reference'])
    if abs(expected-row['reward']) > 1e-9:
        raise ValueError('Teacher reward receipt mismatch')
    if 'cost_receipts' in row:
        if not isinstance(row['cost_receipts'], list):
            raise ValueError('Whole-episode receipts must be a list')
        terms = {key: [] for key in ('actor', 'tools', 'retrieval')}
        for receipt in row['cost_receipts']:
            if not isinstance(receipt, dict) or receipt.get('kind') not in terms or not finite(receipt.get('cost')) or receipt['cost'] < 0:
                raise ValueError('Whole-episode cost receipts must be finite and identified')
            terms[receipt['kind']].append(receipt['cost'])
        try:
            totals = {key: math.fsum(values) for key, values in terms.items()}
        except OverflowError as error:
            raise ValueError('Whole-episode receipts overflowed') from error
        if any(abs(totals[key]-row['costs'][key]) > 1e-8 for key in totals):
            raise ValueError('Whole-episode receipt sum differs from logged costs')
    if row.get('status') in ('unreached', 'single_class', 'unsupported'):
        if type(row['interventions']) is not int or row['interventions'] != 0 or row['chosen'] is not None or row['actions'] or row['probabilities'] or row['payload_hashes'] or x.ndim != 1 or x.size:
            raise ValueError('Absorbing observations have no randomized action')
        return
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or x[0] != 1.0:
        raise ValueError('Invalid visible teacher features')
    actions, probabilities = row['actions'], row['probabilities']
    if not isinstance(actions, (list, tuple)) or not actions or any(type(a) is not int or a not in range(3) for a in actions) or len(actions) != len(set(actions)) or type(row['chosen']) is not int or row['chosen'] not in actions:
        raise ValueError('Malformed payload action classes')
    if actions[0] != 0:
        raise ValueError('The first actual class must retain the semantic action identity')
    simplex(probabilities, len(actions), positive=True)
    if 'old_probabilities' in row:
        simplex(row['old_probabilities'], len(actions))
    if not isinstance(row['payload_hashes'], (list, tuple)) or len(row['payload_hashes']) != len(actions) or any(not isinstance(value, str) or not value for value in row['payload_hashes']) or len(set(row['payload_hashes'])) != len(actions):
        raise ValueError('Payload classes must have distinct byte identities')
    if row.get('semantic_payload_hash', row['payload_hashes'][0]) != row['payload_hashes'][0]:
        raise ValueError('Semantic reference differs from the actual payload identity')
    if row.get('target_reference', REFERENCE) != REFERENCE:
        raise ValueError('Adoption targets require the semantic actual payload reference')
    if type(row['interventions']) is not int or row['interventions'] != 1:
        raise ValueError('Teacher reward or single-intervention receipt mismatch')
def frozen_window(records):
    if not records or len({row['snapshot'] for row in records}) != 1:
        raise ValueError('Do not pool labels across changing actor/bank snapshots')
    for field in ('actor', 'bank', 'bottom', 'encoder', 'alignment', 'metric', 'continuation_teacher_sha256', 'horizon', 'penalty', 'reference'):
        values = [row.get(field) for row in records]
        if any(value is not None for value in values) and (any(value is None for value in values) or len({dynamic.digest(value) for value in values}) != 1):
            raise ValueError('Adoption estimation requires one complete frozen '+field+' binding')
    dimensions = {len(row['features']) for row in records if row['interventions'] == 1}
    if len(dimensions) > 1:
        raise ValueError('A frozen window must use one visible feature dimension')
def crossfit(records, *, folds=3, ridge=1.0):
    if type(folds) is not int or folds < 2 or not finite(ridge) or ridge <= 0:
        raise ValueError('Cross-fitting requires multiple folds and positive ridge')
    for row in records:
        validate(row)
    frozen_window(records)
    if any(row['interventions'] != 1 for row in records):
        raise ValueError('Reward-head fitting requires actual randomized observations')
    ids = sorted({row['base_id'] for row in records})
    if len(ids) < 2 or len(ids) != len(records):
        raise ValueError('Require one intervention per independent base question and snapshot')
    fold_of = {identity: i % min(folds, len(ids)) for i, identity in enumerate(ids)}
    results = []
    for row in records:
        pool = [other for other in records if fold_of[other['base_id']] != fold_of[row['base_id']]]
        features = np.asarray([other['features'] for other in pool], dtype=float)
        mean, scale = features.mean(axis=0), features.std(axis=0)
        mean[0], scale[0] = 0.0, 1.0
        scale[scale < 1e-8] = 1.0
        query = (np.asarray(row['features'])-mean)/scale
        prior = sum(other['reward'] for other in pool)/len(pool)
        values, predictions = [], []
        low, high = -row['penalty'], 1.0
        for action, probability in zip(row['actions'], row['probabilities']):
            observed = [other for other in pool if other['chosen'] == action]
            if observed:
                design = (np.asarray([other['features'] for other in observed])-mean)/scale
                target = np.asarray([other['reward'] for other in observed])
                coefficients = np.linalg.solve(design.T @ design+ridge*np.eye(len(query)), design.T @ target)
                prediction = float(query @ coefficients)
            else:
                prediction = prior
            prediction = max(low, min(high, prediction))
            predictions.append(prediction)
            values.append(prediction+(row['reward']-prediction)/probability if row['chosen'] == action else prediction)
        results.append(dict(base_id=row['base_id'], actions=row['actions'], features=row['features'], values=values,
                            relative_values=[value-values[0] for value in values], reference_action=0,
                            reference_payload_hash=row['payload_hashes'][0], target_reference=REFERENCE,
                            predictions=predictions, model_fit_ids=sorted(other['base_id'] for other in pool),
                            model_mean=mean.tolist(), model_scale=scale.tolist(), target_kind='unclipped_doubly_robust_estimate'))
    return results
def fit(records, *, steps=100, rate=.02, regularization=.01, folds=3, ridge=1.0):
    if type(steps) is not int or steps < 1 or not finite(rate) or rate <= 0 or not finite(regularization) or regularization < 0:
        raise ValueError('Invalid teacher optimizer budget')
    targets = crossfit(records, folds=folds, ridge=ridge)
    features = np.asarray([row['features'] for row in targets])
    mean, scale = features.mean(axis=0), features.std(axis=0)
    mean[0], scale[0] = 0.0, 1.0
    scale[scale < 1e-8] = 1.0
    features = (features-mean)/scale
    weights, objective = np.zeros((3, features.shape[1])), []
    for _ in range(steps):
        gradient, value = np.zeros_like(weights), 0.0
        for row, x in zip(targets, features):
            actions = row['actions']
            logits = weights[actions] @ x
            probability = np.exp(logits-logits.max())
            probability /= probability.sum()
            values = np.asarray(row['relative_values'], dtype=float)
            expected = float(probability @ values)
            value += expected/len(targets)
            gradient[actions] += np.outer(probability*(values-expected), x)/len(targets)
        objective.append(value-regularization*float(np.sum(weights**2)))
        weights += rate*(gradient-2*regularization*weights)
    model = dict(format=TEACHER, dimension=features.shape[1], weights=weights.tolist(), mean=mean.tolist(), scale=scale.tolist(),
                 fit_ids=sorted(row['base_id'] for row in records), fitted_snapshot=records[0]['snapshot'], objective=objective,
                 target_kind='cross_fitted_unclipped_dr', target_reference=REFERENCE, adoption_protocol=PROTOCOL,
                 objective_kind='relative_dr_minus_regularization', scope='single_point_training_not_sequential_ope')
    validate_model(model)
    return model, targets
def evaluate(records, model):
    validate_model(model)
    if not records:
        return dict(n=0, ips=None, scope='one_marked_state_only_not_end_to_end_teacher', clipped=False)
    for row in records:
        validate(row)
    frozen_window(records)
    ids = [row['base_id'] for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Evaluation requires independent base questions')
    fitting, pending = set(), [model]
    while pending:
        node = pending.pop()
        if node is not None:
            fitting.update(node.get('fit_ids', []))
            if node.get('format') == MIXTURE:
                pending.extend((node['previous'], node['proposal']))
    if set(ids) & fitting:
        raise ValueError('Teacher evaluation must use independent base questions')
    estimates = []
    for row in records:
        if row['interventions'] == 0:
            estimates.append(row['reward'])
            continue
        classes = [dict(action=a, alpha=ACTIONS[a], alphas=[ACTIONS[a]]) for a in row['actions']]
        probability = probabilities(model, row['features'], classes)
        index = row['actions'].index(row['chosen'])
        estimates.append(probability[index]*row['reward']/row['probabilities'][index])
    return dict(n=len(estimates), ips=sum(estimates)/len(estimates) if estimates else None,
                scope='one_marked_state_only_not_end_to_end_teacher', clipped=False)
def validate_model(model):
    dimensions, active = set(), set()
    def visit(node, level):
        if node is None:
            return
        if not isinstance(node, dict) or id(node) in active or level > 64:
            raise ValueError('Teacher mixtures must be finite acyclic checkpoints')
        if node.get('adoption_protocol', PROTOCOL) != PROTOCOL or node.get('target_reference', REFERENCE) != REFERENCE:
            raise ValueError('Teacher reference or adoption protocol changed')
        active.add(id(node))
        if node.get('format') == MIXTURE:
            if not {'previous', 'proposal'} <= set(node) or not finite(node.get('delta')) or not 0 <= node['delta'] <= 1:
                raise ValueError('Invalid conservative policy mixture')
            visit(node['previous'], level+1)
            visit(node['proposal'], level+1)
        elif node.get('format') == TEACHER:
            dimension = node.get('dimension')
            if type(dimension) is not int or dimension < 1:
                raise ValueError('Teacher dimension must be a positive integer')
            dimensions.add(dimension)
            for field, shape in (('weights', (3, dimension)), ('mean', (dimension,)), ('scale', (dimension,))):
                try:
                    value = np.asarray(node.get(field), dtype=float)
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError('Malformed linear teacher checkpoint') from error
                if value.shape != shape or not np.isfinite(value).all() or field == 'scale' and np.any(value <= 0):
                    raise ValueError('Malformed linear teacher checkpoint')
        else:
            raise ValueError('Unknown teacher checkpoint')
        if 'fit_ids' in node and (not isinstance(node['fit_ids'], list) or any(not isinstance(base, str) or not base for base in node['fit_ids']) or len(set(node['fit_ids'])) != len(node['fit_ids'])):
            raise ValueError('Teacher fitting identities must be unique base questions')
        active.remove(id(node))
    visit(model, 0)
    if len(dimensions) > 1:
        raise ValueError('Teacher mixture branches must share one feature dimension')
def reward_model(records, ridge=1.0):
    if not records or any(row.get('interventions') != 1 for row in records):
        raise ValueError('Reward heads require actual randomized training observations')
    if not finite(ridge) or ridge <= 0:
        raise ValueError('Reward-head regularization must be finite and positive')
    for row in records:
        validate(row)
    frozen_window(records)
    if len({row['base_id'] for row in records}) != len(records):
        raise ValueError('Reward heads require independent base questions')
    features = np.asarray([row['features'] for row in records], dtype=float)
    mean, scale = features.mean(axis=0), features.std(axis=0)
    mean[0], scale[0] = 0.0, 1.0
    scale[scale < 1e-8] = 1.0
    coefficients = []
    prior = np.mean([row['reward'] for row in records])
    for action in range(3):
        selected = [row for row in records if row['chosen'] == action]
        if selected:
            design = (np.asarray([row['features'] for row in selected])-mean)/scale
            target = np.asarray([row['reward'] for row in selected])
            beta = np.linalg.solve(design.T @ design+ridge*np.eye(len(mean)), design.T @ target)
        else:
            beta = np.zeros(len(mean))
            beta[0] = prior
        coefficients.append(beta.tolist())
    return dict(mean=mean.tolist(), scale=scale.tolist(), coefficients=coefficients,
                fit_ids=sorted(row['base_id'] for row in records), frozen=True)
def heldout_targets(row, head):
    if row['base_id'] in head['fit_ids']:
        raise ValueError('Held-out reward model includes its own base question')
    query = (np.asarray(row['features'])-head['mean'])/head['scale']
    values = []
    for action, probability in zip(row['actions'], row['probabilities']):
        prediction = float(query @ np.asarray(head['coefficients'][action]))
        prediction = min(1.0, max(-row['penalty'], prediction))
        values.append(prediction+(row['reward']-prediction)/probability if action == row['chosen'] else prediction)
    return values
def confidence(values, *, horizon, penalty, epsilon, beta=.05, delta_max=.2, overhead=0.0, method='bernstein'):
    parameters = (penalty, epsilon, beta, delta_max, overhead)
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in parameters):
        raise ValueError('Conservative protocol constants must be finite numbers')
    if type(horizon) is not int or horizon < 1 or penalty < 0 or not 0 < epsilon <= 1 or not 0 < beta < 1 or not 0 < delta_max <= 1 or overhead < 0:
        raise ValueError('Invalid predeclared conservative update protocol')
    if method not in CONFIDENCE or method == 'normal' and beta >= .5:
        raise ValueError('Declare a confidence method and a positive one-sided critical value')
    n = len(values)
    width, pmin = 1+penalty, epsilon/3
    if pmin <= 0:
        raise ValueError('The positive declared propensity floor must not underflow')
    bound = width/pmin
    if not math.isfinite(bound):
        raise ValueError('The global confidence envelope must be finite')
    if any(type(z) not in (int, float) or not math.isfinite(z) or abs(z) > bound+1e-8 for z in values):
        raise ValueError('Advantage exceeds the declared reward and propensity bound')
    variance = float(np.var(values, ddof=1)) if n >= 2 and any(z != values[0] for z in values) else 0.0
    mean = float(np.mean(values)) if n else 0.0
    if not math.isfinite(variance) or not math.isfinite(mean):
        raise ValueError('Confidence moments must be numerically representable')
    logterm = math.log(2)-math.log(beta)
    standard_error = math.sqrt(variance/n) if n else None
    range_penalty = 14*bound*logterm/(3*(n-1)) if n >= 2 else None
    critical = -NormalDist().inv_cdf(beta) if method == 'normal' else None
    eligible = n >= 32 and variance > 0 if method == 'normal' else n >= 2
    reason = 'insufficient_holdout' if n < (32 if method == 'normal' else 2) else 'degenerate_variance' if method == 'normal' and variance == 0 else 'evaluated'
    radius = critical*standard_error if method == 'normal' and eligible else math.sqrt(2*variance*logterm/n)+range_penalty if method == 'bernstein' and eligible else None
    lower = horizon*(mean-radius) if radius is not None else None
    curvature = width*horizon*(horizon-1)
    if not math.isfinite(curvature) or lower is not None and not math.isfinite(lower):
        raise ValueError('The finite-horizon confidence calculation overflowed')
    delta = min(delta_max, lower/(2*curvature)) if lower is not None and lower > 0 and curvature else delta_max if lower is not None and lower > 0 else 0.0
    improvement = delta*lower-curvature*delta**2-overhead if lower is not None else -overhead
    if improvement < 0:
        delta, improvement = 0.0, 0.0
    return dict(n=n, mean=mean, variance=variance, advantage_lower=lower, delta=delta, improvement_lower=improvement,
        beta=beta, pmin=pmin, bound=bound, bound_kind=NORMAL if method == 'normal' else BOUND, adoption_protocol=PROTOCOL,
        confidence_method=method, coverage_kind='asymptotic_fixed_proposal' if method == 'normal' else 'finite_sample_iid_global_envelope',
        finite_sample_certificate=method == 'bernstein' and eligible, confidence_eligible=eligible, confidence_reason=reason,
        standard_error=standard_error, critical_value=critical, confidence_radius=radius,
        range_penalty=range_penalty if method == 'bernstein' else 0.0, bernstein_range_penalty=range_penalty,
        minimum_holdout=32 if method == 'normal' else 2,
        largest_centered_share=max((abs(z-mean) for z in values), default=0.0)/math.sqrt((n-1)*variance) if variance > 0 else None,
        curvature=curvature, overhead_penalty=overhead,
        scope='iid_heldout_fixed_actor_bank_environment_only_not_coupled_evolution')
def depth(model):
    return 0 if model is None or model.get('format') != MIXTURE else 1+depth(model['previous'])
def logged(row, previous, epsilon):
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError('Declare the frozen exploration mixture before evaluation')
    classes = [dict(action=a, alpha=ACTIONS[a], alphas=[ACTIONS[a]]) for a in row['actions']]
    old = simplex(probabilities(previous, row['features'], classes), len(classes))
    logged_old = simplex(row.get('old_probabilities', []), len(classes))
    p = simplex(row['probabilities'], len(classes), positive=True)
    if np.max(np.abs(old-logged_old)) > 1e-8:
        raise ValueError('Continuation teacher differs from the frozen logged policy')
    if np.max(np.abs(p-((1-epsilon)*old+epsilon/len(classes)))) > 1e-8:
        raise ValueError('Logged sampling differs from the frozen exploration mixture')
    if np.min(p) < epsilon/3-1e-12:
        raise ValueError('Actual payload propensity falls below the declared floor')
    return classes, old, p
def advantage(row, previous, proposal, head, *, epsilon):
    validate(row)
    if row['interventions'] != 1:
        return 0.0, dict(total_variation=0.0, bound=0.0)
    classes, old, p = logged(row, previous, epsilon)
    new = probabilities(proposal, row['features'], classes)
    receipt = contrast(old, new, p, width=1+row['penalty'])
    values = heldout_targets(row, head)
    result = float(np.asarray(receipt['difference']) @ np.asarray(values))
    if not math.isfinite(result) or abs(result) > receipt['bound']+1e-8:
        raise ValueError('Held-out advantage violates its zero-sum envelope')
    return result, receipt
def conservative(records, previous, *, seed=42, holdout=.5, horizon=6, epsilon=.2, beta=.05,
                 delta_max=.2, chain_limit=4, steps=100, minimum=6, overhead=0.0, used_holdout=(), method='bernstein'):
    if method not in CONFIDENCE:
        raise ValueError('Unknown predeclared teacher confidence method')
    for row in records:
        validate(row)
        if row['interventions'] == 1:
            logged(row, previous, epsilon)
    frozen_window(records)
    if len({row['base_id'] for row in records}) != len(records) or len({row['snapshot'] for row in records}) != 1:
        raise ValueError('One current-window observation per base question is required')
    if any(row.get('horizon', horizon) != horizon or row['penalty'] != records[0]['penalty'] or
           row['reference'] != records[0]['reference'] for row in records):
        raise ValueError('One frozen horizon and whole-episode reward protocol are required')
    if not 0 < holdout < 1 or chain_limit < 1:
        raise ValueError('Invalid fixed holdout split or mixture chain bound')
    fitting = [row for row in records if randomizer(seed, row['base_id'], 'holdout').random() >= holdout]
    holding = [row for row in records if row not in fitting]
    if set(row['base_id'] for row in holding) & set(used_holdout):
        raise ValueError('Do not adaptively reuse a held-out question for another proposal')
    report = dict(fit_ids=[row['base_id'] for row in fitting], holdout_ids=[row['base_id'] for row in holding],
        zeros=sum(row['interventions'] == 0 for row in holding), deployed=False, extra_rollouts=0, targets=[], adoption_protocol=PROTOCOL,
        confidence_method=method)
    observed = [row for row in fitting if row['interventions'] == 1]
    if len(observed) < minimum or len(holding) < 2:
        report.update(reason='insufficient_independent_fit_or_holdout', delta=0.0)
        return deepcopy(previous), report
    proposal, targets = fit(observed, steps=steps)
    head = reward_model(observed)
    advantages, contrasts = [], []
    for row in holding:
        if row['interventions'] == 0:
            advantages.append(0.0)
            continue
        value, envelope = advantage(row, previous, proposal, head, epsilon=epsilon)
        advantages.append(value)
        contrasts.append(dict(base_id=row['base_id'], **envelope))
    receipt = confidence(advantages, horizon=horizon, penalty=records[0]['penalty'], epsilon=epsilon,
        beta=beta, delta_max=delta_max, overhead=overhead, method=method)
    report.update(receipt, targets=targets, proposal_sha256=dynamic.digest(proposal), reward_model_fit_ids=head['fit_ids'],
        contrast_receipts=contrasts, heldout_advantages=advantages, local_envelopes_used_for_confidence=False)
    if depth(previous) >= chain_limit:
        base = distill(previous, observed, steps=steps)
        report.update(delta=0.0, deployed=False, reason='chain_compressed_requires_new_window', bound_transferred=False)
        return base, report
    if receipt['delta'] <= 0:
        report['reason'] = receipt['confidence_reason'] if not receipt['confidence_eligible'] else 'nonpositive_'+method+'_lower_bound'
        return deepcopy(previous), report
    model = dict(format=MIXTURE, previous=deepcopy(previous), proposal=proposal, delta=receipt['delta'],
        fitted_snapshot=records[0]['snapshot'], fit_ids=proposal['fit_ids'], confidence=receipt,
        adoption_protocol=PROTOCOL, target_reference=REFERENCE,
        continuation_teacher_sha256=dynamic.digest(previous))
    report.update(deployed=True, reason='positive_fixed_bottom_lower_bound')
    return model, report
def distill(previous, rows, *, steps=100):
    features = np.asarray([row['features'] for row in rows])
    mean, scale = features.mean(axis=0), features.std(axis=0)
    mean[0], scale[0] = 0.0, 1.0
    scale[scale < 1e-8] = 1.0
    weights = np.zeros((3, len(mean)))
    for _ in range(steps):
        gradient = np.zeros_like(weights)
        for row in rows:
            classes = [dict(action=a, alpha=ACTIONS[a], alphas=[ACTIONS[a]]) for a in row['actions']]
            query = (np.asarray(row['features'])-mean)/scale
            actions = row['actions']
            exp = np.exp(weights[actions] @ query-np.max(weights[actions] @ query))
            probability = exp/exp.sum()
            gradient[actions] += np.outer(np.asarray(probabilities(previous, row['features'], classes))-probability, query)/len(rows)
        weights += .02*gradient
    return dict(format=TEACHER, dimension=len(mean), weights=weights.tolist(), mean=mean.tolist(), scale=scale.tolist(),
        fit_ids=[row['base_id'] for row in rows], fitted_snapshot=rows[0]['snapshot'], objective=[],
        adoption_protocol=PROTOCOL, target_reference=REFERENCE,
        target_kind='compressed_previous_policy', scope='new_window_required_no_old_bound')

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
