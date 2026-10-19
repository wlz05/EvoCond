# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Read-only synthetic checks of conditional information, not task performance.'
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from control import dynamic, relations
FORMAT = 'evocond_information_diagnostic_v1'
def risk(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Require a nonempty empirical event list')
    by_context, by_representation, mapping = {}, {}, {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'context', 'representation', 'feedback'}:
            raise ValueError('Declare context, deterministic representation and bounded feedback')
        c, z, y = row['context'], row['representation'], row['feedback']
        if any(not isinstance(x, str) or not x for x in (c, z)):
            raise ValueError('Context and representation labels must be nonempty strings')
        if isinstance(y, bool) or not isinstance(y, (int, float)) or not math.isfinite(y) or not -1 <= y <= 1:
            raise ValueError('Feedback must be finite and lie in [-1, 1]')
        if c in mapping and mapping[c] != z:
            raise ValueError('The representation must be a deterministic function of context')
        mapping[c] = z
        by_context.setdefault(c, []).append(y)
        by_representation.setdefault(z, []).append(y)
    means = {c: sum(values)/len(values) for c, values in by_context.items()}
    compressed = {z: sum(values)/len(values) for z, values in by_representation.items()}
    mean = sum(row['feedback'] for row in rows)/len(rows)
    noise = sum((row['feedback']-means[row['context']])**2 for row in rows)/len(rows)
    projected = sum((row['feedback']-compressed[row['representation']])**2 for row in rows)/len(rows)
    marginal = sum((row['feedback']-mean)**2 for row in rows)/len(rows)
    bias = sum((means[row['context']]-compressed[row['representation']])**2 for row in rows)/len(rows)
    heterogeneity = sum((means[row['context']]-mean)**2 for row in rows)/len(rows)
    return dict(n=len(rows), conditional_means=means, representation_means=compressed, mean=mean,
        conditional_risk=noise, representation_risk=projected, unconditional_risk=marginal,
        compression_gap=bias, heterogeneity=heterogeneity, identity_error=projected-noise-bias,
        interpretation='empirical_algebra_not_heldout_estimation_or_task_gain')
def filters(cost_a, cost_b, pass_a, pass_b):
    values = (cost_a, cost_b, pass_a, pass_b)
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError('Filter costs and pass rates must be finite numbers')
    if cost_a <= 0 or cost_b <= 0 or not 0 <= pass_a <= 1 or not 0 <= pass_b <= 1:
        raise ValueError('Require positive per-item costs and pass rates in [0, 1]')
    ab, ba = cost_a+pass_a*cost_b, cost_b+pass_b*cost_a
    return dict(cost_a=cost_a, cost_b=cost_b, pass_a=pass_a, pass_b=pass_b,
        cost_ab=ab, cost_ba=ba, cost_difference=ab-ba,
        best_order='ab' if ab < ba else 'ba' if ba < ab else 'tie',
        assumptions='same commuting pure predicates; both orders legal; constant per-item cost',
        interpretation='analytic_cost_example_not_measured_agent_accuracy')
def decisions(rows, masses=None):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Require a complete controlled utility table')
    contexts, representations, values = {}, {}, {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'context', 'representation', 'action', 'utility'}:
            raise ValueError('Declare context, representation, action and utility')
        c, z, a, u = (row[key] for key in ('context', 'representation', 'action', 'utility'))
        if any(not isinstance(x, str) or not x for x in (c, z, a)):
            raise ValueError('Controlled labels must be nonempty strings')
        if type(u) not in (int, float) or not math.isfinite(u) or not -1 <= u <= 1:
            raise ValueError('Utility must be finite and bounded in [-1, 1]')
        if c in representations and representations[c] != z or (c, a) in values:
            raise ValueError('Require deterministic representations and unique action cells')
        representations[c] = z
        contexts.setdefault(c, {})[a] = u
        values[c, a] = u
    actions = sorted({a for c, a in values})
    if len(actions) < 2 or any(set(table) != set(actions) for table in contexts.values()):
        raise ValueError('Every controlled context must expose the same complete action set')
    if masses is None:
        masses = {c: 1.0 for c in contexts}
    if not isinstance(masses, dict) or set(masses) != set(contexts):
        raise ValueError('Declare one positive mass per controlled context')
    if any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in masses.values()):
        raise ValueError('Context masses must be finite positive numbers')
    largest = max(masses.values())
    scaled = {c: w/largest for c, w in masses.items()}
    total = math.fsum(scaled.values())
    weights = {c: w/total for c, w in scaled.items()}
    if any(w <= 0 for w in weights.values()):
        raise ValueError('Positive context support must not underflow during normalization')
    groups = {z: sorted(c for c in contexts if representations[c] == z) for z in sorted(set(representations.values()))}
    group_mass = {z: math.fsum(weights[c] for c in cs) for z, cs in groups.items()}
    averages = {z: {a: math.fsum(weights[c]/group_mass[z]*contexts[c][a] for c in cs) for a in actions} for z, cs in groups.items()}
    pooled = {z: min(actions, key=lambda a: (-table[a], a)) for z, table in averages.items()}
    records = []
    for c, table in sorted(contexts.items()):
        best = min(actions, key=lambda a: (-table[a], a))
        chosen = pooled[representations[c]]
        records.append(dict(context=c, mass=weights[c], best=best, compressed_choice=chosen, regret=table[best]-table[chosen],
            relative_values={a: table[a]-table[actions[0]] for a in actions}))
    regret = math.fsum(weights[row['context']]*row['regret'] for row in records)
    oracle = math.fsum(weights[c]*max(table.values()) for c, table in contexts.items())
    compressed = math.fsum(group_mass[z]*max(table.values()) for z, table in averages.items())
    prediction = math.fsum(weights[c]*(u-averages[representations[c]][a])**2/len(actions) for (c, a), u in values.items())
    intersections, sharing = {}, []
    for z, cs in groups.items():
        common = set(actions)
        for c in cs:
            common &= {a for a, u in contexts[c].items() if u == max(contexts[c].values())}
        intersections[z] = sorted(common)
        loss = math.fsum(weights[c]/group_mass[z]*(max(contexts[c].values())-contexts[c][pooled[z]]) for c in cs)
        sharing.append(dict(representation=z, contexts=cs, mass=group_mass[z], shared_action=pooled[z],
            common_optimal_actions=intersections[z], can_share_decision=bool(common),
            conditional_loss=loss, weighted_loss=group_mass[z]*loss))
    binary = []
    if len(actions) == 2:
        a, b = actions
        for z, cs in groups.items():
            positive = math.fsum(weights[c]/group_mass[z]*max(contexts[c][a]-contexts[c][b], 0) for c in cs)
            negative = math.fsum(weights[c]/group_mass[z]*max(contexts[c][b]-contexts[c][a], 0) for c in cs)
            binary.append(dict(representation=z, mass=group_mass[z], positive=positive, negative=negative,
                conditional_loss=min(positive, negative), weighted_loss=group_mass[z]*min(positive, negative),
                mean_margin=positive-negative, mean_absolute_margin=positive+negative))
    return dict(records=records, mean_regret=regret, oracle_value=oracle, compressed_value=compressed,
        jensen_gap=oracle-compressed, jensen_error=oracle-compressed-regret,
        positive_regret_contexts=sum(row['regret'] > 0 for row in records),
        common_optimal_actions=intersections, sharing_boundary=sharing, binary_contrast=binary,
        conflicting_groups=sum(not row['can_share_decision'] for row in sharing),
        utility_prediction_gap=prediction, compressed_values=averages,
        assumptions='complete controlled utility table; declared positive masses; lexicographic ties; fixed continuation',
        interpretation='controlled_decision_example_not_observed_counterfactuals_or_agent_results')
def refinement(rows, masses=None):
    before = decisions(rows, masses)
    preserved = [dict(row, representation=row['context']) for row in rows]
    after = decisions(preserved, masses)
    return dict(compressed=before, context_preserved=after,
        recoverable_decision_loss=before['mean_regret']-after['mean_regret'],
        intervention='refine_representation_only_same_utilities_actions_and_context_masses',
        interpretation='complete_controlled_table_not_an_online_oracle_or_learned_feature_guarantee')
def controlled(data):
    if not isinstance(data, dict) or set(data)-{'format', 'rows', 'masses'}:
        raise ValueError('Controlled tables require format, rows and optional masses only')
    if data.get('format') != 'evocond_controlled_utilities_v1' or 'rows' not in data:
        raise ValueError('Require a declared complete controlled utility table, not rollout rewards')
    return dict(format='evocond_controlled_decision_diagnostic_v1',
        gpu_started=False, model_loaded=False, files_written=False,
        decision_boundary=refinement(data['rows'], data.get('masses')))
def fiber():
    feedback = [1]*8+[-1]*2
    values = []
    for mask in range(1, (1 << len(feedback))-1):
        selected = [y for i, y in enumerate(feedback) if mask & (1 << i)]
        values.append(sum(selected)/len(selected))
    return dict(feedback=feedback, contexts=2, both_contexts_nonempty=True, assignments=len(values),
        minimum=min(values), maximum=max(values), diameter=max(values)-min(values),
        interpretation='finite_conditional_mean_set_not_a_continuous_interval_or_task_count')
def shared(probability, first, second, selection=.5):
    values = (probability, first, second, selection)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError('Require finite numeric probabilities and declared utility margins')
    if not 0 < probability < 1 or not 0 <= selection <= 1 or not 0 < min(first, second):
        raise ValueError('Require two supported conditions, positive margins and a valid action mixture')
    if max(first, second) > 2:
        raise ValueError('Margins must fit utilities bounded in [-1, 1]')
    left, right = probability*first, (1-probability)*second
    if min(left, right) <= 0:
        raise ValueError('Positive weighted margins must be representable without numeric underflow')
    return dict(probability=probability, margins=[first, second], selection=selection,
        regret=(1-selection)*left+selection*right, minimum_regret=min(left, right),
        optimal_selection=0.0 if left <= right else 1.0,
        assumptions='same representation and actions; fixed continuation; opposite unique optima',
        interpretation='analytic_restricted_selector_bound_not_a_baseline_performance_bound')
def histories(repeats=1, *, collapse=False, decay=1.0):
    if type(repeats) is not int or not 1 <= repeats <= 64:
        raise ValueError('Synthetic repetitions must be integers in [1, 64]')
    if not isinstance(collapse, bool):
        raise ValueError('Collapse must be a boolean')
    memories, statistics = [], []
    feedback = [1.0]*8+[-1.0]*2
    for swapped in (False, True):
        n = 10*repeats
        memory = relations.new(horizon=n, limit=n, bandwidth=.1, decay=decay)
        rows = []
        for i, y in enumerate(feedback*repeats):
            c = int((y < 0) != swapped)
            event = dict(source='a', target='b', source_version='av', target_version='bv',
                features=[0.0 if collapse else float(c)], context={'goal': 'visible condition '+str(c)},
                requirements=None, feedback=y, granularity='terminal')
            memory = relations.observe(memory, [event], base_id='q'+str(i), actor='fixedactor')
            rows.append(dict(context=str(c), representation='pooled' if collapse else str(c), feedback=y))
        memories.append(memory)
        statistics.append(risk(rows))
    return memories, statistics
def aggregate(memory):
    edge = next(iter(memory['edges'].values()))
    return dict(source=edge['source'], target=edge['target'], clock=memory['clock'],
        events=[dict(base_id=e['base_id'], actor=e['actor'], clock=e['clock'],
            feedback=e['feedback'], versions=[(r['source_version'], r['target_version']) for r in e['samples']])
            for e in edge['events']])
def replay(repeats=1, *, collapse=False, decay=1.0):
    memories, statistics = histories(repeats, collapse=collapse, decay=decay)
    receipts = []
    for memory, stats in zip(memories, statistics):
        local = relations.estimate(memory, 'a', 'b', [0.0], {}, audit=True)
        pooled = relations.estimate(memory, 'a', 'b', [0.0], {}, unconditional=True, audit=True)
        erased = relations.estimate(memory, 'a', 'b', [0.0], {}, erase=True, audit=True)
        receipts.append(dict(aggregate_sha256=dynamic.digest(aggregate(memory)),
            local=local, unconditional=pooled, erased=erased, risk=stats))
    left, right = receipts
    return dict(synthetic_bases=10*repeats, collapse=collapse, decay=decay, histories=receipts,
        same_ordered_aggregate=left['aggregate_sha256'] == right['aggregate_sha256'],
        same_unconditional_weight=left['unconditional']['weight'] == right['unconditional']['weight'],
        same_erased_weight=left['erased']['weight'] == right['erased']['weight'],
        conditional_sign_reversal=left['local']['weight'] > 0 > right['local']['weight'],
        scope='production_observe_estimate_primitives_with_legacy_gaussian_not_full_actor_or_teacher')
def transfers():
    rows = [dict(context=actor, representation='shared', action=action, utility=value)
        for actor, pair in (('oldactor', (.6, .9)), ('newactor', (.95, .8)))
        for action, value in zip(('semantic', 'structure'), pair)]
    decision = decisions(rows)
    support = replay()['histories'][0]['local']['weight']
    margins = {actor: pair[1]-pair[0] for actor, pair in (('oldactor', (.6, .9)), ('newactor', (.95, .8)))}
    if not support > 0 or not margins['oldactor'] > 0 > margins['newactor'] or decision['mean_regret'] <= 0:
        raise AssertionError('Historical support must not substitute for current adoption value')
    return dict(synthetic=True, fixed_task_bank_and_candidates=True, historical_support=support,
        actor_adoption_margins=margins, decision=decision,
        scope='controlled_actor_utility_tables_not_measured_model_improvement')
def report(repeats=(1, 2, 8, 32)):
    if not isinstance(repeats, (list, tuple)) or not repeats:
        raise ValueError('Declare unique synthetic repetition counts')
    if any(type(n) is not int or not 1 <= n <= 64 for n in repeats) or len(set(repeats)) != len(repeats):
        raise ValueError('Declare unique synthetic repetition counts')
    growth = [replay(n) for n in repeats]
    if not all(row['same_ordered_aggregate'] and row['same_unconditional_weight'] and
               row['same_erased_weight'] and row['conditional_sign_reversal'] for row in growth):
        raise AssertionError('Conditional reversal witness failed')
    noise = [dict(context=str(c), representation='pooled', feedback=y)
        for c in (0, 1) for y in [1.0]*8+[-1.0]*2]
    negative = risk(noise)
    collapsed = replay(collapse=True)
    if abs(negative['compression_gap']) > 1e-12 or collapsed['conditional_sign_reversal']:
        raise AssertionError('Invariant or erased-feature control failed')
    def table(values):
        return [dict(context=str(c), representation='shared', action=a, utility=u)
            for c, pair in enumerate(values) for a, u in zip(('a', 'b'), pair)]
    crossing = decisions(table(((.9, .6), (.6, .9))))
    invariant = decisions(table(((.9, .5), (.6, .2))))
    if crossing['mean_regret'] <= 0 or invariant['mean_regret'] != 0 or invariant['utility_prediction_gap'] <= 0:
        raise AssertionError('Decision reversal or common-best-action control failed')
    shifted = decisions(table(((1.0, .7), (.4, .7))))
    large_shift = decisions(table(((.95, .85), (.15, .05))))
    small_reversal = decisions(table(((.55, .45), (.45, .55))))
    weighted = decisions(table(((.9, .6), (.6, .9))), {'0': 1, '1': 3})
    sharing_growth = [dict(mass_multiplier=n,
        boundary=refinement(table(((.9, .6), (.6, .9))), {'0': n, '1': n})) for n in repeats]
    if abs(shifted['mean_regret']-crossing['mean_regret']) > 1e-12 or large_shift['mean_regret'] != 0:
        raise AssertionError('Common-mode value shifts must not change the decision loss')
    if small_reversal['mean_regret'] <= 0 or abs(weighted['mean_regret']-.075) > 1e-12:
        raise AssertionError('Signed margins or declared context support were ignored')
    return dict(format=FORMAT, synthetic=True, gpu_started=False, model_loaded=False, files_written=False,
        fixed_sequence_fiber=fiber(), actor_value_transfer=transfers(),
        filter_example=[filters(1, 4, .1, .5), filters(1, 4, .95, .01)], growth=growth,
        invariant_control=negative, collapsed_feature_control=collapsed,
        decision_reversal=crossing, common_best_action_control=invariant,
        common_mode_shift_control=shifted, large_value_shift_control=large_shift,
        small_margin_reversal=small_reversal, weighted_reversal=weighted,
        decision_sharing_growth=sharing_growth,
        shared_decision_bound=shared(.5, .3, .3),
        boundaries=['event clocks and feedback order are identical in each witness pair',
            'synthetic base IDs are distinct labels, not independent benchmark trials',
            'mass multiplication preserves a declared distribution; it is not additional data collection',
            'complete controlled utility tables are separate from observed edge feedback and rollout rewards',
            'growth changes the declared retention budget, not a fixed-memory deployment',
            'empirical partition means are oracle algebra, not a deployable feature extractor',
            'edge feedback predicts observations, not the causal value of adopting structure',
            'negative controls may differ in finite-sample shrinkage but contain no conditional mean information',
            'no task success, benchmark superiority or teacher effect is established'])
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, nargs='+', default=[1, 2, 8, 32])
    parser.add_argument('--table', type=Path, help='Complete controlled utility JSON; no rollout reward inference')
    args = parser.parse_args()
    try:
        result = controlled(json.loads(args.table.read_text(encoding='utf-8-sig'))) if args.table else report(args.repeats)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
if __name__ == '__main__':
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
