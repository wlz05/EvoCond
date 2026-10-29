# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Freeze mechanism contrasts separately from end-to-end training comparisons.'
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
from control import dynamic, stage
from experiment import read, write_new
from sweep import portable
CONTRASTS = (
    ('semantic', 'semantic', None, 'Remove relation influence'),
    ('unconditional', 'unconditional', .5, 'Pool feedback across historical contexts'),
    ('conditional', 'conditional', .5, 'Retain context without identity alignment'),
    ('identity', 'aligned', .5, 'Align identity with directional residuals'),
    ('erased', 'erased', .5, 'Erase event conditions but retain the event budget'),
    ('whole', 'whole', .5, 'Train whole-role alignment instead of identity alignment'),
    ('full', 'full', None, 'Learn adoption from one-point randomized feedback'),
    ('simple', 'simple', None, 'Use raw conditional retrieval and linear contextual adoption'),
    ('bernstein', 'full', None, 'Use the strict finite-sample global-envelope update rule'),
    ('fixedsemantic', 'full', 0.0, 'Use semantic payload in the full representation'),
    ('fixedhalf', 'full', .5, 'Use fixed equal fusion in the full representation'),
    ('fixedstructure', 'full', 1.0, 'Use structural ranking whenever support exists'))
QUESTIONS = (
    ('context', 'Does conditional reversal survive identical aggregate history?', ('unconditional', 'conditional', 'erased')),
    ('identity', 'Does identity alignment preserve task-dependent roles?', ('conditional', 'identity', 'whole')),
    ('adoption', 'Does learned adoption improve net task utility?', ('full', 'fixedsemantic', 'fixedhalf', 'fixedstructure')),
    ('minimal', 'What do learned geometry and strict bounds add to the core?', ('simple', 'full', 'bernstein')),
    ('contrast', 'Do choice reversals matter more than common value shifts?', ('full', 'fixedsemantic', 'fixedhalf')),
    ('support', 'Does unseen context yield actual semantic payload fallback?', ('full', 'identity')),
    ('transfer', 'Does actor change reverse adoption value with the same bank?', ('full', 'fixedsemantic')),
    ('credit', 'Can positive historical support have negative adoption value?', ('full', 'fixedhalf', 'fixedsemantic')))
def recipes(config):
    portable(config)
    if config.get('contextual') is None:
        raise ValueError('Mechanism comparisons require the contextual runtime')
    outputs = []
    for name, variant, alpha, hypothesis in CONTRASTS:
        current = deepcopy(config)
        current['run_id'] = config['run_id']+'_'+name
        current['contextual'].update(variant=variant, fixed_alpha=alpha)
        if name == 'bernstein':
            current['contextual']['confidence_method'] = 'bernstein'
        current['contextual'] = stage.settings(current['contextual'])
        outputs.append(dict(name=name, config=current, hypothesis=hypothesis))
    return outputs
def generate(config, output, run=False):
    rows = recipes(config)
    manifest = dict(format='evocond_mechanism_design_v3', source_sha256=dynamic.digest(config),
        target_arm='teacher', questions=[dict(name=name, question=question, contrasts=list(contrasts))
            for name, question, contrasts in QUESTIONS],
        controls=['same split and judge', 'same candidate and execution budgets', 'same initial actor',
            'fixed actor and bank for mechanism attribution', 'independent matched training for total utility'],
        literature=[dict(name='SEARL', version='2026.acl-long.1125', venue='ACL 2026', review='2026-10-13',
            control='Match dynamic execution and native policy updates; functional sharing is not decision sharing'),
            dict(name='H-EPM', version='2512.07287v3', venue='ICML 2026', review='2026-10-13',
            control='Compare contextual retrieval, not only context-erased graph statistics'),
            dict(name='FRESH', version='2609.28003v1', venue='preprint', review='2026-10-13',
            control='Both actions legal; execution preconditions alone need not identify the best action'),
            dict(name='DeMem', version='2605.10870v1', venue='preprint', review='2026-10-13',
            source='https://arxiv.org/html/2605.10870v1',
            control='Match decision-conflict refinement and feedback budgets; fixed-restart drift handling already exists'),
            dict(name='MemCalib', version='2609.24259v3', venue='preprint', review='2026-10-13',
            source='https://arxiv.org/html/2609.24259v3',
            control='Use independent ideal-use labels; terminal utility does not supply atom-use targets'),
            dict(name='What Should an Agent Remember?', version='2610.00366v1', review='2026-10-11',
            control='Hold retention access fixed before attributing differences to selection'),
            dict(name='A-TMA', version='2607.01935v2', review='2026-10-11',
            control='Distinguish valid evidence exposure from relative adoption utility'),
            dict(name='MemToC', version='2608.26295v1', review='2026-10-11',
            control='Measure correct suggestion adoption and wrong suggestion rejection jointly'),
            dict(name='Judged Useless, Queried Anyway', version='2610.06191v1', review='2026-10-11',
            control='Record actual behavior; judgments alone do not establish action integration')],
        method_contract=dict(name='EvoCond', object='conditional_relation_reuse_under_coevolution',
            unit='conditioned_proposal_on_shared_function_identity',
            estimand='actual_payload_net_utility_relative_to_semantic_under_current_bottom',
            principle='persist_relation_evidence_reconstruct_current_preferences',
            decision_rule='conditional_support_proposes_current_actor_relative_utility_selects',
            update_schedule=dict(relation='query_local_reconstruction', adoption='fixed_execution_window_reestimation'),
            contributions=['identity_sharing_with_condition_preserving_role_reconstruction',
                'receiver_relative_actual_payload_adoption_learning', 'execution_bound_coevolution_updates'],
            guiding_question='Does the preserved relation improve execution by the current actor?',
            inherited_tools=['decision-preserving abstraction', 'context-aware retrieval',
                'doubly robust estimation', 'conservative policy mixture'],
            exclusive_priority_claim=False,
            comparison_status='literature_targets_not_implemented_official_baselines'),
        decision_contract=dict(object='actual payload classes under fixed continuation',
            lineage='decision_preserving_abstraction_and_DeMem_common_optimal_criterion',
            estimand='conditional Jensen gap of maximum utility',
            binary_identity='half mean absolute margin minus half absolute mean margin',
            invariance='per_state_common_value_shift',
            positive_control='opposite optimal actions with positive context support',
            negative_controls=['common optimal action', 'large common value shift', 'nonoptimal pair reversal'],
            observation='complete analytic tables are not recovered counterfactual labels'),
        teacher_contract=dict(target='semantic_actual_payload_centered_cross_fitted_dr',
            protocol='evocond_contrast_adoption_v2', reference='validated_semantic_action_zero_and_payload_hash',
            deployment='validated_bottom_equals_current_actor_bank_geometry_protocol',
            parameter_retention='candidate_only_no_automatic_deployment_or_claimed_warm_start_gain',
            confidence_bound='studentized_independent_holdout_normal_default_bernstein_control',
            receipt='studentized_dr_normal_v1',
            coverage='asymptotic_not_finite_sample_default',
            strict_receipt='zero_sum_dr_global_envelope_v1',
            forbidden='replace uniform envelope with empirical heldout maximum'),
        exposure=dict(log='retrievals.jsonl', format='evocond_retrieval_exposure_v1',
            observed=['candidate access', 'schema drops', 'actual payload bytes', 'set versus order changes'],
            unknown=['target recall without independent annotation', 'adoption utility without intervention']),
        diagnostic=dict(command='python -B diagnostic.py', synthetic=True, actor_loaded=False,
            scope='ordered feedback, weighted choice loss, common shifts and collapsed features; not task evaluation'),
        missing_experiments=['context-invariant negative control', 'common-best-action negative control',
            'common-value-shift versus small-margin-reversal with matched support',
            'correct-structure acceptance versus wrong-structure rejection',
            'both-legal-action reversal with matched presented candidates', 'new capability versus new combination',
            'positive historical support with negative current adoption utility',
            'same task bank and candidates with actor-only adoption-value reversal',
            'retained parameters versus cleared parameters with current-bottom validation',
            'old/new actor crossed with old/new bank', 'memory growth under matched byte and token budgets',
            'decision-conflict memory refinement under matched action and feedback budgets',
            'memory-use calibration on independently annotated use levels',
            'verified recent condition-aware controls under a common execution interface'],
        claim='recipes are not results; separately trained actors do not isolate retrieval effects',
        recipes=[dict(name=row['name'], hypothesis=row['hypothesis'], config_sha256=dynamic.digest(row['config'])) for row in rows])
    if run:
        root = Path(output)
        if root.exists():
            raise FileExistsError('Use a new frozen design directory')
        for row in rows:
            write_new(root/(row['name']+'.json'), row['config'])
        write_new(root/'manifest.json', manifest)
    return manifest
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/contextual.json'))
    parser.add_argument('--output', type=Path, default=Path('configs/mechanisms'))
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(generate(read(args.config), args.output, args.run), indent=2))
if __name__ == '__main__':
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
