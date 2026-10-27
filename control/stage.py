# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Frozen phase retrieval, one-point collection, and barrier-only publication.'
from __future__ import annotations
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from control import bandit, drift, dynamic, fatigue, mixing as fusion, identity, metric, relations, subtasks, whole, windows
from control.context import embedding
from control.context import visible
FORMAT = 'evocond_contextual_stage_v6'
DEFAULTS = dict(recall=12, top_k=3, neighbors=8, horizon=128, limit=64, bandwidth=1.0, decay=1.0,
                shrinkage=1.0, epsilon=.2, rollouts=8, seed=42, variant='full', horizon_steps=7,
                penalty=.1, reference=100.0, actor_input_price=.001, actor_output_price=.002,
                tool_price=1.0, retrieval_price=1.0, alignment_steps=20, teacher_steps=100,
                minimum_labels=6, fixed_alpha=None, identity_shrinkage=2.0, metric={}, holdout=.5,
                beta=.05, confidence_method='normal', delta_max=.2, chain_limit=4, actor_kl_threshold=.02, actor_drift={}, maximum_windows=1000, frozen_batches=2, whole_samples=256)
def settings(value):
    if not isinstance(value, dict) or set(value)-set(DEFAULTS):
        raise ValueError('Unknown contextual protocol fields')
    result = {**DEFAULTS, **value}
    integers = ('recall', 'top_k', 'neighbors', 'horizon', 'limit', 'rollouts', 'alignment_steps', 'teacher_steps', 'minimum_labels',
        'horizon_steps', 'chain_limit', 'maximum_windows', 'frozen_batches', 'whole_samples')
    if any(type(result[key]) is not int or result[key] < 1 for key in integers):
        raise ValueError('Contextual budgets must be positive integers')
    if type(result['seed']) is not int or result['seed'] < 0 or result['rollouts'] < 2 or result['minimum_labels'] < 2:
        raise ValueError('Require an identified grouped learning budget')
    numeric = ('bandwidth', 'decay', 'shrinkage', 'epsilon', 'holdout', 'beta', 'delta_max',
        'identity_shrinkage', 'actor_kl_threshold', 'penalty', 'reference', 'actor_input_price',
        'actor_output_price', 'tool_price', 'retrieval_price')
    if any(not bandit.finite(result[key]) for key in numeric):
        raise ValueError('Contextual probabilities, distances and prices must be finite numbers')
    relations.new(**{key: result[key] for key in ('horizon', 'limit', 'bandwidth', 'decay', 'shrinkage')})
    if not result['top_k'] < result['recall'] or result['variant'] not in fusion.VARIANTS:
        raise ValueError('Invalid semantic candidate pool or ablation')
    if not 0 < result['epsilon'] <= 1 or not 0 < result['holdout'] < 1 or not 0 < result['beta'] < 1 or not 0 < result['delta_max'] <= 1:
        raise ValueError('Invalid intervention sampling protocol')
    if result['confidence_method'] not in bandit.CONFIDENCE or result['confidence_method'] == 'normal' and result['beta']/result['maximum_windows'] >= .5:
        raise ValueError('Invalid predeclared teacher confidence method')
    if not math.isfinite(result['identity_shrinkage']) or result['identity_shrinkage'] <= 0:
        raise ValueError('Cold-start identity shrinkage must be positive')
    if not math.isfinite(result['actor_kl_threshold']) or result['actor_kl_threshold'] <= 0:
        raise ValueError('Declare a positive actor drift threshold before training')
    result['metric'] = metric.options(result['metric'])
    if result['variant'] in ('simple', 'raw'):
        result['metric'].update(protection='identity', residual=1.0)
    result['actor_drift'] = drift.options(result['actor_drift'])
    if result['fixed_alpha'] is not None and (not bandit.finite(result['fixed_alpha']) or result['fixed_alpha'] not in bandit.ACTIONS):
        raise ValueError('Fixed fusion must use a declared three-action ratio')
    for key in ('penalty', 'reference', 'actor_input_price', 'actor_output_price', 'tool_price', 'retrieval_price'):
        if not math.isfinite(result[key]) or result[key] < 0 or (key == 'reference' and result[key] == 0):
            raise ValueError('Invalid fixed accounting prices')
    return result
class Engine:
    'Publish only complete groups; evaluation never updates the shared bank.'
    def __init__(self, runtime):
        self.runtime = runtime
        self.options = settings(runtime.config['contextual'])
        self.encoder, self.encoder_id = runtime.encoder, runtime.config['encoder_id']
        self.phase, self.groups, self.logs = 0, {}, []
        registry = identity.attach(runtime.dispatcher.registry)
        registry.setdefault('relations', relations.new(**{key: self.options[key] for key in ('horizon', 'limit', 'bandwidth', 'decay', 'shrinkage')}))
        runtime.dispatcher.replace_registry(registry, expected_sha256=dynamic.digest(runtime.dispatcher.registry))
        self.alignment = identity.initialize(registry, self.encoder, self.encoder_id)
        if self.options['variant'] in ('simple', 'noidentity'):
            self.alignment.update(frozen=True, objective='fixed_identity_no_alignment_fit')
        else:
            self.alignment = identity.fit(registry, self.encoder, self.alignment, steps=self.options['alignment_steps'])
        self.whole = whole.initialize(self.alignment) if self.options['variant'] == 'whole' else None
        self.metric = metric.initialize(len(embedding(self.encoder, 'context')), self.encoder_id, self.options['metric'])
        self.teacher = None
        self.teacher_bottom = None
        self.actor_updates = []
        self.used_holdout = []
        self.pending_registry = None
        self.actor = runtime.policy_hash
        self.freeze()
    def freeze(self):
        self.registry = deepcopy(self.runtime.dispatcher.registry)
        self.bottom = dynamic.digest(dict(actor=self.actor, encoder=self.encoder_id, adoption_protocol=bandit.PROTOCOL, registry=dynamic.digest(self.registry),
            alignment=dynamic.digest(self.alignment), whole=dynamic.digest(self.whole), metric=dynamic.digest(self.metric), options=self.options))
        self.snapshot = dynamic.digest(dict(format=FORMAT, phase=self.phase, actor=self.actor, encoder=self.encoder_id, adoption_protocol=bandit.PROTOCOL,
            registry=dynamic.digest(self.registry), alignment=dynamic.digest(self.alignment), whole=dynamic.digest(self.whole), metric=dynamic.digest(self.metric),
            teacher=dynamic.digest(self.teacher), teacher_bottom=self.teacher_bottom, options=self.options))
    def deployed_teacher(self):
        return self.teacher if self.teacher is not None and self.teacher_bottom == self.bottom else None
    def adoption_status(self):
        active = self.deployed_teacher() is not None
        return dict(active=active, bottom=self.bottom, validated_bottom=self.teacher_bottom,
            retained_teacher_sha256=dynamic.digest(self.teacher), target_reference=bandit.REFERENCE,
            reason='current_bottom_validated' if active else 'semantic_base' if self.teacher is None else 'bottom_requires_validation')
    def actor_changed(self, actor, receipt=None):
        if actor == self.actor:
            return None
        if self.runtime.contexts or self.pending_registry is not None:
            raise ValueError('Actor alignment requires a completed co-evolution barrier')
        previous, old_teacher = self.actor, dynamic.digest(self.teacher)
        reason, diagnosis = None, None
        try:
            diagnosis = drift.decision(receipt, previous, self.options['actor_kl_threshold'], self.options['actor_drift'])
        except ValueError as error:
            reason = 'missing_or_invalid_actor_drift'
            seconds = receipt.get('elapsed_seconds') if isinstance(receipt, dict) else None
            receipt = dict(valid=False, reason=str(error), actor=previous)
            if isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds >= 0:
                receipt['elapsed_seconds'] = seconds
        if diagnosis is not None and diagnosis['reset']:
            reason = 'actor_drift_rule_exceeded'
        if reason is not None:
            self.teacher = None
            self.teacher_bottom = None
        report = dict(event='teacher_actor_alignment', phase=self.phase, previous_actor=previous, actor=actor,
            threshold=self.options['actor_kl_threshold'], drift=deepcopy(receipt), diagnosis=diagnosis, soft_reset=reason is not None,
            reason=reason or 'retained_candidate_only', previous_teacher=old_teacher, teacher=dynamic.digest(self.teacher),
            chain_depth=bandit.depth(self.teacher), guarantee_transferred=False, requires_new_actor_window=True,
            deployment_authorized=False, scope='new_bottom_requires_value_validation')
        with (self.runtime.root/'actorupdates.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(report, sort_keys=True, allow_nan=False)+'\n')
        self.actor_updates.append(report)
        self.actor = actor
        self.freeze()
        return report
    def begin(self, context):
        base = context['task_id']
        ordinal = self.groups.get(base, 0) if context['split'] == 'train' else 0
        if context['split'] == 'train':
            if ordinal >= self.options['rollouts']:
                raise ValueError('A phase contains more than G rollouts for one base question')
            self.groups[base] = ordinal+1
        context.update(phase=self.phase, phase_snapshot=self.snapshot, ordinal=ordinal,
            exploration_rollout=bandit.chosen_rollout(self.options['seed'], self.phase, base, self.options['rollouts']),
            marked=False, costs=dict(actor=0.0, tools=0.0, retrieval=0.0), cost_receipts=[], pending_relations=[],
            marked_point=bandit.marked_point(self.options['seed'], self.phase, base, self.options['horizon_steps']),
            retrieval_index=0, selection_cache={}, actor_metered=False, mark_status='unreached')
    def charge(self, context, kind, quantities):
        if kind == 'actor':
            if set(quantities) != {'input_tokens', 'output_tokens'} or any(type(x) is not int or x < 0 for x in quantities.values()):
                raise ValueError('Actor accounting requires actual token counts')
            value = quantities['input_tokens']*self.options['actor_input_price']+quantities['output_tokens']*self.options['actor_output_price']
            context['actor_metered'] = True
        elif kind in ('tools', 'retrieval'):
            seconds = quantities.get('seconds')
            if set(quantities) != {'seconds'} or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError('Execution accounting requires measured elapsed seconds')
            value = seconds*self.options['tool_price' if kind == 'tools' else 'retrieval_price']
        else:
            raise ValueError('Unknown cost category')
        context['costs'][kind] += value
        context['cost_receipts'].append(dict(kind=kind, quantities=deepcopy(quantities), cost=value))
    def view(self, context):
        ready = subtasks.ready(context['plan']) if context['plan'] else []
        plan = (context['plan'] or {}).get('tasks', {})
        task = ready[0] if ready else {}
        remaining = ' '.join(row['goal'] for row in plan.values() if row['status'] != 'completed')
        known = context.get('visible_fields', {})
        available = known.get('available')
        if available is None:
            observed = [row['result'] for row in context.get('calls', []) if row.get('success') and row.get('result_available')]
            if observed:
                available = {kind: False for kind in ('string', 'number', 'boolean', 'object', 'array')}
                for value in observed:
                    kind = 'boolean' if type(value) is bool else 'number' if type(value) in (int, float) else 'string' if isinstance(value, str) else 'object' if isinstance(value, dict) else 'array' if isinstance(value, list) else None
                    if kind is not None:
                        available[kind] = True
        return visible(task.get('goal', context['question']), constraints=known.get('constraints'),
            available=available, requirements=known.get('requirements'), slots=known.get('slots', ()),
            progress=dict(completed=sum(row['status'] == 'completed' for row in plan.values()),
                total=len(plan), turn=context['assistant_turn'], budget=known.get('budget'), remaining_goal=remaining))
    def select(self, context, schemas):
        turn = context['assistant_turn']
        if turn in context['selection_cache']:
            return deepcopy(context['selection_cache'][turn])
        started = time.monotonic()
        index = context['retrieval_index']
        if index >= self.options['horizon_steps']:
            raise ValueError('Actual retrieval count exceeds the frozen uniform sampling horizon')
        context['retrieval_index'] += 1
        registry = identity.attach(context['dispatcher'].registry)
        registry['relations'] = deepcopy(self.registry['relations'])
        view = self.view(context)
        result = fusion.rank(registry, self.encoder, self.alignment, view,
            completed=[row['tool_version'] for row in context['invocations'] if row.get('success') and row.get('tool_version')],
            recall=self.options['recall'], top_k=self.options['top_k'], neighbors=self.options['neighbors'], variant=self.options['variant'],
            metric=self.metric, identity_shrinkage=self.options['identity_shrinkage'], whole_alignment=self.whole)
        if result['abstention']['active']:
            receipt = dict(event='structural_memory_self_invalidation', task_id=context['task_id'], split=context['split'],
                turn=turn, phase=self.phase, snapshot=self.snapshot, **result['abstention'])
            with (self.runtime.root/'abstentions.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(receipt, sort_keys=True, allow_nan=False)+'\n')
            print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
        by_version = {}
        for schema in sorted(schemas, key=lambda row: row['name']):
            by_version.setdefault(schema.get('tool_version'), schema)
        def presentation(ranking):
            return [deepcopy(by_version[row['version']]) for row in ranking if row['version'] in by_version]
        classes = bandit.payloads(result['rankings'], lambda ranking: json.dumps(dict(task=view, tools=presentation(ranking)),
            sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False))
        eligible = (context['split'] == 'train' and self.options['variant'] in fusion.LEARNED and self.options['fixed_alpha'] is None and
            context['ordinal'] == context['exploration_rollout'] and index == context['marked_point'])
        marked = eligible and not context['marked'] and result['edge_support'] and len(classes) > 1
        if eligible:
            context['mark_status'] = 'observed' if marked else 'single_class' if len(classes) == 1 else 'unsupported'
        model = self.deployed_teacher() if result['edge_support'] and context['phase_snapshot'] == self.snapshot else None
        fixed = self.options['fixed_alpha']
        if self.options['variant'] not in fusion.LEARNED and fixed is None:
            fixed = 0.0 if self.options['variant'] == 'semantic' else .5
        if not result['edge_support']:
            fixed, marked = 0.0, False
        if fixed is None:
            selected, probabilities = bandit.choose(model, result['features'], classes, explore=marked,
                epsilon=self.options['epsilon'], rng=bandit.randomizer(self.options['seed'], self.phase, context['task_id'], turn, 'action'))
        else:
            selected = next((i for i, row in enumerate(classes) if fixed in row['alphas']), None)
            probabilities = [float(i == selected) for i in range(len(classes))]
            marked = False
            if eligible:
                context['mark_status'] = 'single_class' if len(classes) == 1 else 'unsupported'
        if marked:
            context['marked'] = True
            context['intervention'] = dict(format=bandit.FORMAT, split='train', base_id=context['task_id'],
                snapshot=self.snapshot, bottom=self.bottom, phase=self.phase, actor=self.actor, bank=dynamic.digest(self.registry),
                features=result['features'], actions=[row['action'] for row in classes], probabilities=probabilities,
                chosen=classes[selected]['action'], payload_hashes=[row['payload_hash'] for row in classes],
                semantic_payload_hash=classes[0]['payload_hash'], target_reference=bandit.REFERENCE,
                interventions=1, old_probabilities=bandit.probabilities(model, result['features'], classes),
                penalty=self.options['penalty'], reference=self.options['reference'], rollout=context['ordinal'],
                horizon=self.options['horizon_steps'], marked_point=context['marked_point'], retrieval_index=index,
                point_probability=1/self.options['horizon_steps'], rollout_probability=1/self.options['rollouts'],
                seed=self.options['seed'], continuation_teacher_sha256=dynamic.digest(model),
                encoder=self.encoder_id, alignment=dynamic.digest(self.alignment), metric=dynamic.digest(self.metric),
                relation_support=fatigue.support(result), status='observed')
        output = dict(schemas=presentation(classes[selected]['ranking']) if selected is not None else [],
            note='Historical order is optional. Use current task arguments and visible evidence.', contextual=True,
            alpha=classes[selected]['alpha'] if selected is not None else 0.0, payload_classes=len(classes),
            payload_hash=classes[selected]['payload_hash'] if selected is not None else None,
            snapshot=self.snapshot, edge_support=result['edge_support'], features=result['features'], view=view,
            phase=self.phase, marked=bool(marked), variant=self.options['variant'], candidates=result['candidates'], abstention=result['abstention'])
        output['semantic_payload_hash'] = next(row['payload_hash'] for row in classes if 0.0 in row['alphas'])
        output['adoption'] = self.adoption_status()
        output['adoption'].update(applied=model is not None and fixed is None, deployed_teacher_sha256=dynamic.digest(model))
        from control import exposure
        output['exposure'] = exposure.build(result['access'], classes, selected, schemas)
        telemetry = dict(event='contextual_retrieval', task_id=context['task_id'], split=context['split'], turn=turn,
            phase=self.phase, snapshot=self.snapshot, actor=self.actor, alpha=output['alpha'], marked=output['marked'],
            edge_support=output['edge_support'], payload_hash=output['payload_hash'], semantic_payload_hash=output['semantic_payload_hash'], abstention=output['abstention'],
            teacher_depth=bandit.depth(model), metric_geometry=self.metric['geometry_id'], exposure=output['exposure'],
            relation_support=fatigue.support(result), adoption=output['adoption'])
        with (self.runtime.root/'retrievals.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(telemetry, sort_keys=True, allow_nan=False)+'\n')
        self.charge(context, 'retrieval', dict(seconds=time.monotonic()-started))
        context['selection_cache'][turn] = deepcopy(output)
        context['relation_context'] = dict(features=result['context_features'], context=deepcopy(view), requirements=view['requirements'])
        return output
    def finish(self, context, success):
        if context['split'] != 'train':
            return
        context['terminal_success'] = success
        store = identity.catalogue(context['dispatcher'].registry)
        events = []
        for left, right in zip(context['invocations'], context['invocations'][1:]):
            source, target = left.get('tool_version'), right.get('tool_version')
            if source not in store['versions'] or target not in store['versions'] or not right.get('relation_context'):
                continue
            feedback = context.get('subtask_feedback', {}).get(right.get('subtask'))
            if feedback is None and type(success) is not bool:
                continue
            events.append(dict(source=store['versions'][source], target=store['versions'][target], source_version=source,
                target_version=target, **right['relation_context'], feedback=feedback['value'] if feedback else 2*int(success)-1,
                granularity='subtask' if feedback else 'terminal', scorer=feedback['source'] if feedback else 'terminal_correctness'))
        context['pending_relations'] = events
        if type(success) is not bool:
            raise ValueError('Training publication requires actual terminal correctness')
        if 'intervention' in context:
            row = deepcopy(context['intervention'])
            if not context['actor_metered']:
                raise ValueError('Refuse teacher labels without measured whole-episode actor tokens')
            row.update(success=success, costs=deepcopy(context['costs']), cost_receipts=deepcopy(context['cost_receipts']))
            row['reward'] = bandit.reward(success, row['costs'], penalty=row['penalty'], reference=row['reference'])
            bandit.validate(row)
            self.logs.append(row)
        elif context['ordinal'] == context['exploration_rollout'] and self.options['variant'] in fusion.LEARNED and self.options['fixed_alpha'] is None:
            row = dict(format=bandit.FORMAT, split='train', base_id=context['task_id'], snapshot=self.snapshot,
                bottom=self.bottom, phase=self.phase, actor=self.actor, bank=dynamic.digest(self.registry), features=[], actions=[],
                encoder=self.encoder_id, alignment=dynamic.digest(self.alignment), metric=dynamic.digest(self.metric),
                probabilities=[], chosen=None, payload_hashes=[], interventions=0, status=context['mark_status'],
                horizon=self.options['horizon_steps'], marked_point=context['marked_point'], rollout=context['ordinal'],
                point_probability=1/self.options['horizon_steps'], rollout_probability=1/self.options['rollouts'],
                penalty=self.options['penalty'], reference=self.options['reference'], success=success,
                seed=self.options['seed'], continuation_teacher_sha256=dynamic.digest(self.deployed_teacher()), costs=deepcopy(context['costs']),
                cost_receipts=deepcopy(context['cost_receipts']), reward=bandit.reward(success, context['costs'],
                    penalty=self.options['penalty'], reference=self.options['reference']))
            bandit.validate(row)
            self.logs.append(row)
    def publish(self, contexts):
        training = [row for row in contexts if row['split'] == 'train']
        if not training:
            return
        groups = {}
        started = time.monotonic()
        for row in training:
            if row['phase_snapshot'] != self.snapshot:
                raise ValueError('Publication contains a stale phase')
            groups.setdefault(row['task_id'], []).append(row)
        if any(len(rows) != self.options['rollouts'] for rows in groups.values()):
            raise ValueError('Publish only after all G original rollouts per question finish')
        registry = deepcopy(self.pending_registry if self.pending_registry is not None else self.registry)
        for base, rows in sorted(groups.items()):
            for row in sorted(rows, key=lambda row: row['ordinal']):
                private = row['dispatcher'].registry
                registry['semantic'].update(deepcopy(private['semantic']))
                registry['structure']['nodes'].update(deepcopy(private['structure']['nodes']))
                for name, version in sorted(private['aliases'].items()):
                    if name not in registry['aliases'] or registry['aliases'][name] == version:
                        registry['aliases'][name] = version
                    elif version not in self.registry['semantic']:
                        registry['aliases'][name+'_'+version[:8]] = version
                registry = identity.attach(registry)
                for pair in identity.catalogue(private)['compatibility']:
                    registry = identity.link(registry, pair['source'], pair['target'], pair['receipt'])
            events = [deepcopy(event) for row in rows for event in row['pending_relations']]
            store = identity.catalogue(registry)
            for row in rows:
                if row.get('terminal_success') is True:
                    for call in row['invocations']:
                        if call.get('success') and call.get('tool_version') in store['versions']:
                            canonical = store['versions'][call['tool_version']]
                            store['usage'][canonical] = sorted(set(store['usage'].get(canonical, [])+[base]))
            registry['capabilities'] = store
            for event in events:
                event.update(source=store['versions'][event['source_version']], target=store['versions'][event['target_version']])
            registry['relations'] = relations.observe(registry['relations'], events, base_id=base, actor=self.actor)
        expected = dynamic.digest(self.runtime.dispatcher.registry)
        if expected != dynamic.digest(self.registry):
            raise ValueError('Shared bank mutated while its phase was frozen')
        labels = [row for row in self.logs if row['snapshot'] == self.snapshot]
        report = dict(format=FORMAT, phase=self.phase, snapshot=self.snapshot, base_questions=len(groups), adoption_protocol=bandit.PROTOCOL,
            original_rollouts=len(training), interventions=sum(row['interventions'] for row in labels), base_observations=len(labels), extra_rollouts=0,
            costs={key: sum(row['costs'][key] for row in training) for key in ('actor', 'tools', 'retrieval')},
            teacher_fitted=False, actor_update='native_searl_trainer_only', test_used=False)
        report['memory_sizes'] = dict(read_snapshot=fatigue.inventory(self.registry), pending_publication=fatigue.inventory(registry))
        report['memory_value'] = fatigue.window(labels)
        report['confidence_protocol'] = dict(horizon=self.options['horizon_steps'], penalty=self.options['penalty'],
            epsilon=self.options['epsilon'], beta=self.options['beta']/self.options['maximum_windows'], delta_max=self.options['delta_max'])
        receipts = [receipt for row in training for receipt in row['cost_receipts']]
        report['measured_usage'] = dict(retrieval_seconds=sum(row['quantities']['seconds'] for row in receipts if row['kind'] == 'retrieval'),
            tool_seconds=sum(row['quantities']['seconds'] for row in receipts if row['kind'] == 'tools'),
            actor_input_tokens=sum(row['quantities']['input_tokens'] for row in receipts if row['kind'] == 'actor'),
            actor_output_tokens=sum(row['quantities']['output_tokens'] for row in receipts if row['kind'] == 'actor'))
        boundary = windows.barrier(self.phase+1, self.runtime.config.get('epoch_batches'), self.options['frozen_batches'])
        fit_geometry = boundary and self.options['variant'] not in ('simple', 'raw')
        next_metric = metric.fit(self.metric, registry['relations'], version_compatible=lambda a, b: identity.eligible(registry, a, b)) if fit_geometry else deepcopy(self.metric)
        next_teacher = deepcopy(self.teacher)
        next_teacher_bottom = self.teacher_bottom
        if labels and (not boundary or self.options['frozen_batches'] == 1):
            if self.phase >= self.options['maximum_windows']:
                raise ValueError('Predeclared teacher window confidence budget is exhausted')
            previous = self.deployed_teacher()
            candidate, update = bandit.conservative(labels, previous, seed=self.options['seed'],
                holdout=self.options['holdout'], horizon=self.options['horizon_steps'], epsilon=self.options['epsilon'],
                beta=self.options['beta']/self.options['maximum_windows'], delta_max=self.options['delta_max'],
                chain_limit=self.options['chain_limit'], steps=self.options['teacher_steps'], minimum=self.options['minimum_labels'],
                used_holdout=[base for bottom, base in self.used_holdout if bottom == self.bottom], method=self.options['confidence_method'])
            if update['deployed']:
                next_teacher, next_teacher_bottom = candidate, self.bottom
            elif candidate is not None:
                next_teacher = candidate
                next_teacher_bottom = self.bottom if dynamic.digest(candidate) == dynamic.digest(previous) else None
            report.update(teacher_fitted=bool(update.get('proposal_sha256')), teacher_update=update,
                teacher_transfer='retained_parameters_do_not_authorize_new_bottom', teacher_validated_bottom=next_teacher_bottom)
        report.update(metric_fit_ids=next_metric['fit_ids'], metric_usable_neighborhoods=next_metric.get('usable_neighborhoods', 0),
            alignment_frozen=self.alignment['frozen'], phase_auxiliary_seconds=time.monotonic()-started,
            actor_update_ready=boundary, public_bank_published=boundary, bottom=self.bottom,
            window_role='co_evolution_barrier' if boundary else 'fixed_bottom_teacher_update_then_deployment')
        next_alignment = identity.fit(registry, self.encoder, self.alignment, steps=self.options['alignment_steps'], fit_ids=groups) if boundary and self.options['variant'] not in ('simple', 'noidentity') else deepcopy(self.alignment)
        next_whole = whole.fit(registry, self.encoder, next_alignment, self.whole, next_metric, actor=self.actor,
            neighbors=self.options['neighbors'], recall=self.options['recall'], samples_limit=self.options['whole_samples'],
            steps=self.options['alignment_steps']) if boundary and self.whole is not None else deepcopy(self.whole)
        report['whole_alignment'] = dict(trained=next_whole['trained'], fit_ids=next_whole['fit_ids'], samples=next_whole['samples']) if next_whole else None
        report['phase_auxiliary_seconds'] = time.monotonic()-started
        report['training_auxiliary_cost'] = report['phase_auxiliary_seconds']*self.options['retrieval_price']
        report['storage'] = dict(registry_json_bytes=len(json.dumps(registry, sort_keys=True, allow_nan=False).encode()),
            metric_json_bytes=len(json.dumps(next_metric, sort_keys=True, allow_nan=False).encode()),
            independent_edge_events=sum(len(row['events']) for row in registry['relations']['edges'].values()),
            reference_rows=len(next_metric['pool']), kind='serialized_size_not_peak_process_memory')
        with (self.runtime.root/'interventions.jsonl').open('a', encoding='utf-8') as stream:
            for row in labels:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        path = self.runtime.root/'phases'
        path.mkdir(exist_ok=True)
        with (path/f'phase{self.phase}.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, sort_keys=True, allow_nan=False)
        if boundary:
            self.runtime.dispatcher.replace_registry(registry, expected_sha256=expected)
            self.pending_registry = None
        else:
            self.pending_registry = registry
        self.metric, self.teacher, self.alignment = next_metric, next_teacher, next_alignment
        self.teacher_bottom = next_teacher_bottom
        self.whole = next_whole
        if 'teacher_update' in report:
            self.used_holdout.extend([[self.bottom, base] for base in report['teacher_update']['holdout_ids']])
        self.phase += 1
        self.groups, self.logs = {}, []
        self.freeze()
    def state(self):
        if self.pending_registry is not None:
            raise ValueError('Checkpoint only at the paired actor/memory window barrier')
        return dict(format=FORMAT, phase=self.phase, snapshot=self.snapshot, actor=self.actor, encoder_id=self.encoder_id, adoption_protocol=bandit.PROTOCOL,
            options=self.options, registry_hash=dynamic.digest(self.registry), alignment=deepcopy(self.alignment), whole=deepcopy(self.whole), metric=deepcopy(self.metric),
            teacher=deepcopy(self.teacher), teacher_bottom=self.teacher_bottom,
            used_holdout=deepcopy(self.used_holdout), actor_updates=deepcopy(self.actor_updates))
    def schedule(self):
        return dict(contextual=True, actor_update_ready=self.pending_registry is None,
            phase=self.phase, frozen_batches=windows.width(self.phase, self.runtime.config.get('epoch_batches'), self.options['frozen_batches']), actor=self.actor,
            actor_drift=deepcopy(self.options['actor_drift']), adoption=self.adoption_status())
    def validate_state(self, state, *, registry_hash=None, actor=None):
        if not isinstance(state, dict) or state.get('adoption_protocol') != bandit.PROTOCOL:
            raise ValueError('Adoption protocol changed; require a new frozen window')
        if state.get('format') != FORMAT or state.get('options') != self.options or state.get('encoder_id') != self.encoder_id:
            raise ValueError('Contextual checkpoint protocol or encoder mismatch')
        if state['registry_hash'] != (dynamic.digest(self.runtime.dispatcher.registry) if registry_hash is None else registry_hash):
            raise ValueError('Checkpoint bank and representation do not belong together')
        if state['actor'] != (self.runtime.policy_hash if actor is None else actor):
            raise ValueError('Teacher checkpoint and current actor snapshot differ')
        identity.validate(state['alignment'], self.encoder_id)
        if self.options['variant'] == 'whole':
            whole.validate(state.get('whole', {}), self.alignment['dimension'])
        elif state.get('whole') is not None:
            raise ValueError('Identity-only checkpoints cannot restore a whole-structure projection')
        metric.validate(state['metric'], self.encoder_id)
        bandit.validate_model(state['teacher'])
        binding = state.get('teacher_bottom')
        if binding is not None and (not isinstance(binding, str) or len(binding) != 64 or
            any(char not in '0123456789abcdef' for char in binding) or state['teacher'] is None):
            raise ValueError('Malformed teacher deployment binding')
        if not windows.barrier(state['phase'], self.runtime.config.get('epoch_batches'), self.options['frozen_batches']):
            raise ValueError('Contextual checkpoint is not at an actor-memory barrier')
        expected = dynamic.digest(dict(format=FORMAT, phase=state['phase'], actor=state['actor'], encoder=self.encoder_id,
            adoption_protocol=bandit.PROTOCOL, registry=state['registry_hash'], alignment=dynamic.digest(state['alignment']),
            whole=dynamic.digest(state.get('whole')), metric=dynamic.digest(state['metric']),
            teacher=dynamic.digest(state['teacher']), teacher_bottom=binding, options=self.options))
        if expected != state['snapshot']:
            raise ValueError('Inconsistent frozen contextual checkpoint')
        if not isinstance(state.get('used_holdout'), list) or any(not isinstance(item, list) or len(item) != 2 or any(not isinstance(value, str) or not value for value in item) for item in state['used_holdout']):
            raise ValueError('Checkpoint holdout history must retain its bottom and base identities')
        if not isinstance(state.get('actor_updates'), list) or any(not isinstance(item, dict) for item in state['actor_updates']):
            raise ValueError('Checkpoint actor history must contain identified update records')
    def restore(self, state):
        self.validate_state(state)
        self.phase, self.actor = state['phase'], state['actor']
        self.alignment, self.teacher = deepcopy(state['alignment']), deepcopy(state['teacher'])
        self.teacher_bottom = state.get('teacher_bottom')
        self.whole = deepcopy(state.get('whole'))
        self.metric, self.used_holdout = deepcopy(state['metric']), deepcopy(state['used_holdout'])
        self.actor_updates = deepcopy(state['actor_updates'])
        self.pending_registry = None
        self.groups, self.logs = {}, []
        self.freeze()
        if self.snapshot != state['snapshot']:
            raise ValueError('Inconsistent frozen contextual checkpoint')

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
