# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Identified, split-isolated method adapters for the official online rollout.'
from __future__ import annotations
import ast
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import threading
import time
import unicodedata
import uuid
from control import dynamic, edges, ranking, boundary, subtasks, teacher
from control.joint import Controller
FORMAT = "evocond_searl_runtime_v1"
ARMS = ("searl", "evocond", "teacher")
TEACHER_MODES = ("exact_policy", "frozen_controller_transfer", "semantic_fallback_on_policy_change")
def execution_context_fingerprint(config):
    'Bind teacher labels to control semantics, not paths or experiment names.'
    files = ('runtime.py', 'recovery.py', 'contracts.py', 'binding.py', 'guidance.py', 'ranking.py',
             'edges.py', 'evidence.py', 'subtasks.py', 'joint.py',
             'budget.py', 'primitive.py', 'boundary.py', 'dynamic.py')
    from control.budget import policy as structure_policy
    return dynamic.digest({'format': 'teacher_execution_context_v1', 'recovery': 'executed_ll_gl_gg',
        'value_contracts': bool(config.get('value_contracts', False)),
        'binding_strategy': config.get('binding_strategy', 'reject'),
        'binding_credit_rule': config.get('binding_credit', 'native_only'),
        'guidance_mode': config.get('guidance_mode', 'legacy'),
        'tool_construction': config.get('tool_construction', 'legacy'),
        'structure_policy': structure_policy(config.get('structure_policy')),
        'reuse_boundary_mode': boundary.validate_mode(config.get('reuse_boundary_mode', 'off')),
        'trusted_alias_tools': sorted(config.get('trusted_alias_tools', [])),
        'sample_tool_turns': bool(config.get('sample_tool_turns', True)), 'retrieval_top_k': 3,
        'rollout_contract': config.get('rollout_contract'),
        'official_source_sha256': config.get('official_source_sha256'),
        'query_prefix': config.get('query_prefix', ''), 'passage_prefix': config.get('passage_prefix', ''),
        'action_scope': 'next_retrieval_only', 'label_continuation_policy': 'fixed_semantic_after_one_override',
        'runtime_prompt_contract': 'current_tool_schemas_recovery_state_and_explicit_refs_v2',
        'sources': {name: sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in files}})
def verify_rollout_contract(config, actual):
    "Fail before generation if a frozen method's visible budget changed."
    expected = config.get('rollout_contract')
    if expected is None:
        return
    keys = {'max_turns', 'max_context', 'max_prompt', 'max_response'}
    if not isinstance(expected, dict) or set(expected) != keys:
        raise ValueError('Invalid frozen rollout contract')
    if any(type(expected[k]) is not int or expected[k] <= 0 for k in keys):
        raise ValueError('Frozen rollout budgets must be positive integers')
    if actual != expected:
        raise ValueError('Actual rollout budget differs from the frozen method contract')
def task_identity(question):
    return sha256(" ".join(unicodedata.normalize("NFKC", question).casefold().split()).encode()).hexdigest()
def reward_outcomes(extra_info, count):
    'Read final answer correctness, never infer truth from shaped total reward.'
    values = extra_info.get("answer_correctness")
    if values is None:
        return [None] * count
    if len(values) != count:
        raise ValueError("Final reward records are not aligned with rollout identities")
    outcomes = []
    for value in values:
        if value is None:
            outcomes.append(None)
            continue
        value = float(value)
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("Invalid official answer-correctness value")
        outcomes.append(True if value == 1 else False if value <= 0 else None)
    return outcomes
def schemas(dispatcher):
    return [{**deepcopy(row.get("function", row)), "tool_version": row.get("tool_version")}
            for row in dispatcher.schemas()]
def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("format") != FORMAT or config.get("arm") not in ARMS:
        raise ValueError("Unknown SEARL runtime format or arm")
    if config.get('design') is not None:
        from study import validate
        design = validate(config['design'])
        if design['arm'] != config['arm'] or design['native'] != (config.get('contextual') is None):
            raise ValueError('Runtime adapter differs from the declared study condition')
        if type(config.get('epoch_batches')) is not int or config['epoch_batches'] < 1 or config.get('actor_batches') != design['actor_batches']:
            raise ValueError('Runtime requires the frozen epoch and actor cadence')
    from control.binding import validate_strategy, validate_credit
    validate_strategy(config.get('binding_strategy', 'reject'), enabled=bool(config.get('value_contracts', False)))
    validate_credit(config.get('binding_credit', 'native_only'))
    from control.guidance import validate as validate_guidance
    validate_guidance(config.get('guidance_mode', 'legacy'))
    from control.primitive import validate_mode
    from control.budget import policy as structure_policy
    validate_mode(config.get('tool_construction', 'legacy'))
    structure_policy(config.get('structure_policy'))
    boundary.validate_mode(config.get('reuse_boundary_mode', 'off'))
    for key in ("run_id", "root", "registry_path", "policy_hash"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Runtime requires {key}")
    if not re.fullmatch(r"[0-9a-f]{64}", config["policy_hash"]):
        raise ValueError("Runtime policy_hash must identify initial policy weights")
    root = Path(config["root"]).resolve()
    if not Path(config["registry_path"]).resolve().is_relative_to(root):
        raise ValueError("Each arm needs a private registry inside its runtime root")
    if config.get('contextual') is not None:
        from control.stage import settings
        settings(config['contextual'])
        if config.get('value_contracts') or config.get('collection'):
            raise ValueError('Contextual retrieval uses the common native executor, not rebinding or branch collection')
        for key in ('encoder_path', 'encoder_id'):
            if not config.get(key):
                raise ValueError('Contextual retrieval requires a pinned frozen encoder')
    elif config["arm"] == "teacher":
        for key in ("teacher_path", "alignment_path", "encoder_path", "encoder_id"):
            if not isinstance(config.get(key), str) or not config[key]:
                raise ValueError(f"Teacher runtime requires fitted artifact: {key}")
        if config.get("teacher_policy_mode") not in TEACHER_MODES:
            raise ValueError("Declare an explicit teacher policy scope")
    return config
def plan_from_searl(question, text, epoch=0):
    "Translate the policy's official ST/DAG syntax, without inventing goals."
    tasks = {}
    for match in re.finditer(r"^\s*##\s*(ST\d+):?([^\n]*)\n([\s\S]*?)(?=\n\s*##|\Z)", text, re.M):
        tasks[match[1]] = {"id": match[1], "goal": (match[2]+"\n"+match[3]).strip(),
                           "depends_on": [], "evidence_required": "Observed successful tool execution for this subtask; not final-answer truth"}
    match = re.search(r"##DAG_LIST[^\n]*\n([\s\S]*?)(?=\n##|\Z)", text)
    if not tasks or match is None:
        raise ValueError("Policy did not supply a subtask DAG")
    raw = re.sub(r"^```[^\n]*\n|\n```$", "", match[1].strip())
    raw = re.sub(r'(?<![\w\"\x27])\b(ST\d+)\b(?![\w\"\x27])', r'"\1"', raw)
    edges = ast.literal_eval(raw)
    if not isinstance(edges, list):
        raise ValueError("Policy DAG must be a list")
    for edge in edges:
        nodes = [edge] if isinstance(edge, str) else edge
        if not isinstance(nodes, (list, tuple)) or len(nodes) not in (1, 2) or any(n not in tasks for n in nodes):
            raise ValueError("Policy DAG refers to an unknown subtask")
        if len(nodes) == 2 and nodes[0] not in tasks[nodes[1]]["depends_on"]:
            tasks[nodes[1]]["depends_on"].append(nodes[0])
    return subtasks.make_plan(question, list(tasks.values()), epoch=epoch)
def recovery(history):
    'Use verified runtime receipts rather than interpreting arbitrary prose.'
    visible = []
    for row in history:
        if row["role"] == "tool":
            if row.get('execution_success') is None:
                continue
            visible.append({"role": "tool", "name": "python", "content":
                            ("Tool call success: " if row.get("execution_success") is True else "Tool call failed: ")+row["content"]})
        else:
            visible.append(row)
    query = next((r["content"] for r in reversed(history) if r["role"] == "user"), "")
    return Controller().decide(query, visible, [])
class Runtime:
    def __init__(self, config, dispatcher, *, encoder=None):
        from control.binding import validate_strategy, validate_credit
        validate_strategy(config.get('binding_strategy', 'reject'), enabled=bool(config.get('value_contracts', False)))
        validate_credit(config.get('binding_credit', 'native_only'))
        from control.guidance import validate as validate_guidance
        validate_guidance(config.get('guidance_mode', 'legacy'))
        from control.primitive import validate_mode
        from control.budget import policy as structure_policy
        validate_mode(config.get('tool_construction', 'legacy'))
        self.structure_policy = structure_policy(config.get('structure_policy'))
        self.reuse_boundary_mode = boundary.validate_mode(config.get('reuse_boundary_mode', 'off'))
        if config['arm'] == 'teacher' and not config.get('contextual') and config.get('teacher_policy_mode') not in TEACHER_MODES:
            raise ValueError("Declare an explicit teacher policy scope")
        self.config = deepcopy(config)
        self.arm, self.root = config["arm"], Path(config["root"]).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.dispatcher, self.encoder = dispatcher, encoder
        self.lock = threading.RLock()
        self.contexts = {}
        self.policy_step = 0
        self.native_batches = 0
        self.policy_hash = config["policy_hash"]
        self.alignment = self.teacher = None
        self.engine = None
        if config.get('contextual') is not None:
            self.encoder = encoder or ranking.SentenceTransformerEncoder(config['encoder_path'], encoder_id=config['encoder_id'],
                device=config.get('encoder_device', 'cpu'), query_prefix=config.get('query_prefix', ''), passage_prefix=config.get('passage_prefix', ''))
        elif self.arm == "teacher":
            self.teacher = json.loads(Path(config["teacher_path"]).read_text(encoding="utf-8"))
            self.alignment = json.loads(Path(config["alignment_path"]).read_text(encoding="utf-8"))
            teacher.validate(self.teacher)
            ranking.validate_alignment(self.alignment, encoder_id=config["encoder_id"],
                                                 structure_policy=self.structure_policy, reuse_boundary_mode=self.reuse_boundary_mode)
            if self.teacher["alignment_hash"] != dynamic.digest(self.alignment):
                raise ValueError("Teacher and alignment artifacts are not paired")
            if self.teacher["policy_hash"] != config["policy_hash"]:
                raise ValueError("Teacher provenance does not match initial Qwen policy")
            if self.teacher.get('execution_context_hash') != execution_context_fingerprint(config):
                raise ValueError('Teacher labels were collected under a different or unrecorded execution context')
            self.encoder = encoder or ranking.SentenceTransformerEncoder(
                config["encoder_path"], encoder_id=config["encoder_id"], device=config.get("encoder_device", "cpu"),
                query_prefix=config.get("query_prefix", ""), passage_prefix=config.get("passage_prefix", ""))
        elif config.get('collection'):
            if self.arm != 'searl':
                raise ValueError('Teacher collection is a separate SEARL bootstrap stage')
            self.alignment = json.loads(Path(config['alignment_path']).read_text(encoding='utf-8'))
            ranking.validate_alignment(self.alignment, encoder_id=config['encoder_id'],
                                                 structure_policy=self.structure_policy, reuse_boundary_mode=self.reuse_boundary_mode)
            self.encoder = encoder or ranking.SentenceTransformerEncoder(
                config['encoder_path'], encoder_id=config['encoder_id'], device=config.get('encoder_device', 'cpu'),
                query_prefix=config.get('query_prefix', ''), passage_prefix=config.get('passage_prefix', ''))
        elif config.get('bootstrap_warmup'):
            if self.arm != 'searl':
                raise ValueError('Warmup must be an explicitly identified SEARL collection stage')
            self.encoder = encoder or ranking.SentenceTransformerEncoder(
                config['encoder_path'], encoder_id=config['encoder_id'], device=config.get('encoder_device', 'cpu'),
                query_prefix=config.get('query_prefix', ''), passage_prefix=config.get('passage_prefix', ''))
        identity = {"format": FORMAT, "run_id": config["run_id"], "arm": self.arm,
                    "config_hash": dynamic.digest(config)}
        identity_path = self.root / "runtime_identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Runtime root already belongs to another experiment")
        if not identity_path.exists():
            with identity_path.open("x", encoding="utf-8") as stream:
                json.dump(identity, stream, sort_keys=True)
        if config.get('contextual') is not None:
            from control.stage import Engine
            self.engine = Engine(self)
    def set_policy_step(self, step, policy_drift=None, *, restoring=False):
        if type(step) is not int or step < 0:
            raise ValueError("Invalid actual optimizer step")
        if self.arm == "teacher" and not self.engine and step and self.config["teacher_policy_mode"] == "exact_policy":
            raise ValueError("Qwen changed: refresh paired teacher or declare transfer/semantic fallback")
        if self.engine and self.contexts:
            raise ValueError('Actor updates require the completed-rollout barrier')
        if self.engine and self.engine.pending_registry is not None:
            raise ValueError('Actor must remain frozen until the configured window barrier')
        if not restoring and self.schedule() is not None and not self.schedule()['actor_update_ready']:
            raise ValueError('Actor must remain frozen until the configured window barrier')
        self.policy_step = step
        self.policy_hash = self.config["policy_hash"] if step == 0 else dynamic.digest(
            {"initial_weights": self.config["policy_hash"], "run_id": self.config["run_id"], "optimizer_step": step})
        if self.engine and not restoring:
            return self.engine.actor_changed(self.policy_hash, policy_drift)
    def begin(self, question, plan_text, *, split, episode_id=None):
        if split not in ("train", "validation", "test") or not isinstance(question, str) or not question.strip():
            raise ValueError("Rollout needs visible question and explicit split")
        identity = episode_id or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identity) or identity in self.contexts:
            raise ValueError("Invalid or reused rollout identity")
        dispatcher = self.dispatcher
        private = split != 'train' or bool(self.config.get('collection')) or self.engine is not None
        if private:
            dispatcher = dispatcher.fork(self.root / "ephemeral" / identity / "registry.json")
        try:
            plan, error = plan_from_searl(question, plan_text), None
        except (ValueError, SyntaxError, TypeError) as exc:
            plan, error = None, str(exc)
        self.contexts[identity] = {"split": split, "question": question, "task_id": task_identity(question),
            "history": [{"role": "user", "content": question}], "plan": plan, "plan_error": error,
            "dispatcher": dispatcher, "invocations": [], "selected_versions": {}, "last_subtask": None,
            "completed": False, "execution_complete": False, "policy_step": self.policy_step,
            "private": private, "collection_branch": False, "calls": [], "assistant_turn": 0,
            "binding_audits": [], "started": time.monotonic()}
        if self.engine:
            self.engine.begin(self.contexts[identity])
        elif self.arm != 'searl' or self.config.get('collection'):
            from control.recovery import new_state
            self.contexts[identity]['recovery'] = new_state(question, plan)
        return identity
    def context(self, identity):
        if identity not in self.contexts:
            raise ValueError("Unknown runtime episode; context cannot be inferred across trajectories")
        return self.contexts[identity]
    def revise(self, identity, question, plan_text=None):
        context = self.context(identity)
        if 'recovery' in context:
            from control.recovery import revise_goal, submit_plan
            state = revise_goal(context['recovery'], question)
            if plan_text is not None:
                state = submit_plan(state, {'goal_epoch': state['goal_epoch'], 'base_revision': state['revision'],
                    'question_sha256': state['question_sha256'], 'searl_plan': plan_text})
            context['recovery'], context['plan'] = state, state['plan']
        else:
            context["plan"] = plan_from_searl(question, plan_text, (context["plan"] or {}).get("epoch", 0)+1) if plan_text else None
        context["question"] = question
        context["history"].append({"role": "user", "content": question})
        context["invocations"], context["selected_versions"] = [], {}
        context['calls'] = []
        context.pop('subtask_feedback', None)
        context['last_subtask'], context['plan_error'] = None, None
        if self.engine:
            context['selection_cache'] = {}
            context.pop('relation_context', None)
    def assistant(self, identity, response):
        context = self.context(identity)
        context['assistant_turn'] = context.get('assistant_turn', 0)+1
        context["history"].append({"role": "assistant", "content": response})
        match = re.search(r"<subtask>\s*([^<\n]+)", response)
        label = match[1].strip().replace('ST_', 'ST') if match else ''
        identities = sorted((context['plan'] or {}).get('tasks', {}), key=len, reverse=True)
        context['last_subtask'] = next((identity for identity in identities
                                       if label == identity or label.startswith(identity+':')), None)
    def select(self, identity, *, top_k=3):
        context = self.context(identity)
        dispatcher = context["dispatcher"]
        dispatcher.reload()
        tools = schemas(dispatcher)
        if self.engine:
            result = self.engine.select(context, tools)
            context['selected_versions'] = {row['name']: row.get('tool_version') for row in result['schemas']}
            return {**result, 'policy_step': self.policy_step, 'policy_hash': self.policy_hash,
                    'registry_hash': dynamic.digest(dispatcher.registry), 'plan_error': context['plan_error']}
        decision = recovery(context["history"])
        note = "" if self.arm == "searl" and not self.config.get('collection') else decision.note
        from control.evidence import REFERENCE_NOTE
        note += ' '+REFERENCE_NOTE
        from control.primitive import creation_note
        construction_note = creation_note(self.config.get('tool_construction', 'legacy'))
        if construction_note:
            note += ' '+construction_note
        ranked = None
        teacher_authority = None
        if self.arm == "teacher" or self.config.get('collection'):
            ready = subtasks.ready(context["plan"]) if context["plan"] else []
            query = ready[0]["goal"] if ready else context["question"]
            previous = next((r for r in reversed(context["invocations"]) if r.get("tool_version")), {})
            last_user = max(i for i, r in enumerate(context["history"]) if r["role"] == "user")
            failures = sum(r.get("execution_success") is False for r in context["history"][last_user+1:] if r["role"] == "tool")
            fixed_alpha = self.config.get('fixed_alpha') if self.arm == 'teacher' else 0.0
            if context.get('collection_branch'):
                fixed_alpha = context.pop('next_alpha', 0.0)
                context['interventions'] = context.get('interventions', 0)+int(not context.get('first_retrieval_done', False))
                context['first_retrieval_done'] = True
            if self.teacher:
                changed = self.policy_hash != self.teacher['policy_hash']
                scope = self.config['teacher_policy_mode']
                fallback = changed and scope == 'semantic_fallback_on_policy_change'
                teacher_authority = {'mode': scope, 'policy_changed': changed,
                    'fitted_policy_hash': self.teacher['policy_hash'], 'runtime_policy_identity': self.policy_hash,
                    'identity_kind': 'initial_weight_hash_then_run_optimizer_step_lineage_not_updated_weight_hash',
                    'requested_fixed_alpha': fixed_alpha, 'semantic_fallback': fallback,
                    'reason': 'policy_changed_unrecalibrated' if fallback else 'declared_transfer' if changed else 'paired_initial_policy'}
                if fallback:
                    fixed_alpha = 0.0
            ranked = ranking.rank(dispatcher.registry, self.encoder, self.alignment, self.teacher,
                query=query, top_k=top_k, fixed_alpha=fixed_alpha, policy_hash=self.teacher["policy_hash"] if self.teacher else None, previous_tool=previous.get("name"),
                previous_version=previous.get("tool_version"), turn=len(context["invocations"]),
                structure_policy=self.structure_policy,
                reuse_boundary_mode=self.reuse_boundary_mode, current_user_text=context['question'],
                goal_changed=decision.scale == "gg", prior_failures=failures,
                completed_subtasks=sum(t["status"] == "completed" for t in (context["plan"] or {}).get("tasks", {}).values()))
            from control.guidance import present
            ranked['tools'], guidance = present(ranked, self.config.get('guidance_mode', 'legacy'), top_k=top_k)
            context.setdefault('guidance_presentations', []).append(guidance)
            context.setdefault('structure_presentations', []).append(deepcopy(ranked['structure_receipt']))
            if teacher_authority is not None:
                teacher_authority['effective_alpha'] = ranked['teacher']['structure_weight']
                context.setdefault('teacher_authority', []).append(deepcopy(teacher_authority))
            names = [r["name"] for r in ranked["tools"]]
            tools = sorted((s for s in tools if s["name"] in names), key=lambda s: names.index(s["name"]))
            note += " Rebind all arguments to the latest visible requirements. Historical tool order is optional, not a prerequisite."
        if 'recovery' in context:
            from control.recovery import control_schema
            tools.append(control_schema(context['recovery']))
            state = context['recovery']
            note += ' Runtime recovery: '+json.dumps({'scale': state['last_scale'], 'pending': state['pending'],
                'goal_epoch': state['goal_epoch'], 'revision': state['revision'], 'question_sha256': state['question_sha256']})
        context["selected_versions"] = {s["name"]: s.get("tool_version") for s in tools}
        return {"schemas": tools, "note": note, "scale": decision.scale, "action": decision.action,
                "plan_error": context["plan_error"], "teacher": ranked["teacher"] if ranked else None,
                "policy_step": self.policy_step, "policy_hash": self.policy_hash,
                "teacher_transfer": bool(self.teacher and self.policy_hash != self.teacher["policy_hash"]),
                "teacher_authority": teacher_authority,
                "registry_hash": dynamic.digest(dispatcher.registry),
                "tool_construction": self.config.get('tool_construction', 'legacy'),
                "structure_receipt": deepcopy(ranked['structure_receipt']) if ranked else None,
                "collection_branch": context.get('collection_branch', False),
                "teacher_bootstrap": bool(self.config.get('collection'))}
    def capture_continuation(self, identity):
        context = self.context(identity)
        if not self.config.get('collection') or context['split'] != 'train' or not context['private'] or self.policy_step != 0:
            raise ValueError('Collect only private training prefixes under one unchanged Qwen policy')
        ready = subtasks.ready(context['plan']) if context['plan'] else []
        previous = next((r for r in reversed(context['invocations']) if r.get('tool_version')), {})
        decision = recovery(context['history'])
        last_user = max(i for i, row in enumerate(context['history']) if row['role'] == 'user')
        view = ranking.describe(context['dispatcher'].registry, self.encoder, self.alignment,
            structure_policy=self.structure_policy,
            reuse_boundary_mode=self.reuse_boundary_mode, current_user_text=context['question'],
            query=ready[0]['goal'] if ready else context['question'], previous_tool=previous.get('name'),
            previous_version=previous.get('tool_version'), turn=len(context['invocations']),
            goal_changed=decision.scale == 'gg', prior_failures=sum(row.get('execution_success') is False for row in context['history'][last_user+1:] if row['role'] == 'tool'),
            completed_subtasks=sum(row['status'] == 'completed' for row in (context['plan'] or {}).get('tasks', {}).values()))
        return {'format': 'searl_runtime_continuation_v1', 'context': deepcopy({k: v for k, v in context.items() if k != 'dispatcher'}),
                'registry': context['dispatcher'].registry, 'view': view, 'alignment': deepcopy(self.alignment),
                'policy_hash': self.policy_hash, 'config_hash': dynamic.digest(self.config),
                'execution_context_hash': execution_context_fingerprint(self.config)}
    def resume_continuation(self, snapshot, alpha, seed):
        if not self.config.get('collection') or self.policy_step != 0 or snapshot.get('policy_hash') != self.policy_hash:
            raise ValueError('Continuation policy differs from captured Qwen')
        if snapshot.get('config_hash') != dynamic.digest(self.config) or snapshot.get('format') != 'searl_runtime_continuation_v1':
            raise ValueError('Continuation runtime identity mismatch')
        if alpha not in teacher.ACTIONS or type(seed) is not int or seed < 0:
            raise ValueError('Unsupported paired action or seed')
        context = deepcopy(snapshot['context'])
        if context['split'] != 'train' or not context['private']:
            raise ValueError('Continuation cannot use a held-out or shared mutable prefix')
        identity = uuid.uuid4().hex
        context['dispatcher'] = type(self.dispatcher)(self.root/'ephemeral'/identity/'registry.json', self.dispatcher.backend,
                                                    initial_registry=deepcopy(snapshot['registry']))
        context.update(collection_branch=True, next_alpha=float(alpha), interventions=0, first_retrieval_done=False,
                       branch_seed=seed, branch_alpha=float(alpha), completed=False, execution_complete=False)
        self.contexts[identity] = context
        return identity
    def discard(self, identities):
        for identity in identities:
            if not self.context(identity)['private']:
                raise ValueError('Only private branch contexts may be discarded without final training feedback')
        for identity in identities:
            del self.contexts[identity]
    def prepare_arguments(self, identity, name, arguments, schemas):
        from control.binding import prepare
        context = self.context(identity)
        if self.config.get('value_contracts', False):
            effective, audit = prepare(context['history'], schemas, name, arguments,
                strategy=self.config.get('binding_strategy', 'reject'),
                trusted_alias_tools=self.config.get('trusted_alias_tools', ()))
        else:
            effective = deepcopy(arguments)
            audit = {'strategy': 'disabled', 'tool': name, 'changed': False, 'rewrites': [],
                     'before': {'decision': 'unknown'}, 'after': {'decision': 'unknown', 'blocked': False},
                     'covered_fields': [], 'unverified_fields': sorted(arguments),
                     'semantic_correctness_certified': False}
        if audit['changed'] and context.get('assistant_turn', 0) <= 0:
            raise ValueError('A corrected action needs its actual sampled assistant turn for credit masking')
        audit.update(audit_id=len(context['binding_audits']), assistant_turn=context.get('assistant_turn', 0),
                     executed=False, success=None)
        audit['policy_credit_rule'] = self.config.get('binding_credit', 'native_only')
        context['binding_audits'].append(audit)
        return effective, deepcopy(audit)
    def binding_mask(self, identity):
        if self.config.get('binding_credit', 'native_only') == 'composite':
            return []
        return sorted({2*row['assistant_turn'] for row in self.context(identity)['binding_audits']
                       if row['changed'] and row['executed']})
    def check(self, identity, name, arguments, schemas):
        context = self.context(identity)
        if 'recovery' in context:
            from control.recovery import check_call
            checked = check_call(context['recovery'], name, arguments, subtask=context['last_subtask'],
                                 tool_version=context['selected_versions'].get(name))
            if checked['blocked']:
                return {'decision': 'conflict', **checked}
        if not self.config.get("value_contracts", False):
            return {"decision": "unknown", "blocked": False, "reason": "declared_disabled"}
        from control.contracts import check_call
        return check_call(self.context(identity)["history"], schemas, name, arguments,
                          trusted_alias_tools=self.config.get("trusted_alias_tools", ()))
    def submit_revised_plan(self, identity, payload):
        from control.recovery import submit_plan
        context = self.context(identity)
        if 'recovery' not in context:
            raise ValueError('Plan recovery is not enabled for this arm')
        state = submit_plan(context['recovery'], payload)
        context['recovery'], context['plan'] = state, state['plan']
        context['plan_error'], context['last_subtask'] = None, None
        return {'accepted': True, 'plan': context['plan'], 'revision': state['revision']}
    def observe(self, identity, name, result, *, success, version=None, arguments=None, execution_result=None, bindings=(),
                executed=True, recovery_name=None, recovery_arguments=None, binding_audit=None):
        from control.evidence import call_record
        context = self.context(identity)
        if binding_audit is not None:
            recorded = context['binding_audits'][binding_audit['audit_id']]
            if recorded != binding_audit:
                raise ValueError('Binding audit does not belong to this proposed call')
            recorded.update(executed=bool(executed), success=bool(success) if executed else None)
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, allow_nan=False)
        context["history"].append({"role": "tool", "name": name, "content": text,
                                   "execution_success": success if executed else None, "tool_version": version})
        if not executed:
            return None
        context["invocations"].append({"name": name, "tool_version": version, "success": success, 'subtask': context['last_subtask'],
                                       'relation_context': deepcopy(context.get('relation_context'))})
        ledger_result = result if execution_result is None else execution_result
        available = len(json.dumps(ledger_result, ensure_ascii=False, allow_nan=False).encode()) <= 60000
        record = call_record(uuid.uuid4().hex, name, version, bool(success), arguments or {},
                             ledger_result if available else None, bindings, result_available=available)
        context['calls'].append(record)
        subtask = context["last_subtask"]
        if success and version and context["plan"] and subtask in {t["id"] for t in subtasks.ready(context["plan"])}:
            context["plan"] = subtasks.complete(context["plan"], subtask, observed_tool_version=version,
                evidence=text, epoch=context["plan"]["epoch"])
        if 'recovery' in context and executed:
            from control.recovery import record_call
            context['recovery'] = record_call(context['recovery'], recovery_name or name,
                recovery_arguments if recovery_arguments is not None else arguments or {}, success=bool(success),
                subtask=subtask, tool_version=version, plan=context['plan'])
            context['plan'] = context['recovery']['plan']
        return record
    def subtask_feedback(self, identity, subtask, value, *, source):
        'Accept explicit scorer feedback, never execution success or tool prose.'
        context = self.context(identity)
        if context['split'] != 'train' or context['completed'] or subtask not in (context['plan'] or {}).get('tasks', {}):
            raise ValueError('Subtask correctness requires an active training plan')
        if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError('Subtask scorer feedback must lie in the signed unit interval')
        if not isinstance(source, str) or not source.strip():
            raise ValueError('Subtask correctness requires identified scorer provenance')
        receipts = context.setdefault('subtask_feedback', {})
        receipt = dict(value=float(value), source=source, epoch=context['plan']['epoch'])
        if subtask in receipts and receipts[subtask] != receipt:
            raise ValueError('Conflicting scorer feedback for one subtask')
        receipts[subtask] = receipt
    def finish(self, identity, *, task_success=None):
        'Only direct transition failures are attributed; unknown stays unknown.'
        context = self.context(identity)
        if context["completed"]:
            return
        if self.engine:
            self.engine.finish(context, task_success)
        context['terminal_success'] = task_success
        from control.outcomes import measured
        outcome = measured(context, episode=identity, arm=self.arm, actor=self.policy_hash,
            bank=dynamic.digest(self.dispatcher.registry), elapsed=time.monotonic()-context['started'],
            variant=self.engine.options['variant'] if self.engine else None)
        outcome['runtime_sha256'] = dynamic.digest(self.config)
        with self.lock, (self.root/'outcomes.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(outcome, sort_keys=True, allow_nan=False)+'\n')
        # Append-only evidence survives release_completed(); it does not mutate memory.
        from control.binding import FORMAT as binding_format
        with self.lock, (self.root/'binding_events.jsonl').open('a', encoding='utf-8') as stream:
            json.dump({'format': binding_format, 'episode_id': identity, 'task_id': context['task_id'],
                       'split': context['split'], 'arm': self.arm, 'policy_step': context['policy_step'],
                       'task_success': task_success, 'collection_branch': context.get('collection_branch', False),
                       'guidance_mode': self.config.get('guidance_mode', 'legacy'),
                       'guidance_presentations': context.get('guidance_presentations', []),
                       'tool_construction': self.config.get('tool_construction', 'legacy'),
                       'structure_policy': self.structure_policy,
                       'reuse_boundary_mode': self.reuse_boundary_mode,
                       'teacher_authority': context.get('teacher_authority', []),
                       'structure_presentations': context.get('structure_presentations', []),
                       'audits': context['binding_audits']}, stream, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        if context["split"] == "train" and not context.get('private') and (self.arm == "teacher" or self.config.get('bootstrap_warmup')):
            dispatcher = context["dispatcher"]
            vector = self.encoder.encode_query(context["question"]) if hasattr(self.encoder, "encode_query") else self.encoder.encode(context["question"])
            def feedback(registry):
                registry = edges.advance_training_task(registry, task_id=context["task_id"], split="train")
                from control.evidence import transition_feedback
                calls = context['calls']
                grouped = {}
                for source, target in zip(calls, calls[1:]):
                    if not source.get('tool_version') or not target.get('tool_version'):
                        continue
                    key = (source["tool_version"], target["tool_version"])
                    grouped.setdefault(key, {'source': source, 'target': target, 'feedback': []})['feedback'].append(transition_feedback(source, target))
                for group in grouped.values():
                    source, target = group['source'], group['target']
                    reuse_context = (boundary.capture(registry, source['tool_version'], target['tool_version'],
                        context['question'], trusted_alias_tools=self.config.get('trusted_alias_tools', ()))
                        if self.reuse_boundary_mode == 'explicit' else None)
                    registry = edges.observe_policy_transition(registry, source_name=source["name"], target_name=target["name"],
                        source_version=source["tool_version"], target_version=target["tool_version"], task_id=context["task_id"], split="train",
                        context_vector=vector, encoder_id=self.config["encoder_id"], feedback=group['feedback'],
                        policy_step=context['policy_step'], rollout_id=identity,
                        task_success=task_success, reuse_boundary=reuse_context)
                return registry
            with self.lock:
                dispatcher.transform_registry(feedback)
        context["completed"] = True
        context["execution_complete"] = True
    def release_completed(self):
        'Call only at a coordinator barrier after all rollout workers return.'
        if any(not row["completed"] for row in self.contexts.values()):
            raise ValueError("Cannot release contexts while rollouts remain in flight")
        if self.engine:
            self.engine.publish(list(self.contexts.values()))
        elif self.config.get('design', {}).get('native') and any(row['split'] == 'train' for row in self.contexts.values()):
            self.native_batches += 1
        self.contexts.clear()
    def schedule(self):
        if self.engine:
            return self.engine.schedule()
        count = self.config.get('actor_batches', 1)
        if count == 1:
            return None
        from control import windows, drift
        epoch = self.config.get('epoch_batches')
        return dict(contextual=False, actor_update_ready=windows.barrier(self.native_batches, epoch, count),
            phase=self.native_batches, frozen_batches=windows.width(self.native_batches, epoch, count),
            actor=self.policy_hash, actor_drift=drift.options({}))
    def charge(self, identity, kind, quantities):
        'Record measured costs without changing the native actor reward objective.'
        if self.engine:
            self.engine.charge(self.context(identity), kind, quantities)
    def visible_fields(self, identity, fields):
        'Supply explicit visible constraints; unknown fields stay missing.'
        if not self.engine or not isinstance(fields, dict) or set(fields)-{'constraints', 'available', 'requirements', 'slots', 'budget'}:
            raise ValueError('Invalid contextual visible-field update')
        context = self.context(identity)
        context['visible_fields'] = deepcopy(fields)
        context['selection_cache'] = {}
    def snapshot(self):
        if any(not row["completed"] for row in self.contexts.values()):
            raise ValueError("Finish in-flight rollout contexts before checkpoint")
        result = {"format": FORMAT, "config_hash": dynamic.digest(self.config), "policy_step": self.policy_step,
                  "registry": self.dispatcher.registry, "run_id": self.config["run_id"], "arm": self.arm}
        if self.config.get('design', {}).get('native'):
            if self.schedule() is not None and not self.schedule()['actor_update_ready']:
                raise ValueError('Native cadence checkpoint must be at an actor barrier')
            result['native_batches'] = self.native_batches
        if self.engine:
            result['contextual'] = self.engine.state()
        return result
    def restore(self, state):
        if state.get("format") != FORMAT or state.get("config_hash") != dynamic.digest(self.config):
            raise ValueError("Runtime checkpoint differs from experiment identity")
        dynamic.validate(state["registry"])
        self.dispatcher.reload()
        self.dispatcher.replace_registry(state["registry"], expected_sha256=dynamic.digest(self.dispatcher.registry))
        self.contexts.clear()
        if self.config.get('design', {}).get('native'):
            from control.windows import barrier
            phase = state.get('native_batches')
            if not barrier(phase, self.config.get('epoch_batches'), self.config.get('actor_batches', 1)):
                raise ValueError('Native cadence checkpoint is inside an actor window')
            self.native_batches = phase
        self.set_policy_step(state["policy_step"], restoring=True)
        if self.engine:
            self.engine.restore(state['contextual'])

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
