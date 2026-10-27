# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Freeze and launch isolated three-arm experiments using upstream online GIGPO.'
from __future__ import annotations
import argparse
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit
from control.isolation import canonical_hash, validate_deployment
from control.runtime import load_config as load_runtime
from control.runtime import execution_context_fingerprint
from splits import audit, file_digest, rows
from convert import validate_official_row, question_hash
from control.contracts import publish, strict_json
from namespace import install as install_packages
from source import reviewed
install_packages()
FORMAT = "evocond_searl_experiment_v1"
PIN = "b535d6199602bd9d747fc260bd29c77764799358"
ARMS = ("searl", "evocond", "teacher")
HERE = Path(__file__).resolve().parent
DEFAULTS = dict(gpus=8, batch_size=256, rollouts=8, epochs=1, max_prompt=4096,
                max_response=2048, max_context=32768, max_turns=6, save_freq=5, validation_freq=5,
                learning_rate=1e-6, seed=42, tail_policy='error', temperature=.7, top_p=.98, repetition_penalty=1.05)
def read(path):
    return strict_json(Path(path).read_text(encoding='utf-8-sig'), max_text=64*1024*1024)
def write_new(path, value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    publish(path, payload)
def directory_digest(path, *, model=False):
    path = Path(path).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("Expected an artifact directory")
    if model:
        config = read(path / "config.json")
        if config.get("model_type") != "qwen3":
            raise ValueError("This protocol requires a Qwen3 backbone")
        files = sorted(p for p in path.iterdir() if p.is_file() and
                       (p.suffix == ".safetensors" or p.name in
                        ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
                         "vocab.json", "merges.txt", "model.safetensors.index.json")))
        weights = [p for p in files if p.suffix == ".safetensors"]
        tokenizer = (path / "tokenizer.json").is_file() or all((path / n).is_file() for n in ("vocab.json", "merges.txt"))
        if not weights or not (path / "tokenizer_config.json").is_file() or not tokenizer:
            raise ValueError("Missing Qwen weights or tokenizer configuration")
        index = path / "model.safetensors.index.json"
        if index.exists():
            names = set(read(index).get("weight_map", {}).values())
            if not names or not names <= {p.name for p in weights}:
                raise ValueError("Incomplete model shard index")
    else:
        files = sorted(p for p in path.rglob("*") if p.is_file() and
                       (p.suffix in (".py", ".sh", ".yaml", ".yml", ".pydata", ".jinja2") or p.relative_to(path).as_posix() == 'configs/packages.json') and
                       not set(p.relative_to(path).parts) & {".git", "__pycache__", "runs", "outputs", "tests", "baselines", "archives", "assets"}
                       and not any(part.startswith(("tmp", ".evocond-backup-")) for part in p.relative_to(path).parts))
    if not files:
        raise ValueError("Empty artifact directory")
    inventory = {p.relative_to(path).as_posix(): file_digest(p) for p in files}
    return {"sha256": canonical_hash(inventory), "files": inventory}
def settings(value, *, collection=False):
    if not isinstance(value, dict):
        raise ValueError('Experiment budgets must be an object')
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown budget fields: {sorted(unknown)}")
    result = {**DEFAULTS, **value}
    for key in set(DEFAULTS) - {"learning_rate", "tail_policy", "temperature", "top_p", "repetition_penalty"}:
        if type(result[key]) is not int or result[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid budget: {key}")
    if type(result["learning_rate"]) not in (float, int) or not 0 < result["learning_rate"] < 1:
        raise ValueError("Invalid learning rate")
    if result['tail_policy'] not in ('error', 'keep'):
        raise ValueError('Declare error or keep for the final training batch')
    for key, low, high in (('temperature', 0., 2.), ('top_p', 0., 1.), ('repetition_penalty', 0., 2.)):
        if type(result[key]) not in (int, float) or not low < result[key] <= high:
            raise ValueError('Invalid generation setting: '+key)
    if not collection and result["rollouts"] < 2:
        raise ValueError("Group-relative training requires at least two rollouts")
    if result["batch_size"] % result["gpus"]:
        raise ValueError("Training batch size must be divisible by GPU count")
    if result["max_context"] < result["max_prompt"] + result["max_response"]:
        raise ValueError("Context budget must fit initial prompt and one response")
    return result
def resolved(value, base=None):
    path = Path(value)
    if not path.is_absolute() and base is not None:
        path = Path(base)/path
    return str(path.resolve(strict=True))
def window_budget(budget, options, training_rows):
    from control.windows import layout
    count = options['frozen_batches']
    schedule = layout(training_rows, budget['batch_size'], count,
        tail=budget.get('tail_policy', 'error'), epochs=budget['epochs'])
    if budget['save_freq'] % count or budget['validation_freq'] % count:
        raise ValueError('Checkpoint and validation frequencies must land on frozen-window barriers')
    if options['horizon_steps'] != budget['max_turns']+1:
        raise ValueError('Uniform marking horizon must equal the actual native retrieval cap')
    return schedule

def arms(config):
    if config.get('design') is None:
        return ARMS
    from study import validate
    return (validate(config['design'])['arm'],)
def validate_teacher_splits(teacher, alignment, identities):
    fitting, calibration = set(teacher["train_ids"]), set(teacher["calibration_ids"])
    if not (fitting | calibration) <= identities["train"]:
        raise ValueError("Teacher contains questions outside this experiment's training split")
    if fitting != set(alignment["train_ids"]) or calibration != set(alignment["calibration_ids"]):
        raise ValueError("Teacher and alignment fit/calibration identities differ")
    for split in ("validation", "test"):
        if set(teacher[split+"_ids"]) != identities[split] or set(alignment[split+"_ids"]) != identities[split]:
            raise ValueError(f"Teacher/alignment {split} identities differ from frozen dataset")
def teacher_options(teacher):
    allowed = {"teacher_path", "alignment_path", "encoder_id", "teacher_policy_mode", "value_contracts", "fixed_alpha", "binding_strategy", "trusted_alias_tools", "binding_credit", "guidance_mode", "tool_construction", "structure_policy", "reuse_boundary_mode"}
    if not isinstance(teacher, dict) or set(teacher) - allowed:
        raise ValueError("Unknown teacher configuration fields")
    if "value_contracts" in teacher and type(teacher["value_contracts"]) is not bool:
        raise ValueError("value_contracts must be a boolean")
    from control.binding import validate_strategy, validate_credit
    validate_strategy(teacher.get('binding_strategy', 'reject'), enabled=teacher.get('value_contracts', True))
    validate_credit(teacher.get('binding_credit', 'native_only'))
    from control.guidance import validate as validate_guidance
    validate_guidance(teacher.get('guidance_mode', 'legacy'))
    from control.primitive import validate_mode
    from control.budget import policy as structure_policy
    validate_mode(teacher.get('tool_construction', 'legacy'))
    structure_policy(teacher.get('structure_policy'))
    from control.boundary import validate_mode as boundary_mode
    boundary_mode(teacher.get('reuse_boundary_mode', 'off'))
    aliases = teacher.get('trusted_alias_tools', [])
    if not isinstance(aliases, list) or any(not isinstance(name, str) or not name for name in aliases):
        raise ValueError('trusted_alias_tools must be an explicit operator-approved list')
    alpha = teacher.get("fixed_alpha")
    if alpha is not None and (type(alpha) not in (float, int) or alpha not in (0., .25, .5, .75, 1.)):
        raise ValueError("fixed_alpha must be null (learned) or a declared five-action weight")
def rollout_contract(budget):
    return {key: budget[key] for key in ("max_turns", "max_context", "max_prompt", "max_response")}
def freeze(config, *, collection=False, base=None):
    if not isinstance(config, dict) or config.get("format") != FORMAT:
        raise ValueError("Unknown experiment configuration")
    result = json.loads(json.dumps(config))
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", result.get("run_id", "")):
        raise ValueError("Invalid run_id")
    result["budget"] = settings(result.get("budget", {}), collection=collection)
    if result.get('design') is not None:
        from study import validate, setup
        result['design'] = validate(result['design'])
        setup(result)
        if result['design']['native'] and (result.get('contextual') is not None or result.get('teacher')):
            raise ValueError('Native SEARL keeps its tool graph and cannot use a contextual retrieval override')
    for key in ("official_root", "model", "encoder", "isolation_config", "isolation_receipt"):
        result[key] = resolved(result[key], base)
    root = Path(result["official_root"])
    version = reviewed(root, PIN)
    if version != PIN:
        raise ValueError("Official checkout must use the reviewed pinned commit")
    for filename in ("main_searl.sh", "verl/trainer/train.py", "envs/tools/creation.py"):
        if not (root / filename).is_file():
            raise ValueError(f"Incomplete official source: {filename}")
    parsed = urlsplit(result.get("search_url", ""))
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.path != "/retrieve"
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Explicit retrieval endpoint ending in /retrieve is required")
    validate_deployment(read(result["isolation_config"]), read(result["isolation_receipt"]))
    result["datasets"] = {s: resolved(result["datasets"][s], base) for s in ("train", "validation", "test")}
    paths = {s: Path(p) for s, p in result["datasets"].items()}
    if any(p.suffix != ".parquet" for p in paths.values()):
        raise ValueError("Upstream online trainer requires prepared Parquet datasets")
    split = audit(paths, "question")
    if not split["disjoint"]:
        raise ValueError("Train/validation/test questions overlap")
    identities = {}
    for name, path in paths.items():
        identities[name] = set()
        for row in rows(path):
            identity = question_hash(validate_official_row(row))
            if identity in identities[name]:
                raise ValueError(f"Duplicate independent question inside {name}")
            identities[name].add(identity)
    if (identities["train"] & (identities["validation"] | identities["test"]) or
            identities["validation"] & identities["test"]):
        raise ValueError("Normalized Unicode question identities overlap across splits")
    if split["datasets"]["train"]["rows"] < result["budget"]["batch_size"]:
        raise ValueError("Training dataset is smaller than one frozen batch")
    if result.get("dataset_provenance") not in ("author_preprocessed", "adapted_upstream"):
        raise ValueError("Declare author_preprocessed or adapted_upstream data provenance")
    fingerprints = {"model": directory_digest(result["model"], model=True),
                    "official_source": directory_digest(root), "method_source": directory_digest(HERE),
                    "encoder": directory_digest_encoder(Path(result["encoder"])),
                    "isolation": file_digest(Path(result["isolation_config"]))}
    position_limit = read(Path(result["model"]) / "config.json").get("max_position_embeddings")
    if result.get('design') is not None:
        from study import actor
        actor(read(Path(result['model'])/'config.json'))
    if type(position_limit) is not int or result["budget"]["max_context"] > position_limit:
        raise ValueError("Declared context exceeds the model's recorded position limit")
    if collection:
        if result.get("teacher") or result["budget"]["rollouts"] != 1:
            raise ValueError("Teacher bootstrap uses one rollout and no fitted teacher")
        return {"config": result, "split_audit": split, "fingerprints": fingerprints,
                "upstream_commit": PIN, "claim": "frozen-policy training-only teacher bootstrap; no Qwen optimizer update"}
    if result.get('design', {}).get('native'):
        from control.windows import layout
        count = result['design']['actor_batches']
        schedule = layout(split['datasets']['train']['rows'], result['budget']['batch_size'], count,
            tail=result['budget']['tail_policy'], epochs=result['budget']['epochs'])
        if any(result['budget'][key] % count for key in ('save_freq', 'validation_freq')):
            raise ValueError('Native cadence controls save and validate only at actor barriers')
        return dict(config=result, split_audit=split, fingerprints=fingerprints, upstream_commit=PIN,
            schedule=schedule, arm_methods={'searl': 'native_tool_graph' if count == 1 else 'native_tool_graph_actor_cadence'},
            claim='pinned SEARL tool-graph baseline on the declared eight-GPU execution environment')
    if result.get('contextual') is not None:
        from control.stage import settings as contextual_settings
        options = contextual_settings(result['contextual'])
        if options['rollouts'] != result['budget']['rollouts'] or result.get('teacher'):
            raise ValueError('Use the same native G rollouts and no legacy teacher artifacts')
        result['contextual'] = options
        schedule = window_budget(result['budget'], options, split['datasets']['train']['rows'])
        methods = ({'teacher': result['design']['condition']} if result.get('design') is not None else
            dict(searl='semantic_control', evocond='conditional_identity_fixed_fusion', teacher=options['variant']))
        return {'config': result, 'split_audit': split, 'fingerprints': fingerprints, 'upstream_commit': PIN,
                'schedule': schedule, 'arm_methods': methods,
                'claim': 'matched contextual controls using native SEARL actor updates; not official SEARL reproduction; live validation required'}
    teacher = result.get("teacher", {})
    teacher_options(teacher)
    for key in ("teacher_path", "alignment_path"):
        teacher[key] = resolved(teacher[key], base)
    if teacher.get("teacher_policy_mode") not in ("frozen_controller_transfer", "semantic_fallback_on_policy_change"):
        raise ValueError("Online experiment requires explicit frozen transfer or semantic fallback; no implicit stale calibration")
    if not teacher.get("encoder_id"):
        raise ValueError("Teacher encoder provenance is required")
    # Artifacts are sealed before launch; no test-driven fallback or refitting.
    fingerprints["teacher"] = {key: file_digest(Path(teacher[key])) for key in ("teacher_path", "alignment_path")}
    from control.teacher import validate
    from control.ranking import validate_alignment
    fitted, alignment = read(teacher["teacher_path"]), read(teacher["alignment_path"])
    validate(fitted)
    from control.budget import policy as structure_policy
    validate_alignment(alignment, encoder_id=teacher["encoder_id"],
                       structure_policy=structure_policy(teacher.get("structure_policy")),
                       reuse_boundary_mode=teacher.get('reuse_boundary_mode', 'off'))
    validate_teacher_splits(fitted, alignment, identities)
    if fitted["policy_hash"] != fingerprints["model"]["sha256"] or fitted["alignment_hash"] != canonical_hash(alignment):
        raise ValueError("Teacher training provenance does not match initial policy/alignment")
    context = {**teacher, "value_contracts": teacher.get("value_contracts", True), "sample_tool_turns": True,
               "rollout_contract": rollout_contract(result["budget"]),
               "official_source_sha256": fingerprints["official_source"]["sha256"]}
    if fitted.get("execution_context_hash") != execution_context_fingerprint(context):
        raise ValueError("Teacher labels were collected under different execution controls or source versions")
    return {"config": result, "split_audit": split, "fingerprints": fingerprints,
            "upstream_commit": PIN, "claim": "modified online SEARL environment; not verified paper reproduction"}
def directory_digest_encoder(path):
    files = sorted(p for p in path.rglob("*") if p.is_file() and not set(p.parts) & {".cache", ".git"})
    if not files or not any(p.suffix in (".safetensors", ".bin") for p in files):
        raise ValueError("Encoder weights are missing")
    return canonical_hash({p.relative_to(path).as_posix(): file_digest(p) for p in files})
def runtimes(frozen, root):
    config, output = frozen["config"], {}
    for arm in arms(config):
        folder = root / arm / "runtime"
        value = dict(format="evocond_searl_runtime_v1", arm=arm, run_id=config["run_id"]+"_"+arm,
                     root=str(folder), registry_path=str(folder / "registry.json"),
                     policy_hash=frozen["fingerprints"]["model"]["sha256"], value_contracts=False,
                     sample_tool_turns=True, rollout_contract=rollout_contract(config["budget"]),
                     official_source_sha256=frozen["fingerprints"]["official_source"]["sha256"])
        if config.get('design') is not None:
            value.update(design=deepcopy(config['design']), epoch_batches=frozen['schedule']['batches_per_epoch'],
                actor_batches=config['design']['actor_batches'])
        if config.get('contextual') is not None:
            options = deepcopy(config['contextual'])
            options['variant'] = 'semantic' if arm == 'searl' else 'aligned' if arm == 'evocond' else options['variant']
            options['horizon_steps'] = config['budget']['max_turns']+1
            value.update(contextual=options, encoder_path=config['encoder'], encoder_id=frozen['fingerprints']['encoder'],
                         tool_construction='legacy', value_contracts=False, binding_credit='native_only')
            if frozen.get('schedule') is not None:
                value['epoch_batches'] = frozen['schedule']['batches_per_epoch']
        elif arm == "teacher":
            value.update(config["teacher"], encoder_path=config["encoder"])
            value["value_contracts"] = bool(config["teacher"].get("value_contracts", True))
        output[arm] = value
    return output
def checkpoint(root, arm, value):
    root = Path(root).resolve(strict=True)
    path = Path(value).resolve(strict=True)
    if not path.is_relative_to(root / arm / "checkpoints") or not re.fullmatch(r"global_step_[1-9][0-9]*", path.name):
        raise ValueError("Resume checkpoint must belong to this arm and experiment")
    if not (path/'actor').is_dir():
        raise ValueError('Incomplete paired checkpoint: missing actor directory')
    for item in ("data.pt", "method_runtime/runtime.json", "method_runtime/graphrag.json", "method_runtime/subtasks.graphml"):
        if not (path / item).is_file():
            raise ValueError(f"Incomplete paired checkpoint: missing or invalid {item}")
    return str(path)
def launch_spec(frozen, root, arm, *, stage="train", split="validation", resume=None, attempt="attempt"):
    if arm not in arms(frozen['config']):
        raise ValueError('Arm does not belong to this frozen study condition')
    if stage == "train" and split != "validation":
        raise ValueError("Training can monitor validation only; test is reserved for explicit final evaluation")
    config, budget = frozen["config"], frozen["config"]["budget"]
    arm_root = root / arm
    ckpt = checkpoint(root, arm, resume) if resume else None
    if stage == "evaluate" and not ckpt:
        raise ValueError("Evaluation requires an explicit paired checkpoint")
    if stage == "train" and not ckpt and (arm_root / "checkpoints").exists() and any((arm_root / "checkpoints").iterdir()):
        raise ValueError("Existing checkpoints: specify --resume, never restart in-place")
    env = dict(os.environ)
    env.update(MODEL_PATH=config["model"], TRAIN_FILES=json.dumps(config["datasets"]["train"]),
               VAL_FILES=json.dumps(config["datasets"][split if stage == "evaluate" else "validation"]),
               RESULT_DIR=str(arm_root / "checkpoints"), LOG_PATH=str(arm_root / "logs" / (attempt+".log")),
               PROJECT_NAME=config["run_id"]+"_"+arm, EXP_NAME=stage+"_"+attempt,
               SEARL_GPUS_PER_NODE=str(budget["gpus"]), PYTHON_BIN=sys.executable,
               MAX_PROMPT_LENGTH=str(budget["max_prompt"]), MAX_RESPONSE_LENGTH=str(budget["max_response"]),
               SEARL_SEARCH_URL=config["search_url"], RL_FACTORY_EMBEDDING_MODEL=config["encoder"],
               SEARL_ISOLATION_CONFIG=config["isolation_config"], SEARL_ISOLATION_RECEIPT=config["isolation_receipt"],
               SEARL_DYNAMIC_REGISTRY=str(arm_root / "runtime" / "registry.json"),
               PYTHONPATH=os.pathsep.join([str(HERE), config["official_root"], env.get("PYTHONPATH", "")]),
               WANDB_DIR=str(arm_root / "wandb"))
    env['PYTHONHASHSEED'] = str(budget['seed'])
    params = {"data.train_batch_size": budget["batch_size"], "data.seed": budget["seed"],
              "data.keep_tail": budget.get('tail_policy') == 'keep',
              "actor_rollout_ref.actor.state_masking": True,
              "actor_rollout_ref.actor.optim.lr": budget["learning_rate"],
              "actor_rollout_ref.rollout.seed": budget["seed"],
              "actor_rollout_ref.rollout.max_tokens": budget["max_response"],
              "actor_rollout_ref.rollout.max_model_len": budget["max_context"],
              "actor_rollout_ref.rollout.n": budget["rollouts"],
              "actor_rollout_ref.rollout.temperature": budget.get('temperature', .7),
              "actor_rollout_ref.rollout.top_p": budget.get('top_p', .98),
              "actor_rollout_ref.rollout.repetition_penalty": budget.get('repetition_penalty', 1.05),
              "actor_rollout_ref.rollout.val_kwargs.n": 1,
              "actor_rollout_ref.rollout.val_kwargs.do_sample": False,
              "actor_rollout_ref.rollout.val_kwargs.temperature": 0.0,
              "actor_rollout_ref.rollout.val_kwargs.top_p": 1.0,
              "actor_rollout_ref.env.enable_thinking": False,
              "data.filter_overlong_prompts": False,
              "data.validation_shuffle": False,
              "actor_rollout_ref.rollout.max_turns": budget["max_turns"],
              "actor_rollout_ref.env.max_turns": budget["max_turns"],
              "actor_rollout_ref.env.runtime_config_path": str(arm_root / "runtime_config.json"),
              "actor_rollout_ref.env.evaluation_split": split,
              "trainer.total_epochs": budget["epochs"], "trainer.save_freq": budget["save_freq"],
              "trainer.test_freq": budget["validation_freq"], "trainer.resume_mode": "resume_path" if ckpt else "disable",
              "trainer.resume_from_path": ckpt, "trainer.val_only": stage == "evaluate",
              "trainer.validation_data_dir": str(arm_root / "evaluation" / split / attempt),
              "ray_init.namespace": config["run_id"]+"_"+arm}
    command = ["bash", "main_searl.sh"]+[f"++{key}={json.dumps(value)}" for key, value in params.items()]
    return command, env
def readiness(gpus):
    if sys.platform != "linux":
        raise RuntimeError("Actual distributed launch requires the inspected Linux GPU host")
    missing = [name for name in ("torch", "ray", "vllm", "transformers", "pyarrow", "mcp", "sentence_transformers")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing training dependencies: {missing}")
    import torch
    if torch.cuda.device_count() < gpus:
        raise RuntimeError("Fewer visible GPUs than the frozen protocol requires")
def environment_identity():
    import torch
    return {"python": sys.version, "packages": {name: importlib.metadata.version(name) for name in
            ("torch", "ray", "vllm", "transformers", "pyarrow", "mcp", "sentence-transformers")},
            "cuda": torch.version.cuda,
            "gpus": [{"name": torch.cuda.get_device_properties(i).name,
                      "memory": torch.cuda.get_device_properties(i).total_memory}
                     for i in range(torch.cuda.device_count())]}

@contextmanager
def allocation(root, workspace, identity, *, shared=False):
    payload = json.dumps(identity, sort_keys=True, allow_nan=False)
    locked = strict_json(payload)
    owned = []
    paths = ([workspace/'runs/.gpu.lock'] if shared else [])+[root/'.launch.lock']
    try:
        for path in paths:
            publish(path, payload)
            owned.append(path)
        yield
    finally:
        for path in reversed(owned):
            try:
                owner = read(path) if path.is_file() else None
            except (OSError, ValueError):
                continue
            if owner == locked:
                path.unlink()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "status", "train", "evaluate"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--config")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--resume")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument('--workspace', type=Path, default=Path('.'))
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    root = (workspace/args.root).resolve()
    protocol = root / "protocol.json"
    if args.stage == "prepare":
        if not args.config:
            parser.error("prepare requires --config")
        frozen = freeze(read(workspace/args.config), base=workspace)
        if root.exists() and any(root.iterdir()):
            if protocol.is_file() and read(protocol) == frozen and all(
                    (root/arm/'runtime_config.json').is_file() and read(root/arm/'runtime_config.json') == value
                    for arm, value in runtimes(frozen, root).items()):
                print(json.dumps(dict(prepared=True, reused=True, root=str(root), gpu_started=False)))
                return
            raise FileExistsError("Existing experiment differs; prepare requires a new empty directory")
        if args.run:
            write_new(protocol, frozen)
            for arm, value in runtimes(frozen, root).items():
                write_new(root / arm / "runtime_config.json", value)
                load_runtime(root / arm / "runtime_config.json")
        print(json.dumps({"prepared": args.run, "root": str(root), "protocol_sha256": canonical_hash(frozen),
                          "gpu_started": False, "claim": frozen["claim"]}))
        return
    frozen = read(protocol)
    if args.stage == "status":
        print(json.dumps({"root": str(root), "claim": frozen["claim"], "arms": {arm: {
            "checkpoints": sorted(p.name for p in (root / arm / "checkpoints").glob("global_step_*")),
            "active_lock": (root / ".launch.lock").exists()} for arm in arms(frozen['config'])}}))
        return
    if args.arm is None:
        parser.error("train/evaluate require --arm; launch arms sequentially on shared GPUs")
    if args.arm not in arms(frozen['config']):
        parser.error('Requested arm does not belong to the frozen study condition')
    if args.stage == "train" and args.split != "validation":
        parser.error("Training may only monitor validation; use evaluate --split test --allow-test for final evaluation")
    if args.split == "test" and args.stage == "evaluate" and not args.allow_test:
        parser.error("Test remains sealed. Explicit --allow-test is required for final evaluation")
    if freeze(frozen["config"]) != frozen:
        raise ValueError("Frozen code, model, data or fitted artifacts changed; create a new experiment")
    if read(root / args.arm / "runtime_config.json") != runtimes(frozen, root)[args.arm]:
        raise ValueError("Arm runtime configuration changed after freezing")
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    resume = str((workspace/args.resume).resolve()) if args.resume else None
    command, env = launch_spec(frozen, root, args.arm, stage=args.stage, split=args.split, resume=resume, attempt=attempt)
    print(json.dumps({"run": args.run, "arm": args.arm, "stage": args.stage, "command": command,
                      "log": env["LOG_PATH"], "qwen_updates": args.stage == "train"}))
    if not args.run:
        return
    readiness(frozen["config"]["budget"]["gpus"])
    record = root / args.arm / "attempts" / (attempt+".json")
    identity = dict(pid=os.getpid(), attempt=attempt, arm=args.arm, root=str(root))
    with allocation(root, workspace, identity, shared=frozen['config'].get('design') is not None):
        environment = environment_identity()
        environment_file = root / "environment.json"
        if environment_file.exists():
            if read(environment_file) != environment:
                raise ValueError("GPU/dependency environment changed between arms or attempts")
        else:
            write_new(environment_file, environment)
        write_new(record, {"command": command, "stage": args.stage, "split": args.split,
                           "protocol_sha256": canonical_hash(frozen), "resume": args.resume,
                           "environment_sha256": canonical_hash(environment)})
        result = subprocess.run(command, cwd=frozen["config"]["official_root"], env=env, check=False)
        write_new(record.with_suffix(".result.json"), {"exit_code": result.returncode})
        raise SystemExit(result.returncode)
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
