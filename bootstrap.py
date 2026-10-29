# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Build a teacher with the pinned official solver, without updating Qwen.'
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import experiment as experiment
from convert import _pyarrow, question_hash, read_rows, validate_official_row
from teacher import fit_frozen_teacher, split_groups
from control import dynamic, ranking
FORMAT = "evocond_teacher_bootstrap_v1"
def artifact_rows(path, *, max_characters=64*1024*1024):
    'Read recorded registries larger than the untrusted tool-proposal limit.'
    def unique(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate key in bootstrap artifact")
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("Nonfinite bootstrap artifact")
    with Path(path).open(encoding="utf-8") as stream:
        while True:
            line = stream.readline(max_characters+1)
            if not line:
                break
            if len(line) > max_characters or not line.strip():
                raise ValueError("Bootstrap artifact row is blank or exceeds the explicit size limit")
            row = json.loads(line, object_pairs_hook=unique, parse_constant=invalid)
            if not isinstance(row, dict):
                raise ValueError("Bootstrap artifact row must be an object")
            yield row
def partitions(training, validation, test, spec):
    if not isinstance(spec, dict) or set(spec) != {"fit_queries", "calibration_queries", "seed"}:
        raise ValueError("Explicit teacher partition counts and seed are required")
    if any(type(spec[k]) is not int or spec[k] < (0 if k == "seed" else 1) for k in spec):
        raise ValueError("Invalid teacher partition counts/seed")
    identities = {}
    for split, data in (("train", training), ("validation", validation), ("test", test)):
        ids = [question_hash(validate_official_row(row)) for row in data]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Empty or duplicate independent questions")
        identities[split] = ids
    if set(identities["train"]) & (set(identities["validation"]) | set(identities["test"])) or set(identities["validation"]) & set(identities["test"]):
        raise ValueError("Teacher partitions cannot borrow held-out questions")
    count = spec["fit_queries"] + spec["calibration_queries"]
    if count > len(identities["train"]):
        raise ValueError("Not enough independent training questions; no automatic shrinking")
    order = sorted(identities["train"])
    random.Random(spec["seed"]).shuffle(order)
    fit, calibration = order[:spec["fit_queries"]], order[spec["fit_queries"]:count]
    result = {"train": fit+calibration, "teacher_fit": fit, "teacher_calibration": calibration,
              "validation": sorted(identities["validation"]), "test": sorted(identities["test"])}
    split_groups(result)
    return result
def freeze(config):
    required = {"format", "experiment", "partition", "encoder_id", "seeds", "capture_turn"}
    if not isinstance(config, dict) or not required <= set(config) or set(config)-required-{'binding_strategy', 'trusted_alias_tools', 'binding_credit', 'guidance_mode', 'tool_construction', 'structure_policy', 'reuse_boundary_mode'} or config.get("format") != FORMAT:
        raise ValueError("Unknown teacher bootstrap configuration")
    experiment.teacher_options({'binding_strategy': config.get('binding_strategy', 'reject'),
                                'guidance_mode': config.get('guidance_mode', 'legacy'),
                                'tool_construction': config.get('tool_construction', 'legacy'),
                                'structure_policy': config.get('structure_policy'),
                                'reuse_boundary_mode': config.get('reuse_boundary_mode', 'off'),
                                'binding_credit': config.get('binding_credit', 'native_only'),
                                'trusted_alias_tools': config.get('trusted_alias_tools', [])})
    if not isinstance(config["encoder_id"], str) or not config["encoder_id"].strip():
        raise ValueError("Pinned encoder_id is required")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Distinct nonnegative common seeds are required")
    frozen = experiment.freeze(config["experiment"], collection=True)
    budget = frozen["config"]["budget"]
    if type(config["capture_turn"]) is not int or not 1 <= config["capture_turn"] < budget["max_turns"]-1:
        raise ValueError("Capture turn must leave continuation budget")
    data = {split: list(read_rows(Path(path), "parquet")) for split, path in frozen["config"]["datasets"].items()}
    splits = partitions(data["train"], data["validation"], data["test"], config["partition"])
    if any(len(splits[key]) % budget["batch_size"] for key in ("teacher_fit", "train")):
        raise ValueError("Teacher subset sizes must be multiples of frozen batch_size; upstream drops partial train batches")
    return {"format": FORMAT, "input_config": deepcopy(config), "experiment": frozen, "splits": splits,
            "claim": "train-only frozen-policy teacher collection; extra teacher compute must be reported"}, data["train"]
def materialize(root, protocol, training):
    pa, pq = _pyarrow()
    selected = {question_hash(row["question"]): row for row in training}
    tables = {name: pa.Table.from_pylist([selected[key] for key in protocol["splits"][part]])
              for name, part in (("teacher_fit", "teacher_fit"), ("teacher_all", "train"))}
    root.mkdir(parents=True, exist_ok=False)
    experiment.write_new(root / "splits.json", protocol["splits"])
    hashes = {}
    for name, table in tables.items():
        path = root / (name+".parquet")
        pq.write_table(table, path, compression="zstd")
        hashes[name] = experiment.file_digest(path)
    experiment.write_new(root / "protocol.json", {**protocol, "subset_sha256": hashes})
def verified(root):
    stored = experiment.read(root / "protocol.json")
    current, _ = freeze(stored["input_config"])
    if {k: v for k, v in stored.items() if k != "subset_sha256"} != current:
        raise ValueError("Bootstrap code/model/data/config changed; use a new bootstrap root")
    for name, expected in stored["subset_sha256"].items():
        if experiment.file_digest(root / (name+".parquet")) != expected:
            raise ValueError("Frozen teacher subset changed")
    if experiment.read(root / "splits.json") != stored["splits"]:
        raise ValueError("Teacher split identities changed")
    return stored
def runtime_config(protocol, root, stage):
    common = protocol["experiment"]
    folder = root / stage / "runtime"
    value = {"format": "evocond_searl_runtime_v1", "arm": "searl", "root": str(folder),
             "run_id": common["config"]["run_id"]+"_teacher_"+stage,
             "registry_path": str(folder / "registry.json"),
             "policy_hash": common["fingerprints"]["model"]["sha256"],
             "sample_tool_turns": True,
             "rollout_contract": experiment.rollout_contract(common["config"]["budget"]),
             "official_source_sha256": common["fingerprints"]["official_source"]["sha256"],
             "value_contracts": stage == "collect", "collection": stage == "collect", "bootstrap_warmup": stage == "warmup",
             "encoder_id": protocol["input_config"]["encoder_id"],
             "encoder_path": common["config"]["encoder"]}
    value['tool_construction'] = protocol['input_config'].get('tool_construction', 'legacy')
    value['structure_policy'] = protocol['input_config'].get('structure_policy')
    value['reuse_boundary_mode'] = protocol['input_config'].get('reuse_boundary_mode', 'off')
    if stage == "collect":
        value["alignment_path"] = str(root / "alignment" / "alignment.json")
        value['binding_strategy'] = protocol['input_config'].get('binding_strategy', 'reject')
        value['binding_credit'] = protocol['input_config'].get('binding_credit', 'native_only')
        value['guidance_mode'] = protocol['input_config'].get('guidance_mode', 'legacy')
        value['trusted_alias_tools'] = protocol['input_config'].get('trusted_alias_tools', [])
    return value
def command(protocol, root, stage):
    frozen = protocol["experiment"]
    cmd, env = experiment.launch_spec(frozen, root, "searl", attempt="teacher_"+stage)
    params = {"actor_rollout_ref.env.runtime_config_path": str(root / stage / "runtime_config.json"),
              "actor_rollout_ref.rollout.n": 1, "actor_rollout_ref.rollout.mode": "sync",
              "actor_rollout_ref.rollout.val_kwargs.n": 1,
              "trainer.teacher_collect_path": str(root / stage),
              "trainer.teacher_collect_mode": "warmup" if stage == "warmup" else "pairs",
              "trainer.teacher_collect_seeds": protocol["input_config"]["seeds"],
              "trainer.teacher_capture_turn": protocol["input_config"]["capture_turn"],
              "trainer.val_before_train": False, "trainer.val_only": False,
              "trainer.resume_mode": "disable", "trainer.resume_from_path": None,
              "ray_init.namespace": frozen["config"]["run_id"]+"_teacher_"+stage}
    cmd = [arg for arg in cmd if not any(arg.startswith("++"+key+"=") for key in params)]
    cmd += [f"++{key}={json.dumps(value)}" for key, value in params.items()]
    folder = root / stage
    env.update(TRAIN_FILES=json.dumps(str(root / ("teacher_fit.parquet" if stage == "warmup" else "teacher_all.parquet"))),
               SEARL_DYNAMIC_REGISTRY=str(folder / "runtime/registry.json"),
               RESULT_DIR=str(folder / "unused_checkpoints"), LOG_PATH=str(root / (stage+".log")),
               PROJECT_NAME=frozen["config"]["run_id"]+"_teacher_bootstrap",
               EXP_NAME=stage, WANDB_DIR=str(folder / "wandb"))
    return cmd, env
def completed(root, stage):
    report = experiment.read(root / stage / "audit.json")
    if report.get("complete") is not True:
        raise ValueError(f"{stage} did not complete")
    marker = experiment.read(root / (stage+"_launch.json"))
    if marker["protocol_sha256"] != experiment.canonical_hash(experiment.read(root / "protocol.json")):
        raise ValueError("Stage belongs to another bootstrap protocol")
    result = experiment.read(root / (stage+"_result.json"))
    expected = {"audit.json", "registry.json", "snapshots.jsonl"} if stage == "warmup" else {"audit.json", "pairs.jsonl"}
    if result.get("exit_code") != 0 or set(result.get("outputs", {})) != expected:
        raise ValueError("Stage has no complete successful execution receipt")
    for name, checksum in result["outputs"].items():
        if experiment.file_digest(root / stage / name) != checksum:
            raise ValueError("Completed bootstrap artifacts changed")
    return report
def launch(root, protocol, stage, execute=False):
    if (root / stage).exists() or (root / (stage+"_launch.json")).exists():
        raise FileExistsError("Stage already exists; inspect it rather than overwrite or mix partial runs")
    seed_registry = None
    if stage == "collect":
        completed(root, "warmup")
        alignment = experiment.read(root / "alignment/alignment.json")
        ranking.validate_alignment(alignment, encoder_id=protocol["input_config"]["encoder_id"])
        provenance = experiment.read(root / "alignment/provenance.json")
        if (provenance["artifact_sha256"] != dynamic.digest(alignment) or
                provenance["protocol_sha256"] != experiment.canonical_hash(protocol) or
                provenance["snapshot_sha256"] != experiment.file_digest(root / "warmup/snapshots.jsonl")):
            raise ValueError("Alignment provenance differs from this warmup")
        if set(alignment["train_ids"]) != set(protocol["splits"]["teacher_fit"]):
            raise ValueError("Alignment not fitted on this teacher partition")
        seed_registry = experiment.read(root / "warmup/registry.json")
        dynamic.validate(seed_registry)
    cmd, env = command(protocol, root, stage)
    info = {"stage": stage, "run": execute, "qwen_updates": False, "command": cmd,
            "protocol_sha256": experiment.canonical_hash(protocol)}
    if not execute:
        return info
    experiment.readiness(protocol["experiment"]["config"]["budget"]["gpus"])
    lock = root / ".launch.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "stage": stage}, stream)
    try:
        identity = experiment.environment_identity()
        envpath = root / "environment.json"
        if envpath.exists() and experiment.read(envpath) != identity:
            raise ValueError("Bootstrap environment changed between stages")
        if not envpath.exists():
            experiment.write_new(envpath, identity)
        experiment.write_new(root / (stage+"_launch.json"), {**info, "started": datetime.now(timezone.utc).isoformat(),
                             "environment_sha256": experiment.canonical_hash(identity)})
        runtime = runtime_config(protocol, root, stage)
        experiment.write_new(root / stage / "runtime_config.json", runtime)
        if seed_registry is not None:
            experiment.write_new(Path(runtime["registry_path"]), seed_registry)
        result = subprocess.run(cmd, cwd=protocol["experiment"]["config"]["official_root"], env=env, check=False)
        output_names = ("audit.json", "registry.json", "snapshots.jsonl") if stage == "warmup" else ("audit.json", "pairs.jsonl")
        output_hashes = {name: experiment.file_digest(root / stage / name) for name in output_names} if result.returncode == 0 else {}
        experiment.write_new(root / (stage+"_result.json"), {"exit_code": result.returncode, "outputs": output_hashes})
        if result.returncode:
            raise RuntimeError(f"Official {stage} exited {result.returncode}; inspect {env['LOG_PATH']}")
        completed(root, stage)
        return info
    finally:
        lock.unlink()
def alignment(root, protocol, execute=False, encoder=None):
    completed(root, "warmup")
    target = root / "alignment"
    if target.exists():
        raise FileExistsError("Alignment is immutable; use a new bootstrap root to refit")
    snapshots = list(artifact_rows(root / "warmup/snapshots.jsonl"))
    fit = set(protocol["splits"]["teacher_fit"])
    if not snapshots or any(row.get("query_id") not in fit for row in snapshots):
        raise ValueError("Alignment snapshots must come only from teacher-fit questions")
    if not execute:
        return {"stage": "alignment", "run": False, "snapshots": len(snapshots), "encoder_loaded": False}
    encoder = encoder or ranking.SentenceTransformerEncoder(
        protocol["experiment"]["config"]["encoder"], encoder_id=protocol["input_config"]["encoder_id"], device="cpu")
    pairs, seen = [], set()
    for row in snapshots:
        for pair in ranking.build_alignment_pairs(row["registry"], encoder,
                query_id=row["query_id"], query=row["query"],
                structure_policy=protocol['input_config'].get('structure_policy'),
                reuse_boundary_mode=protocol['input_config'].get('reuse_boundary_mode', 'off'),
                current_user_text=row.get('current_user_text')):
            key = pair["query_id"], pair["tool_version"]
            if key not in seen:
                pairs.append(pair)
                seen.add(key)
    groups = split_groups(protocol["splits"])
    artifact = ranking.prepare_alignment(pairs, encoder_id=protocol["input_config"]["encoder_id"],
        train_ids=groups["teacher_fit"], calibration_ids=groups["teacher_calibration"],
        validation_ids=groups["validation"], test_ids=groups["test"])
    experiment.write_new(target / "alignment.json", artifact)
    experiment.write_new(target / "provenance.json", {"snapshot_sha256": experiment.file_digest(root / "warmup/snapshots.jsonl"),
        "protocol_sha256": experiment.canonical_hash(protocol), "pairs": len(pairs), "artifact_sha256": dynamic.digest(artifact)})
    return {"stage": "alignment", "run": True, "pairs": len(pairs), "output": str(target)}
def fit(root, protocol, execute=False):
    completed(root, "collect")
    target = root / "teacher"
    if target.exists():
        raise FileExistsError("Fitted teacher is immutable")
    pairs_path = root / "collect/pairs.jsonl"
    pairs = list(artifact_rows(pairs_path))
    model, report = fit_frozen_teacher(protocol["splits"], pairs, experiment.read(root / "alignment/alignment.json"))
    if model["policy_hash"] != protocol["experiment"]["fingerprints"]["model"]["sha256"]:
        raise ValueError("Teacher continuation labels came from a different Qwen policy")
    if execute:
        experiment.write_new(target / "teacher.json", model)
        experiment.write_new(target / "training_report.json", {**report,
            "protocol_sha256": experiment.canonical_hash(protocol), "pairs_sha256": experiment.file_digest(pairs_path)})
    return {"stage": "fit", "run": execute, "output": str(target), "report": report,
            "qwen_updated": False, "note": "A fitted teacher is not evidence of improvement; evaluate all arms on held-out questions"}
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "warmup", "alignment", "collect", "fit", "status"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.stage == "prepare":
        if args.config is None:
            parser.error("prepare requires --config")
        if root.exists():
            raise FileExistsError("Bootstrap requires a new directory")
        protocol, training = freeze(experiment.read(args.config))
        if args.run:
            materialize(root, protocol, training)
        result = {"stage": "prepare", "run": args.run, "root": str(root), "qwen_updated": False,
                  "counts": {s: len(v) for s, v in protocol["splits"].items()}}
    elif args.stage == "status":
        result = {"root": str(root), "active_lock": (root / ".launch.lock").exists(),
                  "stages": {s: (root / s / f).is_file() for s, f in
                             (("warmup", "audit.json"), ("alignment", "alignment.json"), ("collect", "audit.json"), ("teacher", "teacher.json"))}}
    else:
        protocol = verified(root)
        if args.stage in ("warmup", "collect"):
            result = launch(root, protocol, args.stage, args.run)
        elif args.stage == "alignment":
            result = alignment(root, protocol, args.run)
        else:
            result = fit(root, protocol, args.run)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
