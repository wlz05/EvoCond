# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Collect paired training data, fit a task gate, and evaluate on sealed tasks.'
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import random
from statistics import mean
import subprocess
import sys
from typing import Any
from control.gate import ARMS, calibrate, decide, evaluate, fit, initial_user, normalize, validate_model
from sample import TASKS, seen_indices
from panel import (PROJECT, bench_args, completed, digest,
                                port_free, server_args, stop_server, wait_for_server)
def write_new(path: Path, value: Any) -> None:
    'Atomic installation, refusing changes to an existing frozen artifact.'
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise FileExistsError(f"Refusing to replace a frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".pending")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)
def source_hashes() -> dict[str, str]:
    files = list(PROJECT.glob("*.py"))
    for name in ("control", "flow", "model", "state", "tests"):
        files.extend((PROJECT / name).glob("*.py"))
    return {p.relative_to(PROJECT).as_posix(): digest(p) for p in sorted(files)}
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("all", "prepare", "collect", "train", "evaluate", "status"), default="all")
    parser.add_argument("--root", default="runs/gate20260927_v1")
    parser.add_argument("--queries", default="assets/data/costbench/CostBench_queries.json")
    parser.add_argument("--memory", default="runs/joint20260925/memory.json")
    parser.add_argument("--model", default="assets/model")
    parser.add_argument("--exclude-root", default="runs")
    parser.add_argument("--train-per-task", type=int, default=6)
    parser.add_argument("--validation-per-task", type=int, default=3)
    parser.add_argument("--test-per-task", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--refinement", type=int, default=2)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()
def make_manifest(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    if not root.is_relative_to(PROJECT) or root == PROJECT:
        raise ValueError("Gate output must be a dedicated subdirectory of this project")
    sizes = {"train": args.train_per_task, "validation": args.validation_per_task, "test": args.test_per_task}
    if any(x < 1 for x in sizes.values()) or args.tokens < 1 or args.refinement < 0 or not 1 <= args.port <= 65535:
        raise ValueError("Invalid split sizes or inference budget")
    query_path, memory_path = Path(args.queries), Path(args.memory)
    queries = json.loads(query_path.read_text(encoding="utf-8"))
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    metadata = memory.get("metadata") or {}
    training = metadata.get("train_indices")
    if not isinstance(queries, list) or not isinstance(training, list) or not training or any(type(i) is not int or not 0 <= i < len(queries) for i in training):
        raise ValueError("Invalid query dataset or memory training provenance")
    if len(set(training)) != len(training) or metadata.get("source") != "costbench_traces_v1" or metadata.get("query_sha256") != digest(query_path):
        raise ValueError("Memory and query dataset do not match")
    signature = {"version": 1, "scope": "initial_user_episode_gate_not_edge_gate",
                 "sizes_per_task": sizes, "seed": args.seed, "arms": list(ARMS),
                 "modes": ["none", "preference_change"], "tokens": args.tokens,
                 "refinement": args.refinement, "memory_extra_cost": 0.5,
                 "query_sha256": digest(query_path), "memory_sha256": digest(memory_path),
                 "model_config_sha256": digest(Path(args.model) / "config.json"),
                 "source_sha256": source_hashes()}
    path = root / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if any(manifest.get(k) != v for k, v in signature.items()):
            raise ValueError("Frozen workflow differs from code/data/config; use a new --root, never overwrite old results")
        assigned = [i for sample in manifest["splits"].values() for i in sample["indices"]]
        if len(set(assigned)) != len(assigned) or set(assigned) & set(training):
            raise ValueError("Frozen splits overlap each other or memory training")
        return manifest
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Nonempty gate output without manifest; inspect before proceeding")
    excluded = seen_indices(Path(args.exclude_root), queries) | set(training)
    rng = random.Random(args.seed)
    groups: dict[str, list[int]] = {task: [] for task in TASKS}
    seen_texts: set[str] = set()
    # Exclude normalized duplicates of previously inspected tasks as well.
    def public_text(row: dict[str, Any]) -> str:
        value = row.get("user_requirements") or row.get("query") or ""
        return normalize(value if isinstance(value, str) else json.dumps(value, sort_keys=True))
    for i in excluded:
        if 0 <= i < len(queries):
            seen_texts.add(public_text(queries[i]))
    for i, row in enumerate(queries):
        if not isinstance(row, dict) or str(row.get("task", "")).strip().lower() not in TASKS:
            raise ValueError(f"Invalid CostBench task: {i}")
        if i not in excluded and public_text(row) and public_text(row) not in seen_texts:
            groups[row["task"].strip().lower()].append(i)
    splits = {name: {"by_task": {}, "indices": [], "seed": args.seed,
                      "query_sha256": signature["query_sha256"], "memory_sha256": signature["memory_sha256"],
                      "memory_extra_cost": 0.5} for name in sizes}
    for task in sorted(TASKS):
        pool = groups[task]
        rng.shuffle(pool)
        distinct = []
        for i in pool:
            text = public_text(queries[i])
            if text not in seen_texts:
                distinct.append(i)
                seen_texts.add(text)
        if len(distinct) < sum(sizes.values()):
            raise ValueError(f"Only {len(distinct)} unused distinct {task} tasks; need {sum(sizes.values())}. No automatic shrinking or reuse.")
        offset = 0
        for name, size in sizes.items():
            selected = sorted(distinct[offset:offset + size])
            splits[name]["by_task"][task] = selected
            splits[name]["indices"].extend(selected)
            offset += size
    for sample in splits.values():
        sample["indices"].sort()
    return {**signature, "excluded_indices": sorted(excluded), "splits": splits,
            "note": "Test outcomes may only be evaluated after gate.json is frozen; no automatic refitting."}
def initialize(root: Path, manifest: dict[str, Any]) -> None:
    write_new(root / "manifest.json", manifest)
    for name, sample in manifest["splits"].items():
        write_new(root / name / "sample.json", sample)
def collect(args: argparse.Namespace, name: str) -> None:
    root = Path(args.root).resolve()
    command = [sys.executable, "-B", 'panel.py', "--sample", str(root / name / "sample.json"),
               "--memory", args.memory, "--queries", args.queries, "--model", args.model,
               "--port", str(args.port), "--tokens", str(args.tokens), "--refinement", str(args.refinement),
               "--output", str(root / name / "panel"), "--arms", *ARMS,
               "--modes", "none", "preference_change", "--run"]
    print(f"COLLECT {name}: fixed-Qwen paired policy rollouts; not Qwen training", flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)
def load_examples(root: Path, name: str, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    examples, hashes = [], {}
    protocol_path = root / name / "panel" / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = {k: manifest[k] for k in ("query_sha256", "memory_sha256", "model_config_sha256",
                                        "tokens", "refinement", "memory_extra_cost", "arms", "modes")}
    expected.update(indices=manifest["splits"][name]["indices"], threads=1,
                    sample_sha256=digest(root / name / "sample.json"))
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Paired panel protocol does not match the frozen gate workflow: {name}")
    sources = protocol.get("code_sha256")
    if not isinstance(sources, dict) or not sources or any(manifest["source_sha256"].get(k) != v for k, v in sources.items()):
        raise ValueError(f"Paired panel source hashes differ: {name}")
    hashes[protocol_path.relative_to(root).as_posix()] = digest(protocol_path)
    for mode in manifest["modes"]:
        for index in manifest["splits"][name]["indices"]:
            texts, identities, outcomes = [], [], {}
            for arm in ARMS:
                destination = root / name / "panel" / arm / mode / f"q{index}"
                data = completed(destination)
                if data is None:
                    raise RuntimeError(f"Incomplete paired dataset: {destination}")
                path = next(destination.glob("results_*.json"))
                hashes[path.relative_to(root).as_posix()] = digest(path)
                row = data["results"][0]
                success = (row.get("evaluation") or {}).get("success")
                cost = (((data.get("stats") or {}).get("scores") or {}).get("cost_metrics") or {}).get("total_agent_cost")
                steps = row.get("total_steps")
                if type(success) is not bool or type(steps) is not int or steps < 0 or isinstance(cost, bool) or not isinstance(cost, (int, float)) or not isfinite(cost) or cost < 0:
                    raise ValueError(f"Invalid outcome label: {path}")
                text = initial_user(row.get("conversation_history") or [])
                if not normalize(text):
                    raise ValueError(f"Missing visible initial user input: {path}")
                texts.append(text)
                identities.append((row.get("query_id"), row.get("tool_creation_seed")))
                outcomes[arm] = {"success": success, "cost": float(cost), "steps": steps}
            if len(set(texts)) != 1:
                raise ValueError(f"Arms received different initial user inputs: {name}/{mode}/q{index}")
            if any(identity != identities[0] for identity in identities[1:]):
                raise ValueError(f"Arms used different query identities/tool seeds: {name}/{mode}/q{index}")
            examples.append({"index": index, "mode": mode, "text": texts[0], "outcomes": outcomes})
    return examples, hashes
def train(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    # Deliberately never calls load_examples(..., 'test', ...).
    training, train_hashes = load_examples(root, "train", manifest)
    validation, val_hashes = load_examples(root, "validation", manifest)
    if {normalize(r["text"]) for r in training} & {normalize(r["text"]) for r in validation}:
        raise ValueError("Train/validation initial prompts overlap after ID normalization")
    provenance = {"manifest_sha256": digest(root / "manifest.json"),
                  "query_sha256": manifest["query_sha256"], "memory_sha256": manifest["memory_sha256"],
                  "model_config_sha256": manifest["model_config_sha256"],
                  "train_indices": manifest["splits"]["train"]["indices"],
                  "validation_indices": manifest["splits"]["validation"]["indices"],
                  "test_indices": manifest["splits"]["test"]["indices"],
                  "train_result_sha256": train_hashes, "validation_result_sha256": val_hashes}
    gate_path = root / "gate.json"
    if gate_path.exists():
        model = json.loads(gate_path.read_text(encoding="utf-8"))
        validate_model(model)
        if model.get("provenance") != provenance:
            raise ValueError("Gate training inputs changed; refusing to reuse or overwrite fitted model")
    else:
        if (root / "test" / "panel").exists() or (root / "live_gate").exists():
            raise ValueError("Test execution already started but frozen gate is missing; do not refit on this split")
        model = fit(training, manifest["seed"])
        calibration = calibrate(model, validation)
        static_scores = {
            arm: {"correct": sum(r["outcomes"][arm]["success"] for r in validation),
                  "cost": sum(r["outcomes"][arm]["cost"] for r in validation),
                  "steps": sum(r["outcomes"][arm]["steps"] for r in validation)}
            for arm in ARMS
        }
        model["validation_best_static"] = min(ARMS, key=lambda arm: (
            -static_scores[arm]["correct"], static_scores[arm]["cost"],
            static_scores[arm]["steps"], ARMS.index(arm)))
        model["validation_static_scores"] = static_scores
        model["provenance"] = provenance
        model["validation_selection"] = calibration
        validate_model(model)
        write_new(gate_path, model)
    report = {"scope": model["scope"], "trained_parameters": 3 * model["dimension"],
              "qwen_updated": False, "gate_sha256": digest(gate_path),
              "calibration": model["validation_selection"],
              "training": evaluate(model, training), "validation": evaluate(model, validation)}
    write_new(root / "training_report.json", report)
    print(json.dumps({"gate": str(gate_path), "qwen_updated": False,
                      "threshold": model["threshold"], "validation": {k: v for k, v in report["validation"].items() if k != "decisions"}}, ensure_ascii=False), flush=True)
    if model["threshold"] is None:
        print("Validation did not support non-Filter interventions. Gate stays Filter; do NOT lower thresholds using test results.", flush=True)
    return model
def live_test(args: argparse.Namespace, manifest: dict[str, Any], model: dict[str, Any]) -> None:
    root = Path(args.root).resolve()
    gate_path = root / "gate.json"
    gate_hash = digest(gate_path)
    gate_root = root / "live_gate"
    live_protocol = {"manifest_sha256": digest(root / "manifest.json"), "gate_sha256": gate_hash,
                     "source_sha256": manifest["source_sha256"], "evaluation_kind": "live_initial_user_gate"}
    write_new(gate_root / "protocol.json", live_protocol)
    sample = root / "test" / "sample.json"
    fixed = argparse.Namespace(model=args.model, port=args.port, memory=args.memory, queries=args.queries,
                               threads=1, tokens=args.tokens, refinement=args.refinement, sample=str(sample))
    pending = [(mode, i) for mode in manifest["modes"] for i in manifest["splits"]["test"]["indices"]
               if completed(gate_root / mode / f"q{i}") is None]
    if pending:
        if not port_free(args.port):
            raise RuntimeError(f"Port {args.port} occupied. Stop only the known old model service, then resume.")
        trace = gate_root / "events.jsonl"
        command = server_args("joint_family", fixed, trace, 0.5) + ["--gate", str(gate_path)]
        with (gate_root / "serve.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_for_server(process, "joint_family", fixed, manifest["memory_sha256"], 0.5, expected_gate_hash=gate_hash)
                for mode, index in pending:
                    destination = gate_root / mode / f"q{index}"
                    command = bench_args("joint_family", mode, index, fixed, destination, 0.5) + ["--gate", str(gate_path)]
                    trace_start = trace.stat().st_size if trace.exists() else 0
                    with (gate_root / f"{mode}_q{index}.log").open("w", encoding="utf-8") as query_log:
                        subprocess.run(command, cwd=PROJECT, stdout=query_log, stderr=subprocess.STDOUT, check=True)
                    data = completed(destination)
                    if data is None:
                        raise RuntimeError(f"No live gate result: {destination}")
                    row = data["results"][0]
                    with (gate_root / "episodes.jsonl").open("a", encoding="utf-8") as mapping:
                        mapping.write(json.dumps({"mode": mode, "index": index, "trace_start_byte": trace_start,
                            "trace_end_byte": trace.stat().st_size if trace.exists() else 0,
                            "selection": decide(model, initial_user(row.get("conversation_history") or []))}) + "\n")
                    print(f"gate/{mode}/q{index}: correct={(row.get('evaluation') or {}).get('success')} steps={row.get('total_steps')}", flush=True)
            finally:
                stop_server(process)
    test, _ = load_examples(root, "test", manifest)
    train_rows, _ = load_examples(root, "train", manifest)
    val_rows, _ = load_examples(root, "validation", manifest)
    old_texts = {normalize(r["text"]) for r in train_rows + val_rows}
    if old_texts & {normalize(r["text"]) for r in test}:
        raise ValueError("Test initial prompts overlap gate training/validation; evaluation is invalid")
    records = []
    for row in test:
        data = completed(gate_root / row["mode"] / f"q{row['index']}")
        r = data["results"][0]
        if initial_user(r.get("conversation_history") or []) != row["text"]:
            raise ValueError("Live gate and static controls received different initial inputs")
        success = (r.get("evaluation") or {}).get("success")
        cost = data["stats"]["scores"]["cost_metrics"]["total_agent_cost"]
        if type(success) is not bool or isinstance(cost, bool) or not isinstance(cost, (int, float)) or not isfinite(cost) or cost < 0 or type(r.get("total_steps")) is not int:
            raise ValueError("Malformed live-gate test metrics")
        outcome = {"success": success, "cost": cost, "steps": r["total_steps"]}
        selection = decide(model, row["text"])
        reference = row["outcomes"][selection["arm"]]
        records.append({"index": row["index"], "mode": row["mode"], "selection": selection,
                        "outcomes": {**row["outcomes"], "gate": outcome},
                        "live_differs_from_selected_static": outcome != reference})
    report: dict[str, Any] = {"evaluation_kind": "live_frozen_gate_heldout_test", "gate_sha256": gate_hash,
                              "unique_queries": len(manifest["splits"]["test"]["indices"]),
                              "n_per_arm": len(records), "qwen_updated": False, "conditions": {}, "paired_vs_filter": {},
                              "gate_selections": dict(Counter(r["selection"]["arm"] for r in records)),
                              "replay_disagreements": sum(r["live_differs_from_selected_static"] for r in records),
                              "warning": "Small grouped test; cost is benchmark cost, excludes controller/LLM wall time. No SOTA or safety guarantee.",
                              "records": records}
    for arm in (*ARMS, "gate"):
        report["conditions"][arm] = {}
        report["paired_vs_filter"][arm] = {}
        for mode in (*manifest["modes"], "overall"):
            rows = [r for r in records if mode == "overall" or r["mode"] == mode]
            outcomes = [r["outcomes"][arm] for r in rows]
            report["conditions"][arm][mode] = {"n": len(rows), "correct": sum(r["success"] for r in outcomes),
                "mean_steps_all": mean(r["steps"] for r in outcomes), "mean_cost_all": mean(r["cost"] for r in outcomes)}
            gains = [r["index"] for r in rows if r["outcomes"][arm]["success"] and not r["outcomes"]["filter"]["success"]]
            losses = [r["index"] for r in rows if r["outcomes"]["filter"]["success"] and not r["outcomes"][arm]["success"]]
            report["paired_vs_filter"][arm][mode] = {"gain": len(gains), "loss": len(losses),
                "unique_gain_queries": sorted(set(gains)), "unique_loss_queries": sorted(set(losses)),
                "mean_cost_delta_all": mean(r["outcomes"][arm]["cost"] - r["outcomes"]["filter"]["cost"] for r in rows)}
    constant = model["validation_best_static"]
    report["validation_selected_static"] = constant
    report["gate_vs_validation_selected_static"] = {
        "gain": sum(r["outcomes"]["gate"]["success"] and not r["outcomes"][constant]["success"] for r in records),
        "loss": sum(r["outcomes"][constant]["success"] and not r["outcomes"]["gate"]["success"] for r in records),
        "mean_cost_delta_all": mean(r["outcomes"]["gate"]["cost"] - r["outcomes"][constant]["cost"] for r in records)}
    write_new(root / "test_report.json", report)
    print(json.dumps({"report": str(root / "test_report.json"), "gate_selections": report["gate_selections"],
                      "overall": {a: modes["overall"] for a, modes in report["conditions"].items()},
                      "gate_vs_filter": report["paired_vs_filter"]["gate"]["overall"]}, ensure_ascii=False), flush=True)
def status(root: Path, manifest: dict[str, Any]) -> None:
    result = {}
    for name, sample in manifest["splits"].items():
        total = len(sample["indices"]) * len(manifest["modes"]) * len(ARMS)
        done = sum(completed(root / name / "panel" / arm / mode / f"q{i}") is not None
                   for arm in ARMS for mode in manifest["modes"] for i in sample["indices"])
        result[name] = {"queries": len(sample["indices"]), "completed": done, "episodes": total}
    result["live_gate"] = {"completed": sum(completed(root / "live_gate" / mode / f"q{i}") is not None
                           for mode in manifest["modes"] for i in manifest["splits"]["test"]["indices"]),
                           "episodes": len(manifest["modes"]) * len(manifest["splits"]["test"]["indices"])}
    print(json.dumps({"root": str(root), "progress": result, "gate_exists": (root / "gate.json").is_file(),
                      "qwen_training": False, "gate_training": "768 linear regression parameters on CPU",
                      "inference_feature": "initial visible user message only; no blocker mode or answers"}, ensure_ascii=False), flush=True)
def main() -> None:
    args = arguments()
    manifest = make_manifest(args)
    root = Path(args.root).resolve()
    status(root, manifest)
    if not args.run or args.stage == "status":
        print("Read-only preflight. Add --run to freeze splits/collect/fit/evaluate; no GPU used by this preflight.")
        return
    initialize(root, manifest)
    if args.stage in ("collect", "all"):
        collect(args, "train")
        collect(args, "validation")
    model = None
    if args.stage in ("train", "all", "evaluate"):
        model = train(root, manifest)
    if args.stage in ("evaluate", "all"):
        if args.stage == "all" and model["threshold"] is None:
            print("NOT PROMOTED: validation selected Filter-only. Training is complete; sealed test is left untouched. Inspect training_report.json, do not tune on test.", flush=True)
            return
        collect(args, "test")
        live_test(args, manifest, model)
    print(f"Stage {args.stage} completed. Resume with the identical command and root; do not modify frozen artifacts.", flush=True)
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
