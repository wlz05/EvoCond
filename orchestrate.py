# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Collect paired CostBench policy rollouts, then train three separate Qwen adapters.'
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
from control.trajectory import ARMS, adapter_digest, episode_record, verify_record
from gate import write_new
from panel import PROJECT, bench_args, digest, port_free, stop_server
from workflow import MODES, runner_lock, wait_server
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("status", "collect", "smoke", "train", "all"), nargs="?", default="status")
    parser.add_argument("--state-root", default="runs/state20260927_v1")
    parser.add_argument("--root", default="runs/qwen_three_arm_v1")
    parser.add_argument("--model", default="assets/model")
    parser.add_argument("--queries", default="assets/data/costbench/CostBench_queries.json")
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()
def frozen(args: argparse.Namespace) -> tuple[Path, Path, dict]:
    import math
    root, state_root = Path(args.root).resolve(), Path(args.state_root).resolve(strict=True)
    if root == PROJECT or not root.is_relative_to(PROJECT) or root == state_root or not state_root.is_relative_to(PROJECT):
        raise ValueError("Use distinct, dedicated output and state directories inside EvoCond")
    if args.rollouts < 2 or args.rollouts > 8 or not math.isfinite(args.temperature) or not 0 < args.temperature <= 2 or not 1 <= args.port <= 65535:
        raise ValueError("Invalid collection budget")
    old_path = state_root / "manifest.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    controller = state_root / 'controller.json'
    from control.controller import validate
    state = json.loads(controller.read_text(encoding="utf-8"))
    validate(state)
    data = Path(args.queries).resolve(strict=True)
    model = Path(args.model).resolve(strict=True)
    config, tokenizer = model / "config.json", model / "tokenizer_config.json"
    if old.get("query_sha256") != digest(data) or old.get("model_config_sha256") != digest(config):
        raise ValueError("State manifest, query data, and model do not match")
    current_weights = {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                       for p in sorted(model.glob("*.safetensors"))}
    if not current_weights or current_weights != old.get("model_weight_stats"):
        raise ValueError("Qwen checkpoint weights differ from the frozen state manifest")
    train = old["splits"]["train"]["indices"]
    validation = old["splits"]["validation"]["indices"]
    test = old["splits"]["test"]["indices"]
    provenance = state.get("provenance") or {}
    if provenance.get("manifest_sha256") != digest(old_path) or provenance.get("supervised_train_indices") != train:
        raise ValueError("State controller was not trained from this split")
    if set(train) & (set(validation) | set(test)) or set(validation) & set(test):
        raise ValueError("Train/validation/test identities overlap")
    sample = state_root / "train" / "sample.json"
    if json.loads(sample.read_text(encoding="utf-8")) != old["splits"]["train"]:
        raise ValueError("Frozen training sample differs from state manifest")
    from bench import repository
    repo = repository(None)
    for name in ("env/run.py", "env/config/travel_config.yaml", "env/data/static/search_spaces/test/search_space.json"):
        if not (repo / name).is_file():
            raise FileNotFoundError(repo / name)
    files = ("serve.py", "bench.py", 'orchestrate.py', 'update.py',
             'control/trajectory.py', 'control/controller.py')
    protocol = {"format": "evocond_three_arm_qlora_v1", "arms": list(ARMS), "modes": list(MODES),
                "train_indices": train, "validation_indices": validation, "test_indices": test,
                "rollouts_per_group": args.rollouts, "temperature": args.temperature,
                "port": args.port, "tokens": old["tokens"], "refinement": old["refinement"],
                "query_sha256": digest(data), "state_manifest_sha256": digest(old_path),
                "state_controller_sha256": digest(controller), "model_config_sha256": digest(config),
                "tokenizer_config_sha256": digest(tokenizer),
                "model_weight_stats": old["model_weight_stats"],
                "code_sha256": {name: digest(PROJECT / name) for name in files},
                "arm_notes": {"searl_adapted": "unguarded CostBench agent plus grouped outcome QLoRA; not official SEARL",
                              "evocond": "original full/noaudit controller plus grouped outcome QLoRA",
                              "teacher_dual": "partial learned dual-view state controller plus grouped outcome QLoRA"}}
    existing = root / "protocol.json"
    if existing.is_file() and json.loads(existing.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Frozen three-arm protocol changed; use a new output directory")
    if not existing.is_file() and root.exists() and any(root.iterdir()):
        raise FileExistsError("Output is nonempty without its protocol")
    return root, state_root, protocol
def record_path(root: Path, arm: str, mode: str, index: int, rollout: int) -> Path:
    return root / "episodes" / arm / mode / f"q{index}" / f"r{rollout}" / "record.json"
def checked_record(path: Path, arm: str, group: str, rollout: int, protocol: dict) -> dict | None:
    if not path.is_file():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    if (row.get("arm"), row.get("group"), row.get("rollout"), row.get("query_sha256")) != (arm, group, rollout, protocol["query_sha256"]):
        raise ValueError(f"Existing episode identity differs: {path}")
    verify_record(row)
    return row
def result_identity(record: dict) -> tuple:
    row = json.loads(Path(record["source"]["result"]).read_text(encoding="utf-8"))["results"][0]
    from control.gate import initial_user
    return row.get("query_id"), row.get("tool_creation_seed"), initial_user(row.get("conversation_history") or [])
def verify_paired_training(root: Path, protocol: dict, rollouts: int) -> None:
    for mode in MODES:
        for index in protocol["train_indices"]:
            identities = [result_identity(checked_record(record_path(root, arm, mode, index, rollout),
                                                 arm, f"q{index}/{mode}", rollout, protocol))
                          for arm in ARMS for rollout in range(rollouts)]
            if any(value != identities[0] for value in identities[1:]):
                raise ValueError(f"Train rollouts did not share the same query/seed: {mode}/q{index}")
def arm_server(arm: str, args: argparse.Namespace, state_root: Path, trace: Path) -> list[str]:
    command = [sys.executable, "-B", "serve.py", "--model", args.model, "--port", str(args.port),
               "--trace", trace.relative_to(PROJECT).as_posix(), "--trace-context",
               "--training-temperature", str(args.temperature)]
    if arm != "searl_adapted":
        command += ["--evocond", "--noaudit", "--levels", "full"]
    if arm == "teacher_dual":
        command += ["--state-controller", str(state_root / 'controller.json'), "--state-policy", "learned"]
    return command
def collect(args: argparse.Namespace, root: Path, state_root: Path, protocol: dict) -> None:
    fixed = argparse.Namespace(queries=args.queries, port=args.port, threads=1,
                               tokens=protocol["tokens"], refinement=protocol["refinement"],
                               sample=str(state_root / "train" / "sample.json"))
    for arm in ARMS:
        pending = [(mode, index, rollout) for mode in MODES for index in protocol["train_indices"]
                   for rollout in range(args.rollouts)
                   if checked_record(record_path(root, arm, mode, index, rollout), arm, f"q{index}/{mode}", rollout, protocol) is None]
        if not pending:
            print(f"{arm}: frozen rollouts complete", flush=True)
            continue
        if not port_free(args.port):
            raise RuntimeError(f"Port {args.port} is occupied; stop the old service in terminal 1")
        trace = root / "events" / f"{arm}.jsonl"
        relative = trace.relative_to(PROJECT)
        expected = {"status": "ok", "trace_context": True, "training_temperature": args.temperature,
                    "evocond": arm != "searl_adapted", "levels": None if arm == "searl_adapted" else "full",
                    "noaudit": arm != "searl_adapted", "memory_sha256": None,
                    "state_controller_sha256": protocol["state_controller_sha256"] if arm == "teacher_dual" else None,
                    "adapter_sha256": None}
        log_path = root / "logs" / f"serve_{arm}.log"
        print(f"{arm}: {len(pending)} train rollouts pending", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(arm_server(arm, args, state_root, trace), cwd=PROJECT,
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_server(process, args, expected)
                for mode, index, rollout in pending:
                    folder = record_path(root, arm, mode, index, rollout).parent
                    if folder.exists():
                        import uuid
                        preserved = root / "interrupted" / f"{arm}_{mode}_q{index}_r{rollout}_{uuid.uuid4().hex}"
                        preserved.parent.mkdir(exist_ok=True)
                        folder.rename(preserved)
                        print(f"Preserved incomplete episode: {preserved}", flush=True)
                    folder.mkdir(parents=True, exist_ok=False)
                    start = trace.stat().st_size if trace.exists() else 0
                    base = "baseline" if arm == "searl_adapted" else "evocond"
                    command = bench_args(base, mode, index, fixed, folder, 0.0)
                    if arm == "teacher_dual":
                        command += ["--state-controller", str(state_root / 'controller.json'),
                                    "--state-policy", "learned", "--state-split", "train"]
                    query_log = root / "logs" / f"{arm}_{mode}_q{index}_r{rollout}.log"
                    started = time.monotonic()
                    with query_log.open("w", encoding="utf-8") as stream:
                        result = subprocess.run(command, cwd=PROJECT, stdout=stream,
                                                stderr=subprocess.STDOUT, check=False)
                    if result.returncode:
                        raise RuntimeError(f"Episode failed; inspect {query_log}")
                    files = list(folder.glob("results_*.json"))
                    if len(files) != 1:
                        raise RuntimeError(f"Expected one benchmark result in {folder}")
                    end = trace.stat().st_size
                    row = episode_record(files[0], trace, start, end, arm, f"q{index}/{mode}", rollout,
                                         protocol["model_config_sha256"], protocol["tokenizer_config_sha256"],
                                         protocol["query_sha256"])
                    write_new(folder / "record.json", row)
                    print(f"{arm}/{mode}/q{index}/r{rollout}: success={row['success']} cost={row['tool_cost']:.2f} seconds={time.monotonic()-started:.1f}", flush=True)
            finally:
                stop_server(process)
    for arm in ARMS:
        rows = [checked_record(record_path(root, arm, mode, index, rollout), arm,
                               f"q{index}/{mode}", rollout, protocol)
                for mode in MODES for index in protocol["train_indices"] for rollout in range(args.rollouts)]
        if any(row is None for row in rows):
            raise RuntimeError(f"Incomplete {arm} training data")
        content = "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n"
        path = root / f"{arm}_rollouts.jsonl"
        if path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Frozen rollout file changed: {path}")
        else:
            path.write_text(content, encoding="utf-8")
    verify_paired_training(root, protocol, args.rollouts)
def verify_smoke_reload(args: argparse.Namespace, root: Path, arm: str, target: Path) -> None:
    adapter_hash = adapter_digest(target)
    marker = target / "reload_verified.json"
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != {"adapter_sha256": adapter_hash}:
            raise ValueError(f"Smoke reload marker no longer matches adapter: {target}")
        return
    if not port_free(args.port):
        raise RuntimeError(f"Cannot smoke-reload adapter while port {args.port} is occupied")
    reload_trace = root / "events" / f"reload_{arm}.jsonl"
    command = arm_server(arm, args, Path(args.state_root).resolve(), reload_trace)
    command += ["--adapter", str(target)]
    reload_log = root / "logs" / f"reload_{arm}.log"
    with reload_log.open("a", encoding="utf-8") as stream:
        server = subprocess.Popen(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
        try:
            wait_server(server, args, {"status": "ok", "adapter_sha256": adapter_hash,
                                       "evocond": arm != "searl_adapted"})
        finally:
            stop_server(server)
    write_new(marker, {"adapter_sha256": adapter_hash})
def train(args: argparse.Namespace, root: Path, protocol: dict, smoke: bool = False) -> None:
    from update import config, smoke_group
    verify_paired_training(root, protocol, args.rollouts)
    for arm in ARMS:
        source = root / f"{arm}_rollouts.jsonl"
        target = root / ("smoke_adapters" if smoke else "adapters") / arm
        if not source.is_file():
            raise FileNotFoundError(f"Collect all train rollouts first: {source}")
        settings = argparse.Namespace(arm=arm, rollouts=str(source), base_model=args.model,
                                      output=str(target), seed=20260928, learning_rate=1e-6,
                                      lora_r=8, max_prompt=4096, max_completion=2048,
                                      smoke=smoke)
        records, train_protocol = config(settings)
        if smoke:
            train_protocol["smoke_group"] = smoke_group(records)
        expected_groups = {f"q{index}/{mode}" for mode in MODES for index in protocol["train_indices"]}
        actual_groups = {row["group"] for row in records}
        if len(records) != len(expected_groups) * args.rollouts or actual_groups != expected_groups:
            raise ValueError("Arm has an incomplete or non-train training group set")
        for mode in MODES:
            for index in protocol["train_indices"]:
                rows = [row for row in records if row["group"] == f"q{index}/{mode}"]
                if {row["rollout"] for row in rows} != set(range(args.rollouts)):
                    raise ValueError("Arm has a missing or duplicate stochastic rollout")
                for row in rows:
                    folder = record_path(root, arm, mode, index, row["rollout"]).parent.resolve()
                    if Path(row["source"]["result"]).resolve().parent != folder:
                        raise ValueError("Training source is outside its frozen train episode folder")
        if target.exists():
            report = target / "training_report.json"
            if not report.is_file() or json.loads(report.read_text(encoding="utf-8")).get("protocol") != train_protocol:
                raise FileExistsError(f"Adapter is incomplete or from another frozen protocol: {target}")
            adapter_digest(target)
            if smoke:
                verify_smoke_reload(args, root, arm, target)
            print(f"{arm}: verified adapter already complete", flush=True)
            continue
        command = [sys.executable, "-B", 'update.py', "--arm", arm,
                   "--rollouts", str(source), "--base-model", args.model,
                   "--output", str(target), "--run"]
        if smoke:
            command.append("--smoke")
        log = root / "logs" / f"update_{arm}{'_smoke' if smoke else ''}.log"
        with log.open("a", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=PROJECT, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"Qwen update failed; inspect {log}")
        adapter_digest(target)
        if smoke:
            verify_smoke_reload(args, root, arm, target)
        print(json.dumps({"arm": arm, "adapter": str(target), "smoke": smoke,
                          "optimizer_steps": json.loads((target / "training_report.json").read_text())["optimizer_steps"]}), flush=True)
def main() -> None:
    args = arguments()
    root, state_root, protocol = frozen(args)
    progress = {arm: sum(checked_record(record_path(root, arm, mode, index, rollout), arm,
                                        f"q{index}/{mode}", rollout, protocol) is not None
                         for mode in MODES for index in protocol["train_indices"] for rollout in range(args.rollouts))
                for arm in ARMS}
    total = len(protocol["train_indices"]) * len(MODES) * args.rollouts
    print(json.dumps({"root": str(root), "completed": progress, "episodes_per_arm": total,
                      "adapter_exists": {arm: (root / "adapters" / arm / "training_report.json").is_file() for arm in ARMS},
                      "run": args.run, "stage": args.stage}, ensure_ascii=False), flush=True)
    if not args.run or args.stage == "status":
        print("Read-only preflight; no Qwen parameters changed.")
        return
    if args.stage in ("smoke", "train", "all"):
        from importlib.util import find_spec
        missing = [name for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes")
                   if find_spec(name) is None]
        if missing:
            raise RuntimeError(f"Qwen update dependencies missing: {missing}")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen update requires a visible CUDA GPU")
    write_new(root / "protocol.json", protocol)
    (root / "logs").mkdir(exist_ok=True)
    (root / "events").mkdir(exist_ok=True)
    with runner_lock(root):
        if args.stage in ("collect", "all"):
            collect(args, root, state_root, protocol)
        if args.stage in ("smoke", "all"):
            train(args, root, protocol, smoke=True)
        if args.stage in ("train", "all"):
            if any(not (root / "smoke_adapters" / arm / "reload_verified.json").is_file() for arm in ARMS):
                raise RuntimeError("Finish the three-arm one-group smoke update before full Qwen training")
            train(args, root, protocol)
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
