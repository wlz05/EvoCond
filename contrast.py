# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Paired validation of three separately trained Qwen LoRA adapters.'
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from statistics import mean
import time
from control.trajectory import ARMS, adapter_digest
from gate import write_new
from panel import PROJECT, bench_args, digest, port_free, stop_server
from workflow import MODES, runner_lock, wait_server
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", default="runs/qwen_three_arm_v1")
    parser.add_argument("--state-root", default="runs/state20260927_v1")
    parser.add_argument("--output", default="runs/qwen_three_arm_v1/validation_eval")
    parser.add_argument("--model", default="assets/model")
    parser.add_argument("--queries", default="assets/data/costbench/CostBench_queries.json")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()
def protocol(args: argparse.Namespace) -> tuple[Path, Path, dict]:
    train_root = Path(args.train_root).resolve(strict=True)
    state_root = Path(args.state_root).resolve(strict=True)
    output = Path(args.output).resolve()
    if not output.is_relative_to(PROJECT) or output == PROJECT or not 1 <= args.port <= 65535:
        raise ValueError("Use a dedicated validation output inside EvoCond")
    manifest = json.loads((train_root / "protocol.json").read_text(encoding="utf-8"))
    state = json.loads((state_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "evocond_three_arm_qlora_v1" or manifest.get("arms") != list(ARMS):
        raise ValueError("Three-arm training protocol is missing or wrong")
    if (manifest["query_sha256"] != digest(Path(args.queries)) or
        manifest["model_config_sha256"] != digest(Path(args.model) / "config.json") or
        manifest["state_manifest_sha256"] != digest(state_root / "manifest.json") or
        manifest["state_controller_sha256"] != digest(state_root / 'controller.json') or
        manifest["validation_indices"] != state["splits"]["validation"]["indices"]):
        raise ValueError("Validation assets differ from frozen training protocol")
    sample = state_root / "validation" / "sample.json"
    if json.loads(sample.read_text(encoding="utf-8")) != state["splits"]["validation"]:
        raise ValueError("Validation sample differs from frozen split")
    adapters = {}
    for arm in ARMS:
        path = train_root / "adapters" / arm
        report = json.loads((path / "training_report.json").read_text(encoding="utf-8"))
        trained = report.get("protocol") or {}
        if (report.get("arm") != arm or report.get("format") != "evocond_single_gpu_group_pg_v1" or
            trained.get("smoke") is not False or
            trained.get("model_config_sha256") != manifest["model_config_sha256"] or
            trained.get("tokenizer_config_sha256") != manifest["tokenizer_config_sha256"] or
            trained.get("rollouts_sha256") != digest(train_root / f"{arm}_rollouts.jsonl")):
            raise ValueError(f"Adapter {arm} does not match its frozen rollout data")
        adapters[arm] = adapter_digest(path)
    frozen = {"format": "evocond_three_arm_validation_v1", "train_protocol_sha256": digest(train_root / "protocol.json"),
              "state_manifest_sha256": digest(state_root / "manifest.json"),
              "code_sha256": {name: digest(PROJECT / name) for name in
                              ('contrast.py', "serve.py", "bench.py", 'control/trajectory.py')},
              "query_sha256": manifest["query_sha256"], "indices": manifest["validation_indices"],
              "modes": list(MODES), "arms": list(ARMS), "adapter_sha256": adapters,
              "tokens": manifest["tokens"], "refinement": manifest["refinement"], "port": args.port,
              "warning": "Exposed development validation only; SEARL arm is adapted, teacher arm partial."}
    existing = output / "protocol.json"
    if existing.exists() and json.loads(existing.read_text(encoding="utf-8")) != frozen:
        raise ValueError("Validation protocol changed; choose a new output directory")
    if not existing.exists() and output.exists() and any(output.iterdir()):
        raise FileExistsError("Output is nonempty without its protocol")
    return output, state_root, frozen
def completed(folder: Path) -> dict | None:
    meta = folder / "episode.json"
    if not meta.is_file():
        return None
    files = list(folder.glob("results_*.json"))
    if len(files) != 1:
        raise ValueError(f"Incomplete validation result: {folder}")
    audit = json.loads(meta.read_text(encoding="utf-8"))
    if audit.get("result_sha256") != digest(files[0]):
        raise ValueError(f"Validation result changed: {folder}")
    data = json.loads(files[0].read_text(encoding="utf-8"))
    row = data["results"][0]
    from workflow import outcome
    return outcome(row, data)
def identity(folder: Path) -> tuple:
    file = next(folder.glob("results_*.json"))
    row = json.loads(file.read_text(encoding="utf-8"))["results"][0]
    from control.gate import initial_user
    return row.get("query_id"), row.get("tool_creation_seed"), initial_user(row.get("conversation_history") or [])
def evaluate(args: argparse.Namespace, output: Path, state_root: Path, frozen: dict) -> None:
    fixed = argparse.Namespace(queries=args.queries, port=args.port, threads=1,
                               tokens=frozen["tokens"], refinement=frozen["refinement"],
                               sample=str(state_root / "validation" / "sample.json"))
    train_root = Path(args.train_root).resolve(strict=True)
    for arm in ARMS:
        pending = [(mode, index) for mode in MODES for index in frozen["indices"]
                   if completed(output / arm / mode / f"q{index}") is None]
        if not pending:
            print(f"{arm}: validation complete", flush=True)
            continue
        if not port_free(args.port):
            raise RuntimeError(f"Port {args.port} occupied; stop terminal 1's old model server")
        adapter = train_root / "adapters" / arm
        command = [sys.executable, "-B", "serve.py", "--model", args.model, "--adapter", str(adapter),
                   "--port", str(args.port), "--trace", (output / "events" / f"{arm}.jsonl").relative_to(PROJECT).as_posix()]
        if arm != "searl_adapted":
            command += ["--evocond", "--noaudit", "--levels", "full"]
        if arm == "teacher_dual":
            command += ["--state-controller", str(state_root / 'controller.json'), "--state-policy", "learned"]
        expected = {"status": "ok", "adapter_sha256": frozen["adapter_sha256"][arm],
                    "training_temperature": 0.0, "trace_context": False,
                    "evocond": arm != "searl_adapted", "noaudit": arm != "searl_adapted",
                    "levels": None if arm == "searl_adapted" else "full", "memory_sha256": None,
                    "state_controller_sha256": digest(state_root / 'controller.json') if arm == "teacher_dual" else None}
        with (output / "logs" / f"serve_{arm}.log").open("a", encoding="utf-8") as stream:
            server = subprocess.Popen(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
            try:
                wait_server(server, args, expected)
                for mode, index in pending:
                    folder = output / arm / mode / f"q{index}"
                    if folder.exists():
                        raise FileExistsError(f"Incomplete validation folder retained: {folder}")
                    base = "baseline" if arm == "searl_adapted" else "evocond"
                    bench = bench_args(base, mode, index, fixed, folder, 0.0)
                    bench += ["--adapter", str(adapter)]
                    if arm == "teacher_dual":
                        bench += ["--state-controller", str(state_root / 'controller.json'),
                                  "--state-policy", "learned", "--state-split", "validation"]
                    log = output / "logs" / f"{arm}_{mode}_q{index}.log"
                    started = time.monotonic()
                    with log.open("w", encoding="utf-8") as query_stream:
                        result = subprocess.run(bench, cwd=PROJECT, stdout=query_stream,
                                                stderr=subprocess.STDOUT, check=False)
                    if result.returncode:
                        raise RuntimeError(f"Validation episode failed; inspect {log}")
                    files = list(folder.glob("results_*.json"))
                    if len(files) != 1:
                        raise RuntimeError(f"Missing result: {folder}")
                    write_new(folder / "episode.json", {"result_sha256": digest(files[0])})
                    print(f"{arm}/{mode}/q{index}: {completed(folder)} seconds={time.monotonic()-started:.1f}", flush=True)
            finally:
                stop_server(server)
def summarize(output: Path, frozen: dict) -> dict:
    rows = []
    for mode in MODES:
        for index in frozen["indices"]:
            outcomes = {arm: completed(output / arm / mode / f"q{index}") for arm in ARMS}
            if any(value is None for value in outcomes.values()):
                raise RuntimeError("Validation is incomplete")
            identities = [identity(output / arm / mode / f"q{index}") for arm in ARMS]
            if any(value != identities[0] for value in identities[1:]):
                raise ValueError(f"Three arms did not receive the same problem/seed: {mode}/q{index}")
            rows.append({"mode": mode, "index": index, "outcomes": outcomes})
    overall = {arm: {"correct": sum(row["outcomes"][arm]["success"] for row in rows),
                     "n": len(rows),
                     "mean_steps_all": mean(row["outcomes"][arm]["steps"] for row in rows),
                     "mean_tool_cost_all": mean(row["outcomes"][arm]["cost"] for row in rows)} for arm in ARMS}
    report = {"split": "exposed_development_validation", "protocol_sha256": digest(output / "protocol.json"),
              "overall": overall, "records": rows,
              "warning": "Not official SEARL, not full teacher idea, not a fresh test; tool cost excludes Qwen inference cost."}
    write_new(output / "validation_report.json", report)
    return report
def main() -> None:
    args = arguments()
    output, state_root, frozen = protocol(args)
    count = sum(completed(output / arm / mode / f"q{index}") is not None
                for arm in ARMS for mode in MODES for index in frozen["indices"])
    print(json.dumps({"output": str(output), "complete": count,
                      "episodes": len(ARMS)*len(MODES)*len(frozen["indices"]), "run": args.run}), flush=True)
    if not args.run:
        print("Read-only preflight; no GPU evaluation started.")
        return
    write_new(output / "protocol.json", frozen)
    (output / "logs").mkdir(exist_ok=True)
    (output / "events").mkdir(exist_ok=True)
    with runner_lock(output):
        evaluate(args, output, state_root, frozen)
    report = summarize(output, frozen)
    print(json.dumps({"overall": report["overall"], "warning": report["warning"]}), flush=True)
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
