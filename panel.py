# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Run a frozen, paired CostBench panel with five separately hosted arms.'
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import socket
from statistics import mean
import subprocess
import sys
import time
from typing import Any
from urllib.request import ProxyHandler, build_opener
ARMS = ("baseline", "filter", "evocond", "joint_exact", "joint_family")
MODES = ("none", "preference_change", "cost_change", "steplen_change", "ban_tool")
PROJECT = Path(__file__).resolve().parent
def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="runs/joint20260925/sample.json")
    parser.add_argument("--memory", default="runs/joint20260925/memory.json")
    parser.add_argument("--queries", default="assets/data/costbench/CostBench_queries.json")
    parser.add_argument("--model", default="assets/model")
    parser.add_argument("--output", default="runs/joint20260925/confirm12_v2")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--refinement", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--run", action="store_true", help="Actually run the panel; otherwise only validate and print the fixed plan")
    return parser.parse_args()
def protocol(args: argparse.Namespace) -> dict[str, Any]:
    if len(set(args.arms)) != len(args.arms) or len(set(args.modes)) != len(args.modes):
        raise ValueError("Arms and modes must not repeat")
    if not 1 <= args.port <= 65535 or args.tokens < 1 or args.refinement < 0 or args.threads < 1:
        raise ValueError("Invalid port or benchmark budget")
    sample_path = Path(args.sample).resolve(strict=True)
    memory_path = Path(args.memory).resolve(strict=True)
    query_path = Path(args.queries).resolve(strict=True)
    model_config = (Path(args.model) / "config.json").resolve(strict=True)
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    queries = json.loads(query_path.read_text(encoding="utf-8"))
    indices = sample.get("indices") if isinstance(sample, dict) else None
    if not isinstance(indices, list) or not indices or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("Frozen sample indices are invalid")
    if not isinstance(queries, list) or max(indices) >= len(queries):
        raise ValueError("Frozen sample contains an out-of-range query index")
    by_task = sample.get("by_task")
    if not isinstance(by_task, dict) or sorted(i for group in by_task.values() for i in group) != sorted(indices):
        raise ValueError("Frozen sample task groups do not match its indices")
    if any(not isinstance(queries[i], dict) or queries[i].get("task", "").strip().lower() != task for task, group in by_task.items() for i in group):
        raise ValueError("Frozen sample task labels do not match the query file")
    if sample.get("query_sha256") != digest(query_path) or sample.get("memory_sha256") != digest(memory_path):
        raise ValueError("Frozen sample does not match the query file and memory")
    allowance = sample.get("memory_extra_cost")
    if isinstance(allowance, bool) or not isinstance(allowance, (int, float)) or not isfinite(allowance) or allowance < 0:
        raise ValueError("Frozen sample has an invalid memory cost allowance")
    metadata = memory.get("metadata") if isinstance(memory, dict) else None
    training = metadata.get("train_indices") if isinstance(metadata, dict) else None
    if not isinstance(training, list) or not training or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in training) or len(set(training)) != len(training):
        raise ValueError("Frozen memory lacks valid training provenance")
    if metadata.get("source") != "costbench_traces_v1" or metadata.get("query_sha256") != digest(query_path) or set(training).intersection(indices):
        raise ValueError("Memory is not from this dataset or overlaps the frozen sample")
    from bench import repository
    repo = repository(None)
    for required in (repo / "env/data/static/search_spaces/test/search_space.json", repo / "env/config/travel_config.yaml", repo / "env/run.py"):
        if not required.is_file():
            raise FileNotFoundError(f"Required CostBench file is missing: {required}")
    source_files = ("bench.py", "serve.py", "control/guard.py", 'control/family.py', 'control/bridge.py', 'panel.py')
    return {
        "protocol_version": 1,
        "sample_sha256": digest(sample_path),
        "memory_sha256": digest(memory_path),
        "query_sha256": digest(query_path),
        "model_config_sha256": digest(model_config),
        "code_sha256": {name: digest(PROJECT / name) for name in source_files},
        "indices": sorted(indices),
        "arms": list(args.arms),
        "modes": list(args.modes),
        "memory_extra_cost": float(allowance),
        "port": args.port,
        "tokens": args.tokens,
        "refinement": args.refinement,
        "threads": args.threads,
    }
def server_args(arm: str, args: argparse.Namespace, trace: Path, allowance: float) -> list[str]:
    # Journal.local rejects absolute paths; serve.py runs with PROJECT as cwd.
    relative_trace = trace.resolve().relative_to(PROJECT)
    command = [sys.executable, "-B", "serve.py", "--model", args.model, "--port", str(args.port), "--trace", relative_trace.as_posix()]
    if arm != "baseline":
        command.extend(("--evocond", "--noaudit", "--levels", "filter" if arm == "filter" else "full"))
    if arm.startswith("joint_"):
        command.extend(("--memory", str(Path(args.memory).resolve()), "--memory-extra-cost", str(allowance), "--memory-backoff", "family" if arm == "joint_family" else "exact"))
    return command
def bench_args(arm: str, mode: str, index: int, args: argparse.Namespace, destination: Path, allowance: float) -> list[str]:
    command = [sys.executable, "-B", "bench.py", "--queries", str(Path(args.queries).resolve()), "--url", f"http://127.0.0.1:{args.port}/v1", "--start", str(index), "--end", str(index + 1), "--threads", str(args.threads), "--tokens", str(args.tokens), "--refinement", str(args.refinement), "--sample", str(Path(args.sample).resolve()), "--expect-arm", "joint" if arm.startswith("joint_") else arm, "--output", str(destination)]
    if mode != "none":
        command.extend(("--blocker", "--blockmode", mode))
    if arm.startswith("joint_"):
        command.extend(("--memory", str(Path(args.memory).resolve()), "--memory-extra-cost", str(allowance), "--memory-backoff", "family" if arm == "joint_family" else "exact"))
    return command
def completed(destination: Path) -> dict[str, Any] | None:
    if not destination.exists():
        return None
    files = list(destination.glob("results_*.json"))
    if not files and not any(destination.iterdir()):
        return None
    if len(files) != 1:
        raise FileExistsError(f"Existing output is not one complete result; inspect it manually: {destination}")
    data = json.loads(files[0].read_text(encoding="utf-8"))
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise ValueError(f"Existing output is malformed: {files[0]}")
    return data
def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(1.0)
        return connection.connect_ex(("127.0.0.1", port)) != 0
def wait_for_server(process: subprocess.Popen[Any], arm: str, args: argparse.Namespace, memory_hash: str, allowance: float, timeout: float = 180.0, expected_gate_hash: str | None = None) -> None:
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{arm} server exited during startup; inspect its log")
        try:
            with opener.open(f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
                health = json.loads(response.read())
        except (OSError, ValueError):
            time.sleep(2)
            continue
        expected_evo = arm != "baseline"
        expected_levels = None if arm == "baseline" else "filter" if arm == "filter" else "full"
        expected_hash = memory_hash if arm.startswith("joint_") else None
        expected_backoff = "family" if arm == "joint_family" else "exact" if arm == "joint_exact" else None
        if health.get("status") != "ok" or health.get("evocond") != expected_evo or health.get("levels") != expected_levels or health.get("noaudit") != expected_evo or health.get("memory_sha256") != expected_hash or health.get("memory_backoff") != expected_backoff:
            raise RuntimeError(f"{arm} service health does not match the frozen protocol: {health}")
        if arm.startswith("joint_") and health.get("memory_extra_cost") != allowance:
            raise RuntimeError(f"{arm} service uses the wrong memory cost allowance")
        if health.get("gate_sha256") != expected_gate_hash:
            raise RuntimeError("Static/gated service mismatch; refusing to mix policies")
        return
    raise TimeoutError(f"{arm} server did not become healthy within {timeout:g} seconds")
def stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
def prepare_output(root: Path, state: dict[str, Any]) -> None:
    manifest = root / "protocol.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != state:
            raise ValueError(f"Existing panel protocol differs; choose a new output directory: {root}")
        return
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output directory is not empty and has no protocol manifest: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
def summarize(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for arm in state["arms"]:
        rows[arm] = {}
        for mode in state["modes"]:
            batch = {}
            for index in state["indices"]:
                data = completed(root / arm / mode / f"q{index}")
                if data is None:
                    raise RuntimeError(f"Cannot summarize incomplete panel: {arm}/{mode}/q{index}")
                result = data["results"][0]
                metrics = ((data.get("stats") or {}).get("scores") or {}).get("cost_metrics") or {}
                raw_cost = metrics.get("total_agent_cost")
                if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)) or not isfinite(raw_cost):
                    raise ValueError(f"Missing or invalid agent cost in {arm}/{mode}/q{index}")
                batch[index] = {
                    "success": bool((result.get("evaluation") or {}).get("success")),
                    "status": result.get("status"),
                    "steps": int(result.get("total_steps") or 0),
                    "cost": float(raw_cost),
                }
            rows[arm][mode] = batch
    report: dict[str, Any] = {"protocol_sha256": digest(root / "protocol.json"), "n_per_condition": len(state["indices"]), "conditions": {}, "paired": {}}
    for arm, modes in rows.items():
        report["conditions"][arm] = {}
        for mode, batch in modes.items():
            values = list(batch.values())
            report["conditions"][arm][mode] = {
                "correct": sum(row["success"] for row in values),
                "n": len(values),
                "mean_steps_all": round(mean(row["steps"] for row in values), 3),
                "mean_cost_all": round(mean(row["cost"] for row in values), 3),
                "statuses": {status: sum(str(row["status"]) == status for row in values) for status in sorted({str(row["status"]) for row in values})},
            }
    for first, second in (("baseline", "filter"), ("filter", "evocond"), ("evocond", "joint_exact"), ("joint_exact", "joint_family")):
        if first not in rows or second not in rows:
            continue
        report["paired"][f"{second}_minus_{first}"] = {}
        for mode in state["modes"]:
            left, right = rows[first][mode], rows[second][mode]
            indices = state["indices"]
            report["paired"][f"{second}_minus_{first}"][mode] = {
                "gain": sum(not left[i]["success"] and right[i]["success"] for i in indices),
                "loss": sum(left[i]["success"] and not right[i]["success"] for i in indices),
                "mean_step_delta_all": round(mean(right[i]["steps"] - left[i]["steps"] for i in indices), 3),
                "mean_cost_delta_all": round(mean(right[i]["cost"] - left[i]["cost"] for i in indices), 3),
            }
    return report
def main() -> None:
    args = arguments()
    state = protocol(args)
    root = Path(args.output).resolve()
    if not root.is_relative_to(PROJECT):
        raise ValueError("--output must be inside the EvoCond project so the trace stays relative")
    total = len(state["indices"]) * len(args.arms) * len(args.modes)
    print(json.dumps({"output": str(root), "queries": len(state["indices"]), "arms": args.arms, "modes": args.modes, "episodes": total, "sample_sha256": state["sample_sha256"], "memory_sha256": state["memory_sha256"], "run": args.run}, ensure_ascii=False))
    if not args.run:
        print("Read-only preflight complete. Add --run to execute; this may require many GPU-backed episodes.")
        return
    prepare_output(root, state)
    (root / "logs").mkdir(exist_ok=True)
    (root / "events").mkdir(exist_ok=True)
    for arm in args.arms:
        pending = [(mode, index) for mode in args.modes for index in state["indices"] if completed(root / arm / mode / f"q{index}") is None]
        if not pending:
            print(f"{arm}: all frozen episodes already complete", flush=True)
            continue
        if not port_free(args.port):
            raise RuntimeError(f"Port {args.port} is already in use; stop the old terminal-1 server before running this panel")
        trace = root / "events" / f"{arm}.jsonl"
        log_path = root / "logs" / f"serve_{arm}.log"
        print(f"Starting {arm}: {len(pending)} episodes pending", flush=True)
        with log_path.open("a", encoding="utf-8") as server_log:
            process = subprocess.Popen(server_args(arm, args, trace, state["memory_extra_cost"]), cwd=PROJECT, stdout=server_log, stderr=subprocess.STDOUT)
            try:
                wait_for_server(process, arm, args, state["memory_sha256"], state["memory_extra_cost"])
                for mode, index in pending:
                    destination = root / arm / mode / f"q{index}"
                    query_log = root / "logs" / f"{arm}_{mode}_q{index}.log"
                    command = bench_args(arm, mode, index, args, destination, state["memory_extra_cost"])
                    trace_start = trace.stat().st_size if trace.exists() else 0
                    with query_log.open("w", encoding="utf-8") as log:
                        result = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=False)
                    if result.returncode:
                        raise RuntimeError(f"{arm}/{mode}/q{index} failed (exit {result.returncode}); inspect {query_log}")
                    data = completed(destination)
                    if data is None:
                        raise RuntimeError(f"{arm}/{mode}/q{index} produced no result; inspect {query_log}")
                    row = data["results"][0]
                    trace_end = trace.stat().st_size if trace.exists() else 0
                    with (root / "events" / f"{arm}_episodes.jsonl").open("a", encoding="utf-8") as mapping:
                        mapping.write(json.dumps({"mode": mode, "index": index, "trace_start_byte": trace_start,
                                                  "trace_end_byte": trace_end, "status": row.get("status")}) + "\n")
                    print(f"{arm}/{mode}/q{index}: status={row.get('status')} correct={(row.get('evaluation') or {}).get('success')} steps={row.get('total_steps')}", flush=True)
            finally:
                stop_server(process)
    report = summarize(root, state)
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Panel complete: {root}; summary: {summary_path}")
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
