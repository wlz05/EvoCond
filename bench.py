# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import argparse
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bench")
    parser.add_argument("--repo")
    parser.add_argument("--queries", default="assets/data/costbench/CostBench_queries.json")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--adapter", help="Expected separately trained LoRA adapter")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--refinement", type=int, default=2)
    parser.add_argument("--output", default="runs/costbench")
    parser.add_argument("--blocker", action="store_true")
    parser.add_argument("--blockmode", choices=("preference_change", "cost_change", "steplen_change", "ban_tool"), default="cost_change")
    parser.add_argument("--memory", help="Frozen CostBench memory used by the running server; rejects training/test overlap")
    parser.add_argument("--memory-extra-cost", type=float, default=0.0, help="Cost allowance configured on the memory-backed server")
    parser.add_argument("--memory-backoff", choices=("exact", "family"), default="exact", help="Memory policy configured on the running service")
    parser.add_argument("--sample", help="Optional frozen sample manifest; requires a one-query range contained in its indices")
    parser.add_argument("--expect-arm", choices=("baseline", "filter", "evocond", "joint"), help="Reject a service started with another ablation mode")
    parser.add_argument("--gate", help="Frozen task gate used by the running service; rejects train/validation overlap")
    parser.add_argument("--state-controller", help="Frozen state-controller artifact; train/test provenance checked")
    parser.add_argument("--state-policy", choices=("fixed", "learned"), default="learned")
    parser.add_argument("--state-experiment", choices=("legacy", "baseline", "prompt_only", "selective"),
                        default="legacy", help="Expected explicit prompt/ranking ablation on the service")
    parser.add_argument("--state-split", choices=("train", "validation", "test"))
    parser.add_argument("--state-value-audit", action="store_true", help="Require the state-controlled service to audit preference values")
    return parser.parse_args()
def repository(raw: str | None) -> Path:
    candidates = [Path(raw)] if raw else [Path("assets/repos/costbench"), Path("assets/repos/CostBench-main")]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("CostBench repository was not found")
def relative(target: Path, root: Path) -> str:
    return Path(os.path.relpath(target, root)).as_posix()
def check_state_experiment(args: argparse.Namespace, health: dict) -> None:
    expected = args.state_experiment
    if expected != "legacy":
        if not args.state_controller or args.state_split not in ("train", "validation"):
            raise ValueError("New state ablations are development-only, not a fresh-test protocol")
        if args.state_value_audit:
            raise ValueError("Prompt/ranking ablations must not mix value auditing")
    actual = health.get("state_experiment")
    if args.state_controller and (actual or "legacy") != expected:
        raise ValueError("Running state-experiment mode differs; do not pool incompatible arms")
def prepare_experiment_output(args: argparse.Namespace, health: dict, queries: Path) -> None:
    if args.state_experiment == "legacy":
        return
    from serve import experiment_source_hashes
    expected_sources = experiment_source_hashes()
    if health.get("state_experiment_source_sha256") != expected_sources:
        raise ValueError("Service code differs from the local diagnostic source; restart with matched code")
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("State ablations need a new output directory; no automatic resume or arm mixing")
    protocol = {"format": "evocond_state_experiment_development_v1",
                "experiment": args.state_experiment, "state_policy": args.state_policy,
                "split": args.state_split, "fresh_test": False,
                "query_sha256": sha256(queries.read_bytes()).hexdigest(),
                "start": args.start, "end": args.end,
                "mode": args.blockmode if args.blocker else "none",
                "tokens": args.tokens, "refinement": args.refinement,
                "service": {key: health.get(key) for key in (
                    "model", "adapter_sha256", "state_controller_sha256", "state_experiment",
                    "state_experiment_source_sha256", "training_temperature")},
                "bench_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "warning": "Development diagnostic only; model path/config is not a full base-weight hash."}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "diagnostic_protocol.json").open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2, allow_nan=False)
def main() -> None:
    args = arguments()
    if args.start < 0 or args.end != -1 and args.end <= args.start:
        raise ValueError("Query range is invalid")
    if args.threads < 1 or args.tokens < 1 or args.refinement < 0:
        raise ValueError("Benchmark parameters must be positive")
    if not isfinite(args.memory_extra_cost) or args.memory_extra_cost < 0:
        raise ValueError("--memory-extra-cost must be finite and nonnegative")
    if args.memory_extra_cost and not args.memory:
        raise ValueError("--memory-extra-cost requires --memory")
    if args.memory_backoff != "exact" and not args.memory:
        raise ValueError("--memory-backoff family requires --memory")
    repo = repository(args.repo)
    queries = Path(args.queries)
    search = repo / "env/data/static/search_spaces/test/search_space.json"
    legacy = repo / "env/data/static/search_spaces/test_search_space.json"
    source = repo / "env/config/travel_config.yaml"
    for path in (queries, search, source, repo / "env/run.py"):
        if not path.is_file():
            raise FileNotFoundError(f"Required benchmark file is missing: {path}")
    sample = None
    if args.sample:
        sample = json.loads(Path(args.sample).read_text(encoding="utf-8"))
        values = sample.get("indices") if isinstance(sample, dict) else None
        if not isinstance(values, list) or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in values) or len(set(values)) != len(values):
            raise ValueError("Frozen sample indices are invalid")
        if args.end != args.start + 1 or args.start not in values:
            raise ValueError("Benchmark range must be one query from the frozen sample")
        if sample.get("query_sha256") != sha256(queries.read_bytes()).hexdigest():
            raise ValueError("Frozen sample was created from another CostBench query file")
    endpoint = urlsplit(args.url)
    if endpoint.scheme != "http" or not endpoint.netloc or endpoint.path.rstrip("/") != "/v1":
        raise ValueError("The model endpoint must be an explicit HTTP /v1 URL")
    with build_opener(ProxyHandler({})).open(f"{endpoint.scheme}://{endpoint.netloc}/health", timeout=5) as response:
        health = json.loads(response.read())
    check_state_experiment(args, health)
    if args.adapter:
        from control.trajectory import adapter_digest
        expected_adapter = adapter_digest(Path(args.adapter).resolve(strict=True))
        if health.get("adapter_sha256") != expected_adapter:
            raise ValueError("Running server loaded another LoRA adapter")
    elif health.get("adapter_sha256") is not None:
        raise ValueError("Running server has a trained adapter; pass --adapter to verify it")
    if args.state_controller:
        if args.gate or args.memory or args.end != args.start + 1 or not args.state_split:
            raise ValueError("State controller needs one query and --state-split; cannot mix old gate/memory")
        from control.controller import validate
        raw = Path(args.state_controller).read_bytes()
        controller = json.loads(raw)
        validate(controller)
        provenance = controller.get("provenance") or {}
        parts = {key: provenance.get(key + "_indices") for key in ("train", "validation", "test")}
        if any(not isinstance(v, list) or not v or any(type(i) is not int or i < 0 for i in v) or len(v) != len(set(v)) for v in parts.values()):
            raise ValueError("State controller lacks split provenance")
        if set(parts["train"]) & set(parts["validation"]) or (set(parts["train"]) | set(parts["validation"])) & set(parts["test"]):
            raise ValueError("State controller partitions overlap")
        if args.start not in parts[args.state_split] or provenance.get("query_sha256") != sha256(queries.read_bytes()).hexdigest():
            raise ValueError("Wrong state-controller query split or dataset")
        if health.get("state_controller_sha256") != sha256(raw).hexdigest() or health.get("state_policy") != args.state_policy:
            raise ValueError("Running state controller does not match the requested artifact/policy")
        if health.get("state_value_audit") is not args.state_value_audit:
            raise ValueError("Running state controller value-audit mode differs")
    elif args.state_value_audit:
        raise ValueError("Value audit requires --state-controller")
    elif health.get("state_controller_sha256"):
        raise ValueError("A state-controlled server cannot be evaluated as an old static arm")
    if args.gate:
        if not args.memory or args.end != args.start + 1:
            raise ValueError("Gate evaluation requires frozen memory and a one-query range")
        gate_path = Path(args.gate)
        gate_raw = gate_path.read_bytes()
        gate = json.loads(gate_raw)
        from control.gate import validate_model
        validate_model(gate)
        provenance = gate.get("provenance") or {}
        training = provenance.get("train_indices")
        validation = provenance.get("validation_indices")
        testing = provenance.get("test_indices")
        if any(not isinstance(values, list) or not values or any(type(i) is not int or i < 0 for i in values) or len(values) != len(set(values)) for values in (training, validation, testing)):
            raise ValueError("Gate split provenance is missing or malformed")
        if set(training) & set(validation) or (set(training) | set(validation)) & set(testing):
            raise ValueError("Gate train/validation/test overlap")
        if args.start not in testing or args.start in set(training) | set(validation):
            raise ValueError("Gate evaluation must use a frozen test query, never training/validation")
        if provenance.get("query_sha256") != sha256(queries.read_bytes()).hexdigest() or provenance.get("memory_sha256") != sha256(Path(args.memory).read_bytes()).hexdigest():
            raise ValueError("Gate dataset/memory does not match")
        if health.get("gate_sha256") != sha256(gate_raw).hexdigest():
            raise ValueError("Running server loaded a different task gate")
    elif health.get("gate_sha256"):
        raise ValueError("A gated server cannot be evaluated as a static arm; pass --gate")
    if args.expect_arm:
        expected = {"baseline": (False, None, None), "filter": (True, "filter", True), "evocond": (True, "full", True), "joint": (True, "full", True)}[args.expect_arm]
        if health.get("evocond") != expected[0] or expected[1] is not None and (health.get("levels"), health.get("noaudit")) != expected[1:]:
            raise ValueError(f"The running service is not the requested {args.expect_arm} arm")
        if (args.expect_arm == "joint") != bool(args.memory):
            raise ValueError("The joint arm alone requires --memory")
    if args.memory:
        if args.end == -1:
            raise ValueError("Memory evaluation requires an explicit --end for split checking")
        memory_path = Path(args.memory)
        raw = memory_path.read_bytes()
        snapshot = json.loads(raw)
        metadata = snapshot.get("metadata") if isinstance(snapshot, dict) else None
        training = metadata.get("train_indices") if isinstance(metadata, dict) else None
        if not isinstance(metadata, dict) or metadata.get("source") != "costbench_traces_v1" or not isinstance(training, list) or not training or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in training) or len(set(training)) != len(training):
            raise ValueError("Memory is missing valid CostBench training-index provenance")
        if set(training).intersection(range(args.start, args.end)):
            raise ValueError("Evaluation range overlaps memory-training indices")
        if metadata.get("query_sha256") != sha256(queries.read_bytes()).hexdigest():
            raise ValueError("Memory was built for a different CostBench query file")
        if sample is not None and sample.get("memory_sha256") != sha256(raw).hexdigest():
            raise ValueError("Frozen sample was paired with another memory snapshot")
        if sample is not None and sample.get("memory_extra_cost") != args.memory_extra_cost:
            raise ValueError("Frozen sample uses another memory cost allowance")
        if health.get("memory_sha256") != sha256(raw).hexdigest():
            raise ValueError("The running service did not load this exact frozen memory snapshot")
        if health.get("memory_extra_cost") != args.memory_extra_cost:
            raise ValueError("The running service uses another memory cost allowance")
        if health.get("memory_backoff") != args.memory_backoff:
            raise ValueError("The running service uses another memory backoff mode")
        if Path(args.output).exists() and any(Path(args.output).iterdir()):
            raise FileExistsError("Memory evaluation needs a new empty output directory")
    elif health.get("memory_sha256"):
        raise ValueError("The running service has frozen memory; pass --memory so the evaluation split is checked")
    prepare_experiment_output(args, health, queries)
    if args.blocker and not legacy.is_file():
        shutil.copyfile(search, legacy)
    import yaml
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = output / "runtime.yaml" if args.state_experiment != "legacy" else Path("runs/costbench.yaml")
    runtime.parent.mkdir(parents=True, exist_ok=True)
    config["paths"]["query_path"] = relative(queries, repo)
    config["paths"]["search_space_path"] = "env/data/static/search_spaces/test/search_space.json"
    config["paths"]["output_dir"] = relative(output, repo)
    config["model"]["model_name"] = args.model
    config["model"]["max_tokens"] = args.tokens
    config["model"]["endpoints"][args.model] = {"base_url": args.url.rstrip("/"), "api_key": "EMPTY"}
    runtime.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8")
    command = [sys.executable, "env/run.py", "--refinement_level", str(args.refinement), "--ban_longest_tool", "--model_name", args.model, "--num_threads", str(args.threads), "--start_index", str(args.start), "--end_index", str(args.end), "--query_path", relative(queries, repo), "--output_dir", relative(output, repo)]
    if args.blocker:
        command.extend(["--use_blocker", "--block_num", "1", "--block_mode", args.blockmode])
    environment = os.environ.copy()
    environment["COSTBENCH_TRAVEL_CONFIG"] = relative(runtime, repo)
    subprocess.run(command, cwd=repo, env=environment, check=True)
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
