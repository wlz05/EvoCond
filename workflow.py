# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Frozen state-controller workflow. Default/status are read-only.'
from __future__ import annotations
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import uuid
from statistics import mean
from urllib.request import ProxyHandler, build_opener
from control.controller import FORMAT, catalog, validate
from control.observation import fit
from control.gate import normalize
from sample import TASKS, seen_indices
from gate import source_hashes, write_new
from panel import PROJECT, bench_args, digest, port_free, server_args, stop_server
TEACHERS = ("evocond", "joint_family")
ARMS = ("filter", "evocond", "joint_family", "dual_fixed", "state_learned")
MODES = ("none", "preference_change")
def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", nargs="?", choices=("all", "prepare", "train", "validate", "test", "status"), default="all")
    p.add_argument("--root", default="runs/state20260927_v1")
    p.add_argument("--previous", default="runs/gate20260927_s54")
    p.add_argument("--queries", default="assets/data/costbench/CostBench_queries.json")
    p.add_argument("--memory", default="runs/joint20260925/memory.json")
    p.add_argument("--model", default="assets/model")
    p.add_argument("--exclude-root", default="runs")
    p.add_argument("--test-size", type=int, default=18)
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument("--tokens", type=int, default=2048)
    p.add_argument("--refinement", type=int, default=2)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--run", action="store_true")
    return p.parse_args()
def public_text(row):
    text = row.get("user_requirements") or row.get("query") or ""
    return normalize(text if isinstance(text, str) else json.dumps(text, sort_keys=True))
def fresh_sample(queries, excluded, size, seed):
    'Round-robin across available domains; report exhausted domains openly.'
    known = {public_text(queries[i]) for i in excluded if 0 <= i < len(queries)}
    groups = {t: [] for t in sorted(TASKS)}
    rng = random.Random(seed)
    order = list(range(len(queries)))
    rng.shuffle(order)
    for i in order:
        row = queries[i]
        text, task = public_text(row), str(row.get("task", "")).strip().lower()
        if task not in groups:
            raise ValueError(f"Invalid task category at {i}")
        if i not in excluded and text and text not in known:
            groups[task].append(i)
            known.add(text)
    available = {t: len(v) for t, v in groups.items()}
    if sum(available.values()) < size:
        raise ValueError(f"Only {sum(available.values())} unused distinct queries remain; need {size}. Availability: {available}. No old test reuse or automatic shrinking.")
    selected = []
    while len(selected) < size:
        for task in groups:
            if groups[task] and len(selected) < size:
                selected.append(groups[task].pop())
    return sorted(selected), available
def make_manifest(args):
    root = Path(args.root).resolve()
    if root == PROJECT or not root.is_relative_to(PROJECT):
        raise ValueError("Output must be a dedicated directory within EvoCond")
    if args.test_size < 2 or args.tokens < 1 or args.refinement < 0 or not 1 <= args.port <= 65535:
        raise ValueError("Invalid sample or inference budget")
    previous = Path(args.previous) / "manifest.json"
    old = json.loads(previous.read_text(encoding="utf-8"))
    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    memory = json.loads(Path(args.memory).read_text(encoding="utf-8"))
    if not isinstance(queries, list) or not queries or any(not isinstance(r, dict) or str(r.get("task", "")).strip().lower() not in TASKS for r in queries):
        raise ValueError("Malformed CostBench query dataset")
    qhash, mhash = digest(Path(args.queries)), digest(Path(args.memory))
    config = digest(Path(args.model) / "config.json")
    if old.get("query_sha256") != qhash or old.get("memory_sha256") != mhash or old.get("model_config_sha256") != config:
        raise ValueError("Previous workflow dataset/memory/model differs")
    if old.get("tokens") != args.tokens or old.get("refinement") != args.refinement:
        raise ValueError("Preserve the previous inference budget")
    parts = {key: old["splits"][key]["indices"] for key in ("train", "validation", "test")}
    for values in parts.values():
        if not isinstance(values, list) or not values or any(type(i) is not int or not 0 <= i < len(queries) for i in values) or len(values) != len(set(values)):
            raise ValueError("Previous split is malformed")
    flat = [i for values in parts.values() for i in values]
    metadata = memory.get("metadata", {})
    memory_train = metadata.get("train_indices", [])
    if metadata.get("source") != "costbench_traces_v1" or metadata.get("query_sha256") != qhash or not isinstance(memory_train, list) or not memory_train or any(type(i) is not int or not 0 <= i < len(queries) for i in memory_train) or len(memory_train) != len(set(memory_train)):
        raise ValueError("Memory lacks valid train-only provenance")
    if len(flat) != len(set(flat)) or set(flat) & set(memory_train):
        raise ValueError("Previous partitions overlap each other or memory training")
    texts = [public_text(queries[i]) for i in flat]
    if any(not x for x in texts) or len(texts) != len(set(texts)):
        raise ValueError("Previous partitions contain normalized duplicate prompts")
    if set(texts) & {public_text(queries[i]) for i in memory_train}:
        raise ValueError("Previous split has normalized prompt overlap with memory training")
    signature = {"format": FORMAT, "query_sha256": qhash, "memory_sha256": mhash,
                 "model_config_sha256": config, "previous_manifest_sha256": digest(previous),
                 "source_sha256": source_hashes(), "tokens": args.tokens, "refinement": args.refinement,
                 "test_size": args.test_size, "seed": args.seed, "port": args.port,
                 "teachers": list(TEACHERS), "arms": list(ARMS), "modes": list(MODES),
                 "model_weight_stats": {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                                        for p in sorted(Path(args.model).glob("*.safetensors"))}}
    path = root / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if any(manifest.get(k) != v for k, v in signature.items()):
            raise ValueError("Frozen source/data/config changed; do not overwrite or reuse this run")
        if any(manifest["splits"][k]["indices"] != sorted(parts[k]) for k in ("train", "validation")):
            raise ValueError("Frozen development identities changed")
        selected = manifest["splits"]["test"]["indices"]
        excluded = set(manifest["excluded_indices"])
        if not (set(flat) | set(memory_train)).issubset(excluded) or len(selected) != args.test_size or len(selected) != len(set(selected)) or any(type(i) is not int or not 0 <= i < len(queries) for i in selected) or set(selected) & excluded:
            raise ValueError("Frozen fresh test overlaps exposed data or is malformed")
        test_texts = [public_text(queries[i]) for i in selected]
        if any(not t for t in test_texts) or len(test_texts) != len(set(test_texts)) or set(test_texts) & {public_text(queries[i]) for i in excluded if 0 <= i < len(queries)}:
            raise ValueError("Frozen fresh test contains exposed/duplicate prompts")
        return manifest
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Output is nonempty without a manifest")
    excluded = seen_indices(Path(args.exclude_root), queries) | set(flat) | set(memory_train)
    selected, availability = fresh_sample(queries, excluded, args.test_size, args.seed)
    splits = {}
    for key, values in (("train", parts["train"]), ("validation", parts["validation"]), ("test", selected)):
        splits[key] = {"indices": sorted(values), "by_task": {
            t: [i for i in sorted(values) if queries[i]["task"].strip().lower() == t] for t in sorted(TASKS)},
            "query_sha256": qhash, "memory_sha256": mhash, "memory_extra_cost": 0.5}
    return {**signature, "splits": splits, "excluded_indices": sorted(excluded),
            "memory_train_indices": memory_train, "old_exposed_test_indices": parts["test"],
            "available_fresh_by_domain": availability,
            "test_sampling": "round_robin_remaining_domains_not_guaranteed_six_domain_balance",
            "warning": "Old validation is development data. Fresh test is sealed. Weight size/mtime is not a full content hash."}
def data_at(root, split, arm, mode, index):
    folder = root / split / arm / mode / f"q{index}"
    meta = folder / "episode.json"
    files = list(folder.glob("results_*.json"))
    if len(files) != 1 or not meta.is_file():
        return None
    try:
        data = json.loads(files[0].read_text(encoding="utf-8"))
        audit = json.loads(meta.read_text(encoding="utf-8"))
        if audit["result_sha256"] != digest(files[0]):
            raise ValueError(f"Completed result changed: {folder}")
        if not isinstance(data.get("results"), list) or len(data["results"]) != 1:
            raise ValueError(f"Expected one query result: {folder}")
        row = data["results"][0]
        outcome(row, data)
        return data, audit
    except json.JSONDecodeError:
        return None
def outcome(row, data):
    result = {"success": (row.get("evaluation") or {}).get("success"),
              "steps": row.get("total_steps"),
              "cost": data["stats"]["scores"]["cost_metrics"]["total_agent_cost"]}
    if type(result["success"]) is not bool or type(result["steps"]) is not int or result["steps"] < 0 or type(result["cost"]) not in (int, float) or not math.isfinite(result["cost"]) or result["cost"] < 0:
        raise ValueError("Malformed episode metrics")
    return result
def status(root, manifest):
    progress = {}
    for split, part in manifest["splits"].items():
        arms = TEACHERS if split == "train" else ARMS
        total = len(part["indices"]) * len(MODES) * len(arms)
        done = sum(data_at(root, split, a, m, i) is not None for a in arms for m in MODES for i in part["indices"])
        progress[split] = {"completed": done, "episodes": total}
    print(json.dumps({"root": str(root), "progress": progress, "maximum_episodes": sum(x["episodes"] for x in progress.values()),
                      "fresh_test_by_domain": {t: len(v) for t, v in manifest["splits"]["test"]["by_task"].items()},
                      "state_controller_exists": (root / 'controller.json').exists(), "qwen_updated": False}, ensure_ascii=False), flush=True)
def process_exists(pid):
    'Probe a lock owner without sending a Windows console control event.'
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("Runner lock PID must be a positive integer")
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE, not terminate.
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER: no such PID.
                return False
            if error == 5:  # Access denied is not evidence of a stale lock.
                return True
            raise ctypes.WinError(error)
        try:
            result = kernel.WaitForSingleObject(handle, 0)
            if result in (0, 258):  # Signaled process exited; timeout means still alive.
                return result == 258
            raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
@contextmanager
def runner_lock(root):
    path = root / "runner.lock"
    token = uuid.uuid4().hex
    if path.exists():
        prior = json.loads(path.read_text())
        if not process_exists(prior["pid"]):
            path.unlink()  # One verified stale lock, never a result directory.
        else:
            raise RuntimeError(f"Runner PID {prior['pid']} still exists; do not launch two runners")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump({"pid": os.getpid(), "token": token}, stream)
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text()).get("token") == token:
            path.unlink()
def wait_server(process, args, expected):
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Server exited; inspect the corresponding logs/serve_*.log")
        try:
            with opener.open(f"http://127.0.0.1:{args.port}/health", timeout=2) as r:
                health = json.loads(r.read())
        except (OSError, ValueError):
            time.sleep(1)
            continue
        if any(health.get(k) != v for k, v in expected.items()):
            raise RuntimeError(f"Wrong server protocol: {health}")
        return
    raise TimeoutError("Model server did not become ready within 180 seconds")
def execute(command, stream):
    process = subprocess.Popen(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT,
                               start_new_session=os.name == "posix")
    try:
        if process.wait() != 0:
            raise RuntimeError("Benchmark failed; inspect its episode log")
    finally:
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            else:
                stop_server(process)
def collect(args, manifest, split):
    root = Path(args.root).resolve()
    indices = manifest["splits"][split]["indices"]
    controller = root / 'controller.json'
    fixed = argparse.Namespace(**vars(args), sample=str(root / split / "sample.json"), threads=1)
    for arm in TEACHERS if split == "train" else ARMS:
        pending = [(m, i) for m in MODES for i in indices if data_at(root, split, arm, m, i) is None]
        if not pending:
            continue
        if not port_free(args.port):
            raise RuntimeError(f"Port {args.port} occupied; stop the old service in its terminal, not arbitrary Python processes")
        new = arm in ("dual_fixed", "state_learned")
        policy = "fixed" if arm == "dual_fixed" else "learned"
        base_arm = "evocond" if new else arm
        trace = root / "events" / f"{split}_{arm}.jsonl"
        command = server_args(base_arm, fixed, trace, 0.5)
        if split == "train":
            command += ["--trace-context"]
        if new:
            command += ["--state-controller", str(controller), "--state-policy", policy]
        expected = {"status": "ok", "evocond": True, "levels": "filter" if arm == "filter" else "full",
                    "noaudit": True, "gate_sha256": None,
                    "memory_sha256": manifest["memory_sha256"] if arm == "joint_family" else None,
                    "memory_backoff": "family" if arm == "joint_family" else None,
                    "memory_extra_cost": 0.5 if arm == "joint_family" else None,
                    "state_controller_sha256": digest(controller) if new else None,
                    "state_policy": policy if new else None, "trace_context": split == "train"}
        print(f"{split}/{arm}: {len(pending)} episodes pending", flush=True)
        with (root / "logs" / f"serve_{split}_{arm}.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_server(process, args, expected)
                for mode, index in pending:
                    folder = root / split / arm / mode / f"q{index}"
                    # Preserve, never overwrite, an interrupted result whose complete result+trace association was.
                    if folder.exists():
                        preserved = root / "interrupted" / f"{split}_{arm}_{mode}_q{index}_{uuid.uuid4().hex}"
                        if not folder.resolve().is_relative_to(root):
                            raise ValueError("Unsafe result directory")
                        preserved.parent.mkdir(exist_ok=True)
                        folder.rename(preserved)
                        print(f"Preserved incomplete output: {preserved}", flush=True)
                    start = trace.stat().st_size if trace.exists() else 0
                    cmd = bench_args(base_arm, mode, index, fixed, folder, 0.5)
                    if new:
                        cmd += ["--state-controller", str(controller), "--state-policy", policy, "--state-split", split]
                    started = time.monotonic()
                    with (root / "logs" / f"{split}_{arm}_{mode}_q{index}.log").open("a", encoding="utf-8") as qlog:
                        execute(cmd, qlog)
                    files = list(folder.glob("results_*.json"))
                    if len(files) != 1:
                        raise RuntimeError(f"Missing/duplicate result in {folder}")
                    data = json.loads(files[0].read_text(encoding="utf-8"))
                    score = outcome(data["results"][0], data)
                    end = trace.stat().st_size if trace.exists() else start
                    with trace.open("rb") as stream:
                        stream.seek(start)
                        raw = stream.read(end-start)
                    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
                    usage = [e for e in events if e.get("action") == "request_usage"]
                    if not usage:
                        raise RuntimeError("Missing server usage trace; cannot silently omit planning/token overhead")
                    if split == "train" and not any(catalog(e.get("tools") or []) for e in events if e.get("action") == "state_training_observation"):
                        raise RuntimeError("Training episode has no parseable visible schemas; stop before spending GPU time on more episodes")
                    audit = {"result_sha256": digest(files[0]), "trace": trace.relative_to(root).as_posix(),
                             "start": start, "end": end, "trace_sha256": sha256(raw).hexdigest(),
                             "elapsed_seconds": time.monotonic()-started,
                             "prompt_tokens": sum(e["prompt_tokens"] for e in usage),
                             "completion_tokens": sum(e["completion_tokens"] for e in usage),
                             "controller_seconds": sum(e["controller_seconds"] for e in usage),
                             "state_plans": sum(e.get("action") == "state_plan" for e in events),
                             "reranks": sum(bool(e.get("reranked")) for e in events)}
                    write_new(folder / "episode.json", audit)
                    print(f"{split}/{arm}/{mode}/q{index}: {score} seconds={audit['elapsed_seconds']:.1f}", flush=True)
            finally:
                stop_server(process)
def training_episodes(root, manifest):
    episodes, hashes = [], {}
    for arm in TEACHERS:
        for mode in MODES:
            for index in manifest["splits"]["train"]["indices"]:
                found = data_at(root, "train", arm, mode, index)
                if found is None:
                    raise RuntimeError("Training collection incomplete")
                data, audit = found
                with (root / audit["trace"]).open("rb") as stream:
                    stream.seek(audit["start"])
                    raw = stream.read(audit["end"]-audit["start"])
                if sha256(raw).hexdigest() != audit["trace_sha256"]:
                    raise ValueError("Training trace changed")
                records = [json.loads(line) for line in raw.splitlines() if line.strip()]
                observations = [e for e in records if e.get("action") == "state_training_observation"]
                if not observations:
                    raise ValueError("Missing actual visible tool schemas; refusing to invent them")
                episodes.append({"index": index, "result": data["results"][0], "observations": observations})
                hashes[f"{arm}/{mode}/q{index}"] = [audit["result_sha256"], audit["trace_sha256"]]
    return episodes, hashes
def train(root, manifest):
    path = root / 'controller.json'
    if path.exists():
        model = json.loads(path.read_text(encoding="utf-8"))
        validate(model)
        if model["provenance"]["manifest_sha256"] != digest(root / "manifest.json"):
            raise ValueError("Trained controller and manifest differ")
        _, hashes = training_episodes(root, manifest)
        if model["provenance"]["training_artifacts"] != hashes:
            raise ValueError("Frozen training episodes changed; do not refit or resume")
        write_new(root / "training_report.json", {**model["training_report"], "controller_sha256": digest(path)})
        return model
    if any((root / "validation" / a).exists() for a in ARMS):
        raise ValueError("Validation started but frozen controller missing; do not refit")
    episodes, hashes = training_episodes(root, manifest)
    print("CPU training: matched-tool cross-view alignment and verified-step imitation gates. Qwen stays frozen.", flush=True)
    model, report = fit(episodes, manifest["seed"])
    model["provenance"] = {"manifest_sha256": digest(root / "manifest.json"),
                           "query_sha256": manifest["query_sha256"],
                           "model_config_sha256": manifest["model_config_sha256"],
                           "train_indices": sorted(set(manifest["splits"]["train"]["indices"]) | set(manifest["memory_train_indices"])),
                           "supervised_train_indices": manifest["splits"]["train"]["indices"],
                           "validation_indices": manifest["splits"]["validation"]["indices"],
                           "test_indices": manifest["splits"]["test"]["indices"], "training_artifacts": hashes}
    model["training_report"] = report.copy()
    write_new(path, model)
    report["controller_sha256"] = digest(path)
    write_new(root / "training_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return model
def summarize(root, manifest, split):
    rows = []
    for mode in MODES:
        for index in manifest["splits"][split]["indices"]:
            outcomes, identities, overhead = {}, [], {}
            for arm in ARMS:
                found = data_at(root, split, arm, mode, index)
                if found is None:
                    raise RuntimeError(f"Incomplete {split}: {arm}/{mode}/q{index}")
                data, audit = found
                row = data["results"][0]
                from control.gate import initial_user
                identities.append((row.get("query_id"), row.get("tool_creation_seed"), initial_user(row.get("conversation_history") or [])))
                outcomes[arm] = outcome(row, data)
                overhead[arm] = {k: audit[k] for k in ("elapsed_seconds", "prompt_tokens", "completion_tokens", "controller_seconds", "state_plans", "reranks")}
            if any(x != identities[0] for x in identities[1:]):
                raise ValueError("Paired query identities, seeds, or initial messages differ")
            rows.append({"index": index, "mode": mode, "outcomes": outcomes, "overhead": overhead})
    overall, paired = {}, {}
    for arm in ARMS:
        values = [r["outcomes"][arm] for r in rows]
        overall[arm] = {"n": len(rows), "correct": sum(v["success"] for v in values),
                        "mean_steps_all": mean(v["steps"] for v in values), "mean_cost_all": mean(v["cost"] for v in values),
                        **{f"mean_{k}": mean(r["overhead"][arm][k] for r in rows) for k in rows[0]["overhead"][arm]}}
    for reference in ARMS[:-1]:
        gains = [r for r in rows if r["outcomes"]["state_learned"]["success"] and not r["outcomes"][reference]["success"]]
        losses = [r for r in rows if not r["outcomes"]["state_learned"]["success"] and r["outcomes"][reference]["success"]]
        paired[reference] = {"gain": len(gains), "loss": len(losses),
                             "gain_queries": sorted({r["index"] for r in gains}), "loss_queries": sorted({r["index"] for r in losses}),
                             "mean_cost_delta": overall["state_learned"]["mean_cost_all"]-overall[reference]["mean_cost_all"]}
    best = min(ARMS[:-2], key=lambda a: (-overall[a]["correct"], overall[a]["mean_cost_all"], overall[a]["mean_steps_all"]))
    report = {"split": split, "unique_queries": len(manifest["splits"][split]["indices"]),
              "overall": overall, "learned_vs": paired, "best_static_on_this_split": best,
              "controller_sha256": digest(root / 'controller.json'), "records": rows,
              "warning": "Tool cost excludes model cost; tokens and wall time reported separately. Small grouped sample, no significance/safety guarantee."}
    if split == "validation":
        # Fixed dual-view is also a control: learned gating must add something beyond the new state.
        promoted = all(paired[a]["loss"] == 0 for a in ("filter", best, "dual_fixed")) and (
            paired[best]["gain"] > 0 or paired[best]["mean_cost_delta"] < -1e-6) and (
            paired["dual_fixed"]["gain"] > 0 or paired["dual_fixed"]["mean_cost_delta"] < -1e-6)
        report.update(promoted=promoted, validation_selected_static=best,
                      promotion_rule="zero observed losses vs Filter/best static/fixed fusion, and accuracy or tool-cost benefit vs best static AND fixed fusion; empirical only")
    else:
        validation = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
        report["validation_selected_static"] = validation["validation_selected_static"]
        report["learned_vs_validation_selected_static"] = paired[validation["validation_selected_static"]]
    write_new(root / f"{split}_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False), flush=True)
    return report
def preflight():
    'Read-only environment check; fail BEFORE loading a GPU model.'
    from importlib.util import find_spec
    from bench import repository
    repo = repository(None)
    for name in ("env/run.py", "env/config/travel_config.yaml", "env/data/static/search_spaces/test/search_space.json"):
        if not (repo / name).is_file():
            raise FileNotFoundError(f"Required CostBench asset missing: {repo / name}")
    if find_spec("numpy") is None:
        raise RuntimeError("CPU training requires NumPy in this Python environment")
def require_validated_controller(root, manifest):
    if not (root / 'controller.json').is_file():
        raise FileNotFoundError("Test stage cannot fit a missing controller; finish training/validation first")
    train(root, manifest)  # Integrity-only resume; refuses missing training data.
    report = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
    if report.get("split") != "validation" or report.get("controller_sha256") != digest(root / 'controller.json'):
        raise ValueError("Validation report belongs to another controller")
    return report
def main():
    args = arguments()
    manifest = make_manifest(args)
    root = Path(args.root).resolve()
    status(root, manifest)
    if not args.run or args.stage == "status":
        if args.stage != "status":
            preflight()
        print("Read-only. --run is required for collection/training. No Qwen weight update.")
        return
    preflight()
    write_new(root / "manifest.json", manifest)
    for key, sample in manifest["splits"].items():
        write_new(root / key / "sample.json", sample)
    for name in ("logs", "events"):
        (root / name).mkdir(exist_ok=True)
    with runner_lock(root):
        if args.stage == "prepare":
            return
        if args.stage in ("all", "train"):
            collect(args, manifest, "train")
        if args.stage in ("all", "train", "validate"):
            train(root, manifest)
        if args.stage in ("all", "validate"):
            collect(args, manifest, "validation")
            summary = summarize(root, manifest, "validation")
        elif args.stage == "test":
            summary = require_validated_controller(root, manifest)
        else:
            return
        if args.stage in ("all", "test"):
            if not summary["promoted"]:
                print("NOT PROMOTED: live validation did not support the learned state gate. Fresh test remains sealed; do not lower thresholds on test.")
                return
            collect(args, manifest, "test")
            summarize(root, manifest, "test")
    print("Workflow completed. Preserve this frozen run.")
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
