# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Evaluate the opt-in value audit on the old development validation split.'
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time
import uuid
from statistics import mean
from bench import repository
from control.controller import validate
from control.gate import initial_user
from gate import write_new
from panel import PROJECT, bench_args, digest, port_free, server_args, stop_server
from workflow import data_at, execute, outcome, runner_lock, wait_server
ARM = "state_value_audit"
MODES = ("none", "preference_change")
REFERENCES = ("filter", "evocond", "joint_family", "dual_fixed", "state_learned")
def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("status", "validate"), default="status")
    parser.add_argument("--root", default="runs/value20260928_v1")
    parser.add_argument("--source", default="runs/state20260927_v1")
    parser.add_argument("--queries", default="assets/data/costbench/CostBench_queries.json")
    parser.add_argument("--model", default="assets/model")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()
def source_protocol(args):
    root = Path(args.root).resolve()
    source = Path(args.source).resolve()
    if root == source or not root.is_relative_to(PROJECT) or not source.is_relative_to(PROJECT):
        raise ValueError("Source and output must be distinct directories inside EvoCond")
    manifest_path = source / "manifest.json"
    report_path = source / "validation_report.json"
    controller_path = source / 'controller.json'
    sample_path = source / "validation" / "sample.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    validate(controller)
    if report.get("split") != "validation" or report.get("controller_sha256") != digest(controller_path):
        raise ValueError("Source validation report/controller mismatch")
    indices = old["splits"]["validation"]["indices"]
    if not isinstance(indices, list) or not indices or len(indices) != len(set(indices)) or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError("Invalid source validation identities")
    if sample.get("indices") != indices or len(report.get("records", [])) != len(indices) * len(MODES):
        raise ValueError("Source validation sample/report is incomplete")
    keys = {(r["mode"], r["index"]) for r in report["records"]}
    if keys != {(mode, index) for mode in MODES for index in indices}:
        raise ValueError("Source validation report has duplicate or missing episodes")
    if controller["provenance"].get("validation_indices") != indices:
        raise ValueError("Controller was trained for a different validation split")
    if old.get("query_sha256") != digest(Path(args.queries)) or sample.get("query_sha256") != old["query_sha256"]:
        raise ValueError("CostBench query dataset differs from source")
    if old.get("model_config_sha256") != digest(Path(args.model) / "config.json"):
        raise ValueError("Qwen model configuration differs from source")
    weight_stats = {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                    for p in sorted(Path(args.model).glob("*.safetensors"))}
    if old.get("model_weight_stats") != weight_stats:
        raise ValueError("Qwen weight files differ from the frozen source run")
    if not 1 <= args.port <= 65535:
        raise ValueError("Invalid port")
    files = ("serve.py", "bench.py", 'control/value.py', 'control/controller.py', 'review.py')
    frozen = {
        "format": "value_audit_development_v1",
        "source_manifest_sha256": digest(manifest_path),
        "source_validation_report_sha256": digest(report_path),
        "controller_sha256": digest(controller_path),
        "query_sha256": old["query_sha256"],
        "model_config_sha256": old["model_config_sha256"],
        "validation_indices": indices,
        "modes": list(MODES),
        "tokens": old["tokens"],
        "refinement": old["refinement"],
        "port": args.port,
        "code_sha256": {name: digest(PROJECT / name) for name in files},
        "warning": "Exposed development validation only; no Qwen update or fresh-test claim.",
    }
    existing = root / "manifest.json"
    if existing.exists():
        if json.loads(existing.read_text(encoding="utf-8")) != frozen:
            raise ValueError("Frozen value-audit run differs; choose a new output directory")
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError("Nonempty output directory has no frozen manifest")
    return root, source, frozen, report, sample
def identity(row):
    return (row.get("query_id"), row.get("tool_creation_seed"),
            initial_user(row.get("conversation_history") or []))
def source_row(source, mode, index, report_record):
    found = data_at(source, "validation", "evocond", mode, index)
    if found is None:
        raise RuntimeError(f"Missing source result evocond/{mode}/q{index}")
    data, _ = found
    if outcome(data["results"][0], data) != report_record["outcomes"]["evocond"]:
        raise ValueError("Source report and frozen result differ")
    return data["results"][0]
def checked_episode(root, source, mode, index, record):
    found = data_at(root, "validation", ARM, mode, index)
    if found is None:
        return None
    data, audit = found
    trace = (root / audit["trace"]).resolve()
    if not trace.is_relative_to(root) or not 0 <= audit["start"] <= audit["end"] <= trace.stat().st_size:
        raise ValueError(f"Invalid trace segment at {mode}/q{index}")
    with trace.open("rb") as stream:
        stream.seek(audit["start"])
        raw = stream.read(audit["end"]-audit["start"])
    if sha256(raw).hexdigest() != audit["trace_sha256"]:
        raise ValueError(f"Completed value-audit trace changed at {mode}/q{index}")
    row = data["results"][0]
    if identity(row) != identity(source_row(source, mode, index, record)):
        raise ValueError(f"Paired identity differs at {mode}/q{index}")
    return data, audit
def progress(root, source, frozen, report):
    records = {(r["mode"], r["index"]): r for r in report["records"]}
    completed = sum(checked_episode(root, source, mode, index, records[mode, index]) is not None
                    for mode in MODES for index in frozen["validation_indices"])
    return {"root": str(root), "completed": completed,
            "episodes": len(records), "report_exists": (root / "validation_report.json").is_file(),
            "qwen_updated": False, "split": "exposed_development_validation"}
def collect(args, root, source, frozen, report, sample):
    repository(None)
    write_new(root / "manifest.json", frozen)
    write_new(root / "validation" / "sample.json", sample)
    (root / "events").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    records = {(r["mode"], r["index"]): r for r in report["records"]}
    pending = [(mode, index) for mode in MODES for index in frozen["validation_indices"]
               if checked_episode(root, source, mode, index, records[mode, index]) is None]
    if not pending:
        return
    if not port_free(args.port):
        raise RuntimeError(f"Port {args.port} is occupied; stop the service using it before this workflow")
    fixed = argparse.Namespace(model=args.model, port=args.port, queries=args.queries,
                               sample=str(root / "validation" / "sample.json"), threads=1,
                               tokens=frozen["tokens"], refinement=frozen["refinement"])
    controller = source / 'controller.json'
    trace = root / "events" / "validation_value_audit.jsonl"
    command = server_args("evocond", fixed, trace, 0.0) + [
        "--state-controller", str(controller), "--state-policy", "learned", "--state-value-audit"]
    expected = {"status": "ok", "evocond": True, "levels": "full", "noaudit": True,
                "state_controller_sha256": frozen["controller_sha256"],
                "state_policy": "learned", "state_value_audit": True,
                "memory_sha256": None, "trace_context": False}
    print(f"Development value audit: {len(pending)} episodes pending", flush=True)
    with runner_lock(root):
        with (root / "logs" / "serve_validation_value_audit.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_server(process, args, expected)
                for mode, index in pending:
                    folder = root / "validation" / ARM / mode / f"q{index}"
                    if folder.exists():
                        preserved = root / "interrupted" / f"{mode}_q{index}_{uuid.uuid4().hex}"
                        if not folder.resolve().is_relative_to(root):
                            raise ValueError("Unsafe result directory")
                        preserved.parent.mkdir(exist_ok=True)
                        folder.rename(preserved)
                        print(f"Preserved incomplete result: {preserved}", flush=True)
                    start = trace.stat().st_size if trace.exists() else 0
                    cmd = bench_args("evocond", mode, index, fixed, folder, 0.0) + [
                        "--state-controller", str(controller), "--state-policy", "learned",
                        "--state-split", "validation", "--state-value-audit"]
                    begun = time.monotonic()
                    with (root / "logs" / f"validation_{mode}_q{index}.log").open("a", encoding="utf-8") as qlog:
                        execute(cmd, qlog)
                    files = list(folder.glob("results_*.json"))
                    if len(files) != 1:
                        raise RuntimeError(f"Missing/duplicate result: {folder}")
                    data = json.loads(files[0].read_text(encoding="utf-8"))
                    row = data["results"][0]
                    if identity(row) != identity(source_row(source, mode, index, records[mode, index])):
                        raise ValueError(f"Paired identity differs at {mode}/q{index}")
                    end = trace.stat().st_size
                    with trace.open("rb") as stream:
                        stream.seek(start)
                        raw = stream.read(end-start)
                    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
                    usage = [event for event in events if event.get("action") == "request_usage"]
                    if not usage:
                        raise RuntimeError("Missing request usage trace")
                    audit = {"result_sha256": digest(files[0]), "trace": trace.relative_to(root).as_posix(),
                             "start": start, "end": end, "trace_sha256": sha256(raw).hexdigest(),
                             "elapsed_seconds": time.monotonic()-begun,
                             "prompt_tokens": sum(e["prompt_tokens"] for e in usage),
                             "completion_tokens": sum(e["completion_tokens"] for e in usage),
                             "controller_seconds": sum(e.get("controller_seconds", 0) for e in usage),
                             "value_audit_seconds": sum(e.get("value_audit_seconds", 0) for e in usage),
                             "value_audits": sum(e.get("value_audits", 0) for e in usage),
                             "value_rejections": sum(e.get("value_rejections", 0) for e in usage)}
                    write_new(folder / "episode.json", audit)
                    print(f"validation/{ARM}/{mode}/q{index}: {outcome(row, data)} "
                          f"audits={audit['value_audits']} rejections={audit['value_rejections']} "
                          f"seconds={audit['elapsed_seconds']:.1f}", flush=True)
            finally:
                stop_server(process)
def summarize(root, source, frozen, source_report):
    records = {(r["mode"], r["index"]): r for r in source_report["records"]}
    rows = []
    for mode in MODES:
        for index in frozen["validation_indices"]:
            original = records[mode, index]
            found = checked_episode(root, source, mode, index, original)
            if found is None:
                raise RuntimeError(f"Incomplete development episode: {mode}/q{index}")
            data, audit = found
            rows.append({"mode": mode, "index": index,
                         "outcome": outcome(data["results"][0], data),
                         "overhead": {key: audit[key] for key in (
                             "elapsed_seconds", "prompt_tokens", "completion_tokens",
                             "controller_seconds", "value_audit_seconds", "value_audits", "value_rejections")}})
    result = {"split": "exposed_development_validation", "n": len(rows),
              "correct": sum(r["outcome"]["success"] for r in rows),
              "mean_steps_all": mean(r["outcome"]["steps"] for r in rows),
              "mean_tool_cost_all": mean(r["outcome"]["cost"] for r in rows),
              "mean_overhead": {key: mean(r["overhead"][key] for r in rows)
                                for key in rows[0]["overhead"]}, "paired": {}, "records": rows,
              "warning": "This split was inspected during development. Tool cost excludes model cost; extra reviewer tokens and wall time are reported separately. No fresh-test claim."}
    for arm in REFERENCES:
        gains = [r for r in rows if r["outcome"]["success"] and not records[r["mode"], r["index"]]["outcomes"][arm]["success"]]
        losses = [r for r in rows if not r["outcome"]["success"] and records[r["mode"], r["index"]]["outcomes"][arm]["success"]]
        result["paired"][arm] = {"gain": len(gains), "loss": len(losses),
                                 "gain_cases": [[r["mode"], r["index"]] for r in gains],
                                 "loss_cases": [[r["mode"], r["index"]] for r in losses],
                                 "mean_tool_cost_delta": result["mean_tool_cost_all"]-source_report["overall"][arm]["mean_cost_all"]}
    write_new(root / "validation_report.json", result)
    return result
def main():
    args = arguments()
    root, source, frozen, report, sample = source_protocol(args)
    print(json.dumps(progress(root, source, frozen, report), ensure_ascii=False), flush=True)
    if args.stage == "status" or not args.run:
        print("Read-only. Use validate --run to execute 24 exposed-development episodes. Qwen stays frozen.")
        return
    collect(args, root, source, frozen, report, sample)
    result = summarize(root, source, frozen, report)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False), flush=True)
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
