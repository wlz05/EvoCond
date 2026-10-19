# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

# Reserve native evaluation hooks without executing generated code on the host.
from __future__ import annotations
from abc import ABC, abstractmethod
import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from catalogue import benchmark, pinned_revision, relative_path
from convert import file_hash, normalized_question, read_rows
class NativeEvaluator(ABC):
    @abstractmethod
    def run(self, public_case: dict, protocol: dict) -> dict:
        raise NotImplementedError("Run the actor with the benchmark's native environment and isolated tools")
    @abstractmethod
    def grade(self, private_reference, trajectory: dict, protocol: dict) -> dict:
        raise NotImplementedError("Use the benchmark's native evaluator; no generic answer-match fallback")
def load_cases(path):
    cases = list(read_rows(path, "jsonl"))
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "prompt", "context", "reference", "group", "group_keys"}:
            raise ValueError("Use normalized dataset_split records")
        if not isinstance(case["id"], str) or not case["id"] or case["id"] in ids:
            raise ValueError("Cases need unique nonempty string IDs")
        ids.add(case["id"])
        normalized_question(case["prompt"])
        if not isinstance(case["group"], str) or not case["group"]:
            raise ValueError("Every case needs an independent group identity")
        if not isinstance(case["group_keys"], list) or not case["group_keys"] or any(not isinstance(key, str) or len(key) != 64 or any(char not in "0123456789abcdef" for char in key) for key in case["group_keys"]):
            raise ValueError("Every case needs explicit group-key fingerprints")
    if not cases:
        raise ValueError("No evaluation cases")
    return cases
def public_case(case):
    return deepcopy({key: case[key] for key in ("id", "prompt", "context")})
def make_protocol(name, data_sha256, source_revision, evaluator_revision, model_id, arm, seed):
    spec = benchmark(name)
    pinned_revision(source_revision)
    pinned_revision(evaluator_revision)
    if len(data_sha256) != 64 or any(char not in "0123456789abcdef" for char in data_sha256):
        raise ValueError("Declare the exact evaluation data SHA256")
    if not model_id.strip() or not arm.strip():
        raise ValueError("Declare a fixed model/checkpoint identity and method arm")
    protocol = {"benchmark": name, "data_sha256": data_sha256, "source_revision": source_revision, "evaluator": spec.evaluation, "evaluator_revision": evaluator_revision, "model_id": model_id, "arm": arm, "seed": seed, "samples_per_task": 1}
    protocol["sha256"] = sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return protocol
def receipt(case_id, result, protocol):
    continuous = benchmark(protocol["benchmark"]).score_kind == "continuous"
    value_key = "score" if continuous else "passed"
    if not isinstance(result, dict) or set(result) != {"status", value_key}:
        raise ValueError(f"Native grade must explicitly contain status and {value_key}")
    value = result[value_key]
    if result["status"] not in {"ok", "error", "timeout"}:
        raise ValueError("Invalid native grade")
    if continuous:
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Native similarity must be finite and within zero and one")
    elif type(value) is not bool:
        raise ValueError("Binary native grades require boolean passed")
    if result["status"] != "ok" and value != 0:
        raise ValueError("Errors and timeouts must have zero score")
    return {"id": case_id, "protocol_sha256": protocol["sha256"], "evaluator": protocol["evaluator"], "evaluator_revision": protocol["evaluator_revision"], "status": result["status"], "score": float(value), "passed": None if continuous else value}
def evaluate(cases, evaluator: NativeEvaluator, protocol):
    for case in cases:
        trajectory = evaluator.run(public_case(case), dict(protocol))
        yield receipt(case["id"], evaluator.grade(case["reference"], trajectory, dict(protocol)), protocol)
def summarize(cases, receipts, protocol):
    expected, observed = {case["id"] for case in cases}, {}
    continuous = benchmark(protocol["benchmark"]).score_kind == "continuous"
    for row in receipts:
        if not isinstance(row, dict) or set(row) != {"id", "protocol_sha256", "evaluator", "evaluator_revision", "status", "passed", "score"}:
            raise ValueError("Unexpected native receipt schema")
        if row["id"] not in expected or row["id"] in observed:
            raise ValueError("Unknown or repeated receipt ID")
        if any(row[key] != protocol[key] for key in ("evaluator", "evaluator_revision")) or row["protocol_sha256"] != protocol["sha256"]:
            raise ValueError("Receipt does not match this frozen evaluation protocol")
        value_key = "score" if continuous else "passed"
        verified = receipt(row["id"], {key: row[key] for key in ("status", value_key)}, protocol)
        if type(row["score"]) not in (int, float) or row["score"] != verified["score"] or row["passed"] != verified["passed"]:
            raise ValueError("Receipt score does not match its native grade")
        observed[row["id"]] = row
    if not expected or set(observed) != expected:
        raise ValueError("Incomplete receipts; missing cases must not disappear from the denominator")
    passed = None if continuous else sum(row["passed"] for row in observed.values())
    return {"protocol": protocol, "metric": benchmark(protocol["benchmark"]).metric, "n": len(cases), "groups": len({case['group'] for case in cases}), "passed": passed, "score": sum(row["score"] for row in observed.values()) / len(cases), "errors": sum(row["status"] == "error" for row in observed.values()), "timeouts": sum(row["status"] == "timeout" for row in observed.values()), "notice": "Native adapter receipts are trusted inputs, not proof of official reproduction. One sample per task; no inferred pass@k or significance."}
def main():
    parser = argparse.ArgumentParser(description="Export public requests or summarize native receipts. No GPU, code execution or grader fallback.")
    parser.add_argument("stage", choices=("plan", "summarize"))
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--evaluator-revision", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--receipts")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    path, output = relative_path(args.data), relative_path(args.output)
    digest = file_hash(path)
    if digest != args.sha256:
        raise ValueError("Evaluation data SHA256 does not match")
    cases = load_cases(path)
    if file_hash(path) != digest:
        raise ValueError("Evaluation data changed during reading")
    protocol = make_protocol(args.benchmark, digest, args.source_revision, args.evaluator_revision, args.model_id, args.arm, args.seed)
    if args.stage == "summarize":
        if not args.receipts:
            raise ValueError("Summarization requires native receipts")
        report = summarize(cases, list(read_rows(relative_path(args.receipts), "jsonl")), protocol)
    else:
        if args.receipts:
            raise ValueError("Plan stage does not consume receipts")
        report = {"protocol": protocol, "cases": len(cases), "native_adapter_required": True, "training_started": False}
    if args.run:
        output.mkdir(parents=True, exist_ok=False)
        with (output / "protocol.json").open("x", encoding="utf-8") as stream:
            json.dump(protocol, stream, indent=2)
        if args.stage == "plan":
            with (output / "requests.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
                for case in cases:
                    stream.write(json.dumps(public_case(case), ensure_ascii=True, allow_nan=False) + "\n")
        with (output / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
