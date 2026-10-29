# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'CPU-only input-intervention audit; not an online evaluation or training run.'
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryFile
from typing import Callable
sys.dont_write_bytecode = True
from control import controller as state_controller
FORMAT = "evocond_problem_first_input_audit_v1"
MODES = ("baseline", "legacy", "prompt_only", "selective")
SPLITS = ("train", "development")
COUNT_FIELDS = ("prefixes", "payload_changed", "messages_changed", "tools_changed",
                "legacy_not_reranked_but_payload_changed", "reported_not_reranked",
                "controller_errors", "route_changed", "route_same", "route_unknown")
REASONS = {"baseline_requested", "baseline_recovery", "no_type_feasible_candidates",
           "no_distinct_route_clears_margin", "controller_error", "cost_only_prompt",
           "ranked_route_prompt", "state_plan", "state_recovery", "state_abstain"}
CODE_PATHS = ('diagnose.py', 'control/controller.py', "control/guard.py",
              'control/bridge.py', 'control/family.py', 'control/gate.py')
WARNINGS = [
    "Input-intervention audit only: no online execution, correctness, accuracy, or Qwen update.",
    "Split is self-declared; query membership and controller train/test provenance are not verified.",
    "Recorded prefixes are structurally checked, but earlier omitted history cannot be detected.",
    "Route comparison uses explicit event paths only; a missing path is unknown, not unchanged.",
    "A different payload does not establish a different model action, outcome, or causal benefit.",
    "This replays the current source version, not the historical execution that produced the trace.",
    "route_vs_cost_candidate compares the final explicit path with the cost candidate; it is not a rerank flag.",
]
def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()
def _constant(value: str) -> None:
    raise ValueError("Nonfinite JSON number")
def _pairs(values: list[tuple]) -> dict:
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result
def _loads(value: bytes) -> object:
    return json.loads(value, parse_constant=_constant, object_pairs_hook=_pairs)
def _payload(messages: object, tools: object) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("A complete nonempty messages list is required")
    roles = {"system", "developer", "user", "assistant", "tool", "function"}
    if any(not isinstance(m, dict) or m.get("role") not in roles
           or not ("content" in m or "tool_calls" in m) for m in messages):
        raise ValueError("Each recorded message requires a role and visible content/tool_calls")
    if not any(m["role"] == "user" for m in messages):
        raise ValueError("The recorded prefix has no user message; do not reconstruct missing history")
    if not isinstance(tools, list) or any(not isinstance(t, dict) for t in tools):
        raise ValueError("The complete tools list is required, including an explicit empty list")
def _route(event: dict) -> list[str] | None:
    for candidate in (event, event.get("base_event"), event.get("recovery")):
        if isinstance(candidate, dict) and "path" in candidate:
            value = candidate["path"]
            if isinstance(value, list) and all(isinstance(x, str) for x in value):
                return value
    return None
def _compare(left: list[str] | None, right: list[str] | None) -> str:
    if left is None or right is None:
        return "unknown"
    return "same" if left == right else "changed"
def audit(controller: Path, trace: Path, split: str,
          *, record_sink: Callable[[dict], None] | None = None) -> dict:
    'Stream JSONL prefixes, retaining counters but not prompts or records.'
    if split not in SPLITS:
        raise ValueError("Only self-declared train or development traces are accepted; held-out splits are refused")
    controller_bytes = controller.read_bytes()
    model = _loads(controller_bytes)
    if not isinstance(model, dict):
        raise ValueError("Controller must be a JSON object")
    state_controller.validate(model)
    project = Path(__file__).resolve().parent
    code_hashes = {name: sha256((project / name).read_bytes()).hexdigest() for name in CODE_PATHS}
    controller_sha256 = sha256(controller_bytes).hexdigest()
    source_digest = sha256()
    content_digest = sha256()
    counters = {mode: Counter({key: 0 for key in COUNT_FIELDS}) for mode in MODES}
    reasons = {mode: Counter() for mode in MODES}
    lines = observations = ignored = 0
    with trace.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            lines = line_number
            source_digest.update(raw)
            if not raw.strip():
                continue
            try:
                row = _loads(raw)
            except (ValueError, UnicodeError) as error:
                raise ValueError(f"Trace line {line_number}: invalid JSON ({type(error).__name__})") from error
            if not isinstance(row, dict) or not isinstance(row.get("action"), str):
                raise ValueError(f"Trace line {line_number}: unknown event schema; action is required")
            if row["action"] != "state_training_observation":
                ignored += 1
                continue
            if row.get("schema") not in (None, "state_training_observation"):
                raise ValueError(f"Trace line {line_number}: unknown observation schema")
            if "messages" not in row or "tools" not in row:
                raise ValueError(f"Trace line {line_number}: missing original messages/tools; policy_attempts are not a substitute")
            messages, tools = row["messages"], row["tools"]
            try:
                _payload(messages, tools)
            except ValueError as error:
                raise ValueError(f"Trace line {line_number}: {error}") from error
            # Discard labels before any intervention call.
            del row
            source_content = {"messages": messages, "tools": tools}
            content_hash = digest(source_content)
            content_digest.update(bytes.fromhex(content_hash))
            branches = {}
            for mode in MODES:
                try:
                    revised, available, event = state_controller.intervene(
                        deepcopy(messages), deepcopy(tools), deepcopy(model), experiment=mode)
                    _payload(revised, available)
                    if not isinstance(event, dict):
                        raise ValueError("Controller event must be an object")
                    branches[mode] = {
                        "payload_sha256": digest({"messages": revised, "tools": available}),
                        "messages_sha256": digest(revised), "tools_sha256": digest(available),
                        "message_count": len(revised), "tool_count": len(available),
                        "route": _route(event), "event": event,
                    }
                except Exception as error:
                    raise ValueError(f"Trace line {line_number}, branch {mode}: controller replay failed ({type(error).__name__})") from error
            baseline = branches["baseline"]
            record = {"source_line": line_number, "source_line_sha256": sha256(raw).hexdigest(),
                      "controller_sha256": controller_sha256, "content_sha256": content_hash,
                      "input_message_count": len(messages), "input_tool_count": len(tools), "modes": {}}
            for mode, branch in branches.items():
                event, path = branch["event"], branch["route"]
                changed = branch["payload_sha256"] != baseline["payload_sha256"]
                prompt_changed = branch["messages_sha256"] != baseline["messages_sha256"]
                tools_changed = branch["tools_sha256"] != baseline["tools_sha256"]
                path_comparison = _compare(path, baseline["route"])
                cost_path = event.get("baseline_path")
                if not isinstance(cost_path, list) or not all(isinstance(x, str) for x in cost_path):
                    cost_path = None
                reranked = event.get("reranked") if type(event.get("reranked")) is bool else None
                hidden_change = mode == "legacy" and reranked is False and changed
                error_event = event.get("reason") == "controller_error"
                result = {key: branch[key] for key in (
                    "payload_sha256", "messages_sha256", "tools_sha256", "message_count", "tool_count")}
                result.update(payload_changed=changed, messages_changed=prompt_changed,
                              tools_changed=tools_changed, route_vs_baseline=path_comparison,
                              route_sha256=digest(path) if path is not None else None,
                              route_length=len(path) if path is not None else None,
                              route_vs_cost_candidate=_compare(path, cost_path),
                              reported_reranked=reranked,
                              legacy_not_reranked_but_payload_changed=hidden_change,
                              controller_error=error_event)
                record["modes"][mode] = result
                count = counters[mode]
                count.update(prefixes=1, payload_changed=int(changed), messages_changed=int(prompt_changed),
                             tools_changed=int(tools_changed),
                             legacy_not_reranked_but_payload_changed=int(hidden_change),
                             reported_not_reranked=int(reranked is False), controller_errors=int(error_event))
                count[f"route_{path_comparison}"] += 1
                # Do not copy arbitrary event text or controller error messages.
                reason = event.get("reason", event.get("action", "unspecified"))
                reasons[mode][reason if isinstance(reason, str) and reason in REASONS else "other"] += 1
            observations += 1
            if record_sink is not None:
                record_sink(record)
    if not observations:
        raise ValueError("No state_training_observation prefixes found; no intervention audit was performed")
    return {"format": FORMAT, "source": str(trace.resolve()),
            "source_sha256": source_digest.hexdigest(), "controller": str(controller.resolve()),
            "controller_sha256": controller_sha256, "controller_content_sha256": digest(model),
            "source_code_sha256": code_hashes, "code_bundle_sha256": digest(code_hashes),
            "policy": "learned", "python_version": sys.version.split()[0],
            "observed_content_sha256": content_digest.hexdigest(),
            "declared_split": split, "split_identity": "not_verified",
            "controller_split_provenance": "not_verified", "query_membership": "not_verified",
            "input_fields": ["messages", "tools"], "labels_used": False,
            "online_evaluation": False, "accuracy_available": False, "qwen_updated": False,
            "source_lines": lines, "prefixes": observations, "ignored_non_observation_events": ignored,
            "modes": {mode: {**dict(counters[mode]), "reasons": dict(reasons[mode])} for mode in MODES},
            "warnings": WARNINGS}
def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True,
                        help="Self-declared only; not a verified split identity")
    parser.add_argument("--output", type=Path, help="New JSON report path; never overwritten")
    parser.add_argument("--run", action="store_true", help="Save input audit only; does not train or evaluate Qwen")
    args = parser.parse_args(argv)
    if args.run and args.output is None:
        parser.error("--run requires --output NEWFILE")
    try:
        if args.run:
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError("Output exists; refusing to overwrite it")
            if not args.output.parent.is_dir():
                raise ValueError("Output parent directory must already exist")
            # Stage per-prefix records on disk, never accumulate them in RAM.
            with TemporaryFile(mode="w+", encoding="utf-8", dir=args.output.parent) as staged:
                def emit(record: dict) -> None:
                    staged.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                summary = audit(args.controller, args.trace, args.split, record_sink=emit)
                staged.seek(0)
                with args.output.open("x", encoding="utf-8") as output:
                    output.write('{"records":[')
                    separator = ""
                    for row in staged:
                        output.write(separator + row.rstrip("\n"))
                        separator = ",\n"
                    output.write('],"summary":')
                    json.dump(summary, output, ensure_ascii=False, allow_nan=False)
                    output.write("}\n")
            summary = {**summary, "saved": str(args.output.resolve())}
        else:
            summary = {**audit(args.controller, args.trace, args.split), "saved": None}
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        return summary
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"Input audit refused: {error}\n")
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
