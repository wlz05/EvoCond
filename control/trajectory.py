# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Validate train-only policy rollouts and compute grouped outcome advantages.'
from __future__ import annotations
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any
ARMS = ("searl_adapted", "evocond", "teacher_dual")
def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
def adapter_digest(path: Path) -> str:
    report, weights = path / "training_report.json", path / "adapter_model.safetensors"
    if not report.is_file() or not weights.is_file():
        raise FileNotFoundError("Trained adapter report or safetensors weights missing")
    return sha256(report.read_bytes() + weights.read_bytes()).hexdigest()
def token_ids(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"Invalid {label} token IDs")
    return value
def reward_for(success: bool, cost: float) -> float:
    if type(success) is not bool or type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
        raise ValueError("Malformed verified episode outcome")
    return (1.0 + 0.1 / (1.0 + cost / 100.0)) if success else 0.0
def episode_record(result_path: Path, trace_path: Path, start: int, end: int,
                   arm: str, group: str, rollout: int, model_hash: str,
                   tokenizer_hash: str, query_hash: str) -> dict:
    if arm not in ARMS or not isinstance(group, str) or not group or type(rollout) is not int or rollout < 0:
        raise ValueError("Invalid episode identity")
    result_path, trace_path = result_path.resolve(strict=True), trace_path.resolve(strict=True)
    data = json.loads(result_path.read_text(encoding="utf-8"))
    results = data.get("results")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise ValueError("Expected one CostBench result")
    row = results[0]
    outcome = row.get("evaluation") or {}
    cost = (((data.get("stats") or {}).get("scores") or {}).get("cost_metrics") or {}).get("total_agent_cost")
    reward = reward_for(outcome.get("success"), cost)
    if row.get("status") not in ("completed", "max_steps_reached"):
        raise ValueError("Unrecognized CostBench episode status")
    if type(start) is not int or type(end) is not int or start < 0 or end <= start or end > trace_path.stat().st_size:
        raise ValueError("Invalid trace byte range")
    with trace_path.open("rb") as stream:
        stream.seek(start)
        raw = stream.read(end - start)
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    attempts = [attempt for event in events if event.get("action") == "state_training_observation"
                for attempt in event.get("policy_attempts", [])]
    accepted = [attempt for attempt in attempts if attempt.get("accepted") is True]
    turns = accepted if accepted else attempts
    if not turns:
        raise ValueError("Episode has no tokenized policy attempts")
    for turn in turns:
        token_ids(turn.get("prompt_ids"), "prompt")
        token_ids(turn.get("completion_ids"), "completion")
    return {"format": "evocond_policy_rollout_v1", "split": "train", "arm": arm,
            "model_config_sha256": model_hash, "tokenizer_config_sha256": tokenizer_hash,
            "query_sha256": query_hash, "group": group, "rollout": rollout,
            "reward": reward, "success": outcome["success"], "tool_cost": cost,
            "turns": [{"prompt_ids": turn["prompt_ids"], "completion_ids": turn["completion_ids"],
                       "accepted": turn.get("accepted") is True} for turn in turns],
            "source": {"result": str(result_path), "result_sha256": digest(result_path),
                       "trace": str(trace_path), "start": start, "end": end,
                       "trace_sha256": sha256(raw).hexdigest()}}
def verify_record(row: dict) -> None:
    source = row.get("source")
    if not isinstance(source, dict) or not all(k in source for k in ("result", "result_sha256", "trace", "start", "end", "trace_sha256")):
        raise ValueError("Rollout lacks verifiable CostBench/trace provenance")
    result, trace = Path(source["result"]), Path(source["trace"])
    if digest(result) != source["result_sha256"]:
        raise ValueError("CostBench result changed after collection")
    with trace.open("rb") as stream:
        stream.seek(source["start"])
        raw = stream.read(source["end"] - source["start"])
    if sha256(raw).hexdigest() != source["trace_sha256"]:
        raise ValueError("Model trace changed after collection")
    actual = episode_record(result, trace, source["start"], source["end"], row["arm"],
                            row["group"], row["rollout"], row["model_config_sha256"],
                            row["tokenizer_config_sha256"], row["query_sha256"])
    if actual != row:
        raise ValueError("Reward or policy tokens differ from verified source episode")
def load_rollouts(path: Path, arm: str, config_hash: str, tokenizer_hash: str,
                  min_group: int = 2) -> tuple[list[dict], dict]:
    if arm not in ARMS or min_group < 2:
        raise ValueError("Unknown arm or invalid group size")
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict] = []
    groups: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    source_sha = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("format") != "evocond_policy_rollout_v1":
            raise ValueError(f"Invalid rollout format at line {number}")
        if row.get("split") != "train" or row.get("arm") != arm:
            raise ValueError(f"Rollout split or arm mismatch at line {number}")
        if row.get("model_config_sha256") != config_hash or row.get("tokenizer_config_sha256") != tokenizer_hash:
            raise ValueError(f"Model or tokenizer provenance mismatch at line {number}")
        if not isinstance(row.get("query_sha256"), str) or len(row["query_sha256"]) != 64:
            raise ValueError(f"Missing frozen query provenance at line {number}")
        if source_sha is None:
            source_sha = row["query_sha256"]
        elif source_sha != row["query_sha256"]:
            raise ValueError("Mixed query sources")
        group, rollout = row.get("group"), row.get("rollout")
        if not isinstance(group, str) or not group or type(rollout) is not int or rollout < 0:
            raise ValueError(f"Invalid group identity at line {number}")
        if (group, rollout) in seen:
            raise ValueError(f"Duplicate rollout in group {group}")
        seen.add((group, rollout))
        reward = row.get("reward")
        if type(reward) not in (int, float) or not math.isfinite(reward) or not -10 <= reward <= 10:
            raise ValueError(f"Invalid reward at line {number}")
        turns = row.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"Missing policy turns at line {number}")
        for turn in turns:
            if not isinstance(turn, dict) or type(turn.get("accepted")) is not bool:
                raise ValueError(f"Malformed policy turn at line {number}")
            token_ids(turn.get("prompt_ids"), "prompt")
            token_ids(turn.get("completion_ids"), "completion")
        verify_record(row)
        records.append(row)
        groups[group].append(row)
    if not records or any(len(items) < min_group for items in groups.values()):
        raise ValueError("Every training group needs multiple independent rollouts")
    if len({len(items) for items in groups.values()}) != 1:
        raise ValueError("Training groups have unequal rollout budgets")
    nonzero = 0
    for items in groups.values():
        mean = sum(float(item["reward"]) for item in items) / len(items)
        variance = sum((float(item["reward"]) - mean) ** 2 for item in items) / len(items)
        scale = math.sqrt(variance)
        for item in items:
            item["advantage"] = (float(item["reward"]) - mean) / scale if scale > 1e-8 else 0.0
            nonzero += int(abs(item["advantage"]) > 1e-8)
    report = {"arm": arm, "groups": len(groups), "rollouts": len(records),
              "rollouts_per_group": len(next(iter(groups.values()))),
              "nonzero_advantages": nonzero, "query_sha256": source_sha,
              "rollouts_sha256": digest(path)}
    return records, report

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
