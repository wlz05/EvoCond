# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Freeze a task-balanced CostBench sample excluding every locally recorded run.'
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import random
import re
QUERY_DIR = re.compile(r"q(\d+)$")
QUERY_ID = re.compile(r"<Query(\d+)>$")
TASKS = {"location", "transportation", "accommodation", "attraction", "dining", "shopping"}
def seen_indices(root: Path, records: list[dict]) -> set[int]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    seen: set[int] = set()
    ids = {row.get("query_id"): index for index, row in enumerate(records) if isinstance(row.get("query_id"), str)}
    for path in root.rglob("results_*.json"):
        match = QUERY_DIR.fullmatch(path.parent.name)
        if match:
            seen.add(int(match.group(1)))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            raise ValueError(f"Cannot identify query indices in prior result: {path}")
        for result in results:
            query_id = result.get("query_id") if isinstance(result, dict) else None
            index = ids.get(query_id)
            if index is None and isinstance(query_id, str):
                fallback = QUERY_ID.fullmatch(query_id)
                if fallback:
                    index = int(fallback.group(1)) - 1
            if index is None or index < 0 or index >= len(records):
                raise ValueError(f"Cannot identify prior query {query_id!r} in {path}")
            seen.add(index)
    for path in root.rglob("sample.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        values = data.get("indices") if isinstance(data, dict) else None
        if isinstance(values, list):
            seen.update(value for value in values if isinstance(value, int) and not isinstance(value, bool) and value >= 0)
    return seen
def freeze(queries: Path, prior_root: Path, memory: Path, per_task: int, seed: int, memory_extra_cost: float = 0.0) -> dict:
    if per_task < 1:
        raise ValueError("per_task must be positive")
    if not isfinite(memory_extra_cost) or memory_extra_cost < 0:
        raise ValueError("memory_extra_cost must be finite and nonnegative")
    query_bytes = queries.read_bytes()
    records = json.loads(query_bytes)
    if not isinstance(records, list):
        raise ValueError("CostBench queries must be a list")
    memory_bytes = memory.read_bytes()
    snapshot = json.loads(memory_bytes)
    metadata = snapshot.get("metadata") if isinstance(snapshot, dict) else None
    training = metadata.get("train_indices") if isinstance(metadata, dict) else None
    if not isinstance(training, list) or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in training) or metadata.get("query_sha256") != sha256(query_bytes).hexdigest():
        raise ValueError("Memory metadata does not match the query dataset")
    excluded = seen_indices(prior_root, records) | set(training)
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("task"), str):
            raise ValueError(f"Malformed task at query index {index}")
        if index not in excluded:
            groups.setdefault(record["task"].strip().lower(), []).append(index)
    if set(groups) != TASKS:
        raise ValueError("A six-task sample cannot be frozen from the remaining queries")
    rng = random.Random(seed)
    by_task = {}
    for task, available in sorted(groups.items()):
        if len(available) < per_task:
            raise ValueError(f"Not enough unused {task} queries for a frozen sample")
        by_task[task] = sorted(rng.sample(available, per_task))
    selected = sorted(index for values in by_task.values() for index in values)
    return {"seed": seed, "per_task": per_task, "by_task": by_task, "indices": selected, "excluded_count": len(excluded), "query_sha256": sha256(query_bytes).hexdigest(), "memory_sha256": sha256(memory_bytes).hexdigest(), "memory_extra_cost": memory_extra_cost}
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="assets/data/costbench/CostBench_queries.json")
    parser.add_argument("--exclude-root", default="runs")
    parser.add_argument("--memory", required=True)
    parser.add_argument("--per-task", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--memory-extra-cost", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace a frozen sample: {destination}")
    sample = freeze(Path(args.queries), Path(args.exclude_root), Path(args.memory), args.per_task, args.seed, args.memory_extra_cost)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "indices": sample["indices"], "excluded_count": sample["excluded_count"], "sample_sha256": sha256(destination.read_bytes()).hexdigest()}, ensure_ascii=False))
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
