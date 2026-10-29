# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Build a frozen, train-only EVOLT-style soft memory from CostBench traces.'
from __future__ import annotations
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from typing import Any
from control.bridge import _vector
def indices_from(path: Path) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("indices") if isinstance(data, dict) else data
    if not isinstance(values, list) or not values or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in values) or len(set(values)) != len(values):
        raise ValueError("Index file needs a nonempty, unique list of nonnegative integers")
    return sorted(values)
def indices_in(runs: Path) -> list[int]:
    values = []
    for path in runs.iterdir():
        if path.is_dir() and path.name.startswith("q"):
            if not path.name[1:].isdigit():
                raise ValueError(f"Unexpected query directory: {path}")
            values.append(int(path.name[1:]))
    if not values or len(set(values)) != len(values):
        raise ValueError("Training run directory needs unique q<index> subdirectories")
    return sorted(values)
def calls_from(result: dict[str, Any]) -> tuple[str, ...]:
    calls: list[str] = []
    for message in result.get("conversation_history") or []:
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function", call)
            name = function.get("name") if isinstance(function, dict) else None
            if isinstance(name, str) and name.strip():
                calls.append(name.strip())
    return tuple(calls)
def build(runs: Path, indices: list[int], queries: Path, dim: int = 64) -> dict[str, Any]:
    if dim < 8:
        raise ValueError("Memory vector dimension must be at least eight")
    query_bytes = queries.read_bytes()
    query_rows = json.loads(query_bytes)
    if not isinstance(query_rows, list) or max(indices) >= len(query_rows):
        raise ValueError("Training indices exceed the CostBench query file")
    vectors: dict[str, list[list[float]]] = defaultdict(list)
    trials: dict[str, list[bool]] = defaultdict(list)
    edges: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"positive": 0, "negative": 0, "tasks": set()})
    tasks: list[str] = []
    hashes: dict[str, str] = {}
    for index in indices:
        paths = list((runs / f"q{index}").glob("results_*.json"))
        if len(paths) != 1:
            raise ValueError(f"Expected exactly one result for q{index}, found {len(paths)}")
        raw = paths[0].read_bytes()
        data = json.loads(raw)
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
            raise ValueError(f"Result for q{index} is not a single CostBench episode")
        result = results[0]
        task = str(result.get("query_id") or "").strip()
        if not task or task in tasks:
            raise ValueError(f"Missing or duplicate query identity for q{index}")
        tasks.append(task)
        hashes[str(index)] = sha256(raw).hexdigest()
        record = query_rows[index]
        if not isinstance(record, dict):
            raise ValueError(f"Query q{index} is malformed")
        recorded_id = record.get("query_id")
        if isinstance(recorded_id, str) and recorded_id and recorded_id != task:
            raise ValueError(f"Query identity mismatch for q{index}")
        requirement = record.get("user_requirements") or record.get("query") or result.get("user_requirements") or result.get("query")
        text = requirement if isinstance(requirement, str) else json.dumps(requirement, ensure_ascii=False, sort_keys=True)
        if not text or text == "null":
            raise ValueError(f"Query q{index} has no semantic text")
        sequence = calls_from(result)
        if not sequence:
            raise ValueError(f"Result for q{index} has no tool calls")
        success = (result.get("evaluation") or {}).get("success") is True
        for name in set(sequence):
            trials[name].append(success)
            if success:
                vectors[name].append(list(_vector(text, dim)))
        for source, target in set(zip(sequence, sequence[1:])):
            if source == target:
                continue
            edge = edges[source, target]
            edge["positive" if success else "negative"] += 1
            edge["tasks"].add(task)
    tools = []
    for name in sorted(trials):
        rows = vectors[name]
        if rows:
            averaged = [sum(row[index] for row in rows) / len(rows) for index in range(dim)]
            size = sqrt(sum(value * value for value in averaged))
            vector = [value / size for value in averaged] if size else list(_vector(name, dim))
        else:
            vector = list(_vector(name, dim))
        outcomes = trials[name]
        tools.append({"key": name, "name": name, "state": "active", "vector": vector, "trials": len(outcomes), "quality": (2 * sum(outcomes) / len(outcomes)) - 1})
    graph_edges = [
        {"source": source, "target": target, "kind": "supports", "positive": row["positive"], "negative": row["negative"], "tasks": sorted(row["tasks"])}
        for (source, target), row in sorted(edges.items())
    ]
    return {
        "metadata": {"source": "costbench_traces_v1", "train_indices": indices, "query_sha256": sha256(query_bytes).hexdigest(), "result_sha256": hashes, "relation_semantics": "outcome_association_not_prerequisite"},
        "graph": {"tools": tools, "edges": graph_edges, "tasks": tasks},
    }
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="Training-only directory containing q<index>/results_*.json")
    parser.add_argument("--indices", help="Optional frozen JSON list or {'indices': [...]} of training indices; defaults to all q<index> folders under --runs")
    parser.add_argument("--queries", default="assets/data/costbench/CostBench_queries.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen memory: {output}")
    runs = Path(args.runs)
    selected = indices_from(Path(args.indices)) if args.indices else indices_in(runs)
    state = build(runs, selected, Path(args.queries))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "train_samples": len(state["graph"]["tasks"]), "train_indices": selected, "tools": len(state["graph"]["tools"]), "edges": len(state["graph"]["edges"]), "edges_with_3_tasks": sum(len(edge["tasks"]) >= 3 for edge in state["graph"]["edges"]), "sha256": sha256(output.read_bytes()).hexdigest()}, ensure_ascii=False))
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
