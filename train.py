# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from control.router import ACTIONS, SIZE
from state.record import local, rows
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("--data", default="data/pilot.jsonl")
    parser.add_argument("--output", default="runs/router.json")
    parser.add_argument("--ridge", type=float, default=0.01)
    return parser.parse_args()
def arrays(items: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray([item["features"] for item in items], dtype=np.float64)
    utilities = np.asarray([[item["utilities"][label] for label in ACTIONS] for item in items], dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != SIZE or not np.isfinite(features).all() or not np.isfinite(utilities).all():
        raise ValueError("Training rows are invalid")
    return features, utilities
def metrics(features: np.ndarray, utilities: np.ndarray, mean: np.ndarray, scale: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> dict[str, float | int]:
    scores = (features - mean) / scale @ weights.T + bias
    chosen = np.argmax(scores, axis=1)
    oracle = np.argmax(utilities, axis=1)
    regret = utilities[np.arange(len(utilities)), oracle] - utilities[np.arange(len(utilities)), chosen]
    return {"rows": len(features), "accuracy": float(np.mean(chosen == oracle)), "regret": float(np.mean(regret))}
def main() -> None:
    args = arguments()
    if not np.isfinite(args.ridge) or args.ridge <= 0.0:
        raise ValueError("Ridge penalty must be positive")
    records = [dict(item) for item in rows(args.data)]
    groups = {name: [item for item in records if item.get("split") == name] for name in ("train", "valid", "test")}
    if any(not group for group in groups.values()):
        raise ValueError("Training splits must be nonempty")
    trainx, trainy = arrays(groups["train"])
    mean = trainx.mean(axis=0)
    scale = trainx.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (trainx - mean) / scale
    design = np.column_stack((normalized, np.ones(len(normalized))))
    penalty = np.eye(design.shape[1]) * args.ridge
    penalty[-1, -1] = 0.0
    solution = np.linalg.solve(design.T @ design + penalty, design.T @ trainy)
    weights = solution[:-1].T
    bias = solution[-1]
    report = {name: metrics(*arrays(group), mean, scale, weights, bias) for name, group in groups.items()}
    model = {"version": 1, "labels": list(ACTIONS), "mean": mean.tolist(), "scale": scale.tolist(), "weights": weights.tolist(), "bias": bias.tolist(), "metrics": report}
    path = local(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
