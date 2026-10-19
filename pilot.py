# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import argparse
import json
from pathlib import Path
from random import Random
from control.router import ACTIONS
KINDS = ACTIONS
COSTS = {"continue": 0.0, "repair:ll": 0.05, "bridge:ll": 0.2, "bridge:gl": 0.2, "global:gl": 0.45, "global:gg": 0.45, "audit:ll": 0.15, "audit:gl": 0.15, "audit:gg": 0.15}
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pilot")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="data/pilot.jsonl")
    return parser.parse_args()
def clip(value: float) -> float:
    return max(-1.0, min(1.0, value))
def features(kind: str, rng: Random, step: int) -> list[float]:
    violation = [max(0.0, rng.gauss(0.02, 0.03)) for _ in range(3)]
    difference = [clip(rng.gauss(-0.05, 0.08)) for _ in range(3)]
    health = [min(1.0, max(0.0, rng.gauss(0.92, 0.04))) for _ in range(3)]
    if kind in {"repair:ll", "bridge:ll", "bridge:gl", "global:gl", "global:gg"}:
        index = {"repair:ll": 0, "bridge:ll": 0, "bridge:gl": 1, "global:gl": 1, "global:gg": 2}[kind]
        severe = kind in {"bridge:ll", "global:gl", "global:gg"}
        violation[index] = min(1.0, max(0.4, rng.gauss(0.9 if severe else 0.6, 0.06)))
        difference[index] = clip(rng.gauss(0.7 if severe else 0.2, 0.08))
    if kind.startswith("audit:"):
        index = {"audit:ll": 0, "audit:gl": 1, "audit:gg": 2}[kind]
        health[index] = max(0.0, min(0.45, rng.gauss(0.2, 0.1)))
    budget = min(1.0, max(0.05, rng.random()))
    return [1.0, *violation, *difference, *health, budget, min(1.0, step / 64.0)]
def utilities(kind: str, rng: Random) -> dict[str, float]:
    result: dict[str, float] = {}
    for action in ACTIONS:
        cost = max(0.0, COSTS[action] + rng.gauss(0.0, 0.01))
        result[action] = 1.0 - cost if action == kind else -1.0 - cost
    return result
def split(template: int) -> str:
    fold = template % 10
    return "train" if fold < 7 else "valid" if fold == 7 else "test"
def main() -> None:
    args = arguments()
    if args.samples < 700:
        raise ValueError("Pilot needs at least seven hundred samples")
    rng = Random(args.seed)
    path = Path(args.output)
    if path.is_absolute():
        raise ValueError("Output path must be relative")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for index in range(args.samples):
            kind = KINDS[index % len(KINDS)]
            template = index % 100
            outcomes = utilities(kind, rng)
            target = max(ACTIONS, key=lambda action: outcomes[action])
            row = {"template": f"template{template}", "split": split(template), "features": features(kind, rng, index % 65), "utilities": outcomes, "target": target, "seed": args.seed, "kind": kind}
            stream.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
    print(json.dumps({"rows": args.samples, "output": str(path), "templates": 100}, ensure_ascii=True))
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
