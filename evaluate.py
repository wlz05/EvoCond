# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import argparse
import json
from control.router import ACTIONS, Router
from state.record import document, rows
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="evaluate")
    parser.add_argument("--data", default="data/pilot.jsonl")
    parser.add_argument("--model", default="runs/router.json")
    parser.add_argument("--split", default="test")
    return parser.parse_args()
def main() -> None:
    args = arguments()
    model = Router.build(document(args.model))
    items = [item for item in rows(args.data) if item.get("split") == args.split]
    if not items:
        raise ValueError("Evaluation split is empty")
    confusion = {label: {other: 0 for other in ACTIONS} for label in ACTIONS}
    regret = 0.0
    correct = 0
    for item in items:
        target = str(item["target"])
        scores = model.scores(item["features"])
        chosen = max(ACTIONS, key=lambda label: scores[label])
        utilities = item["utilities"]
        confusion[target][chosen] += 1
        correct += int(chosen == target)
        regret += max(float(utilities[label]) for label in ACTIONS) - float(utilities[chosen])
    report = {"split": args.split, "rows": len(items), "accuracy": correct / len(items), "regret": regret / len(items), "confusion": confusion}
    print(json.dumps(report, ensure_ascii=True, indent=2))
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
