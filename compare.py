# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Read-only, keyed comparison of EVOLT and EvoCond SEARL-resource outputs.'
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
from eval import NAMES, answer, corpus, digest, local_correct
def predictions(path: Path) -> dict[str, dict]:
    values: dict[str, dict] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("key"), str) or not isinstance(row.get("output"), str):
            raise ValueError(f"Invalid prediction row {number}: {path}")
        if row["key"] in values:
            raise ValueError(f"Duplicate prediction key {row['key']}: {path}")
        values[row["key"]] = row
    if not values:
        raise ValueError(f"Empty prediction file: {path}")
    return values
def compare(data: Path, name: str, split: str, evolt: Path, evocond: Path) -> dict:
    reference = {row.key: row for row in corpus(data, name, split)}
    left, right = predictions(evolt), predictions(evocond)
    if left.keys() != right.keys() or not left.keys() <= reference.keys():
        raise ValueError("The two arms must contain exactly the same recognized sample keys")
    source_hash = digest(data)
    for method, values in (("EVOLT", left), ("EvoCond", right)):
        for row in values.values():
            protocol = row.get("protocol")
            if not isinstance(protocol, dict) or protocol.get("data_sha256") != source_hash or protocol.get("dataset") != name or protocol.get("split") != split:
                raise ValueError(f"{method} data protocol differs from the provided benchmark")
    if next(iter(left.values()))["protocol"].get("backend") != "searl":
        raise ValueError("EVOLT predictions did not use the SEARL tool backend")
    left_model = {row["protocol"].get("model_config_sha256") for row in left.values()}
    right_model = {row["protocol"].get("model_config_sha256") for row in right.values()}
    if len(left_model) != 1 or None in left_model or left_model != right_model:
        raise ValueError("Backbone model config differs between paired arms")
    for field, counterpart in (("tokens", "tokens"), ("tool_rounds", "rounds"), ("search_url", "search_url"), ("sandbox_url", "sandbox_url")):
        left_values = {row["protocol"].get(field) for row in left.values()}
        right_values = {row["protocol"].get(counterpart) for row in right.values()}
        if len(left_values) != 1 or None in left_values or left_values != right_values:
            raise ValueError(f"Paired inference protocol mismatch: {field}")
    counts = {"EVOLT": 0, "EvoCond": 0, "evocond_only": 0, "evolt_only": 0, "both": 0, "neither": 0}
    tools = {"EVOLT": 0, "EvoCond": 0}
    for key in sorted(left):
        a, b = left[key], right[key]
        expected = reference[key].answer
        a_final, b_final = answer(a["output"]), answer(b["output"])
        a_ok = bool(a_final) and local_correct(a_final, expected)
        b_ok = bool(b_final) and local_correct(b_final, expected)
        counts["EVOLT"] += int(a_ok)
        counts["EvoCond"] += int(b_ok)
        counts["both" if a_ok and b_ok else "evocond_only" if b_ok else "evolt_only" if a_ok else "neither"] += 1
        tools["EVOLT"] += len(re.findall(r"<tool_call>.*?</tool_call>", a["output"], flags=re.DOTALL | re.IGNORECASE))
        tools["EvoCond"] += len(re.findall(r"<tool_call>.*?</tool_call>", b["output"], flags=re.DOTALL | re.IGNORECASE))
    return {"n": len(left), "correct_local": counts, "tool_calls": tools, "score_warning": "Local deterministic paired diagnostic, not SEARL Qwen3-32B judged pass@1. Same dataset/model config is necessary but prompt and controller differences remain."}
def main() -> None:
    parser = argparse.ArgumentParser(description="Compare existing EVOLT/EvoCond predictions without inference")
    parser.add_argument("--data", required=True)
    parser.add_argument("--name", required=True, choices=NAMES)
    parser.add_argument("--split", default="test", choices=("valid", "test"))
    parser.add_argument("--evolt", required=True)
    parser.add_argument("--evocond", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(Path(args.data), args.name, args.split, Path(args.evolt), Path(args.evocond)), ensure_ascii=True, indent=2))
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
