# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Verified, suffix-windowed QLoRA pilot over frozen three-arm rollouts.'
from __future__ import annotations
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Any
from update import config as verified_config, train
from control.trajectory import ARMS
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--base-model", default="assets/model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--max-prompt", type=int, default=4096)
    parser.add_argument("--max-completion", type=int, default=2048)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()
def prepare(args: argparse.Namespace) -> tuple[list[dict], dict[str, Any]]:
    if args.max_prompt < 1 or args.max_completion < 1:
        raise ValueError("Window and completion budgets must be positive")
    source_args = argparse.Namespace(**vars(args))
    source_args.max_prompt = 2**31 - 1
    source_args.max_completion = 2**31 - 1
    records, protocol = verified_config(source_args)
    informative = {row["group"] for row in records if abs(row["advantage"]) > 1e-8}
    group_load: dict[str, int] = defaultdict(int)
    source_max_prompt = source_max_completion = windowed = discarded = turns = 0
    token_hash = sha256()
    for row in records:
        for turn in row["turns"]:
            prompt, completion = turn["prompt_ids"], turn["completion_ids"]
            source_max_prompt = max(source_max_prompt, len(prompt))
            source_max_completion = max(source_max_completion, len(completion))
            if len(completion) > args.max_completion:
                raise ValueError(f"Completion exceeds budget in {row['group']}/r{row['rollout']}")
            if len(prompt) > args.max_prompt:
                discarded += len(prompt) - args.max_prompt
                turn["prompt_ids"] = prompt[-args.max_prompt:]
                windowed += 1
            turns += 1
            if row["group"] in informative:
                group_load[row["group"]] += len(turn["prompt_ids"]) + len(completion)
            encoded = json.dumps([row["group"], row["rollout"], turn["prompt_ids"], completion],
                                 separators=(",", ":")).encode("ascii")
            token_hash.update(len(encoded).to_bytes(8, "big"))
            token_hash.update(encoded)
    protocol["max_prompt"] = args.max_prompt
    protocol["max_completion"] = args.max_completion
    protocol["algorithm"] = "grouped_outcome_pg_suffix_window_no_reference_kl_not_searl_exact"
    protocol["context_window"] = {"rule": "last_n_prompt_tokens_only_no_completion_truncation",
                                  "source_max_prompt": source_max_prompt,
                                  "source_max_completion": source_max_completion,
                                  "windowed_turns": windowed, "total_turns": turns,
                                  "discarded_prompt_tokens": discarded,
                                  "derived_all_rollouts_tokens_sha256": token_hash.hexdigest()}
    protocol["informative_groups"] = len(informative)
    protocol["limitation"] = ("Source episodes remain full-context and immutable; optimizer sees only the "
                              "prompt suffix. This is a resource-limited pilot, not official SEARL.")
    if args.smoke:
        group = min(sorted(informative), key=lambda key: group_load[key])
        records = [row for row in records if row["group"] == group]
        protocol["smoke_group"] = group
    selected_hash = sha256()
    for row in records:
        for turn in row["turns"]:
            encoded = json.dumps([row["group"], row["rollout"], turn["prompt_ids"],
                                  turn["completion_ids"]], separators=(",", ":")).encode("ascii")
            selected_hash.update(len(encoded).to_bytes(8, "big"))
            selected_hash.update(encoded)
    protocol["selected_training_tokens_sha256"] = selected_hash.hexdigest()
    return records, protocol
def main() -> None:
    args = arguments()
    records, protocol = prepare(args)
    target = Path(args.output).resolve()
    project = Path(__file__).resolve().parent
    if target == project or not target.is_relative_to(project):
        raise ValueError("Output must be a new directory inside EvoCond")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite an adapter or partial run: {target}")
    print(json.dumps({"output": str(target), "arm": args.arm, "run": args.run,
                      "smoke": args.smoke, "smoke_group": protocol.get("smoke_group"),
                      "selected_records": len(records),
                      "selected_turns": sum(len(row["turns"]) for row in records
                                            if abs(row["advantage"]) > 1e-8),
                      "groups_with_signal": protocol["informative_groups"],
                      "window": protocol["context_window"]}, ensure_ascii=False), flush=True)
    if not args.run:
        print("Read-only verified preflight; no Qwen weights changed.")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="windowed-staging-", dir=target.parent))
    report = train(records, protocol, staging)
    staging.rename(target)
    print(json.dumps({"saved": str(target), "optimizer_steps": report["optimizer_steps"]}), flush=True)
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
