# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Single-GPU, train-only LoRA outcome policy update from verified rollouts.'
from __future__ import annotations
import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import tempfile
from typing import Any
from control.trajectory import ARMS, digest, load_rollouts
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
    parser.add_argument("--smoke", action="store_true", help="Update one informative group only, saving a separate smoke adapter")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()
def config(args: argparse.Namespace) -> tuple[list[dict], dict[str, Any]]:
    model = Path(args.base_model).resolve(strict=True)
    cfg = model / "config.json"
    tok = model / "tokenizer_config.json"
    if not cfg.is_file() or not tok.is_file():
        raise FileNotFoundError("Qwen model config or tokenizer config is missing")
    identity = json.loads(cfg.read_text(encoding="utf-8"))
    if identity.get("model_type") != "qwen3" or identity.get("hidden_size") != 2560 or identity.get("num_hidden_layers") != 36:
        raise ValueError("This trainer expects the Qwen3-4B backbone")
    if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or
            args.lora_r < 1 or args.max_prompt < 1 or args.max_completion < 1):
        raise ValueError("Invalid training budget")
    records, report = load_rollouts(Path(args.rollouts), args.arm, digest(cfg), digest(tok))
    for row in records:
        for turn in row["turns"]:
            if len(turn["prompt_ids"]) > args.max_prompt or len(turn["completion_ids"]) > args.max_completion:
                raise ValueError("A policy turn exceeds the frozen training token budget")
    if report["nonzero_advantages"] == 0:
        raise ValueError("All grouped advantages are zero; no policy update is possible")
    protocol = {"format": "evocond_single_gpu_group_pg_v1", "arm": args.arm,
                "base_model": str(model), "model_config_sha256": digest(cfg),
                "tokenizer_config_sha256": digest(tok), "rollouts": str(Path(args.rollouts).resolve()),
                "rollouts_sha256": report["rollouts_sha256"], "query_sha256": report["query_sha256"],
                "seed": args.seed, "learning_rate": args.learning_rate, "lora_r": args.lora_r,
                "max_prompt": args.max_prompt, "max_completion": args.max_completion,
                "smoke": bool(getattr(args, "smoke", False)),
                "algorithm": "grouped_outcome_policy_gradient_no_reference_kl_not_searl_exact",
                "data": report}
    return records, protocol
def smoke_group(records: list[dict]) -> str:
    eligible = {row["group"] for row in records if abs(row["advantage"]) > 1e-8}
    if not eligible:
        raise ValueError("No informative grouped rollout exists for the smoke update")
    longest = {group: max(len(turn["prompt_ids"]) + len(turn["completion_ids"])
                          for row in records if row["group"] == group for turn in row["turns"])
               for group in eligible}
    return max(sorted(eligible), key=lambda group: longest[group])
def train(records: list[dict], protocol: dict, output: Path) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as error:
        raise RuntimeError("Install compatible torch, transformers, peft, accelerate and bitsandbytes") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to update Qwen3-4B")
    torch.manual_seed(protocol["seed"])
    random.seed(protocol["seed"])
    model_dir = protocol["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=quantization,
                                                device_map={"": 0}, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=protocol["lora_r"], lora_alpha=2*protocol["lora_r"],
                                            lora_dropout=0.0, bias="none", target_modules="all-linear",
                                            task_type="CAUSAL_LM"))
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=protocol["learning_rate"])
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_group[row["group"]].append(row)
    order = sorted(by_group)
    random.Random(protocol["seed"]).shuffle(order)
    steps, processed, skipped, losses = 0, 0, 0, []
    for key in order:
        items = by_group[key]
        processed += 1
        if not any(abs(row['advantage']) >= 1e-8 for row in items):
            skipped += 1
            print(json.dumps({'arm': protocol['arm'], 'group': key, 'skipped': 'zero_group_advantage',
                              'processed_groups': processed, 'optimizer_steps': steps}), flush=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        for row in items:
            advantage = row["advantage"]
            if abs(advantage) < 1e-8:
                continue
            for turn in row["turns"]:
                prompt = turn["prompt_ids"]
                completion = turn["completion_ids"]
                ids = torch.tensor([prompt + completion], device="cuda", dtype=torch.long)
                output_logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
                logits = output_logits[0, len(prompt)-1:-1, :]
                target = ids[0, len(prompt):]
                if logits.shape[0] != target.shape[0]:
                    raise RuntimeError("Policy token alignment failed")
                token_loss = functional.cross_entropy(logits, target, reduction="mean")
                weighted = token_loss * (advantage / (len(items) * len(row["turns"])))
                weighted.backward()
                losses.append(float(weighted.detach().cpu()))
                del ids, output_logits, logits, target, token_loss, weighted
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        steps += 1
        print(json.dumps({"arm": protocol["arm"], "group": key, "optimizer_step": steps}), flush=True)
    if processed != len(order) or steps + skipped != processed:
        raise RuntimeError("Training stopped before all groups were processed")
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    report = {"format": protocol["format"], "arm": protocol["arm"],
              "optimizer_steps": steps, "mean_weighted_turn_loss": sum(losses)/len(losses) if losses else None,
              'processed_groups': processed, 'zero_advantage_groups_skipped': skipped,
              'historical_counter_compatible': False,
              "protocol": protocol, "warning": "Single-pass grouped outcome policy gradient; no step-anchor reward, reference KL, or official SEARL equivalence."}
    (output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return report
def main() -> None:
    args = arguments()
    records, protocol = config(args)
    if args.smoke:
        group = smoke_group(records)
        records = [row for row in records if row["group"] == group]
        protocol["smoke_group"] = group
    target = Path(args.output).resolve()
    project = Path(__file__).resolve().parent
    if not target.is_relative_to(project) or target == project:
        raise ValueError("Output must be a new directory inside EvoCond")
    print(json.dumps({"target": str(target), "protocol": protocol, "run": args.run}, ensure_ascii=False), flush=True)
    if not args.run:
        print("Read-only policy update preflight; no Qwen weights changed.")
        return
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite an existing model or partial run: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="policy-staging-", dir=target.parent))
    report = train(records, protocol, staging)
    staging.rename(target)
    print(json.dumps({"saved": str(target), "optimizer_steps": report["optimizer_steps"]}, ensure_ascii=False), flush=True)
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
