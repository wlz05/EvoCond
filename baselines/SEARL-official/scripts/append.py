# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'\n\u5c06 plan.yaml \u4e2d\u7b2c 2-30 \u884c\u8ffd\u52a0\u5230 prompt \u5b57\u6bb5\u7684 system prompt\uff0c\u5e76\u751f\u6210\u65b0 parquet\u3002\n'

import json
import math
import os
from pathlib import Path
import pandas as pd

SRC = Path(os.environ.get("RL_FACTORY_APPEND_PLAN_SRC", "data/train.parquet"))
DST = Path(os.environ.get("RL_FACTORY_APPEND_PLAN_DST", "data/train_plan.parquet"))
PLAN_PATH = Path(os.environ.get("RL_FACTORY_PLAN_PROMPT", "prompt/plan.yaml"))

def load_plan_text() -> str:
    '\u8bfb\u53d6 plan.yaml \u7684\u7b2c 2-30 \u884c\u4f5c\u4e3a system prompt \u7247\u6bb5\u3002'
    lines = PLAN_PATH.read_text().splitlines()

    sliced = lines[1:30]
    return "\n".join(sliced)

def normalize_prompt(prompt):
    '\n    \u5c06 prompt \u5f52\u4e00\u5316\u4e3a list[dict]\uff0c\u907f\u514d pyarrow \u7ed3\u6784\u6df7\u6742\u62a5\u9519\u3002\n    \u652f\u6301 None/NaN\u3001dict\u3001list\u3001JSON \u5b57\u7b26\u4e32\u3001\u666e\u901a\u5b57\u7b26\u4e32\u7b49\u3002\n    '
    if prompt is None or (isinstance(prompt, float) and math.isnan(prompt)):
        return []

    if isinstance(prompt, list):
        return prompt

    if isinstance(prompt, dict):
        return [prompt]

    if isinstance(prompt, str):
        try:
            parsed = json.loads(prompt)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:

            return [{"role": "system", "content": prompt}]

    return [{"role": "system", "content": str(prompt)}]

def add_plan_to_prompt(prompt, plan_text: str):
    '\n    \u5c06 plan_text \u8ffd\u52a0\u5230 prompt \u7684 system \u6bb5\u3002\n    prompt \u9884\u671f\u4e3a list[dict]\uff0c\u4f46\u505a\u4e86\u5065\u58ee\u6027\u5904\u7406\u3002\n    '
    prompt = normalize_prompt(prompt)

    if prompt and isinstance(prompt[0], dict) and prompt[0].get("role", "").lower() == "system":
        prompt[0]["content"] = f"{prompt[0].get('content', '')}\n\n{plan_text}"
        return prompt

    return [{"role": "system", "content": plan_text}] + prompt

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"\u6e90\u6587\u4ef6\u4e0d\u5b58\u5728: {SRC}")
    if not PLAN_PATH.exists():
        raise FileNotFoundError(f"plan \u6587\u4ef6\u4e0d\u5b58\u5728: {PLAN_PATH}")

    plan_text = load_plan_text()
    df = pd.read_parquet(SRC)
    df["prompt"] = df["prompt"].apply(lambda p: add_plan_to_prompt(p, plan_text))
    df.to_parquet(DST, index=False)
    print(f"\u5b8c\u6210: {len(df)} \u884c\u5199\u5165 {DST}")

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
