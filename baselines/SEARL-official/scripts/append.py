# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Append plan.yaml lines 2-30 to system prompts and write a new Parquet file.'

import json
import math
import os
from pathlib import Path
import pandas as pd

SRC = Path(os.environ.get("RL_FACTORY_APPEND_PLAN_SRC", "data/train.parquet"))
DST = Path(os.environ.get("RL_FACTORY_APPEND_PLAN_DST", "data/train_plan.parquet"))
PLAN_PATH = Path(os.environ.get("RL_FACTORY_PLAN_PROMPT", "prompt/plan.yaml"))

def load_plan_text() -> str:
    'Read plan.yaml lines 2-30 as a system-prompt fragment.'
    lines = PLAN_PATH.read_text().splitlines()

    sliced = lines[1:30]
    return "\n".join(sliced)

def normalize_prompt(prompt):
    'Normalize None, NaN, mappings, lists, JSON text and plain text into prompt message mappings.'
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
    'Append plan_text to the system message after normalizing the prompt.'
    prompt = normalize_prompt(prompt)

    if prompt and isinstance(prompt[0], dict) and prompt[0].get("role", "").lower() == "system":
        prompt[0]["content"] = f"{prompt[0].get('content', '')}\n\n{plan_text}"
        return prompt

    return [{"role": "system", "content": plan_text}] + prompt

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Source file does not exist: {SRC}")
    if not PLAN_PATH.exists():
        raise FileNotFoundError(f"Plan file does not exist: {PLAN_PATH}")

    plan_text = load_plan_text()
    df = pd.read_parquet(SRC)
    df["prompt"] = df["prompt"].apply(lambda p: add_plan_to_prompt(p, plan_text))
    df.to_parquet(DST, index=False)
    print(f"Completed: {len(df)} rows written to {DST}")

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
