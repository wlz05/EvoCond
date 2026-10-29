# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import subprocess
import sys
def run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], check=True)
def main() -> None:
    run("check.py")
    run("pilot.py", "--samples", "6000", "--output", "data/pilot.jsonl")
    run("train.py", "--data", "data/pilot.jsonl", "--output", "runs/router.json")
    run("evaluate.py", "--data", "data/pilot.jsonl", "--model", "runs/router.json", "--split", "test")
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
