# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from environment import inspect, lock
PACKAGES = ("accelerate>=1.5", "datasets>=3.4", "fastapi>=0.115", "huggingface_hub>=0.30", "numpy>=2.0", "peft>=0.15", "safetensors>=0.5", "transformers>=4.51,<5", "trl>=0.17", "uvicorn>=0.34")
def main() -> None:
    parser = argparse.ArgumentParser(description='Read-only dependency preflight; installation requires --run.')
    parser.add_argument('--profile', choices=('native', 'legacy'), default='native')
    parser.add_argument('--requirements', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.profile == 'legacy':
        if args.requirements is not None:
            parser.error('Legacy profile does not accept a native dependency lock')
        command = [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', *PACKAGES]
        report = dict(profile='legacy', run=args.run, command=command,
                      warning='Legacy CostBench/LoRA dependencies; not the native SEARL environment.')
    elif args.requirements is None:
        path = Path('baselines/SEARL-official/requirements.txt')
        audit = inspect(path) if path.is_file() else None
        report = dict(profile='native', run=False, audit=audit,
                      blocker='Provide a reviewed portable exact lock with --requirements; no implicit upgrades.')
        print(json.dumps(report, ensure_ascii=True, allow_nan=False))
        if args.run:
            raise ValueError(report['blocker'])
        return
    else:
        audit = lock(args.requirements)
        command = [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-r', str(args.requirements)]
        report = dict(profile='native', run=args.run, command=command, audit=audit,
                      warning='Exact pins do not certify GPU compatibility; validate an isolated environment.')
    print(json.dumps(report, ensure_ascii=True, allow_nan=False))
    if args.run:
        subprocess.run(command, check=True)
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
