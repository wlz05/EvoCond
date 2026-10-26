# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Inspect memory value traces and matched actor-by-memory panels on CPU.'
import argparse
import json
from pathlib import Path
from control.fatigue import aggregate, panel, trace

def main():
    parser = argparse.ArgumentParser(description='Measure marginal memory value without changing training.')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--root', type=Path)
    inputs.add_argument('--panel', type=Path)
    inputs.add_argument('--panels', nargs=4, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = trace(args.root) if args.root is not None else aggregate(args.panels) if args.panels else panel(args.panel)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text+'\n')
    print(text)

if __name__ == '__main__':
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
