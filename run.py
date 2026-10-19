# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
SEEDS = (42, 52, 3047, 3407)
def recipes(config):
    result = []
    for seed in SEEDS:
        value = deepcopy(config)
        value['run_id'] = 'evocond'+str(seed)
        value.setdefault('budget', {}).update(gpus=8, seed=seed)
        value.setdefault('contextual', {}).update(seed=seed)
        result.append((seed, value))
    return result
def main():
    parser = argparse.ArgumentParser(description='Prepare and launch the four eight-GPU experiments.')
    parser.add_argument('stage', choices=('configs', 'prepare', 'status', 'train'))
    parser.add_argument('--config', type=Path, default=Path('configs/contextual.json'))
    parser.add_argument('--root', type=Path, default=Path('runs/evocond'))
    parser.add_argument('--arm', choices=('searl', 'evocond', 'teacher'), default='teacher')
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.root.is_absolute():
        parser.error('Use unique seeds and a relative experiment directory')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    # Keep dataset partitions fixed while changing training randomness.
    for seed, value in recipes(config):
        if seed not in args.seeds:
            continue
        folder = args.root/('seed'+str(seed))
        setting = args.root/('seed'+str(seed)+'.json')
        if args.stage == 'configs':
            if args.run:
                setting.parent.mkdir(parents=True, exist_ok=True)
                if setting.exists() and json.loads(setting.read_text(encoding='utf-8')) != value:
                    raise ValueError('Existing seed configuration differs: '+str(setting))
                if not setting.exists():
                    setting.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n', encoding='utf-8')
            print(json.dumps(dict(seed=seed, gpus=8, config=str(setting), written=args.run)))
            continue
        command = [sys.executable, '-B', 'experiment.py', args.stage, '--root', str(folder)]
        if args.stage == 'prepare':
            command += ['--config', str(setting)]
        elif args.stage == 'train':
            command += ['--arm', args.arm]
        if args.run and args.stage != 'status':
            command.append('--run')
        print(json.dumps(dict(seed=seed, command=command, run=args.run)), flush=True)
        if args.run or args.stage == 'status':
            subprocess.run(command, check=True)
if __name__ == '__main__':
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
