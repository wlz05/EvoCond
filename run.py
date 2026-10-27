# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import study
from control.contracts import relative, strict_json
SEEDS = (42, 52, 3047, 3407)
HERE = Path(__file__).resolve().parent
def recipes(config):
    result = []
    for seed in SEEDS:
        value = deepcopy(config)
        value['run_id'] = 'evocond'+str(seed)
        value.setdefault('budget', {}).update(gpus=8, seed=seed)
        value.setdefault('contextual', {}).update(seed=seed)
        result.append((seed, value))
    return result
def checkpoints(path):
    if path is None:
        return {}
    value = strict_json(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict) or set(value) != {'format', 'checkpoints'} or value['format'] != 'evocond_checkpoints_v1':
        raise ValueError('Expected a declared checkpoint map, not automatic test-based selection')
    if not isinstance(value['checkpoints'], dict) or not value['checkpoints'] or any(not isinstance(name, str) or not name for name in value['checkpoints']):
        raise ValueError('Checkpoint entries must use experiment-relative repository paths')
    for item in value['checkpoints'].values():
        relative(HERE, item)
    return value['checkpoints']

def main(argv=None):
    parser = argparse.ArgumentParser(description='Plan matched main and ablation experiments on eight GPUs.')
    parser.add_argument('stage', choices=('configs', 'prepare', 'status', 'train', 'evaluate'))
    parser.add_argument('--config', type=Path, default=Path('configs/contextual.json'))
    parser.add_argument('--root', type=Path, default=Path('runs/reproduction'))
    parser.add_argument('--suite', choices=tuple(study.SUITES), default='main')
    parser.add_argument('--conditions', nargs='+')
    parser.add_argument('--arm', choices=('searl', 'evocond', 'teacher'), help='Main-suite alias for --conditions')
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoints', type=Path)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--allow-test', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    if (args.resume or args.checkpoints) and args.stage not in ('train', 'evaluate'):
        parser.error('Restore arguments apply only to train or evaluate')
    for path in (args.root, args.config, args.resume, args.checkpoints):
        if path is not None:
            try:
                relative(HERE, path.as_posix())
            except ValueError as error:
                parser.error(str(error))
    if args.arm:
        if args.suite != 'main' or args.conditions:
            parser.error('--arm is a main-suite alias; use --conditions for ablations')
        args.conditions = ['searl' if args.arm == 'searl' else 'evocond']
    if args.split == 'test' and (args.stage != 'evaluate' or not args.allow_test):
        parser.error('Test evaluation requires evaluate --split test --allow-test')
    if args.resume and args.checkpoints:
        parser.error('Use either one --resume or a --checkpoints map')
    config = strict_json((HERE/args.config).read_text(encoding='utf-8-sig'))
    values = study.matrix(config, args.suite, args.seeds, args.conditions)
    folder = args.root/args.suite
    if args.stage == 'configs':
        report = study.write(HERE/folder, values) if args.run else study.inventory(values)
        print(json.dumps(dict(root=folder.as_posix(), run=args.run, gpu_started=False, **report)))
        return
    if args.resume and len(values) != 1:
        parser.error('--resume requires exactly one condition and seed')
    selected = checkpoints(HERE/args.checkpoints if args.checkpoints else None)
    if args.checkpoints is not None and set(selected) != set(values):
        parser.error('Checkpoint map must exactly cover the selected condition/seed jobs')
    if args.stage == 'evaluate' and not (args.resume or selected):
        parser.error('Evaluation requires --resume or --checkpoints; select on validation only')
    if args.run and args.stage in ('train', 'evaluate'):
        # Check every job before starting the first GPU process.
        for name, value in values.items():
            protocol = HERE/folder/name/'protocol.json'
            study.prepared(value, strict_json(protocol.read_text(encoding='utf-8-sig'), max_text=64*1024*1024), HERE)
    if args.run and args.stage == 'prepare':
        for name, value in values.items():
            setting = HERE/folder/(name+'.json')
            if strict_json(setting.read_text(encoding='utf-8-sig')) != value:
                raise ValueError('Generated configuration differs from the requested study: '+name)
    if args.run and args.stage in ('train', 'evaluate'):
        from experiment import checkpoint
        for name, value in values.items():
            resume = args.resume.as_posix() if args.resume else selected.get(name)
            if resume:
                checkpoint((HERE/folder/name).resolve(), value['design']['arm'], HERE/resume)
    for name, value in values.items():
        root = folder/name
        setting = folder/(name+'.json')
        if args.stage == 'status' and not (HERE/root/'protocol.json').is_file():
            print(json.dumps(dict(job=name, prepared=False, root=root.as_posix())))
            continue
        command = [sys.executable, '-B', str(HERE/'experiment.py'), args.stage, '--root', root.as_posix(), '--workspace', str(HERE)]
        if args.stage == 'prepare':
            if args.run and not (HERE/setting).is_file():
                raise ValueError('Generate the study configurations first: run.py configs --suite '+args.suite+' --run')
            if (HERE/setting).is_file() and strict_json((HERE/setting).read_text(encoding='utf-8-sig')) != value:
                raise ValueError('Generated configuration differs from the requested study: '+name)
            command += ['--config', setting.as_posix()]
        elif args.stage in ('train', 'evaluate'):
            command += ['--arm', value['design']['arm']]
            resume = args.resume.as_posix() if args.resume else selected.get(name)
            if resume:
                command += ['--resume', resume]
            if args.stage == 'evaluate':
                command += ['--split', args.split]
                if args.allow_test:
                    command.append('--allow-test')
        if args.run and args.stage != 'status':
            command.append('--run')
        print(json.dumps(dict(job=name, condition=value['design']['condition'], seed=value['budget']['seed'],
            command=command, run=args.run)), flush=True)
        if args.run or args.stage == 'status':
            subprocess.run(command, check=True, cwd=HERE)
if __name__ == '__main__':
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
