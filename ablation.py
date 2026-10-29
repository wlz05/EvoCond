# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Plan retrieval x binding controls; never launch GPUs or borrow old labels.'
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import experiment as experiment
import study
def variants(config, teacher_paths):
    if config.get('format') != experiment.FORMAT or not re.fullmatch(r'[A-Za-z0-9_-]{1,50}', config.get('run_id', '')):
        raise ValueError('Expected an experiment config with a short run_id')
    experiment.teacher_options(config.get('teacher'))
    if set(teacher_paths) != {'reject', 'rebind'} or any(not isinstance(p, str) or not p for p in teacher_paths.values()):
        raise ValueError('Two matching teacher paths are required; no automatic old-label reuse')
    if teacher_paths['reject'] == teacher_paths['rebind']:
        raise ValueError('A fitted teacher cannot belong to two different binding contexts')
    result = {}
    for strategy in ('reject', 'rebind'):
        for label, alpha in (('semantic', 0.), ('structural', 1.), ('learned', None)):
            value = deepcopy(config)
            value['run_id'] += '_'+label+'_'+strategy
            value['teacher'].update(value_contracts=True, binding_strategy=strategy,
                                    teacher_path=teacher_paths[strategy], fixed_alpha=alpha)
            result[label+'.'+strategy] = value
    return result
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/contextual.json'))
    parser.add_argument('--teacher-reject')
    parser.add_argument('--teacher-rebind')
    parser.add_argument('--suite', choices=('ablation', 'statistics'), default='ablation')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(study.SEEDS))
    parser.add_argument('--conditions', nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    if not args.teacher_reject and not args.teacher_rebind:
        values = study.matrix(experiment.read(args.config), args.suite, args.seeds, args.conditions)
        if args.run and args.output is None:
            parser.error('--run requires a new --output directory')
        report = study.write(args.output, values) if args.run else study.inventory(values)
        print(json.dumps(dict(run=args.run, gpu_started=False, **report)))
        return
    if not args.teacher_reject or not args.teacher_rebind:
        parser.error('Legacy binding ablations require both matching teacher files')
    values = variants(experiment.read(args.config), {'reject': args.teacher_reject, 'rebind': args.teacher_rebind})
    if args.run:
        if args.output is None:
            parser.error('--run requires a new --output directory')
        args.output.mkdir(parents=True, exist_ok=False)
        for name, value in values.items():
            experiment.write_new(args.output/(name+'.json'), value)
    print(json.dumps({'run': args.run, 'gpu_started': False,
        'conditions': {k: v['run_id'] for k, v in values.items()},
        'next': 'Freeze each config in a new root; train/evaluate arm teacher. Recollect each binding teacher first.',
        'claim': 'Full-training ablations, NOT same-prefix causal replay; alpha=0 still retrieves memory.'}, indent=2))
if __name__ == '__main__':
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
