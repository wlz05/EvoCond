# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Plan full-run fixed-fusion controls; default is read-only, never launches GPUs.'
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import experiment as experiment
def variants(config):
    if config.get('format') != experiment.FORMAT or not re.fullmatch(r'[A-Za-z0-9_-]{1,60}', config.get('run_id', '')):
        raise ValueError('Expected an experiment config with a run_id of at most 60 characters')
    experiment.teacher_options(config.get('teacher'))
    if config['teacher'].get('fixed_alpha') is not None:
        raise ValueError('Start with the learned-teacher config, not another ablation')
    result = {}
    for label, alpha in (('semantic', 0.), ('fixed025', .25), ('fixed050', .5), ('fixed075', .75), ('structural', 1.)):
        value = deepcopy(config)
        value['run_id'] += '_'+label
        value['teacher']['fixed_alpha'] = alpha
        result[label] = value
    return result
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    values = variants(experiment.read(args.config))
    if args.run:
        if args.output is None:
            parser.error('--run requires a new --output directory')
        args.output.mkdir(parents=True, exist_ok=False)
        for name, value in values.items():
            experiment.write_new(args.output/(name+'.json'), value)
    print(json.dumps({'run': args.run, 'gpu_started': False, 'variants': {name: {
        'run_id': value['run_id'], 'fixed_alpha': value['teacher']['fixed_alpha']} for name, value in values.items()},
        'next': 'Prepare each selected config in a NEW experiment root, then train/evaluate arm teacher. '
                'Do not choose the best alpha from test results or reuse a learned-teacher checkpoint as retraining.',
        'limits': 'Fixed-fusion controls only; not no-memory, no-alignment, or no-edge ablations.'}, indent=2))
if __name__ == '__main__':
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
