# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Plan two same-capability ordering controls; never start GPU jobs.'
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import experiment as experiment
def variants(config, teachers):
    if config.get('format') != experiment.FORMAT or not re.fullmatch(r'[A-Za-z0-9_-]{1,50}', config.get('run_id', '')):
        raise ValueError('Expected a short identified experiment configuration')
    experiment.teacher_options(config.get('teacher'))
    if set(teachers) != {'ranked', 'candidate_only'} or any(not isinstance(p, str) or not p for p in teachers.values()):
        raise ValueError('Both matched teachers required')
    if len(set(teachers.values())) != 2:
        raise ValueError('Changed guidance contexts cannot reuse one teacher artifact')
    result = {}
    for mode, path in teachers.items():
        value = deepcopy(config)
        value['run_id'] += '_'+mode
        value['teacher'].update(guidance_mode=mode, teacher_path=path)
        result[mode] = value
    return result
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--teacher-ranked', required=True)
    parser.add_argument('--teacher-candidate-only', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    values = variants(experiment.read(args.config), {'ranked': args.teacher_ranked,
                                                   'candidate_only': args.teacher_candidate_only})
    if args.run:
        if args.output is None:
            parser.error('--run requires a new output directory')
        args.output.mkdir(parents=True, exist_ok=False)
        for name, config in values.items():
            experiment.write_new(args.output/(name+'.json'), config)
    print(json.dumps({'run': args.run, 'gpu_started': False, 'conditions': list(values),
        'scope': 'Shared semantic candidates, history-guided ordering vs canonical ordering; not no-memory.',
        'next': 'Freeze each configuration in a new root and train/evaluate arm teacher.'}, indent=2))
if __name__ == '__main__':
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
