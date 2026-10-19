# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Generate development-only one-factor recipes without launching GPU jobs.'
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path, PureWindowsPath
from control import dynamic, stage
FACTORS = dict(rank=(16, 32, 64), neighbors=(3, 7, 15), absolute=(.5, .65, .8),
    shrinkage=(.5, 1.0, 2.0), residual=(.05, .15, .3), actor_kl_threshold=(.01, .02, .05), chain_limit=(2, 4, 8))
def portable(config):
    paths = [config.get(key) for key in ('official_root', 'model', 'encoder', 'isolation_config', 'isolation_receipt')]
    paths.extend(config.get('datasets', {}).values())
    if any(not isinstance(path, str) or Path(path).is_absolute() or PureWindowsPath(path).is_absolute() for path in paths):
        raise ValueError('Sensitivity recipes require relative file paths')
    if set(config.get('datasets', {})) != {'train', 'validation', 'test'}:
        raise ValueError('Preserve separate train, development and sealed-test paths')
def recipes(config):
    portable(config)
    base = deepcopy(config)
    base['contextual'] = stage.settings(base['contextual'])
    result = [dict(name='base', factor=None, value=None, config=base)]
    for factor, values in FACTORS.items():
        target = base['contextual'] if factor in ('actor_kl_threshold', 'chain_limit') else base['contextual']['metric']
        for value in values:
            if value == target[factor]:
                continue
            current = deepcopy(base)
            options = current['contextual']
            destination = options if factor in ('actor_kl_threshold', 'chain_limit') else options['metric']
            destination[factor] = value
            if factor == 'shrinkage':
                options['shrinkage'] = value
            current['contextual'] = stage.settings(options)
            result.append(dict(name=factor+str(value).replace('.', 'p'), factor=factor, value=value, config=current))
    for mode, residual in (('fixed', .15), ('identity', 1.0), ('unprotected', 0.0)):
        current = deepcopy(base)
        current['contextual']['metric'].update(protection=mode, residual=residual)
        current['contextual'] = stage.settings(current['contextual'])
        result.append(dict(name=mode, factor='protection', value=mode, config=current))
    for row in result:
        row['config']['run_id'] = config['run_id']+'_'+row['name']
    return result
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/contextual.json'))
    parser.add_argument('--output', type=Path, default=Path('runs/sensitivity'))
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    rows = recipes(json.loads(args.config.read_text(encoding='utf-8')))
    manifest = dict(format='evocond_development_sensitivity_v1', selection_split='validation', test_used=False,
        gpu_started=False, actor_training_started=False, rule='one_factor_at_a_time_not_test_set_search',
        shrinkage_rule='shared_relation_and_metric_prior_varied_together',
        recipes=[dict(name=row['name'], factor=row['factor'], value=row['value'], config_sha256=dynamic.digest(row['config'])) for row in rows])
    if args.run:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError('Use a new directory; frozen sensitivity recipes cannot be overwritten')
        args.output.mkdir(parents=True, exist_ok=True)
        for row in rows:
            with (args.output/(row['name']+'.json')).open('x', encoding='utf-8') as stream:
                json.dump(row['config'], stream, indent=2, sort_keys=True, allow_nan=False)
        with (args.output/'manifest.json').open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(dict(written=args.run, output=str(args.output), **manifest), sort_keys=True, allow_nan=False))
if __name__ == '__main__':
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
