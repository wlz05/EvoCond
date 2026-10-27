# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from copy import deepcopy
import json
from pathlib import Path
from control.dynamic import digest

FORMAT = 'evocond_study_v1'
SEEDS = (42, 52, 3047, 3407)
SUITES = {
    'main': ('searl', 'pace', 'evocond'),
    'ablation': ('evocond', 'semantic', 'pooled', 'global', 'identity', 'fixed', 'raw'),
    'statistics': ('evocond', 'simple', 'bernstein'),
}
CHANGES = {
    'evocond': {}, 'semantic': {'variant': 'semantic', 'fixed_alpha': 0.0},
    'pooled': {'variant': 'global', 'fixed_alpha': 0.5},
    'global': {'variant': 'global'}, 'identity': {'variant': 'noidentity'},
    'fixed': {'fixed_alpha': 0.5}, 'raw': {'variant': 'raw'},
    'simple': {'variant': 'simple'}, 'bernstein': {'confidence_method': 'bernstein'},
}

def actor(config):
    expected = dict(model_type='qwen3', hidden_size=2560, intermediate_size=9728,
        num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8, vocab_size=151936)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Main and ablation studies require the Qwen3-4B architecture, not a directory-name alias')

def setup(config):
    design = validate(config['design'])
    initialization = config.get('initialization')
    if not isinstance(initialization, dict) or set(initialization) != {'stage', 'source'}:
        raise ValueError('Declare the shared pretrained or coldstart initialization and its source')
    if initialization['stage'] not in ('pretrained', 'coldstart') or not isinstance(initialization['source'], str) or not initialization['source'].strip():
        raise ValueError('Invalid initial actor provenance')
    if not design['native']:
        options = config.get('contextual', {})
        expected = dict(variant='full', fixed_alpha=None, confidence_method='normal', frozen_batches=2)
        expected.update(CHANGES[design['condition']])
        if any(options.get(key) != value for key, value in expected.items()):
            raise ValueError('Study label does not match its declared ablation factor')

def validate(value):
    keys = {'format', 'suite', 'condition', 'arm', 'native', 'actor_batches'}
    if not isinstance(value, dict) or set(value) != keys or value['format'] != FORMAT:
        raise ValueError('Invalid study identity')
    suite, condition = value['suite'], value['condition']
    if suite not in SUITES or condition not in SUITES[suite]:
        raise ValueError('Unknown study condition')
    native = condition in ('searl', 'pace')
    expected = dict(format=FORMAT, suite=suite, condition=condition,
        arm='searl' if native else 'teacher', native=native,
        actor_batches=1 if condition == 'searl' else 2)
    if value != expected or type(value['native']) is not bool or type(value['actor_batches']) is not int:
        raise ValueError('Study label differs from its execution recipe')
    return deepcopy(value)

def matrix(config, suite='main', seeds=SEEDS, conditions=None):
    if suite not in SUITES or not isinstance(config.get('contextual'), dict) or config.get('teacher'):
        raise ValueError('Use a current contextual configuration and a declared suite')
    seeds, conditions = tuple(seeds), tuple(SUITES[suite] if conditions is None else conditions)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed not in SEEDS for seed in seeds):
        raise ValueError('Use unique seeds from 42, 52, 3047, 3407')
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions)-set(SUITES[suite]):
        raise ValueError('Unknown or duplicate study conditions')
    result = {}
    for condition in conditions:
        for seed in seeds:
            value = deepcopy(config)
            native = condition in ('searl', 'pace')
            value['run_id'] = f'{suite}_{condition}_{seed}'
            value['budget'].update(gpus=8, seed=seed, tail_policy='keep')
            value['design'] = validate(dict(format=FORMAT, suite=suite, condition=condition,
                arm='searl' if native else 'teacher', native=native,
                actor_batches=1 if condition == 'searl' else 2))
            value['contextual'].update(variant='full', fixed_alpha=None, confidence_method='normal', seed=seed, frozen_batches=2)
            value['contextual'].update(CHANGES.get(condition, {}))
            if native:
                value.pop('contextual')
            setup(value)
            result[f'{condition}/seed{seed}'] = value
    return result

def inventory(values):
    # Separate shared experimental inputs from the intended interventions.
    result = {}
    for name, value in values.items():
        design = validate(value['design'])
        shared = {key: value.get(key) for key in ('model', 'encoder', 'datasets', 'isolation_config',
            'isolation_receipt', 'search_url', 'evaluation', 'initialization')}
        budget = deepcopy(value['budget'])
        for key in ('seed', 'save_freq', 'validation_freq'):
            budget.pop(key, None)
        result[name] = dict(design=design, seed=value['budget']['seed'], gpus=value['budget']['gpus'],
            shared_inputs=digest(dict(shared=shared, budget=budget)), config_sha256=digest(value),
            actor_batches=design['actor_batches'], collection_batch=value['budget']['batch_size'],
            nominal_update_questions=value['budget']['batch_size']*design['actor_batches'])
    if len({row['shared_inputs'] for row in result.values()}) != 1:
        raise ValueError('Study conditions changed a shared experimental input')
    return dict(format=FORMAT, jobs=result, seeds=sorted({row['seed'] for row in result.values()}),
        train_unit='independent_base_question', evaluation_unit='one_answer_per_base_question')

def write(root, values):
    root = Path(root)
    record = inventory(values)
    planned = {root/(name+'.json'): value for name, value in values.items()}
    planned[root/'matrix.json'] = record
    for path, value in planned.items():
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Existing study configuration differs: '+str(path))
    for path, value in planned.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open('x', encoding='utf-8', newline='\n') as stream:
                json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write('\n')
    return record

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
