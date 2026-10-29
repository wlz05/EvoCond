# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Read verified legacy rollouts; count learning signal, not claimed progress.'
import argparse
from collections import defaultdict
import json
from pathlib import Path
from control.trajectory import load_rollouts
def summarize(records, *, prompt_limit=4096):
    if type(prompt_limit) is not int or prompt_limit <= 0 or not records:
        raise ValueError('Nonempty rollouts and a positive prompt limit required')
    groups = defaultdict(list)
    prompts = []
    active_turns = 0
    for row in records:
        groups[row['group']].append(row)
        prompts.extend(len(turn['prompt_ids']) for turn in row['turns'])
        active_turns += len(row['turns']) * int(abs(row['advantage']) >= 1e-8)
    variation = sum(any(abs(row['advantage']) >= 1e-8 for row in group) for group in groups.values())
    binary = sum(len({row['success'] for row in group}) > 1 for group in groups.values())
    ordered = sorted(prompts)
    return {'groups': len(groups), 'rollouts': len(records), 'turns': len(prompts),
        'groups_with_reward_variation': variation, 'groups_with_correctness_variation': binary,
        'groups_with_only_cost_signal': variation-binary,
        'zero_advantage_groups': len(groups)-variation, 'nonzero_advantage_turns': active_turns,
        'prompt_p95': ordered[int(.95*(len(ordered)-1))], 'max_prompt': max(ordered),
        'turns_over_prompt_limit': sum(n > prompt_limit for n in prompts),
        'discarded_prefix_tokens_if_suffix_windowed': sum(max(0, n-prompt_limit) for n in prompts),
        'claim': 'Verified-source legacy proxy diagnostics, not gradient norm, learning gain, or memory causality.'}
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--prompt-limit', type=int, default=4096)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    with args.input.open(encoding='utf-8') as stream:
        first = next((json.loads(line) for line in stream if line.strip()), None)
    if first is None:
        raise ValueError('Empty rollout source')
    records, provenance = load_rollouts(args.input, first['arm'], first['model_config_sha256'],
                                       first['tokenizer_config_sha256'])
    report = {**summarize(records, prompt_limit=args.prompt_limit), 'provenance': provenance,
              'model_updated': False}
    if args.run:
        if args.output is None:
            parser.error('--run requires a new output file')
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
