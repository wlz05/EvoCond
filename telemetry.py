# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Summarize support withdrawal, actor alignment and measured training overhead.'
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.is_file() else []
def summarize(root):
    retrievals, actors = lines(root/'retrievals.jsonl'), lines(root/'actorupdates.jsonl')
    phases = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((root/'phases').glob('phase*.json'))]
    groups = {}
    for split in ('train', 'validation', 'test'):
        rows = [row for row in retrievals if row['split'] == split]
        blind = [row for row in rows if row['abstention']['active']]
        mismatches = [row for row in blind if row['alpha'] != 0 or row['edge_support'] or row['marked']]
        groups[split] = dict(retrievals=len(rows), independent_questions=len({row['task_id'] for row in rows}),
            blind_count=len(blind), blind_rate=len(blind)/len(rows) if rows else None,
            reasons=dict(Counter(row['abstention']['reason'] for row in blind)), fallback_contract_violations=len(mismatches),
            alpha_counts=dict(Counter(str(row['alpha']) for row in rows)))
    usage = {key: sum(row.get('measured_usage', {}).get(key, 0) for row in phases) for key in
        ('retrieval_seconds', 'tool_seconds', 'actor_input_tokens', 'actor_output_tokens')}
    return dict(format='evocond_support_telemetry_v1', splits=groups, actor_updates=len(actors),
        teacher_soft_resets=sum(row['soft_reset'] for row in actors), reset_reasons=dict(Counter(row['reason'] for row in actors if row['soft_reset'])),
        maximum_teacher_depth=max((row['teacher_depth'] for row in retrievals), default=0),
        actor_drift_seconds=sum(row['drift'].get('elapsed_seconds', 0) for row in actors if row.get('drift')),
        phase_auxiliary_seconds=sum(row['phase_auxiliary_seconds'] for row in phases), measured_usage=usage,
        maximum_serialized_registry_bytes=max((row.get('storage', {}).get('registry_json_bytes', 0) for row in phases), default=0),
        warning='Support diagnostics are not task-success, SOTA, causal or peak-memory measurements; no configuration is selected here.')
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2, sort_keys=True, allow_nan=False))
if __name__ == '__main__':
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
