# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Inspect dependency provenance without installing packages or loading models.'
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
CORE = ('torch', 'ray', 'vllm', 'transformers', 'numpy', 'pyarrow', 'mcp', 'sentence-transformers')
PATTERN = re.compile(r'([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[A-Za-z0-9_,.-]+\])?==([A-Za-z0-9][A-Za-z0-9.+!-]*)')
def normalize(name):
    return re.sub(r'[-_.]+', '-', name).lower()
def inspect(path):
    pins, unresolved = {}, []
    for index, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        value = line.strip()
        if not value or value.startswith('#'):
            continue
        match = PATTERN.fullmatch(value)
        if match is None:
            unresolved.append({'line': index, 'requirement': value, 'reason': 'not_a_portable_exact_pin'})
            continue
        name, version = normalize(match[1]), match[2]
        if name in pins:
            raise ValueError(f'Duplicate distribution pin: {name}')
        pins[name] = version
    versions = {}
    for name in CORE:
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[name] = dict(pinned=pins.get(name), installed=actual,
                              matches=actual is not None and actual == pins.get(name))
    return dict(format='evocond_dependency_audit_v1', exact_pins=pins, unresolved=unresolved,
                core=versions, portable_exact_lock=bool(pins) and not unresolved,
                gpu_verified=False, compatibility_certified=False,
                warning='Metadata comparison is not CUDA, ABI or distributed-runtime validation.')
def lock(path):
    result = inspect(path)
    if not result['portable_exact_lock'] or set(CORE)-set(result['exact_pins']):
        raise ValueError('Native install requires a portable exact lock containing all core distributions')
    return result
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requirements', type=Path, default=Path('baselines/SEARL-official/requirements.txt'))
    args = parser.parse_args()
    result = inspect(args.requirements)
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
if __name__ == '__main__':
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
