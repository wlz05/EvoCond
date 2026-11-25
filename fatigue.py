# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Inspect memory value traces and matched actor-by-memory panels on CPU.'
import argparse
import json
from pathlib import Path
from control.fatigue import aggregate, panel, trace
from control.contracts import publish, strict_json
from control import search

def read(path, *, limit=64*1024*1024):
    return strict_json(path.read_text(encoding='utf-8-sig'), max_text=limit)

def main():
    parser = argparse.ArgumentParser(description='Measure marginal memory value without changing training.')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--root', type=Path)
    inputs.add_argument('--panel', type=Path)
    inputs.add_argument('--panels', nargs=4, type=Path)
    inputs.add_argument('--search-prefixes', type=Path,
        help='Fetch fixed first hops from a JSON manifest with plan, query and optional topk per question hash')
    inputs.add_argument('--search-panel', type=Path, help='Prepared manifest for replay, annotation or paired scoring')
    parser.add_argument('--check-repeat', action='store_true', help='Compare the two old semantic runs before launching other cells')
    parser.add_argument('--hotpot', type=Path, help='Original HotpotQA JSON with supporting facts')
    parser.add_argument('--annotations', type=Path, help='Offline passage aliases and answers per question hash')
    parser.add_argument('--corpus', type=Path, help='Pinned retrieval corpus JSONL')
    for cell in search.CELLS:
        parser.add_argument('--'+cell.replace('_', '-'), type=Path, help='Executed '+cell+' search.jsonl')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    exports = {cell: getattr(args, cell) for cell in search.CELLS if getattr(args, cell) is not None}
    if args.check_repeat:
        if not args.search_panel or set(exports) != {'old_semantic', 'old_repeat'} or args.corpus or args.hotpot or args.annotations:
            parser.error('Replay needs --search-panel, --old-semantic and --old-repeat only')
        result = search.check_repeat(search.load_exports(search.manifest(read(args.search_panel)), exports))
    elif args.search_panel:
        if not args.corpus or bool(args.hotpot) == bool(args.annotations):
            parser.error('Search panels need --corpus and exactly one of --hotpot or --annotations')
        spec = read(args.search_panel)
        if args.hotpot:
            if exports:
                parser.error('Publish --hotpot annotations before scoring the search panel')
            result = search.hotpot(spec, read(args.hotpot, limit=512*1024*1024), args.corpus)
        else:
            if exports.keys() != set(search.CELLS):
                parser.error('Paired scoring needs four cell exports and --old-repeat')
            result = search.panel(spec, read(args.annotations), args.corpus, exports)
    elif args.search_prefixes:
        if not args.output or exports or args.corpus or args.hotpot or args.annotations:
            parser.error('Prefix preparation needs --output and a plan/query manifest only')
        if args.output.exists():
            parser.error('Prefix output already exists')
        result = search.prepare(read(args.search_prefixes))
    else:
        if exports or args.corpus or args.hotpot or args.annotations:
            parser.error('Evidence arguments require --search-panel')
        result = trace(args.root) if args.root is not None else aggregate(args.panels) if args.panels else panel(args.panel)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        publish(args.output, text)
    print(text)

if __name__ == '__main__':
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
