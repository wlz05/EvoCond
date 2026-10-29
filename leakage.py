# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

# Audit normalized split files before training or native evaluation.
import argparse
from itertools import combinations
import json
from catalogue import relative_path
from benchmark import load_cases
from convert import file_hash, normalized_question
def audit(paths):
    if len(paths) < 2 or set(paths) - {"train", "validation", "test"}:
        raise ValueError("Supply at least two named train, validation or test files")
    resolved = [path.resolve() for path in paths.values()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Split paths must be distinct")
    identities, fingerprints, prompts, groups, sources = {}, {}, {}, {}, {}
    for split, path in paths.items():
        digest = file_hash(path)
        cases = load_cases(path)
        if file_hash(path) != digest:
            raise ValueError("Split changed while auditing")
        sources[split] = {"path": path.as_posix(), "sha256": digest, "rows": len(cases)}
        identities[split] = {case["id"] for case in cases}
        fingerprints[split] = {key for case in cases for key in case["group_keys"]}
        prompts[split] = {normalized_question(case["prompt"]) for case in cases}
        groups[split] = {case["group"] for case in cases}
    overlaps = {}
    for left, right in combinations(sorted(paths), 2):
        overlaps[f"{left}:{right}"] = {"ids": len(identities[left] & identities[right]), "group_keys": len(fingerprints[left] & fingerprints[right]), "prompts": len(prompts[left] & prompts[right]), "groups": len(groups[left] & groups[right])}
    return {"sources": sources, "disjoint": not any(any(row.values()) for row in overlaps.values()), "overlaps": overlaps, "notice": "Checks declared identities, groups and normalized prompts, not semantic paraphrases or pretrained-model contamination."}
def main():
    parser = argparse.ArgumentParser(description="Audit normalized dataset splits. No data rewriting; dry-run by default.")
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--{split}")
    parser.add_argument("--output", default='runs/leakage.json')
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    report = audit({split: relative_path(getattr(args, split)) for split in ("train", "validation", "test") if getattr(args, split)})
    if args.run:
        output = relative_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    if not report["disjoint"]:
        raise SystemExit(1)
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
