# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Audit question and declared task-group overlap across train/dev/test files.'
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from convert import read_rows, question_hash, group_keys
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--{split}", required=True)
    parser.add_argument("--field", default="prompt", help="Question field; default prompt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()
def file_digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
def rows(path: Path):
    if path.suffix.lower() not in ('.json', '.jsonl', '.parquet'):
        raise ValueError(f"Unsupported dataset format: {path.suffix}")
    yield from read_rows(path, path.suffix.lower()[1:])
def question(row: dict, field: str) -> str:
    value = row.get(field)
    if isinstance(value, list):
        users = [item.get("content") for item in value
                 if isinstance(item, dict) and item.get("role") == "user"]
        value = users[-1] if users else None
    if isinstance(value, dict):
        value = value.get("content")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Cannot extract a visible question from field {field!r}")
    return value
def audit(paths: dict[str, Path], field: str) -> dict:
    if set(paths) != {'train', 'validation', 'test'} or len({path.resolve(strict=True) for path in paths.values()}) != 3:
        raise ValueError("Three split paths must differ")
    hashes, identities, groups = {}, {}, {}
    try:
        base = Path(os.path.commonpath([path.resolve().parent for path in paths.values()]))
    except ValueError:
        base = None
    for split, path in paths.items():
        before = file_digest(path)
        seen, split_groups = set(), set()
        count, grouped = 0, 0
        for row in rows(path):
            if not isinstance(row, dict):
                raise ValueError("Dataset row must be an object")
            identity = question_hash(question(row, field))
            if identity in seen:
                raise ValueError('Duplicate normalized question inside '+split+'; no silent deduplication')
            seen.add(identity)
            keys = group_keys(row)
            split_groups.update(keys)
            grouped += bool(keys)
            count += 1
        if not count:
            raise ValueError(f"{split} dataset is empty")
        if file_digest(path) != before:
            raise ValueError('Dataset changed during split audit')
        identities[split] = seen
        groups[split] = split_groups
        hashes[split] = {"path": path.resolve().relative_to(base).as_posix() if base is not None else path.name,
                         'path_base': 'split_common_directory' if base is not None else 'dataset_directory', "sha256": before,
                         "rows": count, "unique_questions": len(seen), 'grouped_rows': grouped,
                         'ungrouped_rows': count-grouped, 'unique_group_keys': len(split_groups)}
    overlap = {"train_validation": len(identities["train"] & identities["validation"]),
               "train_test": len(identities["train"] & identities["test"]),
               "validation_test": len(identities["validation"] & identities["test"])}
    group_overlap = {pair: len(groups[left] & groups[right]) for pair, left, right in (
        ('train_validation', 'train', 'validation'), ('train_test', 'train', 'test'),
        ('validation_test', 'validation', 'test'))}
    return {"format": "evocond_searl_split_audit_v3", "field": field,
            "datasets": hashes, "exact_overlap": overlap, 'group_overlap': group_overlap,
            "disjoint": not any(overlap.values()) and not any(group_overlap.values()),
            'group_metadata_complete': all(record['ungrouped_rows'] == 0 for record in hashes.values()),
            'normalization': 'NFKC + casefold + collapsed whitespace',
            "warning": "Checks normalized text and declared task/template groups; uncovered paraphrases and pretraining contamination require separate evidence"}
def main() -> None:
    args = arguments()
    paths = {split: Path(getattr(args, split)).resolve(strict=True)
             for split in ("train", "validation", "test")}
    result = audit(paths, args.field)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["disjoint"]:
        raise SystemExit(2)
    if args.run:
        output = Path(args.output).resolve()
        if output.exists():
            raise FileExistsError("Split audit output already exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
