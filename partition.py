# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

# Prepare explicit grouped splits without guessing fields or borrowing test data.
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import unicodedata
from benchmark import checked
from control.contracts import publish, publish_chunks
from catalogue import benchmark, pinned_revision, relative_path
from convert import file_hash, normalized_question, pointer, read_rows
SPLITS = ("train", "validation", "test")
FORMAT = "evocond_benchmark_splits_v1"
def identity(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("Task and group identities must be nonempty strings or integers")
    return json.dumps(unicodedata.normalize('NFKC', str(value)).strip(), ensure_ascii=True)
def normalize(rows, id_field, prompt_field, group_fields, reference_field, context_field=None):
    if not group_fields or len(group_fields) != len(set(group_fields)):
        raise ValueError("Declare unique group fields; prompt views must share a task group")
    records, parents, labels, ids = [], [], {}, set()
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for row in rows:
        raw_id = pointer(row, id_field)
        identity(raw_id)
        task_id = str(raw_id)
        if task_id in ids:
            raise ValueError("Record IDs must be unique; give prompt views distinct IDs and a shared task group")
        ids.add(task_id)
        prompt = pointer(row, prompt_field)
        canonical = normalized_question(prompt)
        index = len(records)
        parents.append(index)
        keys = [(field, identity(pointer(row, field))) for field in group_fields]
        keys.append(("normalized_prompt", canonical))
        for key in keys:
            if key in labels:
                parents[find(index)] = find(labels[key])
            else:
                labels[key] = index
        fingerprints = [sha256(json.dumps(key, ensure_ascii=True).encode()).hexdigest() for key in keys]
        records.append({"id": task_id, "prompt": prompt, "context": pointer(row, context_field) if context_field else {}, "reference": pointer(row, reference_field), "group_keys": sorted(fingerprints)})
    if not records:
        raise ValueError("Dataset is empty")
    components = {}
    for index, record in enumerate(records):
        components.setdefault(find(index), []).append(record["id"])
    for index, record in enumerate(records):
        record["group"] = sha256(json.dumps(sorted(components[find(index)]), ensure_ascii=True).encode()).hexdigest()
    return checked(records)
def partition(records, fractions, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError('Partition seeds must be nonnegative integers')
    if (len(fractions) != 3 or any(type(value) not in (int, float) or not 0 <= value <= 1 for value in fractions)
            or not math.isclose(math.fsum(fractions), 1.0, rel_tol=0., abs_tol=1e-9)):
        raise ValueError("Fractions must be three finite nonnegative numbers summing to one")
    groups = sorted({row["group"] for row in records}, key=lambda group: (sha256(f"{seed}:{group}".encode()).hexdigest(), group))
    counts = [math.floor(len(groups) * value) for value in fractions]
    order = sorted(range(3), key=lambda index: (-(len(groups) * fractions[index] - counts[index]), index))
    for index in order[:len(groups) - sum(counts)]:
        counts[index] += 1
    if any(fraction > 0 and count == 0 for fraction, count in zip(fractions, counts)):
        raise ValueError("Too few independent groups for the requested splits; no automatic shrinking")
    assignment, start = {}, 0
    for split, count in zip(SPLITS, counts):
        assignment.update({group: split for group in groups[start:start + count]})
        start += count
    return {split: [row for row in records if assignment[row["group"]] == split] for split in SPLITS}
def prepare(args):
    if args.policy not in ('preserve', 'partition-train', 'development-holdout') or args.source_split not in SPLITS:
        raise ValueError('Declare a supported split policy and source split')
    if type(args.seed) is not int or args.seed < 0:
        raise ValueError('Partition seeds must be nonnegative integers')
    spec = benchmark(args.benchmark)
    pinned_revision(args.revision)
    if not isinstance(args.source_uri, str) or not args.source_uri.strip():
        raise ValueError("Declare the upstream source URI")
    source = relative_path(args.input)
    output = relative_path(args.output)
    before = file_hash(source)
    if before != args.sha256:
        raise ValueError("Source SHA256 does not match")
    records = normalize(read_rows(source, args.format), args.id_field, args.prompt_field, args.group_fields.split(","), args.reference_field, args.context_field)
    if file_hash(source) != before:
        raise ValueError("Source changed during preparation")
    if args.policy == "preserve":
        if spec.evaluation_only and args.source_split == "train":
            raise ValueError("This benchmark is reserved for evaluation, not an official training pool")
        splits = {split: records if split == args.source_split else [] for split in SPLITS}
    else:
        fractions = tuple(float(value) for value in args.fractions.split(","))
        if args.policy == "partition-train":
            if spec.evaluation_only or args.source_split != "train":
                raise ValueError("Only an explicitly declared training pool may produce training splits")
        elif len(fractions) != 3 or fractions[0] != 0:
            raise ValueError("Development holdout must use zero training fraction")
        splits = partition(records, fractions, args.seed)
    report = {"format": FORMAT, "benchmark": spec.name, "source": {"path": source.as_posix(), "sha256": before, "uri": args.source_uri, "revision": args.revision, "split": args.source_split}, "policy": args.policy, "official_partition_preserved": args.policy == "preserve", "seed": args.seed, "mapping": {"id": args.id_field, "prompt": args.prompt_field, "groups": args.group_fields.split(","), "reference": args.reference_field, "context": args.context_field}, "splits": {split: {"path": f"{split}.jsonl", "rows": len(rows), "groups": len({row['group'] for row in rows})} for split, rows in splits.items() if rows}, "notice": "Generated partitions are custom research splits, not author splits. Context must contain only information visible to the actor. Native evaluation is required."}
    report['group_identity'] = 'NFKC-normalized trimmed text; numeric and textual forms share identity'
    if args.run:
        json.dumps(report, ensure_ascii=True, allow_nan=False)
        if file_hash(source) != before:
            raise ValueError('Source changed before publishing split files')
        output.mkdir(parents=True, exist_ok=False)
        for split, rows in splits.items():
            if rows:
                report['splits'][split]['sha256'] = publish_chunks(output/f'{split}.jsonl', (
                    (json.dumps(row, ensure_ascii=True, allow_nan=False)+'\n').encode('utf-8')
                    for row in sorted(rows, key=lambda item: item['id'])))
        if file_hash(source) != before or any(file_hash(output/f'{split}.jsonl') != report['splits'][split]['sha256']
                for split in report['splits']):
            raise ValueError('Source or published splits changed before completion')
        publish(output/'split_manifest.json', json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2).encode('utf-8'))
    return report
def arguments():
    parser = argparse.ArgumentParser(description="Normalize local data and preserve or explicitly group its splits. Dry-run by default.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-split", choices=SPLITS, required=True)
    parser.add_argument("--format", choices=("jsonl", "parquet"), default="jsonl")
    parser.add_argument("--policy", choices=("preserve", "partition-train", "development-holdout"), default="preserve")
    parser.add_argument("--fractions", default="0.8,0.1,0.1", help="Group fractions for train,validation,test; holdout requires e.g. 0,0.3,0.7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-field", required=True, help="Explicit JSON pointer for unique record ID")
    parser.add_argument("--prompt-field", required=True)
    parser.add_argument("--group-fields", required=True, help="Comma-separated task/family JSON pointers; variants must share a group")
    parser.add_argument("--reference-field", required=True)
    parser.add_argument("--context-field", help="Public tools or environment metadata only; never answers or private tests")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()
if __name__ == "__main__":
    print(json.dumps(prepare(arguments()), ensure_ascii=True, indent=2))

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
