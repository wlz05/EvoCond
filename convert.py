# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Prepare explicit, provenance-checked splits for the pinned MCPBase loader.'
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import unicodedata
from control.contracts import publish, relative, strict_json
from catalogue import pinned_revision
FORMAT = "searl_dataset_manifest_v1"
AUDIT_FORMAT = "searl_dataset_audit_v1"
SPLITS = ("train", "validation", "test")
SUPPORTED_REWARDS = ("arpo", "MATH-Hard", "jee", "PHYBench")
def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
def normalized_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Every row needs a nonempty visible question")
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())
def question_hash(question: str) -> str:
    return sha256(normalized_question(question).encode("utf-8")).hexdigest()
def grouping(value):
    if (not isinstance(value, dict) or set(value) != {'namespace', 'fields'}
            or not isinstance(value['namespace'], str) or not value['namespace'].strip()
            or value['namespace'] != value['namespace'].strip()
            or not isinstance(value['fields'], dict) or not value['fields']):
        raise ValueError('Grouping requires a stable namespace and explicit identity fields')
    for label, path in value['fields'].items():
        if (not isinstance(label, str) or not label.strip() or label != label.strip()
                or not isinstance(path, str) or not path.startswith('/')):
            raise ValueError('Group labels must be stable names mapped to JSON pointers')
    return value

def mapped_groups(row, specification):
    declaration = grouping(specification)
    result = []
    for label, path in sorted(declaration['fields'].items()):
        value = pointer(row, path)
        if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
            raise ValueError('Mapped group identity must be nonempty text or integer')
        # Share identities across source files, split names and field aliases.
        identity = unicodedata.normalize('NFKC', str(value)).strip()
        payload = json.dumps([declaration['namespace'], label, identity], ensure_ascii=True, separators=(',', ':'))
        result.append(sha256(payload.encode('utf-8')).hexdigest())
    return sorted(result)

def group_keys(row):
    extra = row.get('extra_info')
    keys = extra.get('split_group_keys') if isinstance(extra, dict) else None
    if keys is None:
        return []
    if (not isinstance(keys, list) or not keys
            or any(not isinstance(key, str) or not re.fullmatch('[a-f0-9]{64}', key) for key in keys)
            or len(keys) != len(set(keys))):
        raise ValueError('split_group_keys must contain unique lowercase SHA256 identities')
    return keys

def pointer(row, path: str):
    'Read an explicit RFC 6901-style JSON pointer; no guessed field names.'
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("Mappings must use explicit JSON pointers such as /answer")
    value = row
    for part in path[1:].split("/"):
        if "~" in part.replace("~0", "").replace("~1", ""):
            raise ValueError("Malformed JSON pointer escape")
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not re.fullmatch(r'0|[1-9][0-9]*', part) or int(part) >= len(value):
                raise ValueError(f"Missing mapped list element: {path}")
            value = value[int(part)]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise ValueError(f"Missing mapped field: {path}")
    return value
def _answers(value):
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not values or not all(isinstance(item, str) and item.strip() for item in values):
        raise ValueError("Answers must be a nonempty string or list of strings; no inferred answer extraction")
    return list(values)
def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet input/output requires pyarrow; JSONL dry-run does not") from exc
    return pa, pq
def read_rows(path: Path, source_format: str):
    if source_format == "jsonl":
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise ValueError(f"Blank JSONL row at line {line_number}")
                row = strict_json(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row {line_number} is not an object")
                yield row
    elif source_format == 'json':
        limit = 64*1024*1024
        if path.stat().st_size > limit:
            raise ValueError('JSON array exceeds 64 MiB; use streaming JSONL or Parquet without truncation')
        value = strict_json(path.read_text(encoding='utf-8-sig'), max_text=limit)
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError('JSON input must be an explicit array of row objects; map nested exports first')
        yield from value
    elif source_format == "parquet":
        _, pq = _pyarrow()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    else:
        raise ValueError("Unknown source format")
def inspect_source(path: Path, source_format: str) -> dict:
    'List field shapes and identities without printing question/answer text.'
    path = path.resolve()
    digest = file_hash(path)
    shapes, count = [], 0
    def shape(value, depth=0):
        if depth > 6:
            return "nested"
        if isinstance(value, dict):
            return {key: shape(item, depth + 1) for key, item in value.items()}
        if isinstance(value, list):
            return [shape(value[0], depth + 1)] if value else []
        return type(value).__name__
    for row in read_rows(path, source_format):
        count += 1
        if len(shapes) < 3:
            candidate = shape(row)
            if candidate not in shapes:
                shapes.append(candidate)
    if file_hash(path) != digest:
        raise ValueError("Source changed during inspection")
    return {"path": Path(os.path.relpath(path, Path.cwd())).as_posix(), 'path_base': 'working_directory',
            "format": source_format, "sha256": digest,
            "rows": count, "first_distinct_shapes": shapes,
            "notice": "No field mapping, split assignment or author provenance inferred"}
def read_manifest(path: Path) -> dict:
    manifest = strict_json(path.read_text(encoding="utf-8-sig"))
    if (not isinstance(manifest, dict) or set(manifest) != {"format", "mode", "sources"}
            or manifest["format"] != FORMAT
            or manifest["mode"] not in ("author_preprocessed", "adapted_upstream")
            or not isinstance(manifest["sources"], list) or not manifest["sources"]):
        raise ValueError("Expected format, mode and nonempty sources in the manifest")
    labels, source_paths, namespaces = set(), set(), {}
    for source in manifest["sources"]:
        required = {"name", "path", "sha256", "format", "split", "source_split", "source_uri", "source_revision"}
        optional = {"mapping", "ability", "data_source", "grouping"}
        if not isinstance(source, dict) or not required <= set(source) or set(source) - required - optional:
            raise ValueError("Source metadata is missing or contains unknown keys")
        if not all(isinstance(source[key], str) and source[key].strip() for key in required):
            raise ValueError("All source identity fields must be nonempty strings")
        if source["name"] in labels:
            raise ValueError("Source names must be unique")
        labels.add(source["name"])
        if source["split"] not in SPLITS or source["source_split"] not in SPLITS:
            raise ValueError("Every source must declare train, validation or test")
        if source["split"] != source["source_split"]:
            raise ValueError("Source split reassignment is forbidden; explicitly partition upstream first")
        if source["format"] not in ("json", "jsonl", "parquet"):
            raise ValueError("Only explicit JSON arrays, JSONL and Parquet inputs are supported")
        if len(source["sha256"]) != 64 or any(char not in "0123456789abcdef" for char in source["sha256"]):
            raise ValueError("Source SHA256 must be an exact lowercase digest")
        try:
            pinned_revision(source['source_revision'])
        except ValueError as error:
            raise ValueError('Floating or invalid source revisions are forbidden') from error
        source_path = Path(source["path"])
        if source_path.is_absolute() or PureWindowsPath(source['path']).drive or PureWindowsPath(source['path']).root or '\\' in source['path']:
            raise ValueError('Source paths must use portable relative paths from the manifest directory')
        resolved = (path.parent / source_path).resolve()
        if resolved in source_paths:
            raise ValueError("The same source file cannot be assigned more than once")
        source_paths.add(resolved)
        if not resolved.is_file():
            raise ValueError(f"Source file is missing: {resolved}")
        if manifest["mode"] == "author_preprocessed":
            if source["format"] != "parquet" or set(source) & optional:
                raise ValueError("Author mode preserves Parquet without mapping or relabelling")
        else:
            mapping = source.get("mapping")
            if (not isinstance(mapping, dict) or not {"question", "answer"} <= set(mapping)
                    or set(mapping) - {"question", "answer", "id", "mcp_worthiness", "tool_call_count"}):
                raise ValueError("Adapted input requires explicit question and answer mappings")
            if not all(isinstance(value, str) and value.startswith("/") for value in mapping.values()):
                raise ValueError("Mappings must be JSON pointers")
            if source.get("data_source") != "arpo" or not isinstance(source.get("ability"), str) or not source["ability"].strip():
                raise ValueError("Adapted mode requires explicit arpo reward route and ability")
            if 'grouping' in source:
                declaration = grouping(source['grouping'])
                namespace, fields = declaration['namespace'], set(declaration['fields'])
                if namespace in namespaces and namespaces[namespace] != fields:
                    raise ValueError('One grouping namespace must use the same identity labels across sources')
                namespaces[namespace] = fields
    if manifest["mode"] == "author_preprocessed":
        counts = Counter(source["split"] for source in manifest["sources"])
        if max(counts.values()) > 1:
            raise ValueError("Byte-preserving author mode requires one source Parquet per split")
    return manifest
def validate_official_row(row: dict) -> str:
    'Validate the actual MCPBase reward route rather than a generic trainer.'
    if not isinstance(row, dict) or row.get("data_source") not in SUPPORTED_REWARDS:
        raise ValueError("Unsupported MCPBase data_source")
    question = row.get("question")
    normalized_question(question)
    prompt = row.get("prompt")
    if (not isinstance(prompt, list) or not prompt or
            any(not isinstance(message, dict) or message.get("role") not in ("system", "user")
                or not isinstance(message.get("content"), str) for message in prompt)):
        raise ValueError("Initial prompt must contain visible system/user text only")
    users = [message["content"] for message in prompt if message["role"] == "user"]
    if not users or users[-1] != question:
        raise ValueError("Latest prompt user text must equal the visible question field")
    reward = row.get("reward_model")
    if (not isinstance(reward, dict) or reward.get("style") != "rule"
            or not isinstance(reward.get("ground_truth"), dict)
            or "target" not in reward["ground_truth"]):
        raise ValueError("MCPBase expects reward_model.ground_truth.target and rule style")
    target = reward["ground_truth"]["target"]
    if row["data_source"] == "arpo":
        _answers(target)
    elif not isinstance(target, str) or not target.strip():
        raise ValueError("The preserved non-arpo reward route requires a string target")
    extra = row.get("extra_info")
    if not isinstance(extra, dict) or type(extra.get("index")) is not int or extra["index"] < 0:
        raise ValueError("Every row needs a nonnegative extra_info.index")
    group_keys(row)
    for key in ("mcp_worthiness", "tool_call_count"):
        value = row.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f"Invalid {key}")
    count = row.get("tool_call_count")
    if count is not None and (count < 0 or int(count) != count):
        raise ValueError("Expected tool_call_count must be a nonnegative integer")
    return question
def adapt_row(row: dict, source: dict, row_number: int, split_index: int) -> dict:
    mapping = source["mapping"]
    question = pointer(row, mapping["question"])
    normalized_question(question)
    answers = _answers(pointer(row, mapping["answer"]))
    original_id = pointer(row, mapping["id"]) if "id" in mapping else str(row_number)
    if isinstance(original_id, bool) or not isinstance(original_id, (str, int)) or not str(original_id).strip():
        raise ValueError("Mapped row ID must be text or integer")
    result = {"data_source": source["data_source"], "ability": source["ability"],
              "prompt": [{"role": "user", "content": question}], "question": question,
              "reward_model": {"style": "rule", "ground_truth": {"target": answers}},
              "extra_info": {"index": split_index, "source_name": source["name"],
                             "source_row": row_number, "source_id": str(original_id),
                             "source_sha256": source["sha256"], "source_split": source["source_split"],
                             "source_revision": source["source_revision"], "source_uri": source["source_uri"],
                             "preprocessing": "adapted_upstream_not_author_preprocessed"}}
    if 'grouping' in source:
        result['extra_info']['split_group_keys'] = mapped_groups(row, source['grouping'])
    for key in ("mcp_worthiness", "tool_call_count"):
        if key in mapping:
            result[key] = pointer(row, mapping[key])
    validate_official_row(result)
    return result
def build_plan(manifest_path: Path) -> tuple[dict, dict, dict]:
    manifest_path = manifest_path.resolve()
    manifest_digest = file_hash(manifest_path)
    manifest = read_manifest(manifest_path)
    rows, paths, identities, sources = {}, {}, {}, []
    row_ids, source_ids, groups, coverage = set(), set(), {}, {}
    for source in manifest["sources"]:
        path = (manifest_path.parent / source["path"]).resolve()
        if file_hash(path) != source["sha256"]:
            raise ValueError(f"Source digest mismatch: {source['name']}")
        split = source["split"]
        target = rows.setdefault(split, [])
        statistics = coverage.setdefault(split, dict(grouped_rows=0, ungrouped_rows=0))
        seen = 0
        for row_number, raw in enumerate(read_rows(path, source["format"]), 1):
            row = raw if manifest["mode"] == "author_preprocessed" else adapt_row(raw, source, row_number, len(target))
            if manifest['mode'] == 'adapted_upstream':
                source_key = (source['name'], row['extra_info']['source_id'])
                if source_key in source_ids:
                    raise ValueError('Duplicate mapped source ID within one source; no automatic renumbering')
                source_ids.add(source_key)
            question = validate_official_row(row)
            keys = group_keys(row)
            for group in keys:
                if group in groups and groups[group] != split:
                    raise ValueError(f'Task/template group crosses {groups[group]} and {split}; partition related variants together')
                groups[group] = split
            statistics['grouped_rows' if keys else 'ungrouped_rows'] += 1
            key = question_hash(question)
            if key in identities:
                previous = identities[key]
                raise ValueError(f"Duplicate normalized question: {previous['split']} and {split}; no automatic deduplication")
            identities[key] = {"split": split, "source": source["name"], "row": row_number}
            id_key = (split, row["extra_info"]["index"])
            if id_key in row_ids:
                raise ValueError("Duplicate row index within a split")
            row_ids.add(id_key)
            target.append(row)
            seen += 1
        if not seen:
            raise ValueError(f"Empty source: {source['name']}")
        if file_hash(path) != source["sha256"]:
            raise ValueError("Source changed during validation")
        sources.append({**source, "rows": seen})
        if manifest["mode"] == "author_preprocessed":
            paths[split] = path
    if file_hash(manifest_path) != manifest_digest:
        raise ValueError("Manifest changed during validation")
    audit = {"format": AUDIT_FORMAT, "manifest_sha256": manifest_digest, "mode": manifest["mode"],
             'source_path_base': 'input_manifest_directory',
             "official_author_preprocessing_claim": manifest["mode"] == "author_preprocessed",
             "author_provenance_independently_verified": False,
             "training_inputs_present": "train" in rows and "validation" in rows,
             "counts": {split: len(items) for split, items in rows.items()}, "sources": sources,
             "normalized_question_sha256": identities, "cross_split_duplicates": 0,
             'split_group_coverage': coverage, 'cross_split_group_overlap': 0,
             'grouping_scope': 'Explicit task/template identities only; uncovered variants require upstream grouping',
             "normalization": "NFKC + casefold + collapsed whitespace; no semantic deduplication",
             "question_order": "manifest source order, then source row order; no shuffle or filtering",
             "reward_contract": "MCPBase computes reward_model.ground_truth.target via data_source route",
             "warning": "Schema-valid is not author preprocessing reproduction, reward correctness validation, or training readiness. "
                        "Missing tool-call expectations are not invented; source license/provenance declarations are not verified online."}
    return audit, rows, paths
def _write_json(path: Path, value: dict) -> None:
    publish(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))

def prepared(paths, *, mode=None, counts=None):
    if (not isinstance(paths, dict) or not paths or set(paths)-set(SPLITS)
            or mode not in (None, 'adapted_upstream', 'author_preprocessed')
            or counts is not None and not isinstance(counts, dict)):
        raise ValueError('Declare prepared split paths, provenance and optional row counts')
    verified = {}
    for split, source in paths.items():
        path = Path(source).resolve(strict=True)
        marker, audit_path = path.parent/'preparation_started.json', path.parent/'dataset_audit.json'
        if not marker.exists() and not audit_path.exists() and mode != 'adapted_upstream':
            continue
        if not audit_path.is_file():
            raise ValueError('Converted datasets require a complete dataset_audit.json')
        payload = audit_path.read_bytes()
        audit = strict_json(payload.decode('utf-8-sig'), max_text=64*1024*1024)
        if (not isinstance(audit, dict) or audit.get('format') != AUDIT_FORMAT or audit.get('complete') is not True
                or audit.get('files_written') is not True or audit.get('mode') not in ('author_preprocessed', 'adapted_upstream')
                or mode is not None and audit['mode'] != mode
                or not isinstance(audit.get('manifest_sha256'), str) or not re.fullmatch('[a-f0-9]{64}', audit['manifest_sha256'])):
            raise ValueError('Dataset preparation is incomplete or has different provenance')
        if marker.exists():
            started = strict_json(marker.read_text(encoding='utf-8-sig'))
            if (not isinstance(started, dict) or started.get('complete') is not False
                    or started.get('manifest_sha256') != audit['manifest_sha256']):
                raise ValueError('Dataset completion audit does not match its preparation marker')
        outputs, declared_counts = audit.get('outputs'), audit.get('counts')
        record = outputs.get(split) if isinstance(outputs, dict) else None
        if (not isinstance(record, dict) or not isinstance(declared_counts, dict)
                or type(record.get('rows')) is not int or record['rows'] < 1
                or type(declared_counts.get(split)) is not int
                or record['rows'] != declared_counts.get(split)
                or counts is not None and (type(counts.get(split)) is not int or record['rows'] != counts.get(split))
                or not isinstance(record.get('sha256'), str) or not re.fullmatch('[a-f0-9]{64}', record['sha256'])
                or not isinstance(record.get('path'), str) or not record['path']):
            raise ValueError('Prepared split metadata or row counts do not match')
        name = record['path']
        declared = PureWindowsPath(name) if PureWindowsPath(name).drive else Path(name)
        if name.startswith('/') or declared.is_absolute():
            # Older audits stored absolute paths; content and split identity remain binding.
            matches = declared.name == path.name
        else:
            matches = relative(path.parent, name).resolve() == path
        if not matches or file_hash(path) != record['sha256']:
            raise ValueError('Prepared split path or content SHA256 differs from its completion audit')
        verified[split] = dict(audit_sha256=sha256(payload).hexdigest(), manifest_sha256=audit['manifest_sha256'])
    return verified
def prepare(manifest_path: Path, output: Path | None = None, *, run: bool = False) -> dict:
    manifest_path = Path(manifest_path).resolve()
    if output is not None and output.exists():
        raise ValueError("Output must be a new directory; existing datasets are never overwritten")
    if run and output is None:
        raise ValueError("--run requires --output NEW_DIRECTORY")
    audit, rows, author_paths = build_plan(manifest_path)
    if not run:
        return {**audit, "run": False, "files_written": False}
    pa, pq = _pyarrow()
    output = output.resolve()
    if not output.parent.is_dir():
        raise ValueError("Output parent must already exist")
    # Validate Arrow tables before creating output.
    tables = {} if author_paths else {split: pa.Table.from_pylist(values) for split, values in rows.items()}
    output.mkdir(exist_ok=False)
    _write_json(output / "preparation_started.json", {"manifest_sha256": audit["manifest_sha256"], "complete": False})
    outputs = {}
    try:
        for source in audit["sources"]:
            if file_hash(manifest_path.parent/source['path']) != source["sha256"]:
                raise ValueError("Source changed before materialization")
        for split in rows:
            target = output / f"{split}.parquet"
            staging = output / f".{split}.parquet.incomplete"
            if author_paths:
                shutil.copyfile(author_paths[split], staging)
                expected = next(source["sha256"] for source in audit["sources"] if source["split"] == split)
                if file_hash(staging) != expected or file_hash(author_paths[split]) != expected:
                    raise ValueError("Author source copy checksum mismatch")
            else:
                pq.write_table(tables[split], staging, compression="zstd")
            staging.rename(target)
            outputs[split] = {"path": target.name, "sha256": file_hash(target), "rows": len(rows[split])}
        if file_hash(manifest_path) != audit['manifest_sha256'] or any(
                file_hash(manifest_path.parent/source['path']) != source['sha256'] for source in audit['sources']):
            raise ValueError('Manifest or source changed before conversion completed')
        result = {**audit, "run": True, "files_written": True, "complete": True, "outputs": outputs}
        _write_json(output / "dataset_audit.json", result)
        return result
    except Exception:
        # Preserve incomplete output; require its final audit before launch.
        raise
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--manifest", type=Path)
    choice.add_argument("--inspect-source", type=Path)
    parser.add_argument("--input-format", choices=("json", "jsonl", "parquet"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.inspect_source:
        if args.output or args.run or args.input_format is None:
            parser.error("--inspect-source requires --input-format and refuses --run/--output")
        report = inspect_source(args.inspect_source, args.input_format)
    else:
        if args.input_format is not None:
            parser.error("Input formats belong to each source entry in --manifest mode")
        report = prepare(args.manifest, args.output, run=args.run)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
