# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"Offline three-arm evaluation using the checked-out author's judge implementation."
from __future__ import annotations
import argparse
import ast
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import warnings
from report import paired_summary
from assets import CATALOGUE, verify
from convert import file_hash, pointer, question_hash, read_rows
FORMAT = "searl_judge_manifest_v1"
STUDY_FORMAT = 'evocond_judge_manifest_v2'
ARMS = ("searl", "evocond", "teacher")
HELPERS = ("remove_boxed", "last_boxed_only_string", "evaluation_metrics_HLE")
AUTHOR_FILES = ("evaluation/judge.py", "evaluation/evaluation.py",
                "evaluation/batch.py", "verl/trainer/ppo/trainer.py")
def digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             allow_nan=False, separators=(",", ":")).encode()).hexdigest()
def text_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()
def _json_file(path: Path):
    # Export arrays may exceed the bounded JSON proposal parser's size limit.
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=pairs,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
def _source(base: Path, spec: dict) -> Path:
    if not isinstance(spec, dict) or not {"path", "sha256"} <= set(spec):
        raise ValueError("Every input needs path and sha256")
    if not isinstance(spec["path"], str) or not isinstance(spec["sha256"], str):
        raise ValueError("Input path/hash must be text")
    path = (base / spec["path"]).resolve()
    if not path.is_file() or file_hash(path) != spec["sha256"]:
        raise ValueError(f"Missing input or SHA256 mismatch: {path}")
    return path
def model_identity(model: Path, *, full_weights: bool = False) -> dict:
    'Check the exact pinned Qwen3-32B download, not a directory-name claim.'
    model = model.resolve()
    spec = next(item for item in CATALOGUE if item["key"] == "judge")
    files = []
    known = {item["name"] for item in spec["files"]}
    alternative = [path.name for path in model.glob("*") if path.is_file()
                   and path.suffix in (".safetensors", ".bin", ".gguf") and path.name not in known]
    if alternative:
        raise ValueError("Unexpected alternative model weights; pin an explicit supported model snapshot")
    for entry in spec["files"]:
        path = model / entry["name"]
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError(f"Missing or incomplete pinned judge file: {entry['name']}")
        weight = path.suffix == ".safetensors"
        verified = not weight or full_weights
        if verified and not verify(path, entry):
            raise ValueError(f"Pinned judge checksum mismatch: {entry['name']}")
        files.append({"name": entry["name"], "size": entry["size"], "hash_type": entry["hash"],
                      "expected_checksum": entry["checksum"], "checksum_verified": verified})
    config = _json_file(model / "config.json")
    if config.get("model_type") != "qwen3" or "Qwen3ForCausalLM" not in config.get("architectures", []):
        raise ValueError("Judge config is not the pinned Qwen3 causal model")
    return {"path": str(model), "repo": spec["repo"], "revision": spec["revision"],
            "config_sha256": file_hash(model / "config.json"),
            "architecture": config.get("architectures"), "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"), "files": files,
            "all_weight_checksums_verified": full_weights,
            "note": "Dry-run checks weight presence/sizes only; --run verifies all pinned weight checksums before model loading"}
def author_helpers(root: Path) -> dict:
    "Load the author's exact stdlib-only metric definitions for CPU preflight."
    path = root / "evaluation/evaluation.py"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {node.name: node for node in parsed.body if isinstance(node, ast.FunctionDef)}
    missing = set(HELPERS) - functions.keys()
    if missing:
        raise ValueError(f"Author evaluation API missing functions: {sorted(missing)}")
    module = ast.Module(body=[functions[name] for name in HELPERS], type_ignores=[])
    namespace = {"re": re, "MathLLM": object}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace
class StrictAuthorJudge:
    'Reuse upstream prompt and generation, but fail on broken judging.'
    def __init__(self, backend):
        self.backend = backend
        self.calls = []
    def evaluation_batch(self, items):
        result = []
        batch_size = self.backend.batch_size
        for begin in range(0, len(items), batch_size):
            chunk = items[begin:begin + batch_size]
            messages = [self.backend._build_prompt(prediction, reference, question)
                        for prediction, reference, question in chunk]
            responses = self.backend._call_llm_batch(messages)
            if not isinstance(responses, list) or len(responses) != len(chunk):
                raise RuntimeError("Judge output count does not match input count")
            for item, prompt, raw in zip(chunk, messages, responses):
                if not isinstance(raw, str) or raw.strip() not in ("0", "1"):
                    raise RuntimeError("Judge emitted a non-binary response; evaluation aborted, not scored as zero")
                score = float(raw.strip())
                self.calls.append({"question_sha256": question_hash(item[2]), "prompt_sha256": digest(prompt),
                                   "prediction_sha256": text_hash(item[0]), "reference_sha256": digest(item[1]),
                                   "raw_judge_response": raw, "score": score})
                result.append(score)
        return result
def individual_scores(rows, calls, metric):
    'Replay the exact author parser with recorded scores, never re-call Qwen.'
    class Replay:
        def __init__(self):
            self.position = 0
        def evaluation_batch(self, items):
            values = []
            for prediction, reference, question in items:
                if self.position >= len(calls):
                    raise ValueError("Missing recorded judge response")
                record = calls[self.position]
                if (record["question_sha256"] != question_hash(question) or
                        record["prediction_sha256"] != text_hash(prediction) or
                        record["reference_sha256"] != digest(reference)):
                    raise ValueError("Recorded judge responses are not in parser order")
                if record["score"] not in (0., 1.):
                    raise ValueError("Invalid recorded judge score")
                values.append(record["score"])
                self.position += 1
            return values
    replay, counts, output = Replay(), {}, []
    for row in rows:
        values = metric([row], replay)
        if sum(values[:4]) != 1 or values[2] not in (0, 1):
            raise ValueError("Author parser did not score exactly one output")
        identity = question_hash(row["question"])
        index = counts.get(identity, 0)
        counts[identity] = index+1
        output.append({"question_sha256": identity, "sample_index": index,
                       "output_sha256": text_hash(row["output"]), "success": bool(values[2]),
                       "no_answer": values[0], "boxed_not_found": values[1]})
    if replay.position != len(calls):
        raise ValueError("Unused recorded judge responses")
    return output
def load_author_judge(root: Path, model: Path, *, batch_size: int, max_new_tokens: int,
                      tensor_parallel: int, gpu_memory_utilization: float):
    os.environ["VLLM_TP_SIZE"] = str(tensor_parallel)
    os.environ["VLLM_GPU_MEM_UTIL"] = str(gpu_memory_utilization)
    path = root / "evaluation/judge.py"
    spec = importlib.util.spec_from_file_location("_evocond_pinned_author_judge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return StrictAuthorJudge(module.VLLMJudge(model_path=str(model), temperature=0.0,
                                            max_new_tokens=max_new_tokens, batch_size=batch_size))
def _exports(path: Path, kind: str, *, allow_empty=False) -> list:
    if kind == "jsonl":
        rows = list(read_rows(path, "jsonl"))
    elif kind == "json":
        rows = _json_file(path)
    else:
        raise ValueError("Prediction export format must be jsonl or json")
    if not isinstance(rows, list) or not rows and not allow_empty:
        raise ValueError("Prediction export must be a nonempty row collection")
    for number, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("output"), str):
            raise ValueError(f"Prediction row {number} requires an explicit string output field")
        if "input" not in row and "question" not in row:
            raise ValueError(f"Prediction row {number} is missing input/question for alignment")
    return rows
def arm_names(manifest):
    if not isinstance(manifest, dict):
        raise ValueError('Judge manifest must be an object')
    if manifest.get('format') == FORMAT:
        if not isinstance(manifest.get('arms'), dict) or set(manifest['arms']) != set(ARMS):
            raise ValueError('Legacy judge requires all three arms')
        return ARMS
    arms = manifest.get('arms')
    if manifest.get('format') != STUDY_FORMAT or not isinstance(arms, dict) or not 2 <= len(arms) <= 32:
        raise ValueError('Study judge requires two to thirty-two named conditions')
    if any(not isinstance(name, str) or not re.fullmatch('[a-z][a-z0-9_-]{0,40}', name) for name in arms):
        raise ValueError('Invalid study condition label')
    if type(manifest.get('samples_per_question')) is not int or manifest['samples_per_question'] != 1:
        raise ValueError('Study pass@1 requires one scheduled prediction per base question')
    from study import SEEDS
    if type(manifest.get('training_seed')) is not int or manifest['training_seed'] not in SEEDS:
        raise ValueError('Study scoring must declare its actual training seed')
    return tuple(sorted(arms))

def preflight(manifest_path: Path, official_root: Path, model: Path, *, full_weights=False) -> tuple[dict, dict]:
    manifest_path, official_root = manifest_path.resolve(), official_root.resolve()
    manifest_digest = file_hash(manifest_path)
    manifest = _json_file(manifest_path)
    selected_arms = arm_names(manifest)
    fields = {"format", "split", "dataset_name", "metric", "dataset", "arms", "samples_per_question"}
    if manifest['format'] == STUDY_FORMAT:
        fields.add('training_seed')
    if (not isinstance(manifest, dict) or set(manifest) != fields or manifest["format"] not in (FORMAT, STUDY_FORMAT)
            or manifest["split"] not in ("validation", "test") or manifest["metric"] != "author_overall"
            or not isinstance(manifest["dataset_name"], str) or not manifest["dataset_name"].strip()
            or not isinstance(manifest["arms"], dict)
            or type(manifest["samples_per_question"]) is not int or manifest["samples_per_question"] < 1):
        raise ValueError("Malformed judge manifest: explicit split, dataset, all three arms and sample count required")
    metric_name = re.sub(r"[\s_-]+", "", manifest["dataset_name"].casefold())
    if metric_name in ("hle", "gaia", "webwalker"):
        raise ValueError("These datasets use grouped/special upstream metrics; author_overall is not equivalent")
    hashes = {}
    for relative in AUTHOR_FILES:
        path = official_root / relative
        if not path.is_file():
            raise ValueError(f"Missing author source: {relative}")
        hashes[relative] = file_hash(path)
    author_helpers(official_root)
    identity = model_identity(model, full_weights=full_weights)
    dataset = manifest["dataset"]
    if (not isinstance(dataset, dict) or set(dataset) != {"path", "sha256", "format", "question_pointer", "answer_pointer"}
            or dataset["format"] not in ("jsonl", "parquet")):
        raise ValueError("Dataset requires format and explicit question_pointer/answer_pointer")
    dataset_path = _source(manifest_path.parent, dataset)
    references = {}
    for row in read_rows(dataset_path, dataset["format"]):
        question, answer = pointer(row, dataset["question_pointer"]), pointer(row, dataset["answer_pointer"])
        key = question_hash(question)
        valid_answer = isinstance(answer, str) and bool(answer.strip())
        valid_answer |= isinstance(answer, list) and bool(answer) and all(isinstance(item, str) and item.strip() for item in answer)
        if not valid_answer:
            raise ValueError("Reference must be explicit text or a nonempty text list")
        if key in references:
            raise ValueError("Reference dataset contains duplicate normalized questions")
        references[key] = (question, answer)
    if not references:
        raise ValueError("Reference dataset is empty")
    inputs, prepared = {}, {}
    for arm in selected_arms:
        spec = manifest["arms"][arm]
        if not isinstance(spec, dict) or set(spec) - {"path", "sha256", "format", "alignment"} or "format" not in spec:
            raise ValueError("Each arm needs path/hash/format and optional explicit alignment")
        path = _source(manifest_path.parent, spec)
        rows = _exports(path, spec["format"], allow_empty=manifest['format'] == STUDY_FORMAT)
        mapping = {}
        if "alignment" in spec:
            alignment = spec["alignment"]
            if not isinstance(alignment, dict) or set(alignment) != {"path", "sha256"}:
                raise ValueError("Alignment needs only path and sha256")
            records = _json_file(_source(manifest_path.parent, alignment))
            if not isinstance(records, list):
                raise ValueError("Alignment must be a list of row/input hash/question hash records")
            for item in records:
                if (not isinstance(item, dict) or set(item) != {"row", "input_sha256", "question_sha256"}
                        or type(item["row"]) is not int or not 0 <= item["row"] < len(rows)
                        or item["row"] in mapping
                        or any(not isinstance(item[key], str) or not re.fullmatch(r"[a-f0-9]{64}", item[key])
                               for key in ("input_sha256", "question_sha256"))):
                    raise ValueError("Malformed or duplicate alignment row")
                mapping[item["row"]] = item
            if len(mapping) != len(rows):
                raise ValueError("Explicit alignment must cover every export row")
        counts = {key: 0 for key in references}
        prepared[arm] = []
        for number, row in enumerate(rows):
            source_input = row.get("input", row.get("question"))
            if not isinstance(source_input, str):
                raise ValueError(f"{arm} row {number} input/question must be visible text")
            assigned_keys = []
            if mapping:
                assigned = mapping[number]
                if assigned["input_sha256"] != text_hash(source_input):
                    raise ValueError("Alignment input hash mismatch")
                assigned_keys.append(assigned["question_sha256"])
            if "question" in row:
                assigned_keys.append(question_hash(row["question"]))
            if "question_sha256" in row:
                value = row["question_sha256"]
                if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
                    raise ValueError("Export question_sha256 must be a normalized-question SHA256")
                assigned_keys.append(value)
            if len(set(assigned_keys)) > 1:
                raise ValueError("Export question identity conflicts with its hash/alignment metadata")
            key = assigned_keys[0] if assigned_keys else question_hash(source_input)
            if key not in references:
                raise ValueError(f"Unaligned {arm} row {number}: raw trainer input may include templates. "
                                 "Supply explicit question or alignment records {row,input_sha256,question_sha256}; "
                                 "the wrapper will not guess the first line or silently drop unmatched rows.")
            counts[key] += 1
            question, answer = references[key]
            prepared[arm].append({"question": question, "ground_truth": answer, "output": row["output"]})
        missing = [key for key, count in counts.items() if count == 0]
        if manifest['format'] == STUDY_FORMAT:
            for key in missing:
                question, answer = references[key]
                prepared[arm].append(dict(question=question, ground_truth=answer, output=''))
                counts[key] = 1
        if any(count != manifest["samples_per_question"] for count in counts.values()):
            raise ValueError(f"{arm} does not have the declared sample count for every reference question")
        inputs[arm] = {"path": str(path), "sha256": spec["sha256"], "format": spec["format"],
                       "alignment": spec.get("alignment"), "rows": len(rows), 'missing_scored_zero': len(missing)}
    # A long weight check or concurrent export must not change scored inputs.
    if file_hash(manifest_path) != manifest_digest:
        raise ValueError("Judge manifest changed during preflight")
    _source(manifest_path.parent, dataset)
    for spec in manifest["arms"].values():
        _source(manifest_path.parent, spec)
        if "alignment" in spec:
            _source(manifest_path.parent, spec["alignment"])
    for relative, expected in hashes.items():
        if file_hash(official_root / relative) != expected:
            raise ValueError("Author source changed during preflight")
    protocol = {"format": "searl_judge_protocol_v1", "manifest_sha256": manifest_digest,
                'training_seed': manifest.get('training_seed'),
                "split": manifest["split"], "dataset_name": manifest["dataset_name"], "metric": "author_overall",
                "samples_per_question": manifest["samples_per_question"], "reference_count": len(references),
                "dataset": {**dataset, "path": str(dataset_path)}, "arms": inputs,
                "judge_model": identity, "author_source_sha256": hashes,
                "parser": "author evaluation_metrics_HLE: last assistant marker -> last <answer> -> boxed answer",
                "deviations": ["explicit exact alignment; never silent dropping or first-user-line guessing",
                               "judge exceptions/invalid outputs raise instead of defaulting to zero",
                               "non-binary responses rejected instead of extracting an isolated 0/1 digit"],
                "paper_exact_reproduction_verified": False}
    return protocol, prepared
def _write(path: Path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
def run(manifest_path: Path, official_root: Path, model: Path, output: Path | None = None, *, execute=False,
        batch_size=64, max_new_tokens=8, tensor_parallel=1, gpu_memory_utilization=0.85) -> dict:
    if output is not None and output.exists():
        raise ValueError("Judge output must be a new directory; no overwrite or result reuse")
    if execute and output is None:
        raise ValueError("--run requires --output NEW_DIRECTORY")
    if (type(batch_size) is not int or batch_size < 1 or type(max_new_tokens) is not int or max_new_tokens < 1
            or type(tensor_parallel) is not int or tensor_parallel < 1
            or not isinstance(gpu_memory_utilization, (float, int)) or not 0 < gpu_memory_utilization < 1):
        raise ValueError("Invalid shared judge generation settings")
    protocol, prepared = preflight(manifest_path, official_root, model, full_weights=execute)
    protocol["generation"] = {"temperature": 0.0, "max_new_tokens": max_new_tokens, "batch_size": batch_size,
                              "tensor_parallel": tensor_parallel, "gpu_memory_utilization": gpu_memory_utilization,
                              "chat_template": "author: enable_thinking=False", "stop": ["\n"]}
    protocol["wrapper_sha256"] = file_hash(Path(__file__).resolve())
    if not execute:
        return {"run": False, "model_loaded": False, "protocol": protocol}
    if not output.parent.is_dir():
        raise ValueError("Judge output parent must exist")
    output.mkdir(exist_ok=False)
    _write(output / "judge_protocol.json", protocol)
    judge = load_author_judge(official_root, model, batch_size=batch_size, max_new_tokens=max_new_tokens,
                              tensor_parallel=tensor_parallel, gpu_memory_utilization=gpu_memory_utilization)
    metrics = author_helpers(official_root)["evaluation_metrics_HLE"]
    result = {"run": True, "complete": False, "protocol_sha256": digest(protocol), "arms": {}}
    scored = {}
    for arm in protocol['arms']:
        before = len(judge.calls)
        values = metrics(prepared[arm], judge)
        row = dict(zip(("no_answer", "boxed_not_found", "correct", "false_answer", "acc"), values))
        total = sum(row[key] for key in ("no_answer", "boxed_not_found", "correct", "false_answer"))
        if total != len(prepared[arm]):
            raise RuntimeError("Author metric denominator does not cover all aligned outputs")
        row["n"] = total
        row["judge_calls"] = len(judge.calls) - before
        result["arms"][arm] = row
        _write(output / f"{arm}_judge_calls.json", judge.calls[before:])
        scored[arm] = individual_scores(prepared[arm], judge.calls[before:], metrics)
        if sum(item["success"] for item in scored[arm]) != row["correct"]:
            raise RuntimeError("Per-question replay disagrees with author aggregate")
        _write(output / f"{arm}_scores.json", scored[arm])
    result["paired"] = paired_summary(scored)
    result["complete"] = True
    _write(output / "judge_report.json", result)
    return result
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.manifest, args.official_root, args.model, args.output, execute=args.run,
                         batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                         tensor_parallel=args.tensor_parallel, gpu_memory_utilization=args.gpu_memory_utilization),
                     ensure_ascii=False, indent=2, allow_nan=False))
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
