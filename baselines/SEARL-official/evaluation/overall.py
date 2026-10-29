# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from evaluation.evaluation import (
    prepare_answer_file,
    evaluate_dataset,
    _build_dataset_map,
    _norm_text,
    main_HLE,
    main_gaia,
    main_webwalker,
    main_xbench,
    main_qa,
    process_item,
    process_item_gaia,
    process_item_webwalker,
    process_item_xbench,
    process_item_qa,
    evaluation_metrics_HLE_key,
    evaluation_metrics_gaia_level,
    evaluation_metrics_webwalker_level,
    evaluation_metrics_HLE,
)
from plot import *
import json
import os
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

class DatasetEvaluator:
    'Load reasoning datasets, align answers and dispatch dataset-specific scoring.'
    def __init__(self, input_json_path: str, output_dir: str):
        'Initialize the evaluator output directory.'
        self.input_json_path = input_json_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Evaluation results will be saved to: {self.output_dir}")

        # self.math_llm = MathLLM()

    def _load_json(self, file_path: str) -> list:
        'Read JSON from a file.'
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"ERROR: File not found: {file_path}")
            return []

    def prepare_answer_json_path(self, input_data, dataset_name):
        answer_json = self._load_json(self.input_json_path)
        if dataset_name == "HLE":
            output_path = main_HLE(data=input_data, answer_json=answer_json, output_dir=self.output_dir)
        elif dataset_name == "GAIA":
            output_path = main_gaia(data=input_data, answer_json=answer_json, output_dir=self.output_dir)
        elif dataset_name == "WebWalker":
            output_path = main_webwalker(data=input_data, answer_json=answer_json, output_dir=self.output_dir)
        elif dataset_name == "XBench":
            output_path = main_xbench(data=input_data, answer_json=answer_json, output_dir=self.output_dir)
        else:
            output_path = main_qa(data=input_data, answer_json=answer_json, output_dir=self.output_dir, dataset_name=dataset_name)
        return output_path

    def _filter_and_process(self, dataset_name: str, raw_item: dict, processor, idx: int):
        'Apply the same filtering rules as match_stats.'
        if dataset_name == "HLE":
            if raw_item.get("image") or raw_item.get("category") not in ["Physics", "Math", "Computer Science/AI"]:
                return None
        if dataset_name == "GAIA":
            if raw_item.get("file_name") != "":
                return None
        return processor(raw_item, idx)

    def _load_split_safe(self, dataset_name: str, cfg: dict):
        'Load the requested split, falling back to train when absent.'
        split = cfg["split"]
        if dataset_name in ["HLE", "GAIA", "XBench"]:
            ds = load_dataset(cfg["loader"])
        elif dataset_name == "WebWalker":
            ds = load_dataset("json", data_files=cfg["loader"])
        else:
            ds = load_dataset("json", data_files=cfg["loader"])
        if split in ds:
            return ds[split]

        if "train" in ds:
            return ds["train"]
        first_split = list(ds.keys())[0]
        return ds[first_split]

    def prepare_data(self, dataset_name: str) -> list:
        'Align a named dataset with prediction JSON and return answers with references and metadata.'
        dataset_map = _build_dataset_map()
        if dataset_name not in dataset_map:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Supported options: {list(dataset_map.keys())}")
        cfg = dataset_map[dataset_name]

        input_data = self._load_split_safe(dataset_name, cfg)

        answer_json_path = self.prepare_answer_json_path(input_data, dataset_name)
        print(f"--- Preparing dataset \'{dataset_name}\' ---, {answer_json_path}")
        answer_json = self._load_json(answer_json_path)
        if not answer_json:
            print(f"--- Dataset \'{dataset_name}\' preparation failed ---")
            return []

        config = cfg
        source_data = self._load_split_safe(dataset_name, config)
        processor = config["processor"]

        processed = []
        for idx, item in enumerate(tqdm(source_data, desc=f"Reading {dataset_name} reference answers")):
            processed_item = self._filter_and_process(dataset_name, item, processor, idx)
            if processed_item:
                processed.append(processed_item)

        matched_data = []
        for item in tqdm(answer_json, desc='Aligning questions and answers'):
            inputs_raw = item.get('input', item.get('question', ''))
            inputs_norm = _norm_text(inputs_raw)
            for tup in processed:
                tgt_norm = _norm_text(tup[0])
                if not tgt_norm:
                    continue

                if tgt_norm in inputs_norm or inputs_norm in tgt_norm:
                    item['ground_truth'] = tup[1]
                    item['question'] = tup[0]

                    if dataset_name == 'HLE':
                        item['category'] = tup[2]
                    elif dataset_name == 'GAIA':
                        item['Level'] = tup[2]
                    elif dataset_name == 'WebWalker':
                        item['difficulty_level'] = tup[2]

                    matched_data.append(item)
                    break

        print(f"Matched {len(matched_data)} / {len(answer_json)} records.")
        return matched_data

    def split_log_into_chunks(self, log_file_path, output_dir=None, start_pattern='actor/kl_loss:'):
        'Split a log into files at start_pattern markers under output_dir.'

        if output_dir:
            output_dir = os.path.join(output_dir, "chunks")
        else:
            output_dir = os.path.join(self.output_dir, "chunks")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        chunk_count = 0
        current_chunk_file = None

        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if start_pattern in line.strip():

                        if current_chunk_file:
                            current_chunk_file.close()

                        chunk_count += 1
                        chunk_filename = os.path.join(output_dir, f'chunk_{chunk_count}.log')
                        current_chunk_file = open(chunk_filename, 'w', encoding='utf-8')
                        print(f"Writing a new chunk to: {chunk_filename}")

                    if current_chunk_file:
                        current_chunk_file.write(line)

            if current_chunk_file:
                current_chunk_file.close()

            print(f"\nCompleted; split into {chunk_count} files.")

        except FileNotFoundError:
            print(f"ERROR: File \'{log_file_path}\' not found.")
        except Exception as e:
            print(f"Error: {e}")

    def run_evaluation(self, dataset_name: str, judge: MathLLM | None = None):
        'Run the full evaluation with an optional shared judge and append results to one JSON file.'

        eval_data = self.prepare_data(dataset_name)
        print(len(eval_data), "######")
        if not eval_data:
            print('Data preparation failed; evaluation stopped.')
            return

        print(f"--- Evaluating dataset \'{dataset_name}' ---")

        if judge is None:
            from evaluation.judge import MathLLM
            judge = MathLLM()
            print('WARNING: No judge supplied; created a new instance. Supply one to reuse the model.')

        results_for_current_dataset = defaultdict(dict)
        metric_keys = ["no_answer", "boxed_not_found", "correct", "false_answer", "acc"]

        if dataset_name == 'HLE':
            for category in ["Math", "Computer Science/AI", "Physics"]:
                print(f"\nEvaluating HLE - {category} categories:")
                metrics = evaluation_metrics_HLE_key(eval_data, category=category, math_llm=judge)
                results_for_current_dataset[category] = dict(zip(metric_keys, metrics))

        elif dataset_name == 'GAIA':
            for level in [1, 2, 3]:
                print(f"\nEvaluating GAIA - Level {level}:")
                metrics = evaluation_metrics_gaia_level(eval_data, level=level, math_llm=judge)
                results_for_current_dataset[f'Level {level}'] = dict(zip(metric_keys, metrics))

        elif dataset_name == 'WebWalker':
            levels = sorted(list(set(item['difficulty_level'] for item in eval_data)))
            for level in levels:
                print(f"\nEvaluating WebWalker - Difficulty Level {level}:")
                metrics = evaluation_metrics_webwalker_level(eval_data, level=level, math_llm=judge)
                results_for_current_dataset[f'Difficulty Level {level}'] = dict(zip(metric_keys, metrics))

        elif dataset_name in ["XBench", "2wiki", "bamboogle", "hotpotqa", "musique"]:
            print(f"\nEvaluating {dataset_name}:")
            metrics = evaluation_metrics_HLE(eval_data, math_llm=judge)
            results_for_current_dataset['overall'] = dict(zip(metric_keys, metrics))

        elif dataset_name in ["gsm8k", "math500", "aime24", "aime25"]:
            print(f"\nEvaluating math - {dataset_name}:")
            metrics = evaluation_metrics_HLE(eval_data, math_llm=judge)
            results_for_current_dataset['overall'] = dict(zip(metric_keys, metrics))

        else:
            print(f"'{dataset_name}\' has no evaluation implementation; stopping.")
            return

        answer_path = Path(self.input_json_path)
        sample_key = answer_path.stem

        if answer_path.parent.name == "validation_samples":
            run_name = answer_path.parent.parent.name
        else:
            run_name = answer_path.parent.name

        results_root = Path(os.environ.get("RL_FACTORY_EVAL_RESULTS_DIR", "evaluation/results"))
        run_dir = results_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        results_json_path = run_dir / "evaluation.json"

        all_results = {}
        if results_json_path.exists() and results_json_path.stat().st_size > 0:
            try:
                with open(results_json_path, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Failed to parse existing results; overwriting: {e}")
                all_results = {}

        sample_bucket = all_results.setdefault(sample_key, {})
        sample_bucket[dataset_name] = dict(results_for_current_dataset)

        try:
            with open(results_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
            print(f"Wrote dataset \'{dataset_name}\' results to {results_json_path} under key \'{sample_key}'")
        except Exception as e:
            print(f"Failed to save results: {e}")

        print(f"--- Dataset \'{dataset_name}\' evaluated ---")

def _load_split_safe_global(dataset_name: str, cfg: dict):
    'Load a dataset split safely for standalone CLI preparation.'
    split = cfg["split"]
    if dataset_name in ["HLE", "GAIA", "XBench"]:
        ds = load_dataset(cfg["loader"])
    elif dataset_name == "WebWalker":
        ds = load_dataset("json", data_files=cfg["loader"])
    else:
        ds = load_dataset("json", data_files=cfg["loader"])
    if split in ds:
        return ds[split]
    if "train" in ds:
        return ds["train"]
    first_split = list(ds.keys())[0]
    return ds[first_split]

def _prepare_answer_file_safe(dataset_name: str, answer_json_path: str, output_dir: str, dataset_root_override: str | None = None):
    'Prepare aligned answers with the legacy split fallback.'

    if output_dir is None:
        answer_path = Path(answer_json_path)
        output_dir = str(answer_path.parent / "prepared_answers")
        print(f"[INFO] Using the default output_dir: {output_dir}")
    dataset_map = _build_dataset_map()
    if dataset_name not in dataset_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    cfg = dataset_map[dataset_name].copy()
    cfg["loader"] = dataset_root_override or cfg["loader"]
    input_data = _load_split_safe_global(dataset_name, cfg)

    with open(answer_json_path, "r", encoding="utf-8") as f:
        answer_json = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    prepare_fn = cfg["prepare_fn"]
    if prepare_fn == main_qa:
        return prepare_fn(input_data, answer_json, output_dir, dataset_name)
    return prepare_fn(input_data, answer_json, output_dir)

def cli():
    'Align answers and evaluate them under the selected output directory.'
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help='Dataset name, e.g. HLE/GAIA/WebWalker/XBench/...')
    parser.add_argument("--answers", required=True, help='Prediction JSON before reference alignment')
    parser.add_argument("--output-dir", required=False, help='Output directory for aligned answers and evaluation reports')
    parser.add_argument("--model-path", default=None, help='vLLM weights: Hugging Face model ID or local path')
    parser.add_argument("--dataset-path", default=None, help='Optional dataset path override')
    args = parser.parse_args()

    if args.output_dir is None:
        answer_path = Path(args.answers)
        args.output_dir = str(answer_path.parent / "prepared_answers")
        print(f"[INFO] Using the default output_dir: {args.output_dir}")

    prepared = _prepare_answer_file_safe(
        dataset_name=args.dataset,
        answer_json_path=args.answers,
        output_dir=args.output_dir,
        dataset_root_override=args.dataset_path,
    )
    evaluate_dataset(
        dataset_name=args.dataset,
        prepared_answer_path=prepared,
        model_path=args.model_path,
        output_dir=args.output_dir,
    )

if __name__ == '__main__':
    cli()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
