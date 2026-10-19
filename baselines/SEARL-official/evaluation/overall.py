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
    '\n    \u4e00\u4e2a\u7528\u4e8e\u8bc4\u4f30\u4e0d\u540c\u6570\u5b66\u548c\u903b\u8f91\u63a8\u7406\u6570\u636e\u96c6\u7684\u7c7b\u3002\n    \u5b83\u5c01\u88c5\u4e86\u6570\u636e\u52a0\u8f7d\u3001\u7b54\u6848\u5339\u914d\u548c\u8c03\u7528\u7279\u5b9a\u8bc4\u4f30\u51fd\u6570\u7684\u8fc7\u7a0b\u3002\n    '
    def __init__(self, input_json_path: str, output_dir: str):
        '\n        \u521d\u59cb\u5316\u8bc4\u4f30\u5668\u3002\n\n        Args:\n            output_dir (str): \u7528\u4e8e\u4fdd\u5b58\u8bc4\u4f30\u7ed3\u679c\u7684\u6587\u4ef6\u5939\u8def\u5f84\u3002\n        '
        self.input_json_path = input_json_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"\u8bc4\u4f30\u7ed3\u679c\u5c06\u4fdd\u5b58\u5230: {self.output_dir}")

        # self.math_llm = MathLLM()

    def _load_json(self, file_path: str) -> list:
        '\u4ece\u6587\u4ef6\u8def\u5f84\u52a0\u8f7d JSON \u6570\u636e\u3002'
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[!] \u9519\u8bef: \u627e\u4e0d\u5230\u6587\u4ef6 {file_path}")
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
        '\u4e0e match_stats \u4fdd\u6301\u4e00\u81f4\u7684\u8fc7\u6ee4\u903b\u8f91\u3002'
        if dataset_name == "HLE":
            if raw_item.get("image") or raw_item.get("category") not in ["Physics", "Math", "Computer Science/AI"]:
                return None
        if dataset_name == "GAIA":
            if raw_item.get("file_name") != "":
                return None
        return processor(raw_item, idx)

    def _load_split_safe(self, dataset_name: str, cfg: dict):
        '\u5b89\u5168\u52a0\u8f7d\u6570\u636e\u96c6\uff1a\u82e5\u6307\u5b9a split \u4e0d\u5b58\u5728\u5219\u56de\u9000\u5230 train\u3002'
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
        "\n        \u51c6\u5907\u7528\u4e8e\u8bc4\u4f30\u7684\u6570\u636e\u3002\n        \u52a0\u8f7d\u539f\u59cb\u6570\u636e\u96c6\u548c\u6a21\u578b\u7b54\u6848\uff0c\u5e76\u5c06\u5b83\u4eec\u5339\u914d\u8d77\u6765\u3002\n\n        Args:\n            dataset_name (str): \u6570\u636e\u96c6\u540d\u79f0 (\u4f8b\u5982 'HLE', 'GAIA', 'WebWalker', 'XBench').\n            answer_json_path (str): \u6a21\u578b\u751f\u6210\u7684\u7b54\u6848\u6587\u4ef6\u8def\u5f84\u3002\n\n        Returns:\n            list: \u5305\u542b\u4e86\u6807\u51c6\u7b54\u6848\u548c\u5143\u6570\u636e\u7684\u6a21\u578b\u7b54\u6848\u5217\u8868\u3002\n        "
        dataset_map = _build_dataset_map()
        if dataset_name not in dataset_map:
            raise ValueError(f"\u4e0d\u652f\u6301\u7684\u6570\u636e\u96c6: {dataset_name}\u3002\u652f\u6301\u7684\u9009\u9879\u4e3a: {list(dataset_map.keys())}")
        cfg = dataset_map[dataset_name]

        input_data = self._load_split_safe(dataset_name, cfg)

        answer_json_path = self.prepare_answer_json_path(input_data, dataset_name)
        print(f"--- \u6b63\u5728\u4e3a\u6570\u636e\u96c6 '{dataset_name}' \u51c6\u5907\u6570\u636e ---, {answer_json_path}")
        answer_json = self._load_json(answer_json_path)
        if not answer_json:
            print(f"--- \u6570\u636e\u96c6 '{dataset_name}' \u51c6\u5907\u6570\u636e\u5931\u8d25 ---")
            return []

        config = cfg
        source_data = self._load_split_safe(dataset_name, config)
        processor = config["processor"]

        processed = []
        for idx, item in enumerate(tqdm(source_data, desc=f"\u4ece {dataset_name} \u63d0\u53d6\u6807\u51c6\u7b54\u6848")):
            processed_item = self._filter_and_process(dataset_name, item, processor, idx)
            if processed_item:
                processed.append(processed_item)

        matched_data = []
        for item in tqdm(answer_json, desc='\u5339\u914d\u95ee\u9898\u4e0e\u7b54\u6848'):
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

        print(f"\u6210\u529f\u5339\u914d {len(matched_data)} / {len(answer_json)} \u6761\u6570\u636e\u3002")
        return matched_data

    def split_log_into_chunks(self, log_file_path, output_dir=None, start_pattern='actor/kl_loss:'):
        '\n        \u6839\u636e\u8d77\u59cb\u6a21\u5f0f\u5c06\u4e00\u4e2a\u5927\u65e5\u5fd7\u6587\u4ef6\u5206\u5272\u6210\u591a\u4e2a\u5c0f\u6587\u4ef6\u3002\n\n        :param log_file_path: \u8f93\u5165\u7684\u65e5\u5fd7\u6587\u4ef6\u8def\u5f84\u3002\n        :param start_pattern: \u6807\u5fd7\u7740\u4e00\u4e2a\u65b0 chunk \u5f00\u59cb\u7684\u5b57\u7b26\u4e32\u3002\n        :param output_dir: \u5b58\u653e\u5207\u5206\u540e\u6587\u4ef6\u7684\u76ee\u5f55\u3002\n        '

        if output_dir:
            output_dir = os.path.join(output_dir, "chunks")
        else:
            output_dir = os.path.join(self.output_dir, "chunks")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"\u521b\u5efa\u76ee\u5f55: {output_dir}")

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
                        print(f"\u68c0\u6d4b\u5230\u65b0 chunk\uff0c\u5f00\u59cb\u5199\u5165\u6587\u4ef6: {chunk_filename}")

                    if current_chunk_file:
                        current_chunk_file.write(line)

            if current_chunk_file:
                current_chunk_file.close()

            print(f"\n\u5904\u7406\u5b8c\u6210\uff01\u603b\u5171\u5206\u5272\u51fa {chunk_count} \u4e2a\u6587\u4ef6\u3002")

        except FileNotFoundError:
            print(f"\u9519\u8bef: \u6587\u4ef6 '{log_file_path}' \u672a\u627e\u5230\u3002")
        except Exception as e:
            print(f"\u53d1\u751f\u9519\u8bef: {e}")

    def run_evaluation(self, dataset_name: str, judge: MathLLM | None = None):
        '\n        \u6267\u884c\u5b8c\u6574\u7684\u8bc4\u4f30\u6d41\u7a0b\uff0c\u5e76\u5c06\u7ed3\u679c\u4ee5\u53ef\u7eed\u5199\u7684\u65b9\u5f0f\u4fdd\u5b58\u5230\u5355\u4e2aJSON\u6587\u4ef6\u4e2d\u3002\n\n        Args:\n            dataset_name (str): \u8981\u8bc4\u4f30\u7684\u6570\u636e\u96c6\u540d\u79f0\u3002\n            judge: \u590d\u7528\u7684judge\u5b9e\u4f8b\uff0c\u5982\u679c\u4e3aNone\u5219\u5728\u5185\u90e8\u521b\u5efa\uff08\u907f\u514d\u91cd\u590d\u52a0\u8f7d\u6a21\u578b\uff09\n        '

        eval_data = self.prepare_data(dataset_name)
        print(len(eval_data), "######")
        if not eval_data:
            print('\u6570\u636e\u51c6\u5907\u5931\u8d25\uff0c\u8bc4\u4f30\u7ec8\u6b62\u3002')
            return

        print(f"--- \u5f00\u59cb\u8bc4\u4f30\u6570\u636e\u96c6 '{dataset_name}' ---")

        if judge is None:
            from evaluation.judge import MathLLM
            judge = MathLLM()
            print('\u8b66\u544a: \u672a\u4f20\u5165judge\u5b9e\u4f8b\uff0c\u521b\u5efa\u4e86\u65b0\u7684judge\uff08\u5efa\u8bae\u5916\u90e8\u4f20\u5165\u4ee5\u590d\u7528\u6a21\u578b\uff09')

        results_for_current_dataset = defaultdict(dict)
        metric_keys = ["no_answer", "boxed_not_found", "correct", "false_answer", "acc"]

        if dataset_name == 'HLE':
            for category in ["Math", "Computer Science/AI", "Physics"]:
                print(f"\n\u8bc4\u4f30 HLE - {category} \u5206\u7c7b:")
                metrics = evaluation_metrics_HLE_key(eval_data, category=category, math_llm=judge)
                results_for_current_dataset[category] = dict(zip(metric_keys, metrics))

        elif dataset_name == 'GAIA':
            for level in [1, 2, 3]:
                print(f"\n\u8bc4\u4f30 GAIA - Level {level}:")
                metrics = evaluation_metrics_gaia_level(eval_data, level=level, math_llm=judge)
                results_for_current_dataset[f'Level {level}'] = dict(zip(metric_keys, metrics))

        elif dataset_name == 'WebWalker':
            levels = sorted(list(set(item['difficulty_level'] for item in eval_data)))
            for level in levels:
                print(f"\n\u8bc4\u4f30 WebWalker - Difficulty Level {level}:")
                metrics = evaluation_metrics_webwalker_level(eval_data, level=level, math_llm=judge)
                results_for_current_dataset[f'Difficulty Level {level}'] = dict(zip(metric_keys, metrics))

        elif dataset_name in ["XBench", "2wiki", "bamboogle", "hotpotqa", "musique"]:
            print(f"\n\u8bc4\u4f30 {dataset_name}:")
            metrics = evaluation_metrics_HLE(eval_data, math_llm=judge)
            results_for_current_dataset['overall'] = dict(zip(metric_keys, metrics))

        elif dataset_name in ["gsm8k", "math500", "aime24", "aime25"]:
            print(f"\n\u8bc4\u4f30 math - {dataset_name}:")
            metrics = evaluation_metrics_HLE(eval_data, math_llm=judge)
            results_for_current_dataset['overall'] = dict(zip(metric_keys, metrics))

        else:
            print(f"'{dataset_name}' \u7684\u8bc4\u4f30\u903b\u8f91\u672a\u5b9a\u4e49\uff0c\u8bc4\u4f30\u7ec8\u6b62\u3002")
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
                print(f"\u8bfb\u53d6\u7ed3\u679c\u6587\u4ef6\u65f6\u89e3\u6790\u5931\u8d25\uff0c\u5c06\u8986\u76d6\u5199\u5165: {e}")
                all_results = {}

        sample_bucket = all_results.setdefault(sample_key, {})
        sample_bucket[dataset_name] = dict(results_for_current_dataset)

        try:
            with open(results_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
            print(f"\u5df2\u5c06\u6570\u636e\u96c6 '{dataset_name}' \u7684\u7ed3\u679c\u5199\u5165 {results_json_path}\uff0c\u952e\u4e3a '{sample_key}'")
        except Exception as e:
            print(f"\u4fdd\u5b58\u7ed3\u679c\u65f6\u53d1\u751f\u9519\u8bef: {e}")

        print(f"--- \u6570\u636e\u96c6 '{dataset_name}' \u8bc4\u4f30\u5b8c\u6210 ---")

def _load_split_safe_global(dataset_name: str, cfg: dict):
    '\u72ec\u7acb\u51fd\u6570\u7248\u5b89\u5168\u52a0\u8f7d\uff0c\u4f9b CLI \u76f4\u63a5\u8c03\u7528 prepare_answer_file \u65f6\u4f7f\u7528\u3002'
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
    "\u5e26 split \u56de\u9000\u7684 prepare_answer_file\uff0c\u907f\u514d KeyError('test')\u3002"

    if output_dir is None:
        answer_path = Path(answer_json_path)
        output_dir = str(answer_path.parent / "prepared_answers")
        print(f"[\u63d0\u793a] \u672a\u6307\u5b9a output_dir\uff0c\u4f7f\u7528\u9ed8\u8ba4\u8def\u5f84: {output_dir}")
    dataset_map = _build_dataset_map()
    if dataset_name not in dataset_map:
        raise ValueError(f"\u4e0d\u652f\u6301\u7684\u6570\u636e\u96c6: {dataset_name}")
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
    '\u7b80\u5355 CLI\uff1a\u5bf9\u9f50\u7b54\u6848\u5e76\u8bc4\u4f30\uff0c\u8f93\u51fa\u5230\u6307\u5b9a\u76ee\u5f55\u3002'
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help='\u6570\u636e\u96c6\u540d\u79f0\uff0c\u4f8b\u5982 HLE/GAIA/WebWalker/XBench/...')
    parser.add_argument("--answers", required=True, help='\u6a21\u578b\u751f\u6210\u7684\u7b54\u6848 JSON\uff08\u672a\u5bf9\u9f50 ground_truth\uff09')
    parser.add_argument("--output-dir", required=False, help='\u8f93\u51fa\u76ee\u5f55\uff0c\u5305\u542b\u5bf9\u9f50\u7b54\u6848\u4e0e\u8bc4\u4f30\u62a5\u544a')
    parser.add_argument("--model-path", default=None, help='vLLM \u6a21\u578b\u6743\u91cd\uff08HF \u540d\u79f0\u6216\u672c\u5730\u8def\u5f84\uff09')
    parser.add_argument("--dataset-path", default=None, help='\u53ef\u9009\uff1a\u8986\u76d6\u9ed8\u8ba4\u6570\u636e\u96c6\u8def\u5f84')
    args = parser.parse_args()

    if args.output_dir is None:
        answer_path = Path(args.answers)
        args.output_dir = str(answer_path.parent / "prepared_answers")
        print(f"[\u63d0\u793a] \u672a\u6307\u5b9a output_dir\uff0c\u4f7f\u7528\u9ed8\u8ba4\u8def\u5f84: {args.output_dir}")

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

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
