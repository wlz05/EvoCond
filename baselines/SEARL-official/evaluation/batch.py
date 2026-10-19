# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'\n\u6279\u91cf\u8bc4\u6d4b\u811a\u672c\uff1a\n- \u626b\u63cf\u6307\u5b9a\u6587\u4ef6\u5939\u4e0b\u7684\u6240\u6709JSON\u6587\u4ef6\n- \u5bf9\u6bcf\u4e2aJSON\u6587\u4ef6\u8fd0\u884c\u591a\u4e2a\u6570\u636e\u96c6\u7684\u8bc4\u6d4b\n- \u8f93\u51fa\u7edf\u4e00\u7684JSON\u6587\u4ef6\uff0c\u683c\u5f0f\u4e3a\uff1a{json_filename: {dataset_name: evaluation_results}}\n'

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from collections import defaultdict

from evaluation.evaluation import (
    prepare_answer_file,
    evaluate_dataset,
    _build_dataset_map,
)
from evaluation.judge import MathLLM

class BatchEvaluator:
    '\u6279\u91cf\u8bc4\u6d4b\u5668\uff1a\u5904\u7406\u6587\u4ef6\u5939\u4e0b\u6240\u6709JSON\u6587\u4ef6\u7684\u8bc4\u6d4b\u4efb\u52a1'

    def __init__(
        self,
        input_folder: str,
        output_file: str,
        datasets: List[str],
        model_path: str = None,
        dataset_paths: Dict[str, str] = None,
        file_pattern: str = "answer_*.json",
    ):
        '\n        Args:\n            input_folder: \u8f93\u5165\u6587\u4ef6\u5939\u8def\u5f84\n            output_file: \u8f93\u51faJSON\u6587\u4ef6\u8def\u5f84\n            datasets: \u8981\u8bc4\u6d4b\u7684\u6570\u636e\u96c6\u5217\u8868\n            model_path: vLLM\u6a21\u578b\u8def\u5f84\n            dataset_paths: \u6570\u636e\u96c6\u8def\u5f84\u8986\u76d6\u5b57\u5178 {dataset_name: path}\n            file_pattern: JSON\u6587\u4ef6\u5339\u914d\u6a21\u5f0f\n        '
        self.input_folder = Path(input_folder)
        self.output_file = Path(output_file)
        self.datasets = datasets
        self.model_path = model_path
        self.dataset_paths = dataset_paths or {}
        self.file_pattern = file_pattern

        self.model_name = self.extract_model_name(self.input_folder)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.temp_dir = self.output_file.parent / "temp_evaluation"
        self.temp_dir.mkdir(exist_ok=True)

        print(f"\n{'='*80}")
        print('\u6b63\u5728\u521d\u59cb\u5316Judge\u6a21\u578b\uff08\u5168\u5c40\u590d\u7528\uff0c\u907f\u514d\u91cd\u590d\u52a0\u8f7d\uff09...')
        print(f"\u6a21\u578b\u8def\u5f84: {model_path or 'Default (Qwen/Qwen2.5-7B-Instruct)'}")
        print(f"{'='*80}\n")
        self.judge = MathLLM(model_path=model_path) if model_path else MathLLM()
        print('\u2713 Judge\u6a21\u578b\u521d\u59cb\u5316\u5b8c\u6210\uff0c\u5c06\u5728\u6240\u6709\u8bc4\u6d4b\u4e2d\u590d\u7528\u6b64\u5b9e\u4f8b\n')

    def extract_model_name(self, input_folder: Path) -> str:
        '\n        \u4ece\u8f93\u5165\u6587\u4ef6\u5939\u8def\u5f84\u4e2d\u63d0\u53d6\u6a21\u578b\u540d\u79f0\n        \u4f8b\u5982:\n            /path/to/baseline_gpu6_grpo_prm_new_set/to_do -> baseline_gpu6_grpo_prm_new_set\n            /path/to/grpo_baseline_1gpu/validation_samples -> grpo_baseline_1gpu\n\n        Args:\n            input_folder: \u8f93\u5165\u6587\u4ef6\u5939\u8def\u5f84\n\n        Returns:\n            str: \u6a21\u578b\u540d\u79f0\n        '
        folder_name = input_folder.name

        if folder_name in ["validation_samples", "to_do"]:
            return input_folder.parent.name

        return folder_name

    def find_json_files(self) -> List[Path]:
        '\u67e5\u627e\u6587\u4ef6\u5939\u4e0b\u6240\u6709\u5339\u914d\u7684JSON\u6587\u4ef6'
        if not self.input_folder.exists():
            raise ValueError(f"\u8f93\u5165\u6587\u4ef6\u5939\u4e0d\u5b58\u5728: {self.input_folder}")

        json_files = []

        for json_file in self.input_folder.rglob(self.file_pattern):
            if json_file.is_file():
                json_files.append(json_file)

        print(f"\u5728 {self.input_folder} \u4e2d\u627e\u5230 {len(json_files)} \u4e2aJSON\u6587\u4ef6")
        return sorted(json_files)

    def evaluate_single_file(self, json_file: Path) -> Dict[str, Dict]:
        '\n        \u5bf9\u5355\u4e2aJSON\u6587\u4ef6\u8fdb\u884c\u6240\u6709\u6570\u636e\u96c6\u7684\u8bc4\u6d4b\n\n        Returns:\n            Dict[dataset_name, evaluation_results]\n        '
        results = {}
        file_name = json_file.name

        print(f"\n{'='*80}")
        print(f"\u6b63\u5728\u8bc4\u6d4b\u6587\u4ef6: {file_name}")
        print(f"{'='*80}")

        for dataset_name in self.datasets:
            print(f"\n--- \u6570\u636e\u96c6: {dataset_name} ---")

            try:

                temp_output_dir = self.temp_dir / file_name.replace('.json', '') / dataset_name
                temp_output_dir.mkdir(parents=True, exist_ok=True)

                dataset_path = self.dataset_paths.get(dataset_name)
                prepared_path = prepare_answer_file(
                    dataset_name=dataset_name,
                    answer_json_path=str(json_file),
                    output_dir=str(temp_output_dir),
                    dataset_root_override=dataset_path,
                )

                eval_output_path = evaluate_dataset(
                    dataset_name=dataset_name,
                    prepared_answer_path=prepared_path,
                    model_path=self.model_path,
                    output_dir=str(temp_output_dir),
                    judge=self.judge,
                )

                with open(eval_output_path, 'r', encoding='utf-8') as f:
                    eval_results = json.load(f)

                results[dataset_name] = eval_results
                print(f"\u2713 {dataset_name} \u8bc4\u6d4b\u5b8c\u6210")

            except Exception as e:
                print(f"\u2717 {dataset_name} \u8bc4\u6d4b\u5931\u8d25: {e}")
                results[dataset_name] = {"error": str(e)}

        return results

    def run(self, skip_existing: bool = True):
        '\n        \u6267\u884c\u6279\u91cf\u8bc4\u6d4b\n\n        Args:\n            skip_existing: \u662f\u5426\u8df3\u8fc7\u5df2\u5b58\u5728\u7684\u7ed3\u679c\n        '

        json_files = self.find_json_files()

        if not json_files:
            print('\u672a\u627e\u5230\u4efb\u4f55\u5339\u914d\u7684JSON\u6587\u4ef6')
            return

        all_results = {}
        if skip_existing and self.output_file.exists():
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
                print(f"\n\u8bfb\u53d6\u5230\u5df2\u6709\u7ed3\u679c: {len(all_results)} \u4e2a\u6a21\u578b")
            except json.JSONDecodeError:
                print('\n\u8b66\u544a: \u5df2\u6709\u7ed3\u679c\u6587\u4ef6\u635f\u574f\uff0c\u5c06\u91cd\u65b0\u5f00\u59cb')
                all_results = {}

        if self.model_name not in all_results:
            all_results[self.model_name] = {}

        for json_file in tqdm(json_files, desc=f"\u8bc4\u6d4b {self.model_name}"):
            file_name = json_file.name

            if skip_existing and file_name in all_results[self.model_name]:
                print(f"\n\u8df3\u8fc7\u5df2\u8bc4\u6d4b\u6587\u4ef6: {file_name}")
                continue

            print(f"\n\u5904\u7406\u6587\u4ef6: {file_name}")

            file_results = self.evaluate_single_file(json_file)

            all_results[self.model_name][file_name] = file_results

            self.save_results(all_results)

        print(f"\n{'='*80}")
        print(f"\u6a21\u578b {self.model_name} \u8bc4\u6d4b\u5b8c\u6210\uff01\u7ed3\u679c\u5df2\u4fdd\u5b58\u5230: {self.output_file}")
        print(f"{'='*80}")

        self.print_summary(all_results)

    def save_results(self, results: Dict):
        '\u4fdd\u5b58\u7ed3\u679c\u5230JSON\u6587\u4ef6'
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def print_summary(self, results: Dict):
        '\u6253\u5370\u8bc4\u6d4b\u6458\u8981'
        print("\n" + "="*80)
        print('\u8bc4\u6d4b\u6458\u8981')
        print("="*80)

        for model_name, files_results in results.items():
            print(f"\n\u6a21\u578b: {model_name}")
            for file_name, datasets_results in files_results.items():
                print(f"  \u6587\u4ef6: {file_name}")
                for dataset_name, eval_results in datasets_results.items():
                    if "error" in eval_results:
                        print(f"    - {dataset_name}: \u5931\u8d25 ({eval_results['error']})")
                    else:

                        if isinstance(eval_results, dict):
                            if "overall" in eval_results:
                                acc = eval_results["overall"].get("acc", "N/A")
                                print(f"    - {dataset_name}: acc={acc:.4f}" if isinstance(acc, float) else f"    - {dataset_name}: acc={acc}")
                            else:

                                accs = []
                                for category, metrics in eval_results.items():
                                    if isinstance(metrics, dict) and "acc" in metrics:
                                        accs.append(metrics["acc"])
                                if accs:
                                    avg_acc = sum(accs) / len(accs)
                                    print(f"    - {dataset_name}: avg_acc={avg_acc:.4f}")
                                else:
                                    print(f"    - {dataset_name}: \u5b8c\u6210")

def parse_args():
    parser = argparse.ArgumentParser(
        description='\u6279\u91cf\u8bc4\u6d4b\u5de5\u5177\uff1a\u5904\u7406\u6587\u4ef6\u5939\u4e0b\u6240\u6709JSON\u6587\u4ef6\u7684\u8bc4\u6d4b',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n\u793a\u4f8b\u7528\u6cd5:\n\n1. \u8bc4\u6d4b\u5355\u4e2a\u6587\u4ef6\u5939\u7684\u6240\u6709answer_*.json\u6587\u4ef6\uff1a\n   python batch.py \\\n       --input-folder /path/to/validation_samples \\\n       --output-file results/batch_eval.json \\\n       --datasets 2wiki bamboogle hotpotqa musique\n\n2. \u6307\u5b9a\u81ea\u5b9a\u4e49\u6570\u636e\u96c6\u8def\u5f84\uff1a\n   python batch.py \\\n       --input-folder /path/to/validation_samples \\\n       --output-file results/batch_eval.json \\\n       --datasets gsm8k math500 \\\n       --dataset-path gsm8k=/path/to/gsm8k.jsonl \\\n       --dataset-path math500=/path/to/math500.jsonl\n\n3. \u4f7f\u7528\u81ea\u5b9a\u4e49\u6a21\u578b\u8def\u5f84\uff1a\n   python batch.py \\\n       --input-folder /path/to/validation_samples \\\n       --output-file results/batch_eval.json \\\n       --datasets 2wiki bamboogle \\\n       --model-path /path/to/model\n        '
    )

    parser.add_argument(
        "--input-folder",
        required=True,
        help='\u8f93\u5165\u6587\u4ef6\u5939\u8def\u5f84\uff08\u5305\u542b\u8981\u8bc4\u6d4b\u7684JSON\u6587\u4ef6\uff09'
    )

    parser.add_argument(
        "--output-file",
        required=True,
        help='\u8f93\u51faJSON\u6587\u4ef6\u8def\u5f84'
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help='\u8981\u8bc4\u6d4b\u7684\u6570\u636e\u96c6\u5217\u8868\uff08\u652f\u6301: HLE, GAIA, WebWalker, XBench, 2wiki, bamboogle, hotpotqa, musique, gsm8k, math500, aime24, aime25\uff09'
    )

    parser.add_argument(
        "--model-path",
        default=None,
        help='vLLM\u6a21\u578b\u6743\u91cd\u8def\u5f84\uff08HF\u540d\u79f0\u6216\u672c\u5730\u8def\u5f84\uff09'
    )

    parser.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        help='\u6570\u636e\u96c6\u8def\u5f84\u8986\u76d6\uff08\u683c\u5f0f: dataset_name=path\uff09\uff0c\u53ef\u591a\u6b21\u4f7f\u7528'
    )

    parser.add_argument(
        "--file-pattern",
        default="answer_*.json",
        help='JSON\u6587\u4ef6\u5339\u914d\u6a21\u5f0f\uff08\u9ed8\u8ba4: answer_*.json\uff09'
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help='\u4e0d\u8df3\u8fc7\u5df2\u8bc4\u6d4b\u7684\u6587\u4ef6\uff0c\u91cd\u65b0\u8bc4\u6d4b\u6240\u6709\u6587\u4ef6'
    )

    return parser.parse_args()

def main():
    args = parse_args()

    dataset_paths = {}
    for item in args.dataset_path:
        if "=" not in item:
            print(f"\u8b66\u544a: \u5ffd\u7565\u683c\u5f0f\u9519\u8bef\u7684dataset-path\u53c2\u6570: {item}")
            continue
        dataset_name, path = item.split("=", 1)
        dataset_paths[dataset_name] = path

    evaluator = BatchEvaluator(
        input_folder=args.input_folder,
        output_file=args.output_file,
        datasets=args.datasets,
        model_path=args.model_path,
        dataset_paths=dataset_paths,
        file_pattern=args.file_pattern,
    )

    evaluator.run(skip_existing=not args.no_skip_existing)

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
