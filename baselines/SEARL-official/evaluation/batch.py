# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Evaluate matching JSON files across datasets and write {json_filename: {dataset_name: evaluation_results}}.'

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
    'Evaluate every matching prediction JSON file in a folder.'

    def __init__(
        self,
        input_folder: str,
        output_file: str,
        datasets: List[str],
        model_path: str = None,
        dataset_paths: Dict[str, str] = None,
        file_pattern: str = "answer_*.json",
    ):
        'Configure input/output paths, datasets, vLLM weights, dataset overrides and filename matching.'
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
        print('Initializing the shared judge model...')
        print(f"Model path: {model_path or 'Default (Qwen/Qwen2.5-7B-Instruct)'}")
        print(f"{'='*80}\n")
        self.judge = MathLLM(model_path=model_path) if model_path else MathLLM()
        print('Judge model initialized; all evaluations reuse this instance.\n')

    def extract_model_name(self, input_folder: Path) -> str:
        "Extract the model name from the prediction folder's parent path."
        folder_name = input_folder.name

        if folder_name in ["validation_samples", "to_do"]:
            return input_folder.parent.name

        return folder_name

    def find_json_files(self) -> List[Path]:
        'Find matching JSON files.'
        if not self.input_folder.exists():
            raise ValueError(f"Input folder does not exist: {self.input_folder}")

        json_files = []

        for json_file in self.input_folder.rglob(self.file_pattern):
            if json_file.is_file():
                json_files.append(json_file)

        print(f"Found files in {self.input_folder}: {len(json_files)} JSON files")
        return sorted(json_files)

    def evaluate_single_file(self, json_file: Path) -> Dict[str, Dict]:
        'Evaluate one JSON file across datasets and return results keyed by dataset name.'
        results = {}
        file_name = json_file.name

        print(f"\n{'='*80}")
        print(f"Evaluating file: {file_name}")
        print(f"{'='*80}")

        for dataset_name in self.datasets:
            print(f"\n--- Dataset: {dataset_name} ---")

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
                print(f"PASS {dataset_name} evaluated")

            except Exception as e:
                print(f"FAIL {dataset_name} evaluation failed: {e}")
                results[dataset_name] = {"error": str(e)}

        return results

    def run(self, skip_existing: bool = True):
        'Run batch evaluation, optionally skipping existing results.'

        json_files = self.find_json_files()

        if not json_files:
            print('No matching JSON files found.')
            return

        all_results = {}
        if skip_existing and self.output_file.exists():
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
                print(f"\nLoaded existing results: {len(all_results)} models")
            except json.JSONDecodeError:
                print('\nWARNING: Existing results are corrupt; starting again.')
                all_results = {}

        if self.model_name not in all_results:
            all_results[self.model_name] = {}

        for json_file in tqdm(json_files, desc=f"Evaluating {self.model_name}"):
            file_name = json_file.name

            if skip_existing and file_name in all_results[self.model_name]:
                print(f"\nSkipping evaluated file: {file_name}")
                continue

            print(f"\nProcessing file: {file_name}")

            file_results = self.evaluate_single_file(json_file)

            all_results[self.model_name][file_name] = file_results

            self.save_results(all_results)

        print(f"\n{'='*80}")
        print(f"Model {self.model_name} evaluated; results saved to: {self.output_file}")
        print(f"{'='*80}")

        self.print_summary(all_results)

    def save_results(self, results: Dict):
        'Write results to JSON.'
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def print_summary(self, results: Dict):
        'Print the evaluation summary.'
        print("\n" + "="*80)
        print('Evaluation summary')
        print("="*80)

        for model_name, files_results in results.items():
            print(f"\nModel: {model_name}")
            for file_name, datasets_results in files_results.items():
                print(f"  File: {file_name}")
                for dataset_name, eval_results in datasets_results.items():
                    if "error" in eval_results:
                        print(f"    - {dataset_name}: failed ({eval_results['error']})")
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
                                    print(f"    - {dataset_name}: completed")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate JSON files in a folder.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\npython batch.py --input-folder outputs/validation_samples --output-file results/batch_eval.json --datasets 2wiki bamboogle hotpotqa musique\npython batch.py --input-folder outputs/validation_samples --output-file results/batch_eval.json --datasets gsm8k math500 --dataset-path gsm8k=assets/data/gsm8k.jsonl --dataset-path math500=assets/data/math500.jsonl\npython batch.py --input-folder outputs/validation_samples --output-file results/batch_eval.json --datasets 2wiki bamboogle --model-path assets/models/judge'
    )

    parser.add_argument(
        "--input-folder",
        required=True,
        help='Input folder containing prediction JSON files'
    )

    parser.add_argument(
        "--output-file",
        required=True,
        help='Output JSON file'
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help='Datasets: HLE, GAIA, WebWalker, XBench, 2wiki, bamboogle, hotpotqa, musique, gsm8k, math500, aime24, aime25'
    )

    parser.add_argument(
        "--model-path",
        default=None,
        help='vLLM weights: Hugging Face model ID or local path'
    )

    parser.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        help='Dataset override as dataset_name=path; may be repeated'
    )

    parser.add_argument(
        "--file-pattern",
        default="answer_*.json",
        help='JSON filename pattern (default: answer_*.json)'
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help='Evaluate all files, including completed files'
    )

    return parser.parse_args()

def main():
    args = parse_args()

    dataset_paths = {}
    for item in args.dataset_path:
        if "=" not in item:
            print(f"WARNING: Ignoring malformed dataset-path argument: {item}")
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
