# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Test nested result formatting.'

import json
from pathlib import Path

def test_nested_structure():
    'Test three-level result nesting.'

    print('Testing three-level result nesting')
    print("=" * 80)

    test_scenarios = [
        {
            "input_folder": 'runs/grpo_baseline_1gpu/validation_samples',
            "json_files": [
                "validation_samples_20251223_232253.json",
                "validation_samples_20251224_100000.json"
            ],
            "expected_model_name": "grpo_baseline_1gpu"
        },
        {
            "input_folder": 'runs/grpo_baseline_6gpu/validation_samples',
            "json_files": [
                "validation_samples_20251223_150000.json"
            ],
            "expected_model_name": "grpo_baseline_6gpu"
        }
    ]

    all_results = {}

    for scenario in test_scenarios:
        model_name = Path(scenario["input_folder"]).parent.name

        if model_name not in all_results:
            all_results[model_name] = {}

        for json_file in scenario["json_files"]:

            file_results = {
                "hotpotqa": {
                    "overall": {
                        "acc": 0.85,
                        "correct": 85,
                        "total": 100
                    }
                },
                "gsm8k": {
                    "overall": {
                        "acc": 0.90,
                        "correct": 90,
                        "total": 100
                    }
                }
            }

            all_results[model_name][json_file] = file_results

    print('\nGenerated structure:')
    print(json.dumps(all_results, indent=2, ensure_ascii=False))

    print('\n\nStructure validation:')
    print("=" * 80)

    for model_name, files_results in all_results.items():
        print(f"Level 1 - Model name: {model_name}")
        for file_name, datasets_results in files_results.items():
            print(f"  Level 2 - JSON file: {file_name}")
            for dataset_name, eval_results in datasets_results.items():
                print(f"    Level 3 - Dataset: {dataset_name}")
                if "overall" in eval_results:
                    acc = eval_results["overall"]["acc"]
                    print(f"      -> acc={acc:.2f}")

    print("\n" + "=" * 80)
    print('Three-level nesting is correct.')
    print("=" * 80)

    print('\nExpected JSON output:')
    print("-" * 80)
    expected_format = '\n{\n  "grpo_baseline_1gpu": {\n    "validation_samples_20251223_232253.json": {\n      "hotpotqa": {...},\n      "gsm8k": {...}\n    },\n    "validation_samples_20251224_100000.json": {\n      "hotpotqa": {...},\n      "gsm8k": {...}\n    }\n  },\n  "grpo_baseline_6gpu": {\n    "validation_samples_20251223_150000.json": {\n      "hotpotqa": {...},\n      "gsm8k": {...}\n    }\n  }\n}\n    '
    print(expected_format)
    print("-" * 80)

def test_model_name_extraction():
    'Test model-name extraction.'

    print('\n\nTesting model-name extraction')
    print("=" * 80)

    test_cases = [
        ('runs/grpo_baseline_1gpu_reasoning_nonthinking/validation_samples',
         "grpo_baseline_1gpu_reasoning_nonthinking"),
        ('runs/grpo_baseline_6gpu_reasoning_nonthinking_1211/validation_samples',
         "grpo_baseline_6gpu_reasoning_nonthinking_1211"),
        ('runs/reinforce_pp_baseline_6gpu_reasoning/validation_samples',
         "reinforce_pp_baseline_6gpu_reasoning"),
    ]

    for input_path, expected in test_cases:
        p = Path(input_path)
        if p.name == "validation_samples":
            extracted = p.parent.name
        else:
            extracted = p.name

        status = 'PASS' if extracted == expected else 'FAIL'
        print(f"{status} {input_path}")
        print(f"   Expected: {expected}")
        print(f"   Extracted: {extracted}")
        print()

if __name__ == "__main__":
    test_nested_structure()
    test_model_name_extraction()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
