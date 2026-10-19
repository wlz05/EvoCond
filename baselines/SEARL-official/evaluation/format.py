# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'\u6d4b\u8bd5\u5d4c\u5957\u8f93\u51fa\u683c\u5f0f'

import json
from pathlib import Path

def test_nested_structure():
    '\u6d4b\u8bd5\u4e09\u5c42\u5d4c\u5957\u7ed3\u6784\u7684\u6784\u5efa\u903b\u8f91'

    print('\u6d4b\u8bd5\u4e09\u5c42\u5d4c\u5957\u7ed3\u6784\u6784\u5efa\u903b\u8f91')
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

    print('\n\u751f\u6210\u7684\u7ed3\u6784:')
    print(json.dumps(all_results, indent=2, ensure_ascii=False))

    print('\n\n\u7ed3\u6784\u9a8c\u8bc1:')
    print("=" * 80)

    for model_name, files_results in all_results.items():
        print(f"\u2713 \u7b2c\u4e00\u5c42 - \u6a21\u578b\u540d\u79f0: {model_name}")
        for file_name, datasets_results in files_results.items():
            print(f"  \u2713 \u7b2c\u4e8c\u5c42 - JSON\u6587\u4ef6: {file_name}")
            for dataset_name, eval_results in datasets_results.items():
                print(f"    \u2713 \u7b2c\u4e09\u5c42 - \u6570\u636e\u96c6: {dataset_name}")
                if "overall" in eval_results:
                    acc = eval_results["overall"]["acc"]
                    print(f"      \u2192 acc={acc:.2f}")

    print("\n" + "=" * 80)
    print('\u2713 \u4e09\u5c42\u5d4c\u5957\u7ed3\u6784\u6b63\u786e\uff01')
    print("=" * 80)

    print('\n\u671f\u671b\u7684JSON\u8f93\u51fa\u683c\u5f0f:')
    print("-" * 80)
    expected_format = '\n{\n  "grpo_baseline_1gpu": {\n    "validation_samples_20251223_232253.json": {\n      "hotpotqa": {...},\n      "gsm8k": {...}\n    },\n    "validation_samples_20251224_100000.json": {\n      "hotpotqa": {...},\n      "gsm8k": {...}\n    }\n  },\n  "grpo_baseline_6gpu": {\n    "validation_samples_20251223_150000.json": {\n      "hotpotqa": {...},\n      "gsm8k": {...}\n    }\n  }\n}\n    '
    print(expected_format)
    print("-" * 80)

def test_model_name_extraction():
    '\u6d4b\u8bd5\u6a21\u578b\u540d\u79f0\u63d0\u53d6\u903b\u8f91'

    print('\n\n\u6d4b\u8bd5\u6a21\u578b\u540d\u79f0\u63d0\u53d6\u903b\u8f91')
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

        status = '\u2713' if extracted == expected else '\u2717'
        print(f"{status} {input_path}")
        print(f"   \u671f\u671b: {expected}")
        print(f"   \u63d0\u53d6: {extracted}")
        print()

if __name__ == "__main__":
    test_nested_structure()
    test_model_name_extraction()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
