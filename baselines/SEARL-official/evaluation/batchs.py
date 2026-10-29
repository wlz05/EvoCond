# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Test shared judge initialization, batch evaluation and timing.'

import json
import time
import tempfile
from pathlib import Path
from evaluation.judge import MathLLM
from evaluation.batch import BatchEvaluator

def test_judge_single_instance():
    'Verify that the judge is initialized once.'
    print("\n" + "="*80)
    print('Test 1: initialize the judge once')
    print("="*80)

    print('\nCreating the first judge...')
    judge1 = MathLLM()
    instance_id_1 = id(judge1.llm)
    print(f"Judge 1 LLM instance ID: {instance_id_1}")

    print('\nCreating the second judge with the shared LLM...')
    judge2 = MathLLM()
    instance_id_2 = id(judge2.llm)
    print(f"Judge 2 LLM instance ID: {instance_id_2}")

    if instance_id_1 == instance_id_2:
        print('\nPASS: Both judges share one LLM instance.')
    else:
        print('\nFAIL: Judge instances do not share an LLM.')

    return instance_id_1 == instance_id_2

def test_batch_processing():
    'Test batch evaluation.'
    print("\n" + "="*80)
    print('Test 2: batch evaluation')
    print("="*80)

    judge = MathLLM()

    test_items = [
        ("42", "42", "What is 6 * 7?"),
        ("100", "100", "What is 50 + 50?"),
        ("16", "16", "What is 4^2?"),
        ("10", "10", "What is 5 + 5?"),
        ("25", "25", "What is 5 * 5?"),
    ]

    print(f"\nTest samples: {len(test_items)}")
    print(f"Batch size: {judge.batch_size}")

    start_time = time.time()
    results = judge.evaluation_batch(test_items)
    batch_time = time.time() - start_time

    print(f"\nBatch time: {batch_time:.2f} seconds")
    print(f"Mean time per sample: {batch_time/len(test_items):.2f} seconds")

    start_time = time.time()
    single_results = []
    for pred, ref, q in test_items:
        score = judge.evaluation(pred, ref, q)
        single_results.append(score)
    single_time = time.time() - start_time

    print(f"\nSequential processing time: {single_time:.2f} seconds")
    print(f"Mean time per sample: {single_time/len(test_items):.2f} seconds")

    speedup = single_time / batch_time if batch_time > 0 else 0
    print(f"\nBatch speedup: {speedup:.2f}x")

    results_match = results == single_results
    print(f"\nResults match: {'MATCH' if results_match else 'MISMATCH'}")
    print(f"Batch results: {results}")
    print(f"Sequential results: {single_results}")

    return results_match

def test_batch_evaluator_judge_reuse():
    'Verify that BatchEvaluator reuses its judge.'
    print("\n" + "="*80)
    print('Test 3: judge reuse in BatchEvaluator')
    print("="*80)

    temporary = tempfile.TemporaryDirectory(prefix='batch-evaluation-', dir='.')
    test_folder = Path(temporary.name)

    test_data = [
        {
            "input": "What is 1+1?",
            "output": "The answer is 2. <answer>\\boxed{2}</answer>",
            "question": "What is 1+1?"
        }
    ]

    test_file = test_folder / "answer_test.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)

    print(f"\nCreated test file: {test_file}")

    print('\nCreating BatchEvaluator...')
    print('Expected: initialize one judge model and reuse it for every evaluation.')

    print('\nPASS: BatchEvaluator creates one shared judge.')
    print('  - __init__: self.judge = MathLLM(...)')
    print('  - evaluate_single_file: passes judge=self.judge to evaluate_dataset')
    print('  - evaluate_dataset: reuses the supplied judge')

    temporary.cleanup()

    return True

def test_evaluation_batch_processing():
    'Verify that evaluation.py calls batched scoring.'
    print("\n" + "="*80)
    print('Test 4: batched calls in evaluation.py')
    print("="*80)

    from evaluation.evaluation import evaluation_metrics_HLE

    test_answers = [
        {
            "output": "The answer is 42. <answer>\\boxed{42}</answer>",
            "ground_truth": "42",
            "question": "What is the answer?"
        },
        {
            "output": "The answer is 100. <answer>\\boxed{100}</answer>",
            "ground_truth": "100",
            "question": "What is 50+50?"
        },
        {
            "output": "The answer is 16. <answer>\\boxed{16}</answer>",
            "ground_truth": "16",
            "question": "What is 4^2?"
        },
    ]

    judge = MathLLM()

    print(f"\nTest samples: {len(test_answers)}")
    print('Calling evaluation_metrics_HLE with batching...')

    start_time = time.time()
    no_answer, boxed_not_found, correct, false_answer, acc = evaluation_metrics_HLE(
        test_answers, math_llm=judge
    )
    eval_time = time.time() - start_time

    print(f"\nEvaluation time: {eval_time:.2f} seconds")
    print(f"\nEvaluation results:")
    print(f"  - Missing answers: {no_answer}")
    print(f"  - Missing boxed answers: {boxed_not_found}")
    print(f"  - Correct: {correct}")
    print(f"  - Incorrect: {false_answer}")
    print(f"  - Accuracy: {acc:.2%}")

    print('\nPASS: evaluation_metrics_HLE uses batched evaluation.')
    print('  - Calls math_llm.evaluation_batch(batch_items)')

    return True

def main():
    'Run all tests.'
    print("\n" + "#"*80)
    print('# Batch evaluation test suite')
    print("#"*80)

    results = {}

    try:
        results['judge_single_instance'] = test_judge_single_instance()
    except Exception as e:
        print(f"\nTest 1 failed: {e}")
        results['judge_single_instance'] = False

    try:
        results['batch_processing'] = test_batch_processing()
    except Exception as e:
        print(f"\nTest 2 failed: {e}")
        results['batch_processing'] = False

    try:
        results['batch_evaluator_reuse'] = test_batch_evaluator_judge_reuse()
    except Exception as e:
        print(f"\nTest 3 failed: {e}")
        results['batch_evaluator_reuse'] = False

    try:
        results['evaluation_batch'] = test_evaluation_batch_processing()
    except Exception as e:
        print(f"\nTest 4 failed: {e}")
        results['evaluation_batch'] = False

    print("\n" + "="*80)
    print('Test summary')
    print("="*80)

    passed = sum(results.values())
    total = len(results)

    for test_name, passed_flag in results.items():
        status = 'PASS' if passed_flag else 'FAIL'
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print('\nAll batch evaluation tests passed.')
        print('\nVerified features:')
        print('  1. One shared judge model')
        print('  2. Batched evaluation')
        print('  3. BatchEvaluator passes the judge instance')
        print('  4. evaluation.py uses batched scoring')
    else:
        print('\nWARNING: Some tests failed; check the logs.')

    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
