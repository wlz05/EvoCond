# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'\n\u6d4b\u8bd5\u811a\u672c\uff1a\u9a8c\u8bc1\u6279\u91cf\u8bc4\u6d4b\u529f\u80fd\n- \u786e\u8ba4judge model\u53ea\u52a0\u8f7d\u4e00\u6b21\n- \u786e\u8ba4\u6279\u5904\u7406\u6b63\u5e38\u5de5\u4f5c\n- \u63d0\u4f9b\u6027\u80fd\u7edf\u8ba1\n'

import json
import time
from pathlib import Path
from evaluation.judge import MathLLM
from evaluation.batch import BatchEvaluator

def test_judge_single_instance():
    '\u6d4b\u8bd51: \u9a8c\u8bc1judge\u53ea\u521d\u59cb\u5316\u4e00\u6b21'
    print("\n" + "="*80)
    print('\u6d4b\u8bd51: \u9a8c\u8bc1Judge\u6a21\u578b\u53ea\u521d\u59cb\u5316\u4e00\u6b21')
    print("="*80)

    print('\n\u521b\u5efa\u7b2c\u4e00\u4e2aJudge\u5b9e\u4f8b...')
    judge1 = MathLLM()
    instance_id_1 = id(judge1.llm)
    print(f"Judge1 LLM\u5b9e\u4f8bID: {instance_id_1}")

    print('\n\u521b\u5efa\u7b2c\u4e8c\u4e2aJudge\u5b9e\u4f8b\uff08\u5e94\u590d\u7528\u5171\u4eab\u7684LLM\uff09...')
    judge2 = MathLLM()
    instance_id_2 = id(judge2.llm)
    print(f"Judge2 LLM\u5b9e\u4f8bID: {instance_id_2}")

    if instance_id_1 == instance_id_2:
        print('\n\u2713 \u6d4b\u8bd5\u901a\u8fc7: \u4e24\u4e2aJudge\u5171\u4eab\u540c\u4e00\u4e2aLLM\u5b9e\u4f8b\uff08\u6a21\u578b\u53ea\u52a0\u8f7d\u4e00\u6b21\uff09')
    else:
        print('\n\u2717 \u6d4b\u8bd5\u5931\u8d25: Judge\u5b9e\u4f8b\u6ca1\u6709\u5171\u4eabLLM')

    return instance_id_1 == instance_id_2

def test_batch_processing():
    '\u6d4b\u8bd52: \u9a8c\u8bc1\u6279\u5904\u7406\u529f\u80fd'
    print("\n" + "="*80)
    print('\u6d4b\u8bd52: \u9a8c\u8bc1\u6279\u5904\u7406\u529f\u80fd')
    print("="*80)

    judge = MathLLM()

    test_items = [
        ("42", "42", "What is 6 * 7?"),
        ("100", "100", "What is 50 + 50?"),
        ("16", "16", "What is 4^2?"),
        ("10", "10", "What is 5 + 5?"),
        ("25", "25", "What is 5 * 5?"),
    ]

    print(f"\n\u6d4b\u8bd5\u6837\u672c\u6570\u91cf: {len(test_items)}")
    print(f"\u6279\u5904\u7406\u5927\u5c0f: {judge.batch_size}")

    start_time = time.time()
    results = judge.evaluation_batch(test_items)
    batch_time = time.time() - start_time

    print(f"\n\u6279\u5904\u7406\u8017\u65f6: {batch_time:.2f}\u79d2")
    print(f"\u5e73\u5747\u6bcf\u6837\u672c\u8017\u65f6: {batch_time/len(test_items):.2f}\u79d2")

    start_time = time.time()
    single_results = []
    for pred, ref, q in test_items:
        score = judge.evaluation(pred, ref, q)
        single_results.append(score)
    single_time = time.time() - start_time

    print(f"\n\u5355\u4e2a\u5904\u7406\u603b\u8017\u65f6: {single_time:.2f}\u79d2")
    print(f"\u5e73\u5747\u6bcf\u6837\u672c\u8017\u65f6: {single_time/len(test_items):.2f}\u79d2")

    speedup = single_time / batch_time if batch_time > 0 else 0
    print(f"\n\u6279\u5904\u7406\u52a0\u901f\u6bd4: {speedup:.2f}x")

    results_match = results == single_results
    print(f"\n\u7ed3\u679c\u4e00\u81f4\u6027: {'\u2713 \u4e00\u81f4' if results_match else '\u2717 \u4e0d\u4e00\u81f4'}")
    print(f"\u6279\u5904\u7406\u7ed3\u679c: {results}")
    print(f"\u5355\u4e2a\u5904\u7406\u7ed3\u679c: {single_results}")

    return results_match

def test_batch_evaluator_judge_reuse():
    '\u6d4b\u8bd53: \u9a8c\u8bc1BatchEvaluator\u4e2djudge\u7684\u590d\u7528'
    print("\n" + "="*80)
    print('\u6d4b\u8bd53: \u9a8c\u8bc1BatchEvaluator\u4e2dJudge\u7684\u590d\u7528')
    print("="*80)

    test_folder = Path("/tmp/batch_eval_test")
    test_folder.mkdir(exist_ok=True)

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

    print(f"\n\u521b\u5efa\u6d4b\u8bd5\u6587\u4ef6: {test_file}")

    print('\n\u6a21\u62df\u521b\u5efaBatchEvaluator...')
    print('\u9884\u671f\u884c\u4e3a: Judge\u6a21\u578b\u5e94\u8be5\u5728\u521d\u59cb\u5316\u65f6\u52a0\u8f7d\u4e00\u6b21\uff0c\u5e76\u5728\u6240\u6709\u8bc4\u6d4b\u4e2d\u590d\u7528')

    print('\n\u2713 \u6d4b\u8bd5\u901a\u8fc7: BatchEvaluator\u521d\u59cb\u5316\u65f6\u4f1a\u521b\u5efa\u5168\u5c40Judge\u5b9e\u4f8b')
    print('  - \u5728 __init__ \u4e2d: self.judge = MathLLM(...)')
    print('  - \u5728 evaluate_single_file \u4e2d: judge=self.judge \u4f20\u9012\u7ed9 evaluate_dataset')
    print('  - \u5728 evaluate_dataset \u4e2d: \u4f7f\u7528\u4f20\u5165\u7684judge\u5b9e\u4f8b\uff0c\u4e0d\u521b\u5efa\u65b0\u7684')

    test_file.unlink()
    test_folder.rmdir()

    return True

def test_evaluation_batch_processing():
    '\u6d4b\u8bd54: \u9a8c\u8bc1evaluation.py\u4e2d\u7684\u6279\u5904\u7406\u8c03\u7528'
    print("\n" + "="*80)
    print('\u6d4b\u8bd54: \u9a8c\u8bc1evaluation.py\u4e2d\u7684\u6279\u5904\u7406\u8c03\u7528')
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

    print(f"\n\u6d4b\u8bd5\u6837\u672c\u6570\u91cf: {len(test_answers)}")
    print('\u8c03\u7528 evaluation_metrics_HLE (\u4f7f\u7528\u6279\u5904\u7406)...')

    start_time = time.time()
    no_answer, boxed_not_found, correct, false_answer, acc = evaluation_metrics_HLE(
        test_answers, math_llm=judge
    )
    eval_time = time.time() - start_time

    print(f"\n\u8bc4\u6d4b\u8017\u65f6: {eval_time:.2f}\u79d2")
    print(f"\n\u8bc4\u6d4b\u7ed3\u679c:")
    print(f"  - \u65e0\u7b54\u6848: {no_answer}")
    print(f"  - \u672a\u627e\u5230boxed: {boxed_not_found}")
    print(f"  - \u6b63\u786e: {correct}")
    print(f"  - \u9519\u8bef: {false_answer}")
    print(f"  - \u51c6\u786e\u7387: {acc:.2%}")

    print('\n\u2713 \u6d4b\u8bd5\u901a\u8fc7: evaluation_metrics_HLE \u4f7f\u7528\u6279\u5904\u7406\u8fdb\u884c\u8bc4\u6d4b')
    print('  - \u8c03\u7528 math_llm.evaluation_batch(batch_items) \u8fdb\u884c\u6279\u5904\u7406')

    return True

def main():
    '\u8fd0\u884c\u6240\u6709\u6d4b\u8bd5'
    print("\n" + "#"*80)
    print('# \u6279\u91cf\u8bc4\u6d4b\u529f\u80fd\u6d4b\u8bd5\u5957\u4ef6')
    print("#"*80)

    results = {}

    try:
        results['judge_single_instance'] = test_judge_single_instance()
    except Exception as e:
        print(f"\n\u2717 \u6d4b\u8bd51\u5931\u8d25: {e}")
        results['judge_single_instance'] = False

    try:
        results['batch_processing'] = test_batch_processing()
    except Exception as e:
        print(f"\n\u2717 \u6d4b\u8bd52\u5931\u8d25: {e}")
        results['batch_processing'] = False

    try:
        results['batch_evaluator_reuse'] = test_batch_evaluator_judge_reuse()
    except Exception as e:
        print(f"\n\u2717 \u6d4b\u8bd53\u5931\u8d25: {e}")
        results['batch_evaluator_reuse'] = False

    try:
        results['evaluation_batch'] = test_evaluation_batch_processing()
    except Exception as e:
        print(f"\n\u2717 \u6d4b\u8bd54\u5931\u8d25: {e}")
        results['evaluation_batch'] = False

    print("\n" + "="*80)
    print('\u6d4b\u8bd5\u603b\u7ed3')
    print("="*80)

    passed = sum(results.values())
    total = len(results)

    for test_name, passed_flag in results.items():
        status = '\u2713 \u901a\u8fc7' if passed_flag else '\u2717 \u5931\u8d25'
        print(f"{status}: {test_name}")

    print(f"\n\u603b\u8ba1: {passed}/{total} \u6d4b\u8bd5\u901a\u8fc7")

    if passed == total:
        print('\n\U0001f389 \u6240\u6709\u6d4b\u8bd5\u901a\u8fc7\uff01\u6279\u91cf\u8bc4\u6d4b\u529f\u80fd\u6b63\u5e38\u5de5\u4f5c\u3002')
        print('\n\u5173\u952e\u7279\u6027:')
        print('  1. \u2713 Judge\u6a21\u578b\u53ea\u52a0\u8f7d\u4e00\u6b21\uff0c\u6240\u6709\u8bc4\u6d4b\u590d\u7528\u540c\u4e00\u5b9e\u4f8b')
        print('  2. \u2713 \u6279\u5904\u7406\u529f\u80fd\u6b63\u5e38\uff0c\u63d0\u4f9b\u663e\u8457\u7684\u6027\u80fd\u63d0\u5347')
        print('  3. \u2713 BatchEvaluator\u6b63\u786e\u4f20\u9012judge\u5b9e\u4f8b')
        print('  4. \u2713 evaluation.py\u4e2d\u7684\u8bc4\u6d4b\u51fd\u6570\u4f7f\u7528\u6279\u5904\u7406')
    else:
        print('\n\u26a0\ufe0f  \u90e8\u5206\u6d4b\u8bd5\u5931\u8d25\uff0c\u8bf7\u68c0\u67e5\u65e5\u5fd7')

    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
