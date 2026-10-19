# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'\n\u7edf\u4e00\u7684\u8bc4\u4f30\u811a\u672c\uff1a\n- \u652f\u6301\u901a\u8fc7\u547d\u4ee4\u884c\u6307\u5b9a\u6570\u636e\u96c6\u3001\u6a21\u578b\u7b54\u6848 JSON\u3001\u8f93\u51fa\u76ee\u5f55\n- \u4f9d\u8d56 vLLM \u5224\u5206\uff08\u89c1 gpt_judge.MathLLM\uff09\n'

import argparse
import os
import re
import string
import json
import datasets
import math
from collections import Counter
from typing import Callable, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from evaluation.judge import MathLLM

def _preprocess_latex_basic(s: str) -> str:
    '\n    \u8f7b\u91cf LaTeX \u5f52\u4e00\uff1a\x0crac, \\sqrt, ^, \\cdot, \times, \\div, \\left, \right, \u82b1\u62ec\u53f7\u7b49\n    '
    s = s.strip()

    for k in (r"\left", r"\right", r"\,", r"\!"):
        s = s.replace(k, "")

    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    # \sqrt{a} -> sqrt(a)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    # \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)

    s = s.replace("{", "(").replace("}", ")")

    s = re.sub(r"(?<=[0-9a-zA-Z\)])\^(?=[0-9a-zA-Z\(])", "**", s)

    s = s.replace("\\", "")
    return s

def _in_interval(x: float, lo: float, hi: float, left_closed: bool, right_closed: bool, rel_tol: float) -> bool:

    pad_lo = rel_tol * max(1.0, abs(lo))
    pad_hi = rel_tol * max(1.0, abs(hi))
    left_ok = (x + pad_lo) >= lo if left_closed else (x - pad_lo) > lo
    right_ok = (x - pad_hi) <= hi if right_closed else (x + pad_hi) < hi
    return left_ok and right_ok

def _to_float(expr: str) -> float:
    '\n    \u628a\u8868\u8fbe\u5f0f/LaTeX \u8f6c\u6210 float\u3002\n    \u5c1d\u8bd5\u987a\u5e8f\uff1a\n      1) \u76f4\u63a5 float\n      2) sympy.parse_latex\n      3) \u8f7b\u91cf LaTeX \u9884\u5904\u7406\u540e sympy.sympify\n    '
    s = str(expr).strip().strip("$")

    try:
        return float(s.replace(",", ""))
    except Exception:
        pass

    num = _parse_number_maybe_percent(s)
    if num is not None:
        return num

    try:
        from sympy.parsing.latex import parse_latex as _parse_latex
        expr_obj = _parse_latex(s)
        if getattr(expr_obj, "is_number", False):
            return float(expr_obj.evalf())

        return float(expr_obj.evalf())
    except Exception:
        pass

    s2 = _preprocess_latex_basic(s)
    try:
        import sympy as sp
        expr_obj = sp.sympify(s2, rational=True)
        return float(expr_obj.evalf())
    except Exception:

        return float(s2)

def _parse_number_maybe_percent(s: str):
    '\u5c1d\u8bd5\u89e3\u6790\u7eaf\u6570\u5b57\u6216\u767e\u5206\u6570\uff1b\u6210\u529f\u8fd4\u56de float\uff0c\u5426\u5219\u8fd4\u56de None\u3002'
    s1 = s.strip().strip("$")

    m = re.fullmatch(r"\s*([-+]?\d+(?:\.\d+)?)\s*(\\?%)\s*", s1)
    if m:
        return float(m.group(1)) / 100.0

    try:
        return float(s1.replace(",", ""))
    except Exception:
        return None

def _norm_text(s: str) -> str:
    '\u8f7b\u91cf\u6587\u672c\u5f52\u4e00\uff1astrip/lower/\u538b\u7f29\u7a7a\u767d\u3002'
    s = s or ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def verify_math_answer(answer_float: float,
                       ground_truth_expr: str,
                       rel_tol: float = 1e-4) -> bool:
    '\n    Verifies if a float answer is mathematically close to a LaTeX expression (or equivalent forms).\n    \u652f\u6301:\n      - \u7eaf\u6570/\u8868\u8fbe\u5f0f/LaTeX(\x0crac, \\sqrt, ^, \\cdot, \times, \\left...\right)\n      - \u767e\u5206\u6570: "12%" -> 0.12\n      - \u533a\u95f4: (a,b), [a,b], (a,b], [a,b]\n      - \u591a\u5019\u9009: {a,b,c}\n      - \\pm: 1 \\pm 2  ->  {1+2, 1-2}\n      - \u4f20\u5165 tuple/list/dict \u65f6\u7684\u9c81\u68d2\u5904\u7406\uff08\u4ece dict \u53d6 value/answer/target\uff1b\u4e8c\u5143\u5bb9\u5668\u89c6\u4e3a\u533a\u95f4\uff0c\u5426\u5219\u89c6\u4e3a\u591a\u5019\u9009\uff09\n    \u8bed\u4e49:\n      - ground truth \u662f\u5355\u503c: \u4e0e answer_float \u505a isclose\n      - ground truth \u662f\u533a\u95f4: \u5224\u65ad answer_float \u662f\u5426\u843d\u5728\u533a\u95f4\uff08\u542b\u5f00\u95ed\u7aef\u70b9\uff09\n      - ground truth \u662f\u591a\u5019\u9009: \u4efb\u4e00\u5019\u9009\u5339\u914d\u5373\u4e3a True\n    '

    if not isinstance(answer_float, (int, float)) or not math.isfinite(answer_float):
        return False
    x = float(answer_float)

    gt = ground_truth_expr

    if isinstance(gt, dict):
        for k in ("value", "answer", "target"):
            if k in gt:
                gt = gt[k]
                break

    if isinstance(gt, (list, tuple, set)):
        items = list(gt)
        if len(items) == 2:
            try:
                lo = _to_float(items[0])
                hi = _to_float(items[1])
                lo, hi = (lo, hi) if lo <= hi else (hi, lo)

                return _in_interval(x, lo, hi, left_closed=False, right_closed=False, rel_tol=rel_tol)
            except Exception:

                pass

        return any(verify_math_answer(x, it, rel_tol=rel_tol) for it in items)

    s = str(gt).strip()

    num = _parse_number_maybe_percent(s)
    if num is not None:
        return math.isclose(x, num, rel_tol=rel_tol)

    if r"\pm" in s:
        left, right = s.split(r"\pm", 1)
        cand1 = f"{left}+{right}"
        cand2 = f"{left}-{right}"
        return verify_math_answer(x, cand1, rel_tol) or verify_math_answer(x, cand2, rel_tol)

    m_set = re.fullmatch(r"\s*\{(.+)\}\s*", s)
    if m_set:
        parts = [p.strip() for p in m_set.group(1).split(",")]
        return any(verify_math_answer(x, p, rel_tol) for p in parts)

    m_iv = re.fullmatch(r"\s*([\(\[])\s*([^,]+)\s*,\s*([^\]\)]+)\s*([\)\]])\s*", s)
    if m_iv:
        left_br, a_str, b_str, right_br = m_iv.groups()
        try:
            lo = _to_float(a_str)
            hi = _to_float(b_str)
            if lo > hi:
                lo, hi = hi, lo
            return _in_interval(
                x, lo, hi,
                left_closed=(left_br == "["),
                right_closed=(right_br == "]"),
                rel_tol=rel_tol
            )
        except Exception:

            pass

    try:
        val = _to_float(s)
        return math.isclose(x, val, rel_tol=rel_tol)
    except Exception:
        return False

def extract_boxed_answer_math(solution: str) -> str:

    """Extract final boxed answer like \boxed{...} from the solution."""
    match = re.search(r"\\boxed{(.+?)}", solution)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"No boxed answer found in: {solution}")

def process_item(item: dict, idx: int, source_name: str = "custom_math_jsonl"):

    question = (
        item.get("question")
        or item.get("input")
        or item.get("query")
        or (item.get("prompt")[0]["content"] if isinstance(item.get("prompt"), list) and item.get("prompt") else None)
    )
    if not question:
        assert False, f"No question found in item: {item}"
    level = item.get("level", "Unknown")
    topic = item.get("category", "Unknown")
    index_idx = item.get("id", -1)  # Use idx if index is not available
    solution_raw = item.get("rationale")
    final_answer = item.get("answer") or item.get("output") or item.get("response")
    # print(f"value={repr(final_answer)}, type={type(final_answer)}")
    if not final_answer:
        # print("[!] No final answer found, trying to extract from solution.", final_answer)
        return None

    # return (question, final_answer)
    return (question, final_answer, topic)

def process_item_xbench(item: dict, idx: int, source_name: str = "custom_math_jsonl"):
    question = item["prompt"][0]['content']     ### hle
    # if question == None:
    #     question = item["Question"] ### gaia
    level = item.get("level", "Unknown")
    topic = item.get("category", "Unknown")
    index_idx = item.get("id", -1)  # Use idx if index is not available
    # solution_raw = item["rationale"]
    try:
        final_answer = item['extra_info'].get("answer", "Unknown")
        # print(f"value={repr(final_answer)}, type={type(final_answer)}")
        if not final_answer:
            # print("[!] No final answer found, trying to extract from solution.", final_answer)
            return None
    except Exception as e:
        final_answer = None  # optionally skip or log

    # return (question, final_answer)
    return (question, final_answer)

def process_item_webwalker(item: dict, idx: int, source_name: str = "custom_math_jsonl"):
    question = (
        item.get("question")
        or item.get("input")
        or item.get("query")
        or (item.get("prompt")[0]["content"] if isinstance(item.get("prompt"), list) and item.get("prompt") else None)
    )
    if not question:
        return None
    level = item.get("level", "Unknown")
    topic = item.get("category", "Unknown")
    info = item.get("info", {})
    index_idx = item.get("id", -1)  # Use idx if index is not available
    # solution_raw = item["rationale"]
    difficulty_level = info["difficulty_level"]
    try:
        final_answer = item['answer']
        # print(f"value={repr(final_answer)}, type={type(final_answer)}")
        if not final_answer:
            # print("[!] No final answer found, trying to extract from solution.", final_answer)
            return None
    except Exception as e:
        final_answer = None  # optionally skip or log

    # return (question, final_answer)
    return (question, final_answer, difficulty_level)

def process_item_qa(item: dict, idx: int, source_name: str = "custom_math_jsonl"):

    question = (
        item.get("question")
        or item.get("input")
        or item.get("query")
        or (item.get("prompt")[0]["content"] if isinstance(item.get("prompt"), list) and item["prompt"] else None)
    )
    if not question:
        return None
    level = item.get("level", "Unknown")
    topic = item.get("category", "Unknown")
    index_idx = item.get("id", -1)  # Use idx if index is not available
    # solution_raw = item["rationale"]
    try:
        final_answer = item.get("answer") or item.get("output") or item.get("response")
        # print(f"value={repr(final_answer)}, type={type(final_answer)}")
        if not final_answer:
            # print("[!] No final answer found, trying to extract from solution.", final_answer)
            return None
    except Exception as e:
        final_answer = None  # optionally skip or log

    # return (question, final_answer)
    return (question, final_answer)

def process_item_gaia(item: dict, idx: int, source_name: str = "custom_math_jsonl"):
    question = item["Question"]     ### hle
    # if question == None:
    #     question = item["Question"] ### gaia
    level = item.get("Level", "Unknown")
    topic = item.get("category", "Unknown")
    index_idx = item.get("id", -1)  # Use idx if index is not available
    # solution_raw = item["rationale"]
    try:
        final_answer = item.get("Final answer", "Unknown")
        # print(f"value={repr(final_answer)}, type={type(final_answer)}")
        if not final_answer:
            # print("[!] No final answer found, trying to extract from solution.", final_answer)
            return None
    except Exception as e:
        final_answer = None  # optionally skip or log
    # return (question, final_answer)
    return (question, final_answer, level)

def _extract_user_question(text: str) -> str:
    "\n    Extract the actual user question from the input text.\n    The user question is the line immediately after the 'user' marker.\n\n    Args:\n        text: The full input text (may include prompt template)\n\n    Returns:\n        str: The extracted user question, or the original text if no marker found\n    "
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.strip() == 'user' and i + 1 < len(lines):
            return lines[i + 1].strip()
    # Fallback: return the original text
    return text.strip()

def _prepare_answers_generic(
    data,
    answer_json,
    output_path: str,
    processor: Callable,
    filter_fn: Callable = lambda x: True,
    extra_assign: Callable[[dict, Tuple], None] = lambda item, tup: None,
):
    processed = []
    for idx, item in enumerate(tqdm(data, desc="Processing")):
        if not filter_fn(item):
            continue
        processed_item = processor(item, idx)
        if processed_item is not None:
            processed.append(processed_item)

    new_jsons = []
    matched = 0
    for item in answer_json:
        # Extract actual user question instead of using the entire input
        input_text = item.get("input", item.get("question", ""))
        actual_question = _extract_user_question(input_text)
        inputs_norm = _norm_text(actual_question)

        for tupes in processed:
            tgt = _norm_text(tupes[0])
            if not tgt or not inputs_norm:
                continue
            # Use exact match after normalization to avoid substring false positives
            if tgt == inputs_norm:
                item["ground_truth"] = tupes[1]
                item["question"] = tupes[0]
                extra_assign(item, tupes)
                matched += 1
                new_jsons.append(item)
                break

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_jsons, f, ensure_ascii=False, indent=4)
    print(f"[prepare] matched {matched}/{len(answer_json)} -> {output_path}")
    return output_path

def main_HLE(data: str, answer_json, output_dir):
    return _prepare_answers_generic(
        data=data,
        answer_json=answer_json,
        output_path=os.path.join(output_dir, "answer_hle.json"),
        processor=process_item,
        filter_fn=lambda item: (not item.get("image")) and item.get("category") in ["Physics", "Math", "Computer Science/AI"],
        extra_assign=lambda item, tup: item.update({"category": tup[2]}),
    )

def main_gaia(data: str, answer_json, output_dir):
    return _prepare_answers_generic(
        data=data,
        answer_json=answer_json,
        output_path=os.path.join(output_dir, "answer_gaia.json"),
        processor=process_item_gaia,
        filter_fn=lambda item: item.get("file_name", "") == "",
        extra_assign=lambda item, tup: item.update({"Level": tup[2]}),
    )

def main_webwalker(data: str, answer_json, output_dir):
    return _prepare_answers_generic(
        data=data,
        answer_json=answer_json,
        output_path=os.path.join(output_dir, "answer_webwalker.json"),
        processor=process_item_webwalker,
        extra_assign=lambda item, tup: item.update({"difficulty_level": tup[2]}),
    )

def main_xbench(data: str, answer_json, output_dir):
    return _prepare_answers_generic(
        data=data,
        answer_json=answer_json,
        output_path=os.path.join(output_dir, "answer_xbench.json"),
        processor=process_item_xbench,
    )

def main_qa(data: str, answer_json, output_dir, dataset_name):
    return _prepare_answers_generic(
        data=data,
        answer_json=answer_json,
        output_path=os.path.join(output_dir, f"answer_{dataset_name}.json"),
        processor=process_item_qa,
    )

def remove_boxed(s: str) -> str:
    '\n    Remove the LaTeX \x08oxed{} wrapper from the string.\n\n    Args:\n        s: String potentially containing \x08oxed{}\n\n    Returns:\n        str: String with \x08oxed{} removed\n    '
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    # assert s[:len(left)] == left
    # assert s[-1] == "}"

    return s[len(left):-1]

def normalize_answer(s: str) -> str:
    '\n    Normalize the answer string by removing articles, white spaces, punctuation and converting to lowercase.\n\n    Args:\n        s: String to normalize\n\n    Returns:\n        str: Normalized string\n    '
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def last_boxed_only_string(string: str):
    '\n    Extract the last \x08oxed{} content from the string.\n\n    Args:\n        string: String to extract \x08oxed{} from\n\n    Returns:\n        Optional[str]: The extracted \x08oxed{} content or None if not found\n    '
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def get_f1_score(prediction: str, ground_truths) -> float:
    '\n    Calculate F1 score between prediction and ground truths.\n\n    Args:\n        prediction: The predicted answer\n        ground_truths: The ground truth answer(s)\n\n    Returns:\n        float: F1 score\n    '
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    for ground_truth in ground_truths:

        # --- START: NEW MATH VERIFICATION BLOCK ---
        try:
            # Step 1: Attempt to convert the prediction to a float.
            # If this fails, it's not a numeric answer, and a ValueError will be raised.
            pred_float = float(prediction)

            # Step 2: Call the math verification function.
            # If the ground truth is a valid math expression and close to the prediction,
            # we have a perfect match (F1 = 1.0).
            if verify_math_answer(pred_float, ground_truth):
                # Since we found a perfect match against one of the ground truths,
                # the max possible F1 score is 1.0, so we can return immediately.
                return 1.0
        except (ValueError, TypeError):
            # This triggers if float(prediction) fails or if verify_math_answer has an issue.
            # It means this is not a valid math comparison, so we'll fall through
            # to the standard text-based F1 score calculation below.
            pass
        # --- END: NEW MATH VERIFICATION BLOCK ---

    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])

    return final_metric['f1']

def evaluation_metrics_HLE(answers, math_llm: MathLLM | None = None):
    correct = 0
    math_llm = math_llm or MathLLM()
    def extract_answer(text: str):
        '\n        Extract the <answer>...</answer> block from the LAST assistant response only.\n        This avoids extracting example answers from the prompt.\n\n        Args:\n            text: The text to extract answer from (may include prompt + response)\n\n        Returns:\n            Optional[str]: The extracted answer or None if no match\n        '
        text = text.strip()

        # Find the last occurrence of "assistant" marker to isolate actual model response
        # This handles cases where prompt contains example <answer> tags
        assistant_markers = ['assistant\n', 'Assistant\n', 'assistant:', 'Assistant:']
        last_assistant_pos = -1

        for marker in assistant_markers:
            pos = text.rfind(marker)
            if pos > last_assistant_pos:
                last_assistant_pos = pos

        # If found assistant marker, only search after it
        if last_assistant_pos >= 0:
            search_text = text[last_assistant_pos:]
        else:
            # Fallback: search entire text
            search_text = text

        pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(pattern, search_text, re.DOTALL)
        if not matches:
            return None

        return matches[-1].strip()  # Return last match in assistant response
    boxed_not_found = 0
    no_answer = 0
    false_answer = 0

    batch_items = []
    valid_indices = []

    for idx, item in enumerate(answers):
        extracted = extract_answer(item['output'])
        if extracted:
            boxed = last_boxed_only_string(extracted)
            if boxed:
                cleaned_answer = remove_boxed(boxed)
                if cleaned_answer:
                    batch_items.append((cleaned_answer, item['ground_truth'], item["question"]))
                    valid_indices.append(idx)

    if batch_items:
        batch_scores = math_llm.evaluation_batch(batch_items)

        for idx, score in zip(valid_indices, batch_scores):
            if score == 1.0:
                correct += 1
            else:
                false_answer += 1
    else:

        for item in answers:
            extracted = extract_answer(item['output'])
            if extracted:
                boxed = last_boxed_only_string(extracted)
                if boxed:
                    if not remove_boxed(boxed):
                        boxed_not_found += 1
                else:
                    boxed_not_found += 1
            else:
                no_answer += 1

    boxed_not_found = 0
    no_answer = 0
    for item in answers:
        extracted = extract_answer(item['output'])
        if not extracted:
            no_answer += 1
        elif not last_boxed_only_string(extracted):
            boxed_not_found += 1
        elif not remove_boxed(last_boxed_only_string(extracted)):
            boxed_not_found += 1

    total = no_answer + boxed_not_found + correct + false_answer
    acc = correct / total if total > 0 else 0.0
    return no_answer, boxed_not_found, correct, false_answer, acc

def evaluation_metrics_HLE_key(answers, category, math_llm: MathLLM | None = None):
    correct = 0
    math_llm = math_llm or MathLLM()
    def extract_answer(text: str):
        '\n        Extract the <answer>...</answer> block from the LAST assistant response only.\n        This avoids extracting example answers from the prompt.\n\n        Args:\n            text: The text to extract answer from (may include prompt + response)\n\n        Returns:\n            Optional[str]: The extracted answer or None if no match\n        '
        text = text.strip()

        # Find the last occurrence of "assistant" marker to isolate actual model response
        # This handles cases where prompt contains example <answer> tags
        assistant_markers = ['assistant\n', 'Assistant\n', 'assistant:', 'Assistant:']
        last_assistant_pos = -1

        for marker in assistant_markers:
            pos = text.rfind(marker)
            if pos > last_assistant_pos:
                last_assistant_pos = pos

        # If found assistant marker, only search after it
        if last_assistant_pos >= 0:
            search_text = text[last_assistant_pos:]
        else:
            # Fallback: search entire text
            search_text = text

        pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(pattern, search_text, re.DOTALL)
        if not matches:
            return None

        return matches[-1].strip()  # Return last match in assistant response
    boxed_not_found = 0
    no_answer = 0
    false_answer = 0

    batch_items = []
    valid_indices = []

    for idx, item in enumerate(tqdm(answers)):
        if item['category'] == category:
            extracted = extract_answer(item['output'])
            if extracted:
                boxed = last_boxed_only_string(extracted)
                if boxed:
                    cleaned_answer = remove_boxed(boxed)
                    if cleaned_answer:
                        batch_items.append((cleaned_answer, item['ground_truth'], item["question"]))
                        valid_indices.append(idx)

    if batch_items:
        batch_scores = math_llm.evaluation_batch(batch_items)

        for score in batch_scores:
            if score == 1.0:
                correct += 1
            else:
                false_answer += 1

    for item in answers:
        if item['category'] == category:
            extracted = extract_answer(item['output'])
            if not extracted:
                no_answer += 1
            elif not last_boxed_only_string(extracted):
                boxed_not_found += 1
            elif not remove_boxed(last_boxed_only_string(extracted)):
                boxed_not_found += 1

    total = no_answer + boxed_not_found + correct + false_answer
    acc = correct / total if total > 0 else 0.0
    return no_answer, boxed_not_found, correct, false_answer, acc

def evaluation_metrics_gaia_level(answers, level, math_llm: MathLLM | None = None):
    correct = 0
    math_llm = math_llm or MathLLM()
    def extract_answer(text: str):
        '\n        Extract the <answer>...</answer> block from the LAST assistant response only.\n        This avoids extracting example answers from the prompt.\n\n        Args:\n            text: The text to extract answer from (may include prompt + response)\n\n        Returns:\n            Optional[str]: The extracted answer or None if no match\n        '
        text = text.strip()

        # Find the last occurrence of "assistant" marker to isolate actual model response
        # This handles cases where prompt contains example <answer> tags
        assistant_markers = ['assistant\n', 'Assistant\n', 'assistant:', 'Assistant:']
        last_assistant_pos = -1

        for marker in assistant_markers:
            pos = text.rfind(marker)
            if pos > last_assistant_pos:
                last_assistant_pos = pos

        # If found assistant marker, only search after it
        if last_assistant_pos >= 0:
            search_text = text[last_assistant_pos:]
        else:
            # Fallback: search entire text
            search_text = text

        pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(pattern, search_text, re.DOTALL)
        if not matches:
            return None

        return matches[-1].strip()  # Return last match in assistant response
    boxed_not_found = 0
    no_answer = 0
    false_answer = 0

    batch_items = []

    for item in tqdm(answers):
        if item['Level'] == level:
            extracted = extract_answer(item['output'])
            if extracted:
                boxed = last_boxed_only_string(extracted)
                if boxed:
                    cleaned_answer = remove_boxed(boxed)
                    if cleaned_answer:
                        batch_items.append((cleaned_answer, item['ground_truth'], item["question"]))

    if batch_items:
        batch_scores = math_llm.evaluation_batch(batch_items)

        for idx, score in enumerate(batch_scores):
            cleaned_answer, ground_truth, _ = batch_items[idx]
            if score == 1.0:
                correct += get_f1_score(cleaned_answer, ground_truth)
            else:
                false_answer += 1

    for item in answers:
        if item['Level'] == level:
            extracted = extract_answer(item['output'])
            if not extracted:
                no_answer += 1
            elif not last_boxed_only_string(extracted):
                boxed_not_found += 1
            elif not remove_boxed(last_boxed_only_string(extracted)):
                boxed_not_found += 1

    total = no_answer + boxed_not_found + correct + false_answer
    acc = correct / total if total > 0 else 0.0
    return no_answer, boxed_not_found, correct, false_answer, acc

def evaluation_metrics_webwalker_level(answers, level, math_llm: MathLLM | None = None):
    correct = 0
    math_llm = math_llm or MathLLM()
    def extract_answer(text: str):
        '\n        Extract the <answer>...</answer> block from the LAST assistant response only.\n        This avoids extracting example answers from the prompt.\n\n        Args:\n            text: The text to extract answer from (may include prompt + response)\n\n        Returns:\n            Optional[str]: The extracted answer or None if no match\n        '
        text = text.strip()

        # Find the last occurrence of "assistant" marker to isolate actual model response
        # This handles cases where prompt contains example <answer> tags
        assistant_markers = ['assistant\n', 'Assistant\n', 'assistant:', 'Assistant:']
        last_assistant_pos = -1

        for marker in assistant_markers:
            pos = text.rfind(marker)
            if pos > last_assistant_pos:
                last_assistant_pos = pos

        # If found assistant marker, only search after it
        if last_assistant_pos >= 0:
            search_text = text[last_assistant_pos:]
        else:
            # Fallback: search entire text
            search_text = text

        pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(pattern, search_text, re.DOTALL)
        if not matches:
            return None

        return matches[-1].strip()  # Return last match in assistant response
    boxed_not_found = 0
    no_answer = 0
    false_answer = 0

    batch_items = []

    for item in tqdm(answers):
        if item['difficulty_level'] == level:
            extracted = extract_answer(item['output'])
            if extracted:
                boxed = last_boxed_only_string(extracted)
                if boxed:
                    cleaned_answer = remove_boxed(boxed)
                    if cleaned_answer:
                        batch_items.append((cleaned_answer, item['ground_truth'], item["question"]))

    if batch_items:
        batch_scores = math_llm.evaluation_batch(batch_items)

        for idx, score in enumerate(batch_scores):
            cleaned_answer, ground_truth, _ = batch_items[idx]
            if score == 1.0:
                correct += get_f1_score(cleaned_answer, ground_truth)
            else:
                false_answer += 1

    for item in answers:
        if item['difficulty_level'] == level:
            extracted = extract_answer(item['output'])
            if not extracted:
                no_answer += 1
            elif not last_boxed_only_string(extracted):
                boxed_not_found += 1
            elif not remove_boxed(last_boxed_only_string(extracted)):
                boxed_not_found += 1

    total = no_answer + boxed_not_found + correct + false_answer
    acc = correct / total if total > 0 else 0.0
    return no_answer, boxed_not_found, correct, false_answer, acc

def _build_dataset_map() -> Dict[str, Dict]:
    '\u96c6\u4e2d\u5f0f\u6570\u636e\u96c6\u914d\u7f6e\uff0c\u4fbf\u4e8e CLI / \u4ee3\u7801\u8c03\u7528\u3002'
    eval_root = os.environ.get("RL_FACTORY_EVAL_DATA_ROOT", "data/eval")
    arpo_root = os.environ.get("RL_FACTORY_ARPO_EVAL_ROOT", "data/arpo/evaluation/data")
    return {
        "HLE": {"loader": f"{eval_root}/hle", "split": "train", "processor": process_item, "prepare_fn": main_HLE},
        "GAIA": {"loader": f"{eval_root}/gaia", "split": "train", "processor": process_item_gaia, "prepare_fn": main_gaia},
        "WebWalker": {"loader": f"{eval_root}/web_walker/main-00000-of-00001.jsonl", "split": "train", "processor": process_item_webwalker, "prepare_fn": main_webwalker},
        "XBench": {"loader": f"{eval_root}/xbench", "split": "test", "processor": process_item_xbench, "prepare_fn": main_xbench},
        "2wiki": {"loader": f"{arpo_root}/2wiki/test.jsonl", "split": "train", "processor": process_item_qa, "prepare_fn": main_qa},
        "bamboogle": {"loader": f"{arpo_root}/bamboogle/test.jsonl", "split": "train", "processor": process_item_qa, "prepare_fn": main_qa},
        "hotpotqa": {"loader": f"{arpo_root}/hotpotqa/test.jsonl", "split": "train", "processor": process_item_qa, "prepare_fn": main_qa},
        "musique": {"loader": f"{arpo_root}/musique/test.jsonl", "split": "train", "processor": process_item_qa, "prepare_fn": main_qa},
        "webwalker-search": {"loader": f"{arpo_root}/webwalker/test.jsonl", "split": "train", "processor": process_item_qa, "prepare_fn": main_qa},
        "gsm8k": {"loader": f"{arpo_root}/gsm8k/test.jsonl", "split": "train", "processor": process_item_qa, "prepare_fn": main_qa},
        "math500": {"loader": f"{arpo_root}/math500/test.jsonl", "split": "train", "processor": process_item_qa, "prepare_fn": main_qa},
        "aime24": {"loader": f"{arpo_root}/aime24/test.jsonl", "split": "train", "processor": process_item_qa, "prepare_fn": main_qa},
        "aime25": {"loader": f"{arpo_root}/aime25/test.jsonl", "split": "train", "processor": process_item_qa, "prepare_fn": main_qa},
    }

def prepare_answer_file(dataset_name: str, answer_json_path: str, output_dir: str, dataset_root_override: str | None = None):
    '\u5c06\u6a21\u578b\u7b54\u6848\u4e0e\u6570\u636e\u96c6\u5bf9\u9f50\uff0c\u751f\u6210\u5e26 ground_truth \u7684\u7b54\u6848 JSON\u3002'
    dataset_map = _build_dataset_map()
    if dataset_name not in dataset_map:
        raise ValueError(f"\u4e0d\u652f\u6301\u7684\u6570\u636e\u96c6: {dataset_name}")
    cfg = dataset_map[dataset_name]

    loader_path = dataset_root_override or cfg["loader"]
    split = cfg["split"]
    prepare_fn = cfg["prepare_fn"]

    if dataset_name in ["HLE", "GAIA", "XBench"]:
        input_data = load_dataset(loader_path)[split]
    elif dataset_name == "WebWalker":
        input_data = load_dataset("json", data_files=loader_path)["train"]
    else:
        input_data = load_dataset("json", data_files=loader_path)[split]

    with open(answer_json_path, "r", encoding="utf-8") as f:
        answer_json = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    return prepare_fn(input_data, answer_json, output_dir) if prepare_fn != main_qa else prepare_fn(input_data, answer_json, output_dir, dataset_name)

def evaluate_dataset(dataset_name: str, prepared_answer_path: str, model_path: str | None, output_dir: str, judge: MathLLM | None = None):
    '\u5bf9\u9f50\u540e\u7684\u7b54\u6848\u6587\u4ef6\u8fdb\u884c\u8bc4\u4f30\uff0c\u5e76\u8f93\u51fa\u62a5\u544a JSON\u3002\n\n    Args:\n        dataset_name: \u6570\u636e\u96c6\u540d\u79f0\n        prepared_answer_path: \u51c6\u5907\u597d\u7684\u7b54\u6848\u6587\u4ef6\u8def\u5f84\n        model_path: \u6a21\u578b\u8def\u5f84\uff08\u5f53judge\u4e3aNone\u65f6\u4f7f\u7528\uff09\n        output_dir: \u8f93\u51fa\u76ee\u5f55\n        judge: \u590d\u7528\u7684judge\u5b9e\u4f8b\uff0c\u5982\u679c\u4e3aNone\u5219\u521b\u5efa\u65b0\u7684\uff08\u907f\u514d\u91cd\u590d\u52a0\u8f7d\u6a21\u578b\uff09\n    '
    dataset_map = _build_dataset_map()
    if dataset_name not in dataset_map:
        raise ValueError(f"\u4e0d\u652f\u6301\u7684\u6570\u636e\u96c6: {dataset_name}")

    with open(prepared_answer_path, "r", encoding="utf-8") as f:
        answers = json.load(f)

    if judge is None:
        judge = MathLLM(model_path=model_path) if model_path else MathLLM()

    metric_keys = ["no_answer", "boxed_not_found", "correct", "false_answer", "acc"]

    if dataset_name == "HLE":
        results = {}
        for category in ["Math", "Computer Science/AI", "Physics"]:
            results[category] = dict(zip(metric_keys, evaluation_metrics_HLE_key(answers, category, judge)))
    elif dataset_name == "GAIA":
        results = {f"Level {lvl}": dict(zip(metric_keys, evaluation_metrics_gaia_level(answers, lvl, judge))) for lvl in [1, 2, 3]}
    elif dataset_name == "WebWalker":
        levels = sorted(list(set(item.get("difficulty_level") for item in answers)))
        results = {f"Difficulty Level {lvl}": dict(zip(metric_keys, evaluation_metrics_webwalker_level(answers, lvl, judge))) for lvl in levels}
    else:
        results = {"overall": dict(zip(metric_keys, evaluation_metrics_HLE(answers, judge)))}

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"evaluation_{dataset_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"[evaluation] \u7ed3\u679c\u5199\u5165 {out_path}")
    return out_path

def cli():
    parser = argparse.ArgumentParser(description='\u8bc4\u4f30\u5de5\u5177\uff1a\u6307\u5b9a\u6570\u636e\u96c6\u3001\u7b54\u6848\u6587\u4ef6\u3001\u8f93\u51fa\u76ee\u5f55\uff0c\u751f\u6210\u62a5\u544a\u3002')
    parser.add_argument("--dataset", required=True, help='\u6570\u636e\u96c6\u540d\u79f0\uff0c\u4f8b\u5982 HLE/GAIA/WebWalker/XBench/... ')
    parser.add_argument("--answers", required=True, help='\u6a21\u578b\u751f\u6210\u7684\u7b54\u6848 JSON \u8def\u5f84\uff08\u672a\u5bf9\u9f50 ground_truth\uff09')
    parser.add_argument("--output-dir", required=True, help='\u8f93\u51fa\u76ee\u5f55\uff0c\u5305\u542b\u5bf9\u9f50\u7b54\u6848\u4e0e\u8bc4\u4f30\u62a5\u544a')
    parser.add_argument("--model-path", default=None, help='vLLM \u4f7f\u7528\u7684\u5f00\u6e90\u6a21\u578b\u6743\u91cd\uff08HF \u540d\u79f0\u6216\u672c\u5730\u8def\u5f84\uff09')
    parser.add_argument("--dataset-path", default=None, help='\u53ef\u9009\uff1a\u8986\u76d6\u9ed8\u8ba4\u6570\u636e\u96c6\u52a0\u8f7d\u8def\u5f84')
    args = parser.parse_args()

    prepared = prepare_answer_file(
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

if __name__ == "__main__":
    cli()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
