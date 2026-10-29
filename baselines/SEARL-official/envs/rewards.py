# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import re
import sys
import string
from typing import Union, List, Dict, Any, Optional, Tuple
from collections import Counter
from transformers import AutoTokenizer
import sympy
import math

def verify_math_answer(answer_float: float,
                       ground_truth_expr: str,
                       rel_tol: float = 1e-4) -> bool:
    'Compare an answer with numeric, LaTeX, percentage, interval or alternative targets; preserve interval endpoints and container target rules.'

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

def _parse_number_maybe_percent(s: str):
    'Parse a number or percentage as float; return None on failure.'
    s1 = s.strip().strip("$")

    m = re.fullmatch(r"\s*([-+]?\d+(?:\.\d+)?)\s*(\\?%)\s*", s1)
    if m:
        return float(m.group(1)) / 100.0

    try:
        return float(s1.replace(",", ""))
    except Exception:
        return None

def _in_interval(x: float, lo: float, hi: float, left_closed: bool, right_closed: bool, rel_tol: float) -> bool:

    pad_lo = rel_tol * max(1.0, abs(lo))
    pad_hi = rel_tol * max(1.0, abs(hi))
    left_ok = (x + pad_lo) >= lo if left_closed else (x - pad_lo) > lo
    right_ok = (x - pad_hi) <= hi if right_closed else (x + pad_hi) < hi
    return left_ok and right_ok

def _to_float(expr: str) -> float:
    'Convert an expression using float, parse_latex, then normalized sympify.'
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

def _preprocess_latex_basic(s: str) -> str:
    'Normalize common LaTeX operators, delimiters and braces.'
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

def validate_format(text: str) -> Tuple[bool, str]:
    '\n    Validate if the text follows the required format with paired tags.\n\n    Args:\n        text: The text to validate\n\n    Returns:\n        tuple: (is_valid, reason)\n    '
    # Check if <thinking></thinking>, <answer></answer> is paired
    if text.count('<tool_call>') != text.count('</tool_call>'):
        return False, "<tool_call> </tool_call> not paired"

    if text.count('<subtask>') != text.count('</subtask>'):
        return False, "<subtask> </subtask> not paired"
    else:
        if text.count('<subtask>') == 0 or text.count('</subtask>') == 0:
            return False, "<subtask> or </subtask> not found"

    if text.count('<thinking>') == 0 or text.count('</thinking>') == 0:
       return False, "<thinking> or </thinking> not found"
    else:
        if text.count('<thinking>') == 0 or text.count('</thinking>') == 0:
            return False, "<thinking> or </thinking> not found"

    # Check if \boxed{} is in the answer
    answer_start = text.rfind('<answer>')
    answer_end = text.rfind('</answer>')

    if answer_start == -1 or answer_end == -1:
        return False, "<answer> or </answer> not found"

    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"

    answer_content = text[answer_start:answer_end]

    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\boxed{} format"

    return True, "format is correct"

def validate_format_baseline(text: str) -> Tuple[bool, str]:
    '\n    Validate if the text follows the required format with paired tags.\n\n    Args:\n        text: The text to validate\n\n    Returns:\n        tuple: (is_valid, reason)\n    '
    # Check if <thinking></thinking>, <answer></answer> is paired
    if text.count('<tool_call>') != text.count('</tool_call>'):
        return False, "<tool_call> </tool_call> not paired"

    # if text.count('<answer>') <= 1 or text.count('</answer>') <= 1:
    #     return False, "<answer> or </answer> not found"

    # if text.count('<python>') == 0 or text.count('</python>') == 0:
    #    return False, "<python> or </python> not found"

    # Check if \boxed{} is in the answer
    answer_start = text.rfind('<answer>')
    answer_end = text.rfind('</answer>')

    if answer_start == -1 or answer_end == -1:
        return False, "<answer> or </answer> not found"

    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"

    answer_content = text[answer_start:answer_end]

    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\boxed{} format"

    return True, "format is correct"

def extract_answer(text: str) -> Optional[str]:
    '\n    Extract the last <answer>...</answer> block from the text.\n\n    Args:\n        text: The text to extract answer from\n\n    Returns:\n        Optional[str]: The extracted answer or None if no match\n    '
    text = text.strip()

    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None

    return matches[-1].strip()

def remove_boxed(s: str) -> str:
    '\n    Remove the LaTeX \x08oxed{} wrapper from the string.\n\n    Args:\n        s: String potentially containing \x08oxed{}\n\n    Returns:\n        str: String with \x08oxed{} removed\n    '
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]

def last_boxed_only_string(string: str) -> Optional[str]:
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

def get_f1_score(prediction: str, ground_truths: Union[str, List[str]]) -> float:
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

        # Fallback to string normalization check
        if normalize_answer(prediction) == normalize_answer(ground_truth):
            return 1.0

    final_metric = {"f1": 0.0, "precision": 0.0, "recall": 0.0}

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

    return float(final_metric['f1'])

def compute_score_baseline(solution_str: str, ground_truth: Any, extra_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    '\n    Compute reward score for a solution based on the ground truth.\n\n    Args:\n        data_source: The data source identifier\n        solution_str: The solution string to evaluate\n        ground_truth: The ground truth answer(s)\n        extra_info: Optional additional information\n\n    Returns:\n        Dict[str, Any]: A dictionary containing the score and detailed metrics for wandb tracking\n    '

    result = {
        "score": 0,
        "reason": "",
        "answer": "",
        "f1_score": 0,

        "format_reward": 0.0,
        "tool_penalty": 0.0,
        "answer_correctness": 0.0,
    }

    response = solution_str
    valid_template, reason = validate_format_baseline(response)

    if not valid_template:
        # print(f"--------------------------------bad format: {reason}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        # result["score"] = -1
        # result["reason"] = f"bad format: {reason}"
        # return result["score"]
        format_reward = -0.2
    else:
        format_reward = 0.2

    result["format_reward"] = format_reward

    if extra_info is not None and "tokenizer" in extra_info and extra_info["tokenizer"].eos_token and response.endswith(extra_info["tokenizer"].eos_token):
        response = response[:-len(extra_info["tokenizer"].eos_token)]

    answer_part = extract_answer(response)  # todo_sxh <answer> </answer>
    if answer_part is None or answer_part == r'\boxed{[2, 1]}' or answer_part == r'\boxed{...}':
        print(f"--------------------------------cannot extract answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1
        result["reason"] = "cannot extract answer"
        result["answer_correctness"] = -1.0
        return result

    try:
        answer = remove_boxed(last_boxed_only_string(answer_part))
        result["answer"] = answer
    except Exception as e:
        print(f"--------------------------------find box error: {e}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1
        result["reason"] = f"find box error: {e}"
        result["answer_correctness"] = -1.0
        return result

    f1_score = get_f1_score(answer, ground_truth)
    result["f1_score"] = f1_score
    print(f"f1_score: {f1_score}, answer: {answer}, ground_truth: {ground_truth}")

    # Bonus for using multiple tools correctly
    # if f1_score > 0 and "</search>" in response and "</python>" in response:
    if f1_score > 0:
        print(f"--------------------------------correct answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = f1_score + format_reward
        result["reason"] = f"correct answer, get f1 score: {f1_score}"
        result["answer_correctness"] = f1_score
    else:
        print(f"--------------------------------wrong answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1 + format_reward
        result["reason"] = f"wrong answer but good format: {answer}"
        result["answer_correctness"] = -1.0

    return result

def compute_score(solution_str: str, ground_truth: Any, extra_info: Optional[Dict[str, Any]] = None, mcp_worthiness: Optional[float] = None, tool_call_count: Optional[int] = None) -> Dict[str, Any]:
    '\n    Compute reward score for a solution based on the ground truth.\n\n    Args:\n        data_source: The data source identifier\n        solution_str: The solution string to evaluate\n        ground_truth: The ground truth answer(s)\n        extra_info: Optional additional information\n        mcp_worthiness: Optional MCP worthiness\n        tool_call_count: Optional tool call count\n    Returns:\n        Dict[str, Any]: A dictionary containing the score and detailed metrics for wandb tracking\n    '

    expected_tool_calls = None
    if tool_call_count is not None:
        try:
            expected_tool_calls = int(tool_call_count)
        except (TypeError, ValueError):
            expected_tool_calls = None

    result = {
        "score": 0,
        "reason": "",
        "answer": "",
        "f1_score": 0,

        "format_reward": 0.0,
        "tool_penalty": 0.0,
        "answer_correctness": 0.0,
        "num_tool_calls": 0,
        "expected_tool_calls": expected_tool_calls if expected_tool_calls is not None else 0,
    }

    response = solution_str
    penalty_per_extra = extra_info.get("tool_call_penalty", 0.05) if extra_info else 0.05
    tool_calls = re.findall(r"<tool_call>([\s\S]*?)</tool_call>", response)
    num_tool_calls = len(tool_calls) - 3 # remove the prompt <tool_call>
    result["num_tool_calls"] = num_tool_calls
    # assert mcp_worthiness is not None

    tool_penalty = 0.0
    if expected_tool_calls is None:
        # Keep reward neutral when expectation is missing/invalid.
        tool_penalty = 0.0
    elif abs(num_tool_calls - expected_tool_calls) > 1:
        tool_penalty = -penalty_per_extra * max(abs(num_tool_calls - expected_tool_calls) - 1, 0)
    else:
        tool_penalty = 0.1
    result["tool_penalty"] = tool_penalty
    print(f"tool_penalty: {tool_penalty}, num_tool_calls: {num_tool_calls}, tool_call_count: {expected_tool_calls}")
    valid_template, reason = validate_format(response)

    if not valid_template:
        format_reward = -0.1
    else:
        format_reward = 0.1
    result["format_reward"] = format_reward

    if extra_info is not None and "tokenizer" in extra_info and extra_info["tokenizer"].eos_token and response.endswith(extra_info["tokenizer"].eos_token):
        response = response[:-len(extra_info["tokenizer"].eos_token)]

    answer_part = extract_answer(response)  # todo_sxh <answer> </answer>
    if answer_part is None or answer_part == r'\boxed{[2, 1]}' or answer_part == r'\boxed{...}':
        print(f"--------------------------------cannot extract answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1 + format_reward + tool_penalty
        result["reason"] = "cannot extract answer"
        result["answer_correctness"] = -1.0
        return result

    try:
        answer = remove_boxed(last_boxed_only_string(answer_part))
        result["answer"] = answer
    except Exception as e:
        print(f"--------------------------------find box error: {e}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1 + format_reward + tool_penalty
        result["reason"] = f"find box error: {e}"
        result["answer_correctness"] = -1.0
        return result

    f1_score = get_f1_score(answer, ground_truth)
    result["f1_score"] = f1_score
    print(f"f1_score: {f1_score}, answer: {answer}, ground_truth: {ground_truth}")

    # Bonus for using multiple tools correctly
    # if f1_score > 0 and "</search>" in response and "</python>" in response:
    if f1_score > 0:
        print(f"--------------------------------correct answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = f1_score * 1.5 + format_reward + tool_penalty
        result["reason"] = f"correct answer, get f1 score: {f1_score}"
        result["answer_correctness"] = f1_score
    else:
        print(f"--------------------------------wrong answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1 + format_reward + tool_penalty
        result["reason"] = f"wrong answer but good format: {answer}"
        result["answer_correctness"] = -1.0

    return result

if __name__ == "__main__":
    solution_str = "\frac{14}"
    ground_truth = "7"
    extra_info = None
    print(compute_score_baseline(solution_str, ground_truth))

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
