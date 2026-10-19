# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Measure the baseline evaluation protocol."""
from collections import Counter
from fractions import Fraction
from math import isfinite
import re
import string
import unicodedata
from core.data import Record, identify
def block(text: object) -> str:
    """Extract the final answer block."""
    value = str(text or "").strip()
    positions = [value.casefold().rfind(marker) for marker in ("assistant\n", "assistant:")]
    if max(positions, default=-1) >= 0:
        value = value[max(positions):]
    matches = re.findall(r"<answer>(.*?)</answer>", value, flags=re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else ""
def boxed(text: str) -> str:
    """Extract the final balanced boxed value."""
    markers = ("\\boxed", "\\fbox")
    start, marker = max(((text.rfind(value), value) for value in markers), key=lambda item: item[0])
    if start < 0:
        return ""
    brace = text.find("{", start)
    if brace < 0:
        tail = text[start + len(marker):].strip()
        return tail.split("$")[0].strip()
    depth = 0
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1:pos].strip()
    return ""
def normalize(value: object) -> str:
    """Normalize a textual answer."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(char for char in text if char not in string.punctuation)
    return " ".join(text.split())
def references(value: object) -> list[object]:
    """Flatten common reference containers."""
    if isinstance(value, dict):
        for key in ("target", "answer", "value", "ground_truth"):
            if key in value:
                return references(value[key])
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]
def number(value: object) -> float | None:
    """Parse common numeric answer forms."""
    text = str(value or "").strip().strip("$").replace(",", "")
    if text.endswith("%"):
        try:
            return float(text[:-1]) / 100.0
        except ValueError:
            return None
    match = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", text)
    if match:
        try:
            return float(Fraction(match.group(1)) / Fraction(match.group(2)))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None
def integers(value: object) -> frozenset[int] | None:
    """Parse one comma-separated set of integers."""
    text = str(value or "").strip().strip("$")
    if text.startswith("\\{") and text.endswith("\\}"):
        text = text[2:-2]
    elif text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    parts = [part.strip() for part in text.split(",")]
    if len(parts) < 2 or any(not re.fullmatch(r"[+-]?\d+", part) for part in parts):
        return None
    return frozenset(int(part) for part in parts)
def local(prediction: str, reference: object, question: str = "") -> bool:
    """Provide a deterministic judge fallback."""
    unordered = bool(re.search(r"\ball(?:\s+such)?\s+integers\b", str(question or ""), flags=re.IGNORECASE))
    for value in references(reference):
        if unordered:
            left, right = integers(prediction), integers(value)
            if left is not None and right is not None and left == right:
                return True
        left = number(prediction)
        right = number(value)
        if left is not None and right is not None and abs(left - right) <= 1e-4 * max(1.0, abs(right)):
            return True
        if normalize(prediction) == normalize(value):
            return True
    return False
def prediction(rows: list[dict]) -> dict[str, dict]:
    """Index prediction rows by their stable sample key."""
    output = {}
    for pos, row in enumerate(rows):
        key = str(row.get("key", row.get("id", row.get("index", ""))))
        if not key:
            raise ValueError(f"prediction row {pos} lacks a key")
        if key in output:
            raise ValueError(f"prediction key is duplicated: {key}")
        output[key] = row
    return output
def measure(records: list[Record], values: dict[str, object], judge=None) -> dict:
    """Compute binary pass at one by dataset."""
    if not records:
        raise ValueError("evaluation records are empty")
    rows: dict[str, dict[str, int]] = {}
    modes = set()
    for item in records:
        row = rows.setdefault(item.dataset, {"samples": 0, "noanswer": 0, "nobox": 0, "correct": 0, "wrong": 0, "toolcalls": 0, "tooltasks": 0, "toolfailures": 0})
        row["samples"] += 1
        value = values.get(item.key, {})
        data = value if isinstance(value, dict) else {"output": value}
        text = data.get("output", data.get("prediction", data.get("answer", "")))
        calls = re.findall(r"<tool_call>.*?</tool_call>", str(text or ""), flags=re.DOTALL | re.IGNORECASE)
        responses = re.findall(r"<tool_response[^>]*>(.*?)</tool_response>", str(text or ""), flags=re.DOTALL | re.IGNORECASE)
        row["toolcalls"] += len(calls)
        row["tooltasks"] += int(bool(calls))
        row["toolfailures"] += sum(value.strip().casefold().startswith("error:") for value in responses)
        answer = block(text)
        if not answer:
            row["noanswer"] += 1
            continue
        final = boxed(answer)
        if not final:
            row["nobox"] += 1
            continue
        if "correct" in data:
            score = data["correct"]
            if isinstance(score, bool):
                correct = score
            elif isinstance(score, (int, float)) and isfinite(float(score)) and float(score) in (0.0, 1.0):
                correct = bool(score)
            else:
                raise ValueError("correct labels must be binary")
            modes.add("labels")
        elif judge is not None:
            score = judge(final, item.answer, item.question)
            if not isinstance(score, (bool, int, float)) or not isfinite(float(score)) or float(score) not in (0.0, 1.0):
                raise ValueError("judge scores must be binary")
            correct = float(score) == 1.0
            modes.add("judge")
        else:
            correct = local(final, item.answer, item.question)
            modes.add("local")
        row["correct" if correct else "wrong"] += 1
    total = {"samples": 0, "noanswer": 0, "nobox": 0, "correct": 0, "wrong": 0, "toolcalls": 0, "tooltasks": 0, "toolfailures": 0}
    for row in rows.values():
        row["pass@1"] = row["correct"] / row["samples"]
        row["accuracy"] = row["pass@1"]
        row["toolrate"] = row["tooltasks"] / row["samples"]
        row["failurerate"] = row["toolfailures"] / row["toolcalls"] if row["toolcalls"] else 0.0
        for key in total:
            total[key] += row[key]
    total["pass@1"] = total["correct"] / total["samples"]
    total["accuracy"] = total["pass@1"]
    total["toolrate"] = total["tooltasks"] / total["samples"]
    total["failurerate"] = total["toolfailures"] / total["toolcalls"] if total["toolcalls"] else 0.0
    macro = sum(row["pass@1"] for row in rows.values()) / len(rows)
    mode = next(iter(modes)) if len(modes) == 1 else "mixed" if modes else "none"
    return {"mode": mode, "datasets": rows, "overall": total, "macro": macro}
def ranks(scores: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute tied average ranks across datasets."""
    if not scores:
        raise ValueError("score table is empty")
    methods = list(scores)
    datasets = {identify(name) for row in scores.values() for name in row}
    if "" in datasets or not datasets or any({identify(name) for name in scores[method]} != datasets for method in methods):
        raise ValueError("score table datasets differ")
    totals = Counter({method: 0.0 for method in methods})
    for dataset in datasets:
        values = {method: next(float(score) for name, score in scores[method].items() if identify(name) == dataset) for method in methods}
        if any(not isfinite(value) for value in values.values()):
            raise ValueError("scores must be finite")
        ordered = sorted(set(values.values()), reverse=True)
        for value in ordered:
            tied = [method for method in methods if values[method] == value]
            ranks = [1 + sum(other > value for other in values.values()) + pos for pos in range(len(tied))]
            rank = sum(ranks) / len(ranks)
            for method in tied:
                totals[method] += rank
    return {method: totals[method] / len(datasets) for method in methods}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
