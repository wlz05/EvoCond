# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Measure calibration and selective risk."""
from math import isfinite
def check(scores: list[float], labels: list[int]) -> None:
    if len(scores) != len(labels) or not scores:
        raise ValueError("evaluation samples are invalid")
    if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in scores):
        raise ValueError("evaluation scores are invalid")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("evaluation labels are invalid")
def brier(scores: list[float], labels: list[int]) -> float:
    check(scores, labels)
    return sum((value - label) ** 2 for value, label in zip(scores, labels)) / len(scores)
def ece(scores: list[float], labels: list[int], bins: int = 10) -> float:
    check(scores, labels)
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bin count is invalid")
    total = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        items = [(score, label) for score, label in zip(scores, labels) if low <= score < high or index == bins - 1 and score == high]
        if items:
            confidence = sum(item[0] for item in items) / len(items)
            accuracy = sum(item[1] for item in items) / len(items)
            total += len(items) / len(scores) * abs(confidence - accuracy)
    return total
def aurc(scores: list[float], labels: list[int]) -> float:
    check(scores, labels)
    order = sorted(range(len(scores)), key=lambda pos: abs(scores[pos] - 0.5), reverse=True)
    errors = 0
    area = 0.0
    for rank, pos in enumerate(order, 1):
        errors += int((scores[pos] >= 0.5) != bool(labels[pos]))
        area += errors / rank
    return area / len(order)
def report(scores: list[float], labels: list[int], bins: int = 10) -> dict:
    return {"brier": brier(scores, labels), "ece": ece(scores, labels, bins), "aurc": aurc(scores, labels)}

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
