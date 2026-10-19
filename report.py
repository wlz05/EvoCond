# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Question-cluster paired accuracy intervals, including parser failures.'
from __future__ import annotations
from itertools import combinations
import math
import random
import re
def paired_summary(arms, *, seed=42, resamples=2000):
    if not isinstance(arms, dict) or len(arms) < 2:
        raise ValueError("At least two named arms are required")
    if type(seed) is not int or seed < 0 or type(resamples) is not int or resamples < 100:
        raise ValueError("Explicit bootstrap seed and at least 100 resamples are required")
    grouped, sizes = {}, None
    for arm, rows in arms.items():
        if not isinstance(arm, str) or not arm or not isinstance(rows, list) or not rows:
            raise ValueError("Each arm requires a nonempty score list")
        groups, seen = {}, set()
        for row in rows:
            key, sample = row.get("question_sha256"), row.get("sample_index")
            if (not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key) or
                    type(sample) is not int or sample < 0 or type(row.get("success")) is not bool):
                raise ValueError("Scores require question hash, sample index and boolean success")
            if (key, sample) in seen:
                raise ValueError("Duplicate question/sample identity")
            seen.add((key, sample))
            groups.setdefault(key, {})[sample] = int(row["success"])
        counts = {key: len(values) for key, values in groups.items()}
        if any(set(values) != set(range(len(values))) for values in groups.values()):
            raise ValueError("Sample indices must be complete within each question")
        if sizes is not None and counts != sizes:
            raise ValueError("Arms do not cover identical questions and sample counts")
        sizes = counts
        grouped[arm] = {key: sum(values.values()) / len(values) for key, values in groups.items()}
    keys = sorted(sizes)
    result = {"unit": "independent_question", "questions": len(keys), "samples": sum(sizes.values()),
              "bootstrap_seed": seed, "resamples": resamples, "confidence": 0.95,
              "arm_mean_question_accuracy": {arm: sum(scores.values())/len(keys) for arm, scores in grouped.items()},
              "pairs": {}, "warning": "Question-cluster percentile intervals; no run-seed uncertainty or multiplicity correction. "
              "Repeated samples are not independent questions. No causal or no-regression guarantee."}
    for baseline, candidate in combinations(sorted(grouped), 2):
        delta = [grouped[candidate][key]-grouped[baseline][key] for key in keys]
        rng = random.Random(seed)
        distribution = sorted(sum(rng.choice(delta) for _ in keys)/len(keys) for _ in range(resamples))
        interval = [distribution[math.floor(.025*(resamples-1))], distribution[math.ceil(.975*(resamples-1))]]
        result["pairs"][candidate+"_minus_"+baseline] = {
            "mean_accuracy_delta": sum(delta)/len(delta), "interval": interval,
            "higher_question_mean": sum(value > 0 for value in delta),
            "lower_question_mean": sum(value < 0 for value in delta),
            "tied_question_mean": sum(value == 0 for value in delta),
            "binary_paired_gain_loss": all(count == 1 for count in sizes.values())}
    return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
