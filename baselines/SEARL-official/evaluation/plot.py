# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_log_line(line):
    '\n    \u4ece\u4e00\u884c log \u4e2d\u63d0\u53d6 key:value \u5bf9\n    \u8fd4\u56de dict\n    '

    line = re.sub(r'\x1b\[[0-9;]*m', '', line)
    parts = line.split(" - ")
    metrics = {}
    for part in parts:
        if ":" in part:
            key, val = part.split(":", 1)
            try:
                val = float(val.strip())
            except ValueError:
                continue
            metrics[key.strip()] = val
    return metrics

def collect_metrics(log_dir):
    '\n    \u904d\u5386 log \u6587\u4ef6\u5939\uff0c\u6536\u96c6\u6240\u6709\u6307\u6807\n    '
    all_metrics = defaultdict(list)
    file_list = sorted(
        [f for f in os.listdir(log_dir) if f.startswith("chunk_") and f.endswith(".log")],
        key=lambda x: int(re.search(r"chunk_(\d+)\.log", x).group(1))
    )

    for fname in file_list:
        path = os.path.join(log_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            metrics = parse_log_line(first_line)
            for k, v in metrics.items():
                all_metrics[k].append(v)
    return all_metrics

def plot_metrics(all_metrics, output_dir):
    '\n    \u7ed8\u5236\u6bcf\u4e2a\u6307\u6807\u7684\u66f2\u7ebf\u56fe\n    '
    os.makedirs(output_dir, exist_ok=True)
    for metric, values in all_metrics.items():
        plt.figure(figsize=(10, 5))
        plt.plot(values, marker="o")
        plt.title(metric)
        plt.xlabel("Log Index (chunk_i)")
        plt.ylabel("Value")
        plt.grid(True)
        safe_name = metric.replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(output_dir, f"{safe_name}.png"))
        plt.close()

if __name__ == "__main__":
    log_dir = os.environ.get("RL_FACTORY_LOG_CHUNKS_DIR", "logs/chunks")
    output_dir = os.environ.get("RL_FACTORY_LOG_CURVES_DIR", "logs/curves")

    all_metrics = collect_metrics(log_dir)
    plot_metrics(all_metrics, output_dir)

    print(f"\u7ed8\u56fe\u5b8c\u6210\uff0c\u5171\u751f\u6210 {len(all_metrics)} \u5f20\u56fe\uff0c\u4fdd\u5b58\u5728 {output_dir}")

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
