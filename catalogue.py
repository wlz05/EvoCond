# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

# Declare benchmark roles without downloading data or loading a model.
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
@dataclass(frozen=True)
class Benchmark:
    name: str
    source: str
    evaluation: str
    metric: str
    focus: str
    evaluation_only: bool = True
    score_kind: str = "binary"
BENCHMARKS = {
    "searl_qa": Benchmark("searl_qa", "https://github.com/circles-post/SEARL", "searl_native_judge", "answer_accuracy", "Existing SEARL QA protocol", False),
    "toolhop": Benchmark("toolhop", "https://huggingface.co/datasets/bytedance-research/ToolHop", "toolhop_native", "answer_accuracy", "Multi-hop tool dependencies and route transfer"),
    "toolsandbox": Benchmark("toolsandbox", "https://github.com/apple-aiml-research/ToolSandbox", "toolsandbox_milestones", "milestone_similarity", "State dependencies and goal changes", score_kind="continuous"),
    "bigcodebench": Benchmark("bigcodebench", "https://github.com/bigcode-project/bigcodebench", "bigcodebench_isolated_tests", "pass_at_1", "Executable tool construction and library composition"),
}
def benchmark(name: str) -> Benchmark:
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}; choose {', '.join(BENCHMARKS)}")
    return BENCHMARKS[name]
def pinned_revision(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.casefold() in {"main", "master", "head", "latest"}:
        raise ValueError("Declare a fixed source revision or immutable release")
    return value
def relative_path(value: str):
    from pathlib import Path, PureWindowsPath
    if not value or Path(value).is_absolute() or PureWindowsPath(value).drive or PureWindowsPath(value).root:
        raise ValueError("Use a relative filesystem path")
    return Path(value)
if __name__ == "__main__":
    print(json.dumps({"benchmarks": [asdict(item) for item in BENCHMARKS.values()], "notice": "Reserved integration contracts, not installed native evaluators. Preserve official splits; group prompt variants by task identity."}, indent=2))

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
