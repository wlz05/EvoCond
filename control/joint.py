# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"EvoCond's evidence-aware controller for the SEARL/EVOLT tool protocol."
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence
from control.bridge import EvoltBridge, _cosine, _vector
@dataclass(frozen=True)
class Evidence:
    tool: str
    result: str
    version: int
    source: str
    verified_execution: bool
@dataclass(frozen=True)
class Decision:
    scale: str
    action: str
    available: tuple[str, ...]
    suggested: str | None
    memory_used: bool
    reason: str
    valid_evidence: tuple[Evidence, ...]
    note: str
def evidence(history: Sequence[Mapping[str, Any]]) -> tuple[int, tuple[Evidence, ...]]:
    'Version evidence by user update, never by tool-output type alone.'
    version = -1
    rows: list[Evidence] = []
    for message in history:
        role = message.get("role")
        if role == "user":
            version += 1
        elif role == "tool" and version >= 0:
            name = str(message.get("name") or "").casefold()
            result = str(message.get("content") or "")
            valid = result.startswith("Tool call success") if name == "python" else result.startswith("Following are the search results") if name == "search" else False
            rows.append(Evidence(name, result, version, name, valid))
    return max(0, version), tuple(row for row in rows if row.version == version and row.verified_execution)
class ToolMemory:
    'Rank exact-aligned tools with frozen EVOLT semantics and outcome counts.'
    def __init__(self, state: Mapping[str, Any], min_tasks: int = 3) -> None:
        if min_tasks < 1:
            raise ValueError("min_tasks must be positive")
        self.bridge = EvoltBridge.from_state(state, min_tasks=min_tasks)
        self.min_tasks = min_tasks
        graph = state.get("graph", state)
        if not isinstance(graph, Mapping):
            raise ValueError("EVOLT graph is missing")
        self.raw_tools = {str(row.get("key")): row for row in graph.get("tools", []) if isinstance(row, Mapping)}
        self.sequences = tuple(row for row in graph.get("edges", []) if isinstance(row, Mapping) and row.get("kind") == "precedes")
    def rank(self, query: str, available: Sequence[str], previous: str = "") -> tuple[str | None, str]:
        bridge = self.bridge
        if len(bridge.tasks) < self.min_tasks or not query.strip() or bridge.dim == 0:
            return None, "cold_memory"
        aligned: list[tuple[float, str]] = []
        qvec = _vector(query, bridge.dim)
        previous_key = bridge.by_name.get(previous)
        for name in available:
            key = bridge.by_name.get(name)
            if key is None:
                return None, "unaligned_tool"
            raw = self.raw_tools.get(key, {})
            trials = raw.get("trials", 0)
            quality = raw.get("quality", 0.0)
            if isinstance(trials, bool) or not isinstance(trials, int) or trials < self.min_tasks:
                return None, "insufficient_tool_outcomes"
            if not isinstance(quality, (int, float)) or not isfinite(float(quality)):
                return None, "invalid_tool_outcomes"
            semantic = (1.0 + _cosine(qvec, bridge.tools[key].vector)) / 2.0
            reliability = max(0.0, min(1.0, (float(quality) + 1.0) / 2.0))
            sequence = 0.0
            if previous_key:
                for edge in self.sequences:
                    if edge.get("source") == previous_key and edge.get("target") == key and isinstance(edge.get("tasks"), list) and len(set(edge["tasks"])) >= self.min_tasks:
                        positive, negative = float(edge.get("positive", 0.0)), float(edge.get("negative", 0.0))
                        if all(isfinite(value) and value >= 0 for value in (positive, negative)):
                            sequence = max(sequence, (positive - negative) / (positive + negative + 2.0))
            aligned.append((0.6 * semantic + 0.25 * reliability + 0.15 * sequence, name))
        if not aligned:
            return None, "no_available_tool"
        aligned.sort(key=lambda item: (-item[0], item[1]))
        if len(aligned) > 1 and aligned[0][0] - aligned[1][0] < 0.05:
            return None, "ambiguous_memory"
        return aligned[0][1], "frozen_memory_prior"
class Controller:
    'LL/GL/GG recovery with a cost-free abstaining memory suggestion.'
    def __init__(self, memory: ToolMemory | None = None) -> None:
        self.memory = memory
    def decide(self, query: str, history: Sequence[Mapping[str, Any]], tools: Sequence[str]) -> Decision:
        available = tuple(dict.fromkeys(str(name).casefold() for name in tools))
        version, valid = evidence(history)
        user_count = sum(message.get("role") == "user" for message in history)
        last_user = max((index for index, message in enumerate(history) if message.get("role") == "user"), default=-1)
        replies = [message for message in history[last_user + 1:] if message.get("role") == "tool"]
        failures = 0
        for message in reversed(replies):
            name = str(message.get("name") or "").casefold()
            content = str(message.get("content") or "")
            success = content.startswith("Tool call success") if name == "python" else content.startswith("Following are the search results") if name == "search" else False
            if success:
                break
            failures += 1
        if user_count > 1 and not replies:
            # A newer user request invalidates prior derived evidence. The runner retains the conversation.
            scale, action, reason = "gg", "replan", "user_goal_version_changed"
            note = "EvoCond GG: the latest user request supersedes earlier tool-derived claims. Replan from the latest request; reuse only explicitly unchanged, independently supported facts."
        elif failures >= 2:
            scale, action, reason = "gl", "replan", "repeated_tool_failure"
            note = "EvoCond GL: repeated tool failures. Change the subtask route or answer from verified current evidence; do not repeat identical failing calls."
        elif failures:
            scale, action, reason = "ll", "repair", "tool_failure"
            note = "EvoCond LL: inspect the failed call and repair its arguments, or choose another useful tool. Never invent a successful result."
        else:
            scale, action, reason = "ll", "continue", "no_contract_violation"
            note = "Use only current-request tool observations as evidence. A tool response proves execution, not final-answer correctness."
        suggested = None
        memory_used = False
        if self.memory is not None and available:
            previous = next((row.tool for row in reversed(valid)), "")
            suggested, memory_reason = self.memory.rank(query, available, previous)
            memory_used = suggested is not None
            if memory_used:
                note += f" Historical tool memory softly favors {suggested}; it is optional, may be stale, and does not override current evidence or tool availability."
            else:
                reason += f";{memory_reason}"
        return Decision(scale, action, available, suggested, memory_used, reason, valid, note)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
