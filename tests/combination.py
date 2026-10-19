# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"Offline tests for EvoCond's SEARL-resource evaluation boundary."
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from control.bridge import _vector
from control.joint import Controller, ToolMemory, evidence
from control.tools import Tools
from compare import compare
from eval import Record, answer, calls, corpus, local_correct, solve
def snapshot() -> dict:
    tasks = [f"train-{index}" for index in range(4)]
    return {"graph": {"tasks": tasks, "tools": [
        {"key": "python", "name": "python", "vector": list(_vector("arithmetic calculation", 32)), "quality": 0.8, "trials": 5, "state": "active"},
        {"key": "search", "name": "search", "vector": list(_vector("Wikipedia fact retrieval", 32)), "quality": 0.4, "trials": 5, "state": "active"},
    ], "edges": [
        {"source": "search", "target": "python", "kind": "precedes", "positive": 4, "negative": 0, "tasks": tasks},
    ]}}
class MockModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.visible: list[list[str]] = []
    def generate(self, messages, tools):
        self.visible.append([tool.__name__ for tool in tools])
        return next(self.outputs)
class MockTools:
    def python(self, code: str) -> str:
        return "Tool call success\n\nSTDOUT:\n5"
    def search(self, query: str) -> str:
        return "Following are the search results given by the wiki search engine:\nsearch results: \"Title\"\nText"
class JointTests(unittest.TestCase):
    def test_memory_is_soft_and_abstains_when_cold(self) -> None:
        memory = ToolMemory(snapshot())
        suggestion, reason = memory.rank("arithmetic calculation", ["python", "search"])
        self.assertEqual(suggestion, "python")
        self.assertEqual(reason, "frozen_memory_prior")
        state = snapshot()
        state["graph"]["tasks"] = ["train-0"]
        state["graph"]["edges"][0]["tasks"] = ["train-0"]
        cold = ToolMemory(state)
        self.assertEqual(cold.rank("arithmetic", ["python", "search"]), (None, "cold_memory"))
        self.assertEqual(Controller(memory).decide("arithmetic calculation", [{"role": "user", "content": "2+3"}], ["python", "search"]).available, ("python", "search"))
    def test_only_current_request_successful_observations_survive(self) -> None:
        history = [
            {"role": "user", "content": "Old question"},
            {"role": "tool", "name": "python", "content": "Tool call success\nSTDOUT: 5"},
            {"role": "user", "content": "New question"},
        ]
        version, seen = evidence(history)
        self.assertEqual(version, 1)
        self.assertEqual(seen, ())
        self.assertEqual(Controller().decide("New question", history, ["python"]).scale, "gg")
        history.append({"role": "tool", "name": "search", "content": "Error: unavailable"})
        self.assertEqual(Controller().decide("New question", history, ["python", "search"]).scale, "ll")
        history.append({"role": "tool", "name": "search", "content": "Error: unavailable"})
        self.assertEqual(Controller().decide("New question", history, ["python", "search"]).scale, "gl")
    def test_bad_or_missing_tool_results_are_not_verified(self) -> None:
        history = [
            {"role": "user", "content": "Question"},
            {"role": "tool", "name": "python", "content": "Error: timeout"},
            {"role": "tool", "name": "search", "content": "No relevant documents found"},
            {"role": "tool", "name": "python", "content": "Tool call success\nSTDOUT: 5"},
        ]
        _, seen = evidence(history)
        self.assertEqual([item.tool for item in seen], ["python"])
    def test_paired_runner_executes_same_tool_protocol(self) -> None:
        record = Record("key", "What is 2+3?", "5", "gsm8k")
        model = MockModel(['<tool_call>{"name":"python","arguments":{"code":"print(2+3)"}}</tool_call>', '<answer>\\boxed{5}</answer>'])
        result = solve(record, "joint", model, MockTools(), ToolMemory(snapshot()), 2)
        self.assertTrue(result["correct_local"])
        self.assertEqual(result["tools"], ["python"])
        self.assertTrue(result["events"])
        self.assertEqual(model.visible[0], ["python", "search"])
        self.assertEqual(calls(result["output"]), [("python", {"code": "print(2+3)"})])
    def test_record_keys_match_evolt_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gsm8k.jsonl"
            path.write_text(json.dumps({"id": "x", "question": "What is 2+3?", "answer": "Work #### 5"}) + "\n", encoding="utf-8")
            row = corpus(path, "gsm8k", "test")[0]
            self.assertEqual(row.answer, "5")
            from hashlib import blake2b
            self.assertEqual(row.key, blake2b(b"gsm8k|test|x|What is 2+3?", digest_size=10).hexdigest())
    def test_answer_must_follow_tool_response(self) -> None:
        output = '<tool_response name="search">\n<answer>\\boxed{5}</answer>\n</tool_response>\nI cannot answer.'
        self.assertEqual(answer(output), "")
    def test_local_math_score_ignores_tex_delimiter_sizing(self) -> None:
        expected = r"\left( 3, \frac{\pi}{2} \right)"
        self.assertTrue(local_correct(r"(3, \frac{\pi}{2})", expected))
        self.assertFalse(local_correct(r"(4, \frac{\pi}{2})", expected))
    def test_paired_compare_checks_protocol_and_keys(self) -> None:
        from eval import digest
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / "gsm8k.jsonl"
            data.write_text(json.dumps({"id": "x", "question": "What is 2+3?", "answer": "#### 5"}) + "\n", encoding="utf-8")
            key = corpus(data, "gsm8k", "test")[0].key
            protocol = {"backend": "searl", "data_sha256": digest(data), "dataset": "gsm8k", "split": "test", "model_config_sha256": "same", "tokens": 2048, "tool_rounds": 5, "rounds": 5, "search_url": "http://127.0.0.1:5003/retrieve", "sandbox_url": "http://127.0.0.1:8080"}
            left = base / "left.jsonl"
            right = base / "right.jsonl"
            left.write_text(json.dumps({"key": key, "output": "<answer>\\boxed{4}</answer>", "protocol": protocol}) + "\n", encoding="utf-8")
            right.write_text(json.dumps({"key": key, "output": "<answer>\\boxed{5}</answer>", "protocol": protocol}) + "\n", encoding="utf-8")
            result = compare(data, "gsm8k", "test", left, right)
            self.assertEqual(result["correct_local"]["evocond_only"], 1)
            bad = {**protocol, "data_sha256": "different"}
            right.write_text(json.dumps({"key": key, "output": "<answer>\\boxed{5}</answer>", "protocol": bad}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                compare(data, "gsm8k", "test", left, right)
            wrong_budget = {**protocol, "data_sha256": digest(data), "rounds": 4}
            right.write_text(json.dumps({"key": key, "output": "<answer>\\boxed{5}</answer>", "protocol": wrong_budget}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tool_rounds"):
                compare(data, "gsm8k", "test", left, right)
    def test_external_endpoints_are_explicit(self) -> None:
        with self.assertRaises(ValueError):
            Tools("https://wikipedia.org/retrieve", "http://127.0.0.1:8080")
        with self.assertRaises(ValueError):
            Tools("http://127.0.0.1:5003/other", "http://127.0.0.1:8080")
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
