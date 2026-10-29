# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

# Check split isolation and native evaluation contracts without model execution.
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from catalogue import benchmark, relative_path
from partition import normalize, partition, prepare
from leakage import audit
from benchmark import NativeEvaluator, evaluate, load_cases, make_protocol, public_case, receipt, summarize
from convert import file_hash
class DatasetInterfacesTests(unittest.TestCase):
    def rows(self, count=12):
        return [{"id": f"view{i}", "task": f"task{i}", "question": f"Question {i}", "answer": f"Secret {i}", "tools": {"names": ["search"]}} for i in range(count)]
    def normalized(self, rows=None):
        return normalize(rows or self.rows(), "/id", "/question", ["/task"], "/answer", "/tools")
    def protocol(self):
        return make_protocol("toolhop", "a" * 64, "source-v1", "evaluator-v1", "qwen3-4b/checkpoint-1", "evocond", 42)
    def test_registry_and_relative_paths(self):
        for name in ("toolhop", "toolsandbox", "bigcodebench"):
            self.assertTrue(benchmark(name).evaluation_only)
        self.assertFalse(benchmark("searl_qa").evaluation_only)
        for path in ("/etc/passwd", "C:/models/qwen", "\\rooted", ""):
            with self.assertRaises(ValueError):
                relative_path(path)
    def test_grouped_variants_and_duplicate_prompts(self):
        rows = self.rows()
        rows[1]["task"] = rows[0]["task"]
        rows[2]["question"] = "  QUESTION   1 "
        records = self.normalized(rows)
        self.assertEqual(len({record["group"] for record in records[:3]}), 1)
        parts = partition(records, (0.6, 0.2, 0.2), 5)
        destinations = {record["id"]: split for split, cases in parts.items() for record in cases}
        self.assertEqual(len({destinations[f"view{i}"] for i in range(3)}), 1)
        reordered = partition(self.normalized(list(reversed(rows))), (0.6, 0.2, 0.2), 5)
        self.assertEqual({split: {row["id"] for row in cases} for split, cases in parts.items()}, {split: {row["id"] for row in cases} for split, cases in reordered.items()})
    def test_bad_ratios_and_too_few_groups(self):
        for fractions in ((1, 1, 1), (float("nan"), 0, 1), (-1, 1, 1), (0.5, 0.5)):
            with self.assertRaises(ValueError):
                partition(self.normalized(), fractions, 42)
        with self.assertRaises(ValueError):
            partition(self.normalized(self.rows(2)), (0.8, 0.1, 0.1), 42)
    def test_public_requests_hide_and_do_not_mutate_references(self):
        case = self.normalized()[0]
        public = public_case(case)
        self.assertEqual(set(public), {"id", "prompt", "context"})
        self.assertNotIn("Secret", json.dumps(public))
        public["context"]["names"].append("new")
        self.assertEqual(case["context"]["names"], ["search"])
    def test_receipt_errors_stay_in_denominator(self):
        cases, protocol = self.normalized(), self.protocol()
        receipts = [receipt(case["id"], {"status": "ok", "passed": True}, protocol) for case in cases]
        receipts[0] = receipt(cases[0]["id"], {"status": "timeout", "passed": False}, protocol)
        report = summarize(cases, receipts, protocol)
        self.assertEqual((report["passed"], report["n"], report["timeouts"]), (11, 12, 1))
        for bad in (receipts[:-1], receipts + [receipts[0]]):
            with self.assertRaises(ValueError):
                summarize(cases, bad, protocol)
        with self.assertRaises(ValueError):
            receipt("x", {"status": "error", "passed": True}, protocol)
        changed = deepcopy(receipts)
        changed[0]["evaluator_revision"] = "other"
        with self.assertRaises(ValueError):
            summarize(cases, changed, protocol)
    def test_native_adapter_separates_actor_and_grader(self):
        owner = self
        class FakeNative(NativeEvaluator):
            def run(self, case, protocol):
                owner.assertNotIn("reference", case)
                return {"answer": "mock"}
            def grade(self, reference, trajectory, protocol):
                owner.assertTrue(reference.startswith("Secret"))
                return {"status": "ok", "passed": False}
        cases, protocol = self.normalized(), self.protocol()
        self.assertEqual(summarize(cases, list(evaluate(cases, FakeNative(), protocol)), protocol)["passed"], 0)

    def test_inference_execution_failures_remain_in_the_denominator(self):
        class Failing(NativeEvaluator):
            def run(self, case, protocol):
                if case['id'] == 'view0':
                    raise TimeoutError('request timed out')
                if case['id'] == 'view1':
                    raise RuntimeError('execution failed')
                return {'answer': 'fixture'}
            def grade(self, reference, trajectory, protocol):
                return {'status': 'ok', 'passed': True}
        cases = self.normalized(self.rows(3))
        with self.assertLogs('benchmark', level='WARNING'):
            rows = list(evaluate(cases, Failing(), self.protocol()))
        report = summarize(cases, rows, self.protocol())
        self.assertEqual((report['n'], report['passed'], report['timeouts'], report['errors']), (3, 1, 1, 1))

    def test_grader_failures_and_interruptions_are_not_task_failures(self):
        class Broken(NativeEvaluator):
            def run(self, case, protocol):
                return {}
            def grade(self, reference, trajectory, protocol):
                raise RuntimeError('grader unavailable')
        cases = self.normalized(self.rows(1))
        with self.assertRaisesRegex(RuntimeError, 'grader unavailable'):
            list(evaluate(cases, Broken(), self.protocol()))
        with patch.object(Broken, 'run', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            list(evaluate(cases, Broken(), self.protocol()))
        from control.isolation import SandboxInfrastructureError
        with patch.object(Broken, 'run', side_effect=SandboxInfrastructureError('executor unavailable')):
            with self.assertRaises(SandboxInfrastructureError):
                list(evaluate(cases, Broken(), self.protocol()))

    def test_direct_evaluation_rejects_duplicate_cases_before_execution(self):
        cases = self.normalized(self.rows(1))
        protocol = self.protocol()
        rows = [receipt(cases[0]['id'], {'status': 'ok', 'passed': True}, protocol)]
        with self.assertRaises(ValueError):
            summarize(cases+cases, rows, protocol)

    def test_split_ratios_and_revision_labels_are_not_coerced(self):
        for seed in (True, -1, '42'):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                partition(self.normalized(), (.8, .1, .1), seed)
        with self.assertRaises(ValueError):
            partition(self.normalized(), (True, 0, 0), 42)
        for revision in (' main ', 'refs/heads/main', 'HEAD~1'):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                make_protocol('toolhop', 'a'*64, revision, 'v1', 'qwen', 'arm', 42)
    def test_preparation_dry_run_preserves_test_and_rejects_training(self):
        with tempfile.TemporaryDirectory(dir=".") as folder:
            root = Path(folder).resolve().relative_to(Path.cwd())
            source = root / "source.jsonl"
            source.write_text("\n".join(json.dumps(row) for row in self.rows()) + "\n", encoding="utf-8")
            args = Namespace(benchmark="toolhop", input=source.as_posix(), output=(root / "output").as_posix(), sha256=file_hash(source), revision="v1", source_uri="https://example.test/toolhop", source_split="test", format="jsonl", policy="preserve", fractions="0.8,0.1,0.1", seed=42, id_field="/id", prompt_field="/question", group_fields="/task", reference_field="/answer", context_field="/tools", run=False)
            report = prepare(args)
            self.assertEqual(set(report["splits"]), {"test"})
            self.assertFalse(Path(args.output).exists())
            args.policy = "partition-train"
            with self.assertRaises(ValueError):
                prepare(args)
            args.policy, args.fractions = "development-holdout", "0,0.3,0.7"
            args.run = True
            report = prepare(args)
            self.assertFalse(report["official_partition_preserved"])
            self.assertNotIn("train", report["splits"])
            self.assertTrue(load_cases(Path(args.output) / "test.jsonl"))
            with self.assertRaises(FileExistsError):
                prepare(args)
    def test_wrong_hash_and_floating_revision(self):
        with self.assertRaises(ValueError):
            make_protocol("toolhop", "bad", "v1", "v1", "qwen", "arm", 42)
        with self.assertRaises(ValueError):
            make_protocol("toolhop", "a" * 64, "main", "v1", "qwen", "arm", 42)
    def test_cross_file_task_variants_are_detected(self):
        with tempfile.TemporaryDirectory(dir=".") as folder:
            root = Path(folder).resolve().relative_to(Path.cwd())
            left = self.rows(1)
            right = [{**left[0], "id": "other-view", "question": "Different wording"}]
            paths = {"train": root / "train.jsonl", "test": root / "test.jsonl"}
            for split, rows in zip(paths, (left, right)):
                paths[split].write_text("\n".join(json.dumps(row) for row in self.normalized(rows)) + "\n", encoding="utf-8")
            report = audit(paths)
            self.assertFalse(report["disjoint"])
            self.assertEqual(report["overlaps"]["test:train"]["group_keys"], 1)
            self.assertEqual(report["overlaps"]["test:train"]["prompts"], 0)
    def test_toolsandbox_preserves_native_continuous_similarity(self):
        cases = self.normalized(self.rows(2))
        protocol = make_protocol("toolsandbox", "a" * 64, "v1", "v1", "qwen/checkpoint-1", "evocond", 42)
        receipts = [receipt(case["id"], {"status": "ok", "score": value}, protocol) for case, value in zip(cases, (0.97, 0.5))]
        report = summarize(cases, receipts, protocol)
        self.assertEqual(report["metric"], "milestone_similarity")
        self.assertAlmostEqual(report["score"], 0.735)
        self.assertIsNone(report["passed"])
        for value in (float("nan"), float("inf"), True, -1, 1.1):
            with self.assertRaises(ValueError):
                receipt("x", {"status": "ok", "score": value}, protocol)
        with self.assertRaises(ValueError):
            receipt("x", {"status": "ok", "passed": True}, protocol)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
