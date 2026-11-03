# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from control.trajectory import adapter_digest, digest, episode_record, load_rollouts, trace_bytes, verify_record
class PolicyDataTest(unittest.TestCase):
    def episode(self, root, events=None):
        result, trace = root/'result.json', root/'trace.jsonl'
        result.write_text(json.dumps({'results': [{'status': 'completed', 'evaluation': {'success': True}}],
            'stats': {'scores': {'cost_metrics': {'total_agent_cost': 100.}}}}), encoding='utf-8')
        if events is None:
            events = [{'action': 'state_training_observation', 'policy_attempts': [
                {'prompt_ids': [1, 2], 'completion_ids': [3], 'accepted': True}]}]
        trace.write_text(''.join(json.dumps(event)+'\n' for event in events), encoding='utf-8')
        return result, trace

    def record(self, result, trace, **changes):
        values = dict(start=0, end=trace.stat().st_size, arm='evocond', group='q17/none',
            rollout=0, model_hash='model', tokenizer_hash='tokenizer', query_hash='a'*64)
        return episode_record(result, trace, **(values | changes))

    def load(self, outcomes: list[bool], mutate=None) -> tuple[list[dict], dict]:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for rollout, success in enumerate(outcomes):
                result = root / f"result_{rollout}.json"
                trace = root / f"trace_{rollout}.jsonl"
                result.write_text(json.dumps({"results": [{"status": "completed",
                    "evaluation": {"success": success}}], "stats": {"scores": {
                    "cost_metrics": {"total_agent_cost": 100.0}}}}), encoding="utf-8")
                trace.write_text(json.dumps({"action": "state_training_observation", "policy_attempts": [
                    {"prompt_ids": [1, 2], "completion_ids": [3, 4], "accepted": True}]})+"\n", encoding="utf-8")
                rows.append(episode_record(result, trace, 0, trace.stat().st_size, "evocond",
                                           "q17/none", rollout, "model", "tokenizer", "a"*64))
            if mutate:
                mutate(rows)
            path = root / "rollouts.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows)+"\n", encoding="utf-8")
            return load_rollouts(path, "evocond", "model", "tokenizer")
    def test_grouped_advantage(self) -> None:
        rows, report = self.load([True, False])
        self.assertEqual(report["nonzero_advantages"], 2)
        self.assertAlmostEqual(rows[0]["advantage"], 1.0)
        self.assertAlmostEqual(rows[1]["advantage"], -1.0)
    def test_test_split_is_rejected(self) -> None:
        def mutate(rows):
            rows[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "split or arm"):
            self.load([True, False], mutate)
    def test_tampered_acceptance_is_rejected(self) -> None:
        def mutate(rows):
            rows[0]["turns"][0]["accepted"] = False
        with self.assertRaisesRegex(ValueError, "Reward or policy tokens"):
            self.load([True, False], mutate)
    def test_duplicate_rollout_is_rejected(self) -> None:
        def mutate(rows):
            rows[1]["rollout"] = 0
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load([True, False], mutate)
    def test_mixed_sources_are_rejected(self) -> None:
        def mutate(rows):
            rows[1]["query_sha256"] = "b"*64
        with self.assertRaisesRegex(ValueError, "Mixed query"):
            self.load([True, False], mutate)
    def test_reward_tampering_rejected(self) -> None:
        def mutate(rows):
            rows[0]["reward"] = 9.0
        with self.assertRaisesRegex(ValueError, "Reward or policy tokens"):
            self.load([True, False], mutate)
    def test_all_rejected_attempts_still_form_failure_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result, trace = root / "result.json", root / "trace.jsonl"
            result.write_text(json.dumps({"results": [{"status": "completed", "evaluation": {"success": False}}],
                                          "stats": {"scores": {"cost_metrics": {"total_agent_cost": 0.0}}}}), encoding="utf-8")
            trace.write_text(json.dumps({"action": "state_training_observation", "policy_attempts": [
                {"prompt_ids": [1], "completion_ids": [2], "accepted": False}]})+"\n", encoding="utf-8")
            row = episode_record(result, trace, 0, trace.stat().st_size, "evocond", "q17/none", 0,
                                 "model", "tokenizer", "a"*64)
            self.assertEqual(row["reward"], 0.0)
            self.assertFalse(row["turns"][0]["accepted"])

    def test_trace_ranges_must_cover_complete_existing_jsonl_records(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result, trace = self.episode(root)
            size = trace.stat().st_size
            for start, end in ((True, size), (0, False), (-1, size), (0, size+1),
                    (0, 0), (1, size), (0, size-2), (0., size), (None, size)):
                with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                    self.record(result, trace, start=start, end=end)
            raw = trace.read_bytes()
            trace.write_bytes(raw+raw)
            self.assertEqual(trace_bytes(trace, len(raw), 2*len(raw)), raw)
            self.assertEqual(self.record(result, trace, end=len(raw))['source']['end'], len(raw))

    def test_complete_final_jsonl_record_does_not_require_a_terminal_newline(self):
        with tempfile.TemporaryDirectory() as folder:
            result, trace = self.episode(Path(folder))
            trace.write_bytes(trace.read_bytes().rstrip(b'\r\n'))
            row = self.record(result, trace)
            self.assertEqual(row['source']['trace_sha256'], sha256(trace.read_bytes()).hexdigest())
            verify_record(row)

    def test_malformed_trace_events_and_attempts_are_not_coerced(self):
        valid = {'prompt_ids': [1], 'completion_ids': [2], 'accepted': True}
        invalid = [None, {}, [valid], [{**valid, 'accepted': 'true'}],
            [{key: value for key, value in valid.items() if key != 'accepted'}],
            [valid, {**valid, 'accepted': False, 'prompt_ids': [True]}]]
        with tempfile.TemporaryDirectory() as folder:
            for attempts in invalid:
                with self.subTest(attempts=attempts), self.assertRaises(ValueError):
                    result, trace = self.episode(Path(folder),
                        [{'action': 'state_training_observation', 'policy_attempts': attempts}])
                    self.record(result, trace)
            result, trace = self.episode(Path(folder), [[], {'action': 'other'}])
            with self.assertRaises(ValueError):
                self.record(result, trace)

    def test_result_shape_and_nonfinite_cost_are_checked_before_reward_use(self):
        with tempfile.TemporaryDirectory() as folder:
            result, trace = self.episode(Path(folder))
            original = json.loads(result.read_text(encoding='utf-8'))
            invalid = [[], {}, {**original, 'stats': []},
                {**original, 'results': [{'status': 'completed', 'evaluation': []}]}]
            for payload in invalid:
                result.write_text(json.dumps(payload), encoding='utf-8')
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    self.record(result, trace)
            for cost in (True, -1, float('inf'), 10**1000):
                payload = deepcopy(original)
                payload['stats']['scores']['cost_metrics']['total_agent_cost'] = cost
                result.write_text(json.dumps(payload), encoding='utf-8')
                with self.subTest(cost=cost), self.assertRaises(ValueError):
                    self.record(result, trace)

    def test_result_hash_is_bound_to_the_bytes_that_supplied_the_reward(self):
        with tempfile.TemporaryDirectory() as folder:
            result, trace = self.episode(Path(folder))
            def changed(path):
                payload = json.loads(result.read_text(encoding='utf-8'))
                payload['stats']['scores']['cost_metrics']['total_agent_cost'] = 101.
                result.write_text(json.dumps(payload), encoding='utf-8')
                return digest(path)
            with patch('control.trajectory.digest', side_effect=changed), self.assertRaisesRegex(
                    ValueError, 'source changed during'):
                self.record(result, trace)

    def test_trace_changes_during_collection_cannot_supply_a_frozen_record(self):
        with tempfile.TemporaryDirectory() as folder:
            result, trace = self.episode(Path(folder))
            calls = []
            def changed(path, start, end):
                raw = trace_bytes(path, start, end)
                if not calls:
                    path.write_bytes(raw.replace(b'"completion_ids": [3]', b'"completion_ids": [4]'))
                calls.append(raw)
                return raw
            with patch('control.trajectory.trace_bytes', side_effect=changed), self.assertRaisesRegex(
                    ValueError, 'source changed during'):
                self.record(result, trace)

    def test_rollout_file_hash_cannot_follow_a_different_file_revision(self):
        def changed(row):
            verify_record(row)
            path = Path(row['source']['result']).parent/'rollouts.jsonl'
            with path.open('a', encoding='utf-8') as stream:
                stream.write('\n')
        with patch('control.trajectory.verify_record', side_effect=changed), self.assertRaisesRegex(
                ValueError, 'Rollout file changed during'):
            self.load([True, False])

    def test_verify_record_validates_source_ranges_and_identities_before_reading(self):
        with tempfile.TemporaryDirectory() as folder:
            result, trace = self.episode(Path(folder))
            row = self.record(result, trace)
            for key, value in (('start', -1), ('end', 0), ('result', None), ('trace', []),
                    ('result_sha256', 'missing'), ('trace_sha256', 'g'*64)):
                modified = deepcopy(row)
                modified['source'][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    verify_record(modified)
            for value in (None, {}, {'source': {}}):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    verify_record(value)

    def test_group_size_and_query_hash_are_validated_without_coercion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for size in (True, 2., '2', 0, None):
                with self.subTest(size=size), self.assertRaises(ValueError):
                    load_rollouts(root/'missing.jsonl', 'evocond', 'model', 'tokenizer', min_group=size)
            result, trace = self.episode(root)
            for identity in ('g'*64, 'A'*64, '', None):
                with self.subTest(identity=identity), self.assertRaises(ValueError):
                    self.record(result, trace, query_hash=identity)

    def test_streaming_hashes_preserve_existing_artifact_identities(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report, weights = root/'training_report.json', root/'adapter_model.safetensors'
            report.write_bytes(b'{"optimizer_steps": 1}')
            weights.write_bytes(b'weights'*(1024*170))
            self.assertEqual(digest(weights), sha256(weights.read_bytes()).hexdigest())
            self.assertEqual(adapter_digest(root), sha256(report.read_bytes()+weights.read_bytes()).hexdigest())
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
