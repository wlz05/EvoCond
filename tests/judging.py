# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from copy import deepcopy
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import judge as judge
from tests.checkout import official_root
AUTHOR = official_root()
class Backend:
    batch_size = 2
    def __init__(self, responses=None):
        self.responses = responses
        self.seen = []
    def _build_prompt(self, prediction, reference, question):
        self.seen.append((prediction, reference, question))
        return [{"role": "user", "content": f"{question}|{prediction}|{reference}"}]
    def _call_llm_batch(self, messages):
        return self.responses if self.responses is not None else ["1"] * len(messages)
class JudgeTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.author = self.root / "author"
        for name in judge.AUTHOR_FILES:
            path = self.author / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture source\n", encoding="utf-8")
        self.model = self.root / "Qwen3-32B"
        self.model.mkdir()
        self.manifest_path = self.root / "judge.json"
        self.output = self.root / "result"
        self.helpers_patch = patch.object(judge, "author_helpers", return_value={})
        self.helpers_patch.start()
        self.addCleanup(self.helpers_patch.stop)
        self.identity_patch = patch.object(judge, "model_identity", return_value={"fixture_identity": True})
        self.identity_patch.start()
        self.addCleanup(self.identity_patch.stop)
    def write_json(self, name, content):
        path = self.root / name
        path.write_text(json.dumps(content), encoding="utf-8")
        return {"path": name, "sha256": judge.file_hash(path)}
    def setup_manifest(self, inputs=None):
        dataset = self.root / "refs.jsonl"
        dataset.write_text(json.dumps({"question": "Q?", "answer": "A"}) + "\n", encoding="utf-8")
        arms = {}
        for name in judge.ARMS:
            path = self.root / f"{name}.jsonl"
            rows = inputs if inputs is not None else [{"input": "Q?", "output": "<answer>\\boxed{A}</answer>", "score": -999, "step": 7}]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            arms[name] = {"path": path.name, "sha256": judge.file_hash(path), "format": "jsonl"}
        manifest = {"format": judge.FORMAT, "split": "validation", "dataset_name": "math500",
                    "metric": "author_overall", "samples_per_question": 1,
                    "dataset": {"path": dataset.name, "sha256": judge.file_hash(dataset), "format": "jsonl",
                                "question_pointer": "/question", "answer_pointer": "/answer"}, "arms": arms}
        self.save_manifest(manifest)
        return manifest
    def save_manifest(self, manifest):
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    def test_dry_run_does_not_import_or_load_judge_or_write(self):
        self.setup_manifest()
        before = sorted(path.name for path in self.root.iterdir())
        with patch.object(judge, "load_author_judge", side_effect=AssertionError("GPU load forbidden")):
            result = judge.run(self.manifest_path, self.author, self.model, self.output)
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["run"])
        self.assertFalse(result["protocol"]["paper_exact_reproduction_verified"])
        self.assertEqual(before, sorted(path.name for path in self.root.iterdir()))
    def test_native_jsonl_export_schema_and_score_ignored(self):
        self.setup_manifest()
        protocol, rows = judge.preflight(self.manifest_path, self.author, self.model)
        self.assertEqual(rows["searl"][0], {"question": "Q?", "ground_truth": "A", "output": "<answer>\\boxed{A}</answer>"})
        self.assertNotIn("score", rows["searl"][0])
        self.assertEqual(protocol["reference_count"], 1)
    def test_raw_template_input_requires_explicit_alignment(self):
        self.setup_manifest([{"input": "system\nTemplate\nuser\nQ?\nassistant\n", "output": "answer"}])
        with self.assertRaisesRegex(ValueError, "alignment records"):
            judge.preflight(self.manifest_path, self.author, self.model)
    def test_original_question_metadata_aligns_template_export_without_mapping(self):
        for identity in ({"question": "Q?"}, {"question_sha256": judge.question_hash("Q?")},
                         {"question": "Q?", "question_sha256": judge.question_hash("Q?")}):
            with self.subTest(identity=identity):
                self.setup_manifest([{"input": "system\nTemplate\nuser\nQ?\nassistant\n",
                                      "output": "answer", **identity}])
                _, prepared = judge.preflight(self.manifest_path, self.author, self.model)
                self.assertEqual(prepared["teacher"][0]["question"], "Q?")
    def test_original_question_hash_conflicts_fail(self):
        self.setup_manifest([{"input": "templated", "question": "Q?", "output": "answer",
                              "question_sha256": judge.question_hash("Other question?")}])
        with self.assertRaisesRegex(ValueError, "identity conflicts"):
            judge.preflight(self.manifest_path, self.author, self.model)
    def test_question_metadata_cannot_silently_override_alignment(self):
        manifest = self.setup_manifest([{"input": "templated", "question": "Q?", "output": "answer"}])
        manifest["arms"]["searl"]["alignment"] = self.write_json("map.json", [
            {"row": 0, "input_sha256": judge.text_hash("templated"), "question_sha256": judge.question_hash("Other?")}])
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "identity conflicts"):
            judge.preflight(self.manifest_path, self.author, self.model)
    def test_explicit_alignment_binds_input_hash_not_firstline_guess(self):
        raw = "system\nExample\nuser\nQ?\nassistant\n"
        manifest = self.setup_manifest([{"input": raw, "output": "answer"}])
        for arm in judge.ARMS:
            manifest["arms"][arm]["alignment"] = self.write_json(f"{arm}_map.json", [
                {"row": 0, "input_sha256": judge.text_hash(raw), "question_sha256": judge.question_hash("Q?")}])
        self.save_manifest(manifest)
        _, prepared = judge.preflight(self.manifest_path, self.author, self.model)
        self.assertEqual(prepared["teacher"][0]["question"], "Q?")
    def test_alignment_bad_input_hash_fails(self):
        manifest = self.setup_manifest()
        manifest["arms"]["searl"]["alignment"] = self.write_json("map.json", [
            {"row": 0, "input_sha256": "a" * 64, "question_sha256": judge.question_hash("Q?")}])
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "input hash mismatch"):
            judge.preflight(self.manifest_path, self.author, self.model)
    def test_missing_output_is_error_not_zero(self):
        self.setup_manifest([{"input": "Q?", "score": 1}])
        with self.assertRaisesRegex(ValueError, "output field"):
            judge.preflight(self.manifest_path, self.author, self.model)
    def test_three_arms_required(self):
        manifest = self.setup_manifest()
        del manifest["arms"]["teacher"]
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "all three arms"):
            judge.preflight(self.manifest_path, self.author, self.model)
    def test_unbalanced_repeats_fail(self):
        manifest = self.setup_manifest()
        path = self.root / manifest["arms"]["teacher"]["path"]
        path.write_text(path.read_text() * 2, encoding="utf-8")
        manifest["arms"]["teacher"]["sha256"] = judge.file_hash(path)
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "declared sample count"):
            judge.preflight(self.manifest_path, self.author, self.model)
    def test_reference_duplicates_fail(self):
        manifest = self.setup_manifest()
        path = self.root / "refs.jsonl"
        path.write_text(path.read_text() * 2, encoding="utf-8")
        manifest["dataset"]["sha256"] = judge.file_hash(path)
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "duplicate normalized"):
            judge.preflight(self.manifest_path, self.author, self.model)
    def test_special_grouped_metrics_not_mislabeled_overall(self):
        manifest = self.setup_manifest()
        for name in ("GAIA", "gaia", "GaIa", " hle ", "HLE", "webwalker", "Web-Walker", "web_walker"):
            with self.subTest(dataset=name):
                manifest["dataset_name"] = name
                self.save_manifest(manifest)
                with self.assertRaisesRegex(ValueError, "not equivalent"):
                    judge.preflight(self.manifest_path, self.author, self.model)
    def test_existing_output_not_overwritten(self):
        self.setup_manifest()
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "no overwrite"):
            judge.run(self.manifest_path, self.author, self.model, self.output, execute=True)

    def test_json_reader_accepts_bom_but_rejects_duplicate_and_overflow_numbers(self):
        path = self.root/'input.json'
        path.write_text('{"value":0.5}', encoding='utf-8-sig')
        self.assertEqual(judge._json_file(path), {'value': .5})
        for text in ('{"value":1,"value":2}', '{"value":1e999}', '{"value":NaN}'):
            path.write_text(text, encoding='utf-8')
            with self.subTest(text=text), self.assertRaises(ValueError):
                judge._json_file(path)

    def test_report_publication_preserves_prior_output_and_rejects_partial_json(self):
        path = self.root/'report.json'
        with self.assertRaises(ValueError):
            judge._write(path, {'value': float('nan')})
        self.assertFalse(path.exists())
        with patch('control.contracts.os.link', side_effect=OSError('Interrupted publication')):
            with self.assertRaises(OSError):
                judge._write(path, {'complete': True})
        self.assertFalse(path.exists())
        self.assertEqual(list(self.root.glob('.evocond-*.tmp')), [])
        judge._write(path, {'complete': True})
        payload = path.read_bytes()
        with self.assertRaises(FileExistsError):
            judge._write(path, {'complete': False})
        self.assertEqual(path.read_bytes(), payload)
    def test_transport_preserves_author_prompt_and_binary_results(self):
        backend = Backend()
        wrapped = judge.StrictAuthorJudge(backend)
        result = wrapped.evaluation_batch([("A", ["A", "alias"], "Q?"), ("B", "B", "Next?")])
        self.assertEqual(result, [1.0, 1.0])
        self.assertEqual(backend.seen[0], ("A", ["A", "alias"], "Q?"))
        self.assertEqual(len(wrapped.calls), 2)
    def test_invalid_judge_output_and_missing_count_raise(self):
        for responses in (["Answer: 1"], [""], ["true"], ["10"], []):
            with self.assertRaises(RuntimeError):
                judge.StrictAuthorJudge(Backend(responses)).evaluation_batch([("A", "A", "Q?")])
    def test_judge_exception_not_converted_to_zero(self):
        backend = Backend()
        with patch.object(backend, "_call_llm_batch", side_effect=RuntimeError("CUDA OOM")):
            with self.assertRaisesRegex(RuntimeError, "CUDA OOM"):
                judge.StrictAuthorJudge(backend).evaluation_batch([("A", "A", "Q?")])
    @unittest.skipUnless((AUTHOR / "evaluation/evaluation.py").is_file(), "author checkout unavailable")
    def test_actual_author_parser_and_metric_with_mocked_model(self):
        self.helpers_patch.stop()
        helpers = judge.author_helpers(AUTHOR)
        backend = Backend()
        wrapped = judge.StrictAuthorJudge(backend)
        rows = [{"question": "Q?", "ground_truth": "A", "output": "<answer>example</answer>\nassistant\n<answer>\\boxed{A}</answer>"},
                {"question": "Q2?", "ground_truth": "B", "output": "no answer tags"},
                {"question": "Q3?", "ground_truth": "C", "output": "<answer>C</answer>"}]
        metrics = helpers["evaluation_metrics_HLE"](rows, wrapped)
        self.assertEqual(metrics[:4], (1, 1, 1, 0))
        self.assertAlmostEqual(metrics[4], 1 / 3)
        self.assertEqual(backend.seen, [("A", "A", "Q?")])
    @unittest.skipUnless((AUTHOR / "evaluation/evaluation.py").is_file(), "author checkout unavailable")
    def test_execution_reuses_one_judge_for_all_arms_and_writes_last_report(self):
        self.setup_manifest()
        self.helpers_patch.stop()
        actual_helpers = judge.author_helpers(AUTHOR)
        backend = Backend()
        wrapped = judge.StrictAuthorJudge(backend)
        with patch.object(judge, "author_helpers", return_value=actual_helpers), \
                patch.object(judge, "load_author_judge", return_value=wrapped) as loader:
            result = judge.run(self.manifest_path, self.author, self.model, self.output, execute=True)
        self.assertEqual(loader.call_count, 1)
        self.assertTrue(result["complete"])
        self.assertTrue((self.output / "judge_report.json").is_file())
        self.assertEqual({name: row["acc"] for name, row in result["arms"].items()}, {name: 1.0 for name in judge.ARMS})
    @unittest.skipUnless((AUTHOR / "evaluation/evaluation.py").is_file(), "author checkout unavailable")
    def test_runtime_failure_leaves_protocol_not_success_report(self):
        self.setup_manifest()
        self.helpers_patch.stop()
        actual_helpers = judge.author_helpers(AUTHOR)
        with patch.object(judge, "author_helpers", return_value=actual_helpers), \
                patch.object(judge, "load_author_judge", side_effect=RuntimeError("GPU failed")):
            with self.assertRaisesRegex(RuntimeError, "GPU failed"):
                judge.run(self.manifest_path, self.author, self.model, self.output, execute=True)
        self.assertTrue((self.output / "judge_protocol.json").exists())
        self.assertFalse((self.output / "judge_report.json").exists())
    def test_model_identity_checks_pinned_content_not_name(self):
        self.identity_patch.stop()
        config = self.model / "config.json"
        config.write_text(json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
                                      "hidden_size": 5120, "num_hidden_layers": 64}), encoding="utf-8")
        weight = self.model / "model-00001-of-00001.safetensors"
        weight.write_bytes(b"tiny CPU test fixture")
        catalogue = [{"key": "judge", "repo": "Qwen/Qwen3-32B", "revision": "pinned", "files": [
            {"name": path.name, "size": path.stat().st_size, "hash": "sha256", "checksum": judge.file_hash(path)}
            for path in (config, weight)]}]
        with patch.object(judge, "CATALOGUE", catalogue):
            self.assertFalse(judge.model_identity(self.model)["all_weight_checksums_verified"])
            self.assertTrue(judge.model_identity(self.model, full_weights=True)["all_weight_checksums_verified"])
            weight.write_bytes(b"X" * weight.stat().st_size)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                judge.model_identity(self.model, full_weights=True)
    def test_json_array_validation_export_is_supported(self):
        manifest = self.setup_manifest()
        for arm in judge.ARMS:
            manifest["arms"][arm] = {**self.write_json(f"{arm}.json", [{"input": "Q?", "output": "raw"}]), "format": "json"}
        self.save_manifest(manifest)
        _, prepared = judge.preflight(self.manifest_path, self.author, self.model)
        self.assertEqual(len(prepared["searl"]), 1)
    @unittest.skipUnless((AUTHOR / "verl/trainer/ppo/trainer.py").is_file(), "author checkout unavailable")
    def test_actual_trainer_question_helper_preserves_multiline_and_repeat_order(self):
        source = AUTHOR / "verl/trainer/ppo/trainer.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "validation_question_metadata")
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
        questions = ["First line\nSecond line?", "First line\nSecond line?", "Next?", "Next?"]
        result = namespace[helper.name]({"question": questions, "reward_model": "DO NOT READ"}, 4)
        self.assertEqual(result["question"], questions)
        self.assertEqual(result["question_sha256"], [judge.question_hash(item) for item in questions])
        self.assertIsNot(result["question"], questions)
        for metadata, expected in (({}, 1), ({"question": ["Q?"]}, 2), ({"question": [None]}, 1)):
            with self.assertRaises(ValueError):
                namespace[helper.name](metadata, expected)
    @unittest.skipUnless((AUTHOR / "verl/trainer/ppo/trainer.py").is_file(), "author checkout unavailable")
    def test_actual_trainer_dump_carries_original_question_metadata(self):
        source = AUTHOR / "verl/trainer/ppo/trainer.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        dump = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and node.name == "_dump_generations")
        namespace = {"os": os, "json": json}
        exec(compile(ast.Module(body=[dump], type_ignores=[]), str(source), "exec"), namespace)
        extras = {"reward": [1.0], "question": ["A full\nquestion?"],
                  "question_sha256": [judge.question_hash("A full\nquestion?")]}
        namespace[dump.name](SimpleNamespace(global_steps=7), ["templated"], ["answer"], [1.0], extras, str(self.output))
        row = json.loads((self.output / "7.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(row["question"], "A full\nquestion?")
        self.assertEqual(row["question_sha256"], judge.question_hash(row["question"]))
        self.assertEqual(row["score"], 1.0)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
