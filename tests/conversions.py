# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import experiment
import convert as datasets
class TableFactory:
    @staticmethod
    def from_pylist(values):
        return deepcopy(values)
class FakeArrow:
    Table = TableFactory
class FakeParquet:
    @staticmethod
    def write_table(table, path, compression):
        path.write_text(json.dumps(table), encoding="utf-8")
class DatasetPrepareTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manifest = self.root / "manifest.json"
        self.output = self.root / "prepared"
    def source(self, name, split, question="What is 2+2?", answer="4", rows=None):
        path = self.root / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in
                               (rows if rows is not None else [{"q": question, "a": answer}])), encoding="utf-8")
        return {"name": name, "path": path.name, "sha256": datasets.file_hash(path), "format": "jsonl",
                "split": split, "source_split": split, "source_uri": "https://example.test/data",
                "source_revision": "pinned-revision-123", "mapping": {"question": "/q", "answer": "/a"},
                "data_source": "arpo", "ability": "reasoning"}
    def manifest_file(self, sources, mode="adapted_upstream"):
        self.manifest.write_text(json.dumps({"format": datasets.FORMAT, "mode": mode, "sources": sources}), encoding="utf-8")
        return self.manifest
    def ready_row(self, question="Question?", index=0):
        return {"data_source": "arpo", "question": question, "prompt": [{"role": "user", "content": question}],
                "reward_model": {"style": "rule", "ground_truth": {"target": ["SecretAnswer"]}},
                "extra_info": {"index": index}, "untouched": "author_column"}
    def test_jsonl_dry_run_no_dependency_or_writes(self):
        manifest = self.manifest_file([self.source("train", "train"), self.source("valid", "validation", "Different?")])
        before = sorted(path.name for path in self.root.iterdir())
        with patch.object(datasets, "_pyarrow", side_effect=AssertionError("must stay lazy")):
            report = datasets.prepare(manifest, self.output)
        self.assertFalse(report["files_written"])
        self.assertTrue(report["training_inputs_present"])
        self.assertEqual(report["counts"], {"train": 1, "validation": 1})
        self.assertFalse(report["official_author_preprocessing_claim"])
        self.assertEqual(before, sorted(path.name for path in self.root.iterdir()))
    def test_windows_bom_crlf_and_json_array_use_same_mapping(self):
        source = self.source('train', 'train')
        path = self.root/source['path']
        path.write_bytes(b'\xef\xbb\xbf{"q":"Question", "a":"Answer"}\r\n')
        source['sha256'] = datasets.file_hash(path)
        self.manifest_file([source])
        _, first, _ = datasets.build_plan(self.manifest)
        path = self.root/'train.json'
        path.write_text(json.dumps([{'q': 'Question', 'a': 'Answer'}]), encoding='utf-8-sig')
        source.update(path=path.name, format='json', sha256=datasets.file_hash(path))
        self.manifest_file([source])
        _, second, _ = datasets.build_plan(self.manifest)
        self.assertEqual(first['train'][0]['prompt'], second['train'][0]['prompt'])
        self.assertEqual(first['train'][0]['reward_model'], second['train'][0]['reward_model'])
    def test_json_envelope_is_not_silently_inferred(self):
        path = self.root/'nested.json'
        path.write_text(json.dumps({'data': [{'q': 'Q', 'a': 'A'}]}))
        with self.assertRaisesRegex(ValueError, 'explicit array'):
            list(datasets.read_rows(path, 'json'))
    def test_json_dataset_not_limited_by_single_payload_budget(self):
        path = self.root/'large.json'
        path.write_text(json.dumps([{'q': 'x'*6000, 'a': 'A'} for _ in range(200)]))
        self.assertGreater(path.stat().st_size, 1_000_000)
        self.assertEqual(len(list(datasets.read_rows(path, 'json'))), 200)
    def test_duplicate_mapped_source_ids_are_rejected(self):
        source = self.source('train', 'train', rows=[{'q': 'Q1', 'a': 'A', 'id': 'same'}, {'q': 'Q2', 'a': 'B', 'id': 'same'}])
        source['mapping']['id'] = '/id'
        self.manifest_file([source])
        with self.assertRaisesRegex(ValueError, 'Duplicate mapped source ID'):
            datasets.build_plan(self.manifest)
    def test_adapted_row_matches_mcp_reward_and_no_answer_prompt(self):
        source = self.source("train", "train", answer="SECRET")
        self.manifest_file([source])
        report, rows, _ = datasets.build_plan(self.manifest)
        row = rows["train"][0]
        self.assertEqual(row["reward_model"]["ground_truth"], {"target": ["SECRET"]})
        self.assertNotIn("SECRET", json.dumps(row["prompt"]))
        self.assertNotIn("SECRET", json.dumps(report))
        self.assertEqual(row["extra_info"]["preprocessing"], "adapted_upstream_not_author_preprocessed")
    def test_explicit_pointer_mapping_nested_and_lists(self):
        source = self.source("train", "train", rows=[{"turns": [{"text": "Question"}], "label": {"answers": ["A", "B"]}}])
        source["mapping"] = {"question": "/turns/0/text", "answer": "/label/answers"}
        self.manifest_file([source])
        _, rows, _ = datasets.build_plan(self.manifest)
        self.assertEqual(rows["train"][0]["reward_model"]["ground_truth"]["target"], ["A", "B"])
        self.assertEqual(datasets.pointer({"a/b": {"c~d": "yes"}}, "/a~1b/c~0d"), "yes")
    def test_no_guessed_question_or_answer_keys(self):
        source = self.source("train", "train")
        source["mapping"]["question"] = "/missing"
        self.manifest_file([source])
        with self.assertRaisesRegex(ValueError, "Missing mapped field"):
            datasets.prepare(self.manifest, self.output)
        self.assertFalse(self.output.exists())
    def test_validation_and_test_cannot_be_reassigned_to_train(self):
        for original in ("validation", "test"):
            source = self.source("eval", original)
            source["split"] = "train"
            self.manifest_file([source])
            with self.assertRaisesRegex(ValueError, "reassignment"):
                datasets.prepare(self.manifest)
    def test_no_automatic_training_split_partition(self):
        source = self.source("train", "train")
        source["split"] = "validation"
        self.manifest_file([source])
        with self.assertRaisesRegex(ValueError, "partition upstream first"):
            datasets.prepare(self.manifest)
    def test_crosssplit_normalized_duplicate_fails(self):
        self.manifest_file([self.source("train", "train", "Hello WORLD"),
                            self.source("test", "test", '  \uff28\uff45\uff4c\uff4c\uff4f\nworld  ')])
        with self.assertRaisesRegex(ValueError, "Duplicate normalized question"):
            datasets.prepare(self.manifest, self.output)
        self.assertFalse(self.output.exists())
    def test_intrasplit_duplicate_not_silently_removed(self):
        self.manifest_file([self.source("train", "train", rows=[{"q": "Q", "a": "A"}, {"q": "Q", "a": "B"}])])
        with self.assertRaisesRegex(ValueError, "no automatic deduplication"):
            datasets.prepare(self.manifest)
    def test_group_identity_crosses_files_even_with_different_questions_and_field_aliases(self):
        train = self.source('train', 'train', rows=[dict(q='Original task?', a='A', task=17, template='family')])
        heldout = self.source('heldout', 'test', rows=[dict(q='Rephrased task?', a='A', original='17', template='other')])
        train['grouping'] = dict(namespace='benchmark', fields=dict(task='/task', template='/template'))
        heldout['grouping'] = dict(namespace='benchmark', fields=dict(task='/original', template='/template'))
        self.manifest_file([train, heldout])
        with self.assertRaisesRegex(ValueError, 'Task/template group crosses'):
            datasets.prepare(self.manifest, self.output, run=True)
        self.assertFalse(self.output.exists())

    def test_template_variants_stay_together_and_coverage_is_reported(self):
        train = self.source('train', 'train', rows=[dict(q='Q1', a='A', task='one', template='shared'),
            dict(q='Q2', a='A', task='two', template='shared')])
        train['grouping'] = dict(namespace='benchmark', fields=dict(task='/task', template='/template'))
        heldout = self.source('heldout', 'test', 'Different?')
        audit, rows, _ = datasets.build_plan(self.manifest_file([train, heldout]))
        first, second = (set(datasets.group_keys(row)) for row in rows['train'])
        self.assertEqual(len(first & second), 1)
        self.assertEqual(audit['split_group_coverage']['train'], dict(grouped_rows=2, ungrouped_rows=0))
        self.assertEqual(audit['split_group_coverage']['test'], dict(grouped_rows=0, ungrouped_rows=1))
        heldout = self.source('heldout', 'test', rows=[dict(q='New task same template?', a='A', task='three', template='shared')])
        heldout['grouping'] = train['grouping']
        self.manifest_file([train, heldout])
        with self.assertRaisesRegex(ValueError, 'Task/template group crosses'):
            datasets.build_plan(self.manifest)

    def test_independent_dataset_namespaces_do_not_merge_local_ids(self):
        sources = [self.source(name, split, rows=[dict(q=name, a='A', id=0)])
            for name, split in (('first', 'train'), ('second', 'test'))]
        for source in sources:
            source['grouping'] = dict(namespace=source['name'], fields=dict(task='/id'))
        _, rows, _ = datasets.build_plan(self.manifest_file(sources))
        self.assertNotEqual(datasets.group_keys(rows['train'][0]), datasets.group_keys(rows['test'][0]))

    def test_one_group_namespace_cannot_drop_an_identity_label(self):
        first, second = self.source('first', 'train'), self.source('second', 'test', 'Other?')
        first['grouping'] = dict(namespace='shared', fields=dict(task='/id', template='/family'))
        second['grouping'] = dict(namespace='shared', fields=dict(task='/id'))
        with self.assertRaisesRegex(ValueError, 'same identity labels'):
            datasets.read_manifest(self.manifest_file([first, second]))

    def test_declared_groups_never_fall_back_to_row_numbers(self):
        for value in (None, True, '', ' ', [], {}, 1.5):
            source = self.source('train', 'train', rows=[dict(q='Q', a='A', id=value)])
            source['grouping'] = dict(namespace='shared', fields=dict(task='/id'))
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'group identity'):
                datasets.build_plan(self.manifest_file([source]))
        source = self.source('train', 'train')
        source['grouping'] = dict(namespace='shared', fields=dict(task='/missing'))
        with self.assertRaisesRegex(ValueError, 'Missing mapped field'):
            datasets.build_plan(self.manifest_file([source]))

    def test_group_declaration_and_preserved_keys_are_validated(self):
        for declaration in (None, {}, dict(namespace=' ', fields=dict(task='/id')),
                dict(namespace='shared', fields={}), dict(namespace='shared', fields=dict(task='id'))):
            source = self.source('train', 'train')
            source['grouping'] = declaration
            with self.subTest(declaration=declaration), self.assertRaises(ValueError):
                datasets.read_manifest(self.manifest_file([source]))
        for keys in ([], 'a'*64, ['A'*64], [None], ['a'*64, 'a'*64]):
            row = self.ready_row()
            row['extra_info']['split_group_keys'] = keys
            with self.subTest(keys=keys), self.assertRaisesRegex(ValueError, 'split_group_keys'):
                datasets.validate_official_row(row)

    def test_bad_source_hash_fails(self):
        source = self.source("train", "train")
        source["sha256"] = "a" * 64
        self.manifest_file([source])
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            datasets.prepare(self.manifest)
    def test_floating_revision_fails(self):
        source = self.source("train", "train")
        source["source_revision"] = "main"
        self.manifest_file([source])
        with self.assertRaisesRegex(ValueError, "Floating"):
            datasets.prepare(self.manifest)
    def test_same_file_cannot_be_in_multiple_entries(self):
        source = self.source("train", "train")
        duplicate = {**source, "name": "second", "split": "test", "source_split": "test"}
        self.manifest_file([source, duplicate])
        with self.assertRaisesRegex(ValueError, "same source file"):
            datasets.prepare(self.manifest)
    def test_existing_output_never_overwritten(self):
        self.manifest_file([self.source("train", "train")])
        self.output.mkdir()
        sentinel = self.output / "checkpoint"
        sentinel.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "never overwritten"):
            datasets.prepare(self.manifest, self.output, run=True)
        self.assertEqual(sentinel.read_text(), "preserve")
    def test_missing_pyarrow_before_output_created(self):
        self.manifest_file([self.source("train", "train")])
        with patch.object(datasets, "_pyarrow", side_effect=RuntimeError("pyarrow required")):
            with self.assertRaisesRegex(RuntimeError, "pyarrow"):
                datasets.prepare(self.manifest, self.output, run=True)
        self.assertFalse(self.output.exists())
    def test_mocked_materialization_has_final_audit_and_hashes(self):
        self.manifest_file([self.source("train", "train"), self.source("valid", "validation", "New?")])
        with patch.object(datasets, "_pyarrow", return_value=(FakeArrow, FakeParquet)):
            report = datasets.prepare(self.manifest, self.output, run=True)
        self.assertTrue(report["complete"])
        self.assertTrue((self.output / "dataset_audit.json").is_file())
        for record in report["outputs"].values():
            self.assertEqual(datasets.file_hash(self.output/record['path']), record["sha256"])
        self.assertEqual(report['source_path_base'], 'input_manifest_directory')
        for source in report['sources']:
            self.assertFalse(Path(source['path']).is_absolute())
            self.assertNotIn('resolved_path', source)

    def test_manifest_rejects_machine_specific_absolute_source_paths(self):
        source = self.source('train', 'train')
        for path in (str((self.root/source['path']).resolve()), '/root/train.jsonl', 'C:/data/train.jsonl', 'data\\train.jsonl'):
            self.manifest_file([{**source, 'path': path}])
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'relative paths'):
                datasets.read_manifest(self.manifest)
    def test_writer_error_does_not_publish_complete_audit(self):
        self.manifest_file([self.source("train", "train")])
        with patch.object(datasets, "_pyarrow", return_value=(FakeArrow, FakeParquet)), \
                patch.object(FakeParquet, "write_table", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                datasets.prepare(self.manifest, self.output, run=True)
        self.assertTrue((self.output / "preparation_started.json").is_file())
        self.assertFalse((self.output / "dataset_audit.json").exists())

    def test_final_audit_failure_cannot_leave_a_complete_looking_file(self):
        self.manifest_file([self.source('train', 'train')])
        with patch.object(datasets, '_pyarrow', return_value=(FakeArrow, FakeParquet)), \
                patch('control.contracts.os.fsync', side_effect=[None, OSError('audit disk failure')]):
            with self.assertRaises(OSError):
                datasets.prepare(self.manifest, self.output, run=True)
        self.assertFalse((self.output/'dataset_audit.json').exists())

    def test_prepared_inputs_require_complete_matching_audit(self):
        self.manifest_file([self.source('train', 'train')])
        with patch.object(datasets, '_pyarrow', return_value=(FakeArrow, FakeParquet)):
            report = datasets.prepare(self.manifest, self.output, run=True)
        paths = {'train': self.output/'train.parquet'}
        verified = datasets.prepared(paths, mode='adapted_upstream', counts={'train': 1})
        self.assertIn('train', verified)
        audit_path = self.output/'dataset_audit.json'
        for changes in ({'complete': False}, {'counts': {'train': 2}}, {'counts': {'train': True}}, {'mode': 'author_preprocessed'}):
            audit_path.write_text(json.dumps({**report, **changes}), encoding='utf-8')
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                datasets.prepared(paths, mode='adapted_upstream', counts={'train': 1})
        audit_path.unlink()
        with self.assertRaises(ValueError):
            datasets.prepared(paths, mode='adapted_upstream')
        (self.output/'preparation_started.json').unlink()
        with self.assertRaises(ValueError):
            datasets.prepared(paths, mode='adapted_upstream')

    def test_prepared_data_hash_and_completion_marker_are_bound(self):
        self.manifest_file([self.source('train', 'train')])
        with patch.object(datasets, '_pyarrow', return_value=(FakeArrow, FakeParquet)):
            datasets.prepare(self.manifest, self.output, run=True)
        paths = {'train': self.output/'train.parquet'}
        marker = self.output/'preparation_started.json'
        original = marker.read_bytes()
        marker.write_text(json.dumps({'manifest_sha256': 'a'*64, 'complete': False}), encoding='utf-8')
        with self.assertRaises(ValueError):
            datasets.prepared(paths, mode='adapted_upstream')
        marker.write_bytes(original)
        paths['train'].write_bytes(b'changed after conversion')
        with self.assertRaises(ValueError):
            datasets.prepared(paths, mode='adapted_upstream')

    def test_completed_conversion_can_be_moved_as_one_directory(self):
        self.manifest_file([self.source('train', 'train')])
        with patch.object(datasets, '_pyarrow', return_value=(FakeArrow, FakeParquet)):
            report = datasets.prepare(self.manifest, self.output, run=True)
        self.assertFalse(Path(report['outputs']['train']['path']).is_absolute())
        moved = self.root/'moved'
        self.output.rename(moved)
        self.assertIn('train', datasets.prepared({'train': moved/'train.parquet'}, mode='adapted_upstream'))
        for previous in ('/legacy/server/train.parquet', 'D:/legacy/train.parquet'):
            report['outputs']['train']['path'] = previous
            (moved/'dataset_audit.json').write_text(json.dumps(report), encoding='utf-8')
            self.assertIn('train', datasets.prepared({'train': moved/'train.parquet'}, mode='adapted_upstream'))

    def test_json_pointer_indices_and_source_ids_are_unambiguous(self):
        for pointer in ('/items/01', '/items/\u0661'):
            with self.subTest(pointer=pointer), self.assertRaises(ValueError):
                datasets.pointer({'items': ['a', 'b']}, pointer)
        source = self.source('train', 'train', rows=[{'q': 'Question?', 'a': 'Answer', 'id': ' '}])
        source['mapping']['id'] = '/id'
        self.manifest_file([source])
        with self.assertRaises(ValueError):
            datasets.build_plan(self.manifest)

    def test_source_and_manifest_are_rechecked_before_completion(self):
        self.manifest_file([self.source('train', 'train')])
        original = FakeParquet.write_table
        def write(table, path, compression):
            original(table, path, compression)
            self.manifest.write_text('{}', encoding='utf-8')
        with patch.object(datasets, '_pyarrow', return_value=(FakeArrow, FakeParquet)), \
                patch.object(FakeParquet, 'write_table', side_effect=write), self.assertRaises(ValueError):
            datasets.prepare(self.manifest, self.output, run=True)
        self.assertFalse((self.output/'dataset_audit.json').exists())
    def test_author_parquet_is_byte_preserved(self):
        source = self.source("author", "train")
        path = self.root / "author.parquet"
        path.write_bytes(b"fixture stand-in parquet original bytes")
        for key in ("mapping", "data_source", "ability"):
            source.pop(key)
        source.update(path=path.name, sha256=datasets.file_hash(path), format="parquet")
        self.manifest_file([source], mode="author_preprocessed")
        with patch.object(datasets, "read_rows", return_value=iter([self.ready_row()])), \
                patch.object(datasets, "_pyarrow", return_value=(FakeArrow, FakeParquet)):
            report = datasets.prepare(self.manifest, self.output, run=True)
        self.assertEqual((self.output / "train.parquet").read_bytes(), path.read_bytes())
        self.assertTrue(report["official_author_preprocessing_claim"])
        self.assertFalse(report["author_provenance_independently_verified"])
    def test_author_mode_cannot_silently_map(self):
        self.manifest_file([self.source("author", "train")], mode="author_preprocessed")
        with self.assertRaisesRegex(ValueError, "preserves Parquet"):
            datasets.prepare(self.manifest)
    def test_input_changed_during_read_fails(self):
        source = self.source("train", "train")
        self.manifest_file([source])
        def mutate(*args):
            yield {"q": "Q", "a": "A"}
            (self.root / source["path"]).write_text("modified", encoding="utf-8")
        with patch.object(datasets, "read_rows", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed during"):
                datasets.prepare(self.manifest)
    def test_no_invented_target_from_complex_object(self):
        self.manifest_file([self.source("train", "train", answer={"proof": "final answer 4"})])
        with self.assertRaisesRegex(ValueError, "no inferred answer extraction"):
            datasets.prepare(self.manifest)
    def test_empty_blank_duplicate_key_input_fails(self):
        for text in ("", "\n", '{"q":"Q","q":"other","a":"A"}\n'):
            source = self.source("train", "train")
            path = self.root / source["path"]
            path.write_text(text, encoding="utf-8")
            source["sha256"] = datasets.file_hash(path)
            self.manifest_file([source])
            with self.assertRaises(ValueError):
                datasets.prepare(self.manifest)
    def test_author_reward_route_and_prompt_are_checked(self):
        row = self.ready_row()
        row["data_source"] = "gsm8k"
        with self.assertRaisesRegex(ValueError, "data_source"):
            datasets.validate_official_row(row)
        row = self.ready_row()
        row["prompt"].append({"role": "assistant", "content": "SecretAnswer"})
        with self.assertRaisesRegex(ValueError, "system/user"):
            datasets.validate_official_row(row)
        row = self.ready_row()
        row["reward_model"]["ground_truth"] = "SecretAnswer"
        with self.assertRaisesRegex(ValueError, "ground_truth.target"):
            datasets.validate_official_row(row)
    def test_tool_expectation_is_explicit_not_invented(self):
        source = self.source("train", "train", rows=[{"q": "Q", "a": "A", "calls": 2}])
        self.manifest_file([source])
        _, rows, _ = datasets.build_plan(self.manifest)
        self.assertNotIn("tool_call_count", rows["train"][0])
        source["mapping"]["tool_call_count"] = "/calls"
        self.manifest_file([source])
        _, rows, _ = datasets.build_plan(self.manifest)
        self.assertEqual(rows["train"][0]["tool_call_count"], 2)
    def test_preserve_row_order_and_no_test_claim_from_train(self):
        source = self.source("train", "train", rows=[{"q": "Z?", "a": "1"}, {"q": "A?", "a": "2"}])
        self.manifest_file([source])
        report, rows, _ = datasets.build_plan(self.manifest)
        self.assertEqual([row["question"] for row in rows["train"]], ["Z?", "A?"])
        self.assertFalse(report["training_inputs_present"])
        self.assertNotIn("test", report["counts"])
    def test_inspect_source_prints_shapes_not_answers(self):
        source = self.source("train", "train", question="secret_question", answer="secret_answer")
        report = datasets.inspect_source(self.root / source["path"], "jsonl")
        self.assertFalse(Path(report['path']).is_absolute())
        self.assertEqual(report['path_base'], 'working_directory')
        self.assertEqual(report["first_distinct_shapes"], [{"q": "str", "a": "str"}])
        self.assertEqual(report["rows"], 1)
        self.assertNotIn("secret_question", json.dumps(report))
        self.assertNotIn("secret_answer", json.dumps(report))
    @unittest.skipUnless(importlib.util.find_spec("pyarrow") is not None, "pyarrow not installed; real Parquet roundtrip unverified")
    def test_real_parquet_roundtrip_when_dependency_present(self):
        grouped = self.source('grouped', 'train', rows=[dict(q='Variant?', a='A', task='original')])
        grouped['grouping'] = dict(namespace='benchmark', fields=dict(task='/task'))
        self.manifest_file([self.source("train", "train"), grouped, self.source("valid", "validation", "Validation?")])
        report = datasets.prepare(self.manifest, self.output, run=True)
        self.assertTrue(report["complete"])
        restored = list(datasets.read_rows(self.output / "train.parquet", "parquet"))
        self.assertEqual(restored[0]["question"], "What is 2+2?")
        self.assertEqual(restored[0]["reward_model"]["ground_truth"]["target"], ["4"])
        self.assertFalse(datasets.group_keys(restored[0]))
        self.assertEqual(datasets.group_keys(restored[1]), datasets.mapped_groups(dict(task='original'), grouped['grouping']))

    @unittest.skipUnless(importlib.util.find_spec('pyarrow') is not None, 'pyarrow is required for native freeze integration')
    def test_real_converted_splits_reach_native_freeze_and_reject_missing_audit(self):
        from tests.isolation import deployment
        import study
        sources = [self.source('train', 'train', rows=[{'q': f'Training question {i}?', 'a': str(i), 'task': str(i)} for i in range(256)]),
            self.source('valid', 'validation', rows=[dict(q='Validation question?', a='A', task='validation')]),
            self.source('test', 'test', rows=[dict(q='Test question?', a='A', task='test')])]
        for source in sources:
            source['grouping'] = dict(namespace='benchmark', fields=dict(task='/task'))
        datasets.prepare(self.manifest_file(sources), self.output, run=True)
        config = experiment.read(experiment.HERE/'configs/contextual.json')
        config = study.matrix(config, seeds=[42], conditions=['evocond'])['evocond/seed42']
        model, encoder = self.root/'model', self.root/'encoder'
        model.mkdir()
        encoder.mkdir()
        architecture = dict(model_type='qwen3', hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
            num_attention_heads=32, num_key_value_heads=8, vocab_size=151936, max_position_embeddings=32768)
        (model/'config.json').write_text(json.dumps(architecture), encoding='utf-8')
        for name in ('tokenizer_config.json', 'tokenizer.json'):
            (model/name).write_text('{}', encoding='utf-8')
        # Tiny byte fixtures exercise freezing without loading a model or GPU.
        (model/'model.safetensors').write_bytes(b'CPU_FIXTURE_NOT_MODEL_WEIGHTS')
        (encoder/'model.safetensors').write_bytes(b'CPU_FIXTURE_NOT_ENCODER_WEIGHTS')
        isolation, inspection = deployment()
        for name, value in (('isolation.json', isolation), ('inspection.json', inspection)):
            (self.root/name).write_text(json.dumps(value), encoding='utf-8')
        config.update(model=str(model), encoder=str(encoder), official_root=str(experiment.HERE/'baselines/SEARL-official'),
            isolation_config=str(self.root/'isolation.json'), isolation_receipt=str(self.root/'inspection.json'),
            datasets={name: str(self.output/(name+'.parquet')) for name in datasets.SPLITS})
        frozen = experiment.freeze(config)
        self.assertEqual(set(frozen['fingerprints']['dataset_preparation']), set(datasets.SPLITS))
        self.assertEqual(frozen['split_audit']['datasets']['train']['rows'], 256)
        self.assertTrue(frozen['split_audit']['group_metadata_complete'])
        import pyarrow as pa
        import pyarrow.parquet as pq
        heldout = self.output/'test.parquet'
        original = heldout.read_bytes()
        row = list(datasets.read_rows(heldout, 'parquet'))[0]
        row['extra_info']['split_group_keys'] = datasets.mapped_groups(dict(task='0'), sources[0]['grouping'])
        pq.write_table(pa.Table.from_pylist([row]), heldout)
        with self.assertRaisesRegex(ValueError, 'task/template groups overlap'):
            experiment.freeze(config)
        heldout.write_bytes(original)
        (self.output/'dataset_audit.json').unlink()
        with self.assertRaisesRegex(ValueError, 'dataset_audit'):
            experiment.freeze(config)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
