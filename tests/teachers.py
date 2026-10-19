# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from control.pairing import collect_teacher_policy_pairs, snapshot_hash
from control.ranking import build_alignment_pairs, describe, prepare_alignment, rank
from control.teacher import PROVENANCE_FIELDS
from teacher import fit_frozen_teacher, main, split_groups
from tests.rankings import registry
class Encoder:
    encoder_id = "synthetic-encoder-3-v1"
    def encode(self, text):
        return [1., 0., 0.]
def fixture():
    splits = {"train": [f"q{i}" for i in range(6)],
              "teacher_fit": ["q0", "q1", "q2"],
              "teacher_calibration": ["q3", "q4", "q5"],
              "validation": ["dev"], "test": ["sealed"]}
    memory, encoder = registry(), Encoder()
    alignment_pairs = []
    for query_id in splits["teacher_fit"]:
        alignment_pairs.extend(build_alignment_pairs(memory, encoder, query_id=query_id, query=query_id))
    aligned = prepare_alignment(alignment_pairs, encoder_id=encoder.encoder_id,
                                train_ids=set(splits["teacher_fit"]),
                                calibration_ids=set(splits["teacher_calibration"]),
                                validation_ids={"dev"}, test_ids={"sealed"})
    records = []
    def synthetic_evaluator(request):
        # Unit fixture only; never evidence of live rollout or method quality.
        return {"executed": True, "action_scope": "next_retrieval_only",
                "alpha": request.alpha, "seed": request.seed,
                "reward": float(request.alpha == .25), "success": request.alpha == .25,
                "trace_hash": snapshot_hash([request.query_id, request.alpha, request.seed]),
                **{name: getattr(request, name) for name in PROVENANCE_FIELDS}}
    for query_id in splits["train"]:
        view = describe(memory, encoder, aligned, query=query_id)
        state = {"messages": [{"role": "user", "content": query_id}], "observations": [],
                 "subtasks": {"ready": [query_id]}, "environment_snapshot": {"fixture": True}}
        records.extend(collect_teacher_policy_pairs(
            evaluator=synthetic_evaluator, query_id=query_id, decision_id="turn-0", split="train",
            state=state, registry=memory, candidates=view["candidates"], alignment=aligned,
            features=view["features"], query_embedding=view["query_embedding"],
            encoder_id=encoder.encoder_id, policy_hash="a" * 64, seeds=[1, 2]))
    return splits, memory, encoder, alignment_pairs, aligned, records
class TeacherWorkflowTests(unittest.TestCase):
    def test_alignment_collection_fit_reload_and_live_ranking(self):
        splits, memory, encoder, _, aligned, records = fixture()
        model, report = fit_frozen_teacher(splits, records, aligned)
        restored = json.loads(json.dumps(model, allow_nan=False))
        before = deepcopy(memory)
        result = rank(memory, encoder, aligned, restored, query="new visible question", policy_hash="a" * 64)
        self.assertEqual(report["matched_decisions"], 12)
        self.assertEqual(result["teacher"]["structure_weight"], .25)
        self.assertEqual(memory, before)
        with self.assertRaisesRegex(ValueError, "current frozen Qwen"):
            rank(memory, encoder, aligned, restored, query="new visible question", policy_hash="b" * 64)
    def test_malformed_or_different_alignment_rejected_before_fit(self):
        splits, _, _, _, aligned, records = fixture()
        for field in ("format", "view_schema", "matrix", "dimension"):
            bad = deepcopy(aligned)
            del bad[field]
            updated = deepcopy(records)
            for row in updated:
                row["alignment_hash"] = snapshot_hash(bad)
                for action in row["actions"]:
                    action["alignment_hash"] = row["alignment_hash"]
            with self.subTest(field=field), self.assertRaises(ValueError):
                fit_frozen_teacher(splits, updated, bad)
        bad = deepcopy(aligned)
        bad["matrix"][0][0] += .1
        with self.assertRaisesRegex(ValueError, "freezing this exact alignment"):
            fit_frozen_teacher(splits, records, bad)
    def test_split_overlap_and_missing_calibration_are_rejected(self):
        splits, _, _, _, aligned, records = fixture()
        for change in ({"teacher_calibration": ["q0", "q4", "q5"]},
                       {"test": ["q3"]}, {"train": ["q0", "q1", "q2"]},
                       {"teacher_fit": ["q0", "q0", "q1"]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                split_groups({**splits, **change})
        with self.assertRaisesRegex(ValueError, "three independent"):
            fit_frozen_teacher(splits, [row for row in records if row["query_id"] in splits["teacher_fit"]], aligned)
    def test_cli_read_only_default_and_new_directory_only(self):
        splits, _, encoder, pairs, aligned, records = fixture()
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for name, data in (("splits.json", splits), ("alignment.json", aligned)):
                (root / name).write_text(json.dumps(data), encoding="utf-8")
            for name, data in (("alignment_pairs.jsonl", pairs), ("pairs.jsonl", records)):
                (root / name).write_text("\n".join(json.dumps(row) for row in data), encoding="utf-8")
            for stage, inputs in (("alignment", ["--alignment-pairs", str(root / "alignment_pairs.jsonl"),
                                                 "--encoder-id", encoder.encoder_id]),
                                  ("fit", ["--pairs", str(root / "pairs.jsonl"),
                                           "--alignment", str(root / "alignment.json")])):
                output = root / stage
                command = [stage, "--splits", str(root / "splits.json"), "--output", str(output), *inputs]
                with redirect_stdout(io.StringIO()) as log:
                    main(command)
                self.assertFalse(output.exists())
                self.assertFalse(json.loads(log.getvalue())["run"])
                with redirect_stdout(io.StringIO()):
                    main([*command, "--run"])
                artifact = "alignment.json" if stage == "alignment" else "teacher.json"
                self.assertTrue((output / artifact).is_file())
                provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
                self.assertFalse(provenance["qwen_updated"])
                with self.assertRaises(FileExistsError):
                    main([*command, "--run"])
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
