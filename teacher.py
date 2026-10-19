# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Freeze alignment, then fit an action teacher from matched training continuations.'
from __future__ import annotations
import argparse
import json
from pathlib import Path
from control.dynamic import digest
from control.ranking import prepare_alignment, validate_alignment
from control import teacher
def split_groups(splits: dict) -> dict[str, set[str]]:
    names = ("train", "teacher_fit", "teacher_calibration", "validation", "test")
    parts = {}
    for name in names:
        values = splits.get(name)
        if (not isinstance(values, list) or not values
                or any(not isinstance(x, str) or not x.strip() for x in values)
                or len(values) != len(set(values))):
            raise ValueError(f"{name} requires distinct independent question identities")
        parts[name] = set(values)
    groups = [parts[key] for key in ("teacher_fit", "teacher_calibration", "validation", "test")]
    if any(groups[i] & groups[j] for i in range(4) for j in range(i+1, 4)):
        raise ValueError("Fit, calibration, validation and test groups overlap")
    if parts["train"] != parts["teacher_fit"] | parts["teacher_calibration"]:
        raise ValueError("Teacher fit/calibration must partition the declared training questions")
    return parts
def fit_frozen_teacher(splits: dict, records: list[dict], alignment: dict) -> tuple[dict, dict]:
    groups = split_groups(splits)
    validate_alignment(alignment)
    if not isinstance(records, list) or not records or any(not isinstance(row, dict) for row in records):
        raise ValueError("Executed teacher pair records are required")
    if any(row.get("alignment_hash") != digest(alignment) for row in records):
        raise ValueError("Collect teacher continuations after freezing this exact alignment")
    if set(alignment.get("train_ids", [])) != groups["teacher_fit"]:
        raise ValueError("Alignment was not fitted exclusively on the teacher fit partition")
    if set(alignment.get("calibration_ids", [])) != groups["teacher_calibration"]:
        raise ValueError("Alignment calibration provenance differs")
    model, report = teacher.fit(records, train_ids=groups["teacher_fit"],
                                       calibration_ids=groups["teacher_calibration"],
                                       validation_ids=groups["validation"], test_ids=groups["test"])
    if model["encoder_id"] != alignment.get("encoder_id") or model["embedding_dimension"] != alignment.get("dimension"):
        raise ValueError("Teacher and alignment encoders differ")
    return model, report
def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    align = sub.add_parser("alignment", help="Freeze a train-only alignment before collecting teacher labels")
    align.add_argument("--alignment-pairs", required=True)
    align.add_argument("--encoder-id", required=True)
    teacher = sub.add_parser("fit", help="Fit/calibrate on executed matched training actions")
    teacher.add_argument("--pairs", required=True)
    teacher.add_argument("--alignment", required=True)
    for command in (align, teacher):
        command.add_argument("--splits", required=True)
        command.add_argument("--output", required=True, help="A new artifact directory")
        command.add_argument("--run", action="store_true")
    return parser.parse_args(argv)
def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
def main(argv=None):
    args = arguments(argv)
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    parts = split_groups(splits)
    if args.stage == "alignment":
        artifact = prepare_alignment(_rows(args.alignment_pairs), encoder_id=args.encoder_id,
                                     train_ids=parts["teacher_fit"], calibration_ids=parts["teacher_calibration"],
                                     validation_ids=parts["validation"], test_ids=parts["test"])
        report = {"dimension": artifact["dimension"], "encoder_id": artifact["encoder_id"],
                  "next": "Collect real five-action continuations using this frozen alignment"}
        filename = "alignment.json"
    else:
        alignment = json.loads(Path(args.alignment).read_text(encoding="utf-8"))
        artifact, report = fit_frozen_teacher(splits, _rows(args.pairs), alignment)
        filename = "teacher.json"
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Use a new directory; existing experiment artifacts are immutable")
    provenance = {"stage": args.stage, "artifact_sha256": digest(artifact),
                  "splits_sha256": digest(splits), "report": report,
                  "qwen_updated": False, "official_rollout_started": False}
    print(json.dumps({"output": str(output), "run": args.run, **provenance}, ensure_ascii=False))
    if args.run:
        output.mkdir(parents=True, exist_ok=False)
        for name, value in ((filename, artifact), ("provenance.json", provenance)):
            (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
