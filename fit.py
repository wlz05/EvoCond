# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Fit aligned retrieval and an abstaining teacher from paired train-only logs.'
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
from control.retrieval import fit_alignment
from control.fusion import fit as fit_teacher
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--alignment-pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()
def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
def prepare(splits: dict, pairs: list[dict], alignment_pairs: list[dict]) -> tuple[dict, dict, dict]:
    parts = {key: splits.get(key) for key in ("train", "validation", "test")}
    if any(not isinstance(value, list) or not value or len(value) != len(set(value))
           or any(not isinstance(identity, str) or not identity for identity in value)
           for value in parts.values()):
        raise ValueError("Each split needs distinct nonempty string question IDs")
    train, validation, test = (set(parts[key]) for key in ("train", "validation", "test"))
    if train & (validation | test) or validation & test:
        raise ValueError("Independent question identities overlap")
    identities = sorted(train | validation | test)
    indexed = {identity: index for index, identity in enumerate(identities)}
    teacher_rows = []
    for row in pairs:
        identity = row.get("query_id")
        if identity not in train:
            raise ValueError("Teacher pair is not from the training split")
        teacher_rows.append({"index": indexed[identity], "mode": row.get("mode"),
                             "seed": row.get("seed"), "features": row.get("features"),
                             "semantic_reward": row.get("semantic_reward"),
                             "structural_reward": row.get("structural_reward")})
    model, report = fit_teacher(teacher_rows,
                                train_indices=sorted(indexed[x] for x in train),
                                validation_indices=sorted(indexed[x] for x in validation),
                                test_indices=sorted(indexed[x] for x in test))
    alignment = fit_alignment(alignment_pairs, train_ids=train,
                              validation_ids=validation, test_ids=test)
    return model, alignment, report
def main() -> None:
    args = arguments()
    inputs = {name: Path(getattr(args, name)).resolve(strict=True)
              for name in ("splits", "pairs", "alignment_pairs")}
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory for frozen teacher artifacts")
    model, alignment, report = prepare(json.loads(inputs["splits"].read_text(encoding="utf-8")),
                                       _rows(inputs["pairs"]), _rows(inputs["alignment_pairs"]))
    provenance = {"input_sha256": {name: _digest(path) for name, path in inputs.items()},
                  "teacher_fit": report, "note": "Train fit only; no held-out claim"}
    print(json.dumps({"output": str(output), "run": args.run, **provenance}, ensure_ascii=False))
    if not args.run:
        return
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in (("teacher.json", model), ("alignment.json", alignment),
                          ("provenance.json", provenance)):
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
