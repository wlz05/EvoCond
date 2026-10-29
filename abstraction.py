# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Read-only source/structure diagnostics; no encoder, model or tool execution.'
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
from control import dynamic, edges, boundary
from control.ranking import active_tools
from control.budget import select
from control.primitive import profile
def audit(registry, *, context_vector=None, encoder_id=None, max_edges=2, min_tasks=2, seed=42,
          reuse_boundary_mode="off", current_user_text=None):
    dynamic.validate(registry)
    original = deepcopy(registry)
    active = active_tools(registry)
    profiles = [{"version": row["version"], "name": row["name"],
                 "profile": profile(registry["semantic"][row["version"]])} for row in active]
    views = edges.batch_edge_views(registry, context_vector=context_vector, encoder_id=encoder_id,
        reuse_boundary_mode=reuse_boundary_mode, current_user_text=current_user_text)
    sparse = {"mode": "sparse", "max_incident_edges": max_edges, "min_matching_tasks": min_tasks,
              "min_abs_score": 0.0, "selection": "score", "seed": seed}
    conditions = {"dense": {"mode": "dense"}, "sparse_score": sparse,
                  "sparse_hash": {**sparse, "selection": "hash"}, "no_history_edges": {"mode": "none"}}
    receipts = {name: select(views, policy, allowed_versions=[r["version"] for r in active])[1]
                for name, policy in conditions.items()}
    if registry != original:
        raise RuntimeError("Read-only abstraction audit mutated its snapshot")
    return {"format": "evocond_abstraction_audit_v1", "registry_hash": dynamic.digest(registry),
            "active_tools": len(active), "stored_versions": len(registry["semantic"]),
            "tool_profiles": profiles, "structure_conditions": receipts,
            "reuse_boundary": boundary.receipt(views, reuse_boundary_mode, current_user_text),
            "edge_boundary_checks": {key: row["reuse_boundary"] for key, row in views.items() if "reuse_boundary" in row},
            "query_context_supplied": context_vector is not None, "encoder_id": encoder_id,
            "gpu_used": False, "generated_code_executed": False, "accuracy_measured": False,
            "warning": "No new-task performance or generality claim. No context means zero historical utility. "
                       "Hash and score use the same degree budget, not necessarily the same selected-edge count."}
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--context", type=Path, help="JSON numeric vector of the visible query; no answers")
    parser.add_argument("--encoder-id")
    parser.add_argument("--reuse-boundary-mode", choices=("off", "explicit"), default="off")
    parser.add_argument("--current-user-file", type=Path, help="Trusted visible user text, never generated goals or answers")
    parser.add_argument("--max-edges", type=int, default=2)
    parser.add_argument("--min-tasks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optional new JSON file; existing results are never overwritten")
    args = parser.parse_args()
    if bool(args.context) != bool(args.encoder_id):
        parser.error("--context and --encoder-id must be supplied together")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    context = json.loads(args.context.read_text(encoding="utf-8")) if args.context else None
    report = audit(registry, context_vector=context, encoder_id=args.encoder_id,
                   max_edges=args.max_edges, min_tasks=args.min_tasks, seed=args.seed,
                   reuse_boundary_mode=args.reuse_boundary_mode,
                   current_user_text=args.current_user_file.read_text(encoding='utf-8') if args.current_user_file else None)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
