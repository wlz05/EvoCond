# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Plan matched bootstrap/train configs for construction x historical-view controls.'
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import experiment as experiment
from bootstrap import FORMAT as BOOTSTRAP_FORMAT
from control.budget import policy
def variants(config, bootstrap, destination, *, max_edges=2, min_tasks=2, seed=42):
    if config.get("format") != experiment.FORMAT or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", config.get("run_id", "")):
        raise ValueError("Expected a short identified experiment configuration")
    if bootstrap.get("format") != BOOTSTRAP_FORMAT or not isinstance(bootstrap.get("experiment"), dict):
        raise ValueError("Expected a bootstrap configuration")
    if not isinstance(config.get("teacher"), dict) or not config["teacher"]:
        raise ValueError("Expected explicit teacher training options")
    if bootstrap["experiment"].get("teacher"):
        raise ValueError("Bootstrap must collect without an already fitted teacher")
    training_budget = experiment.settings(config.get("budget", {}))
    collection_budget = experiment.settings(bootstrap["experiment"].get("budget", {}), collection=True)
    if collection_budget["rollouts"] != 1:
        raise ValueError("Bootstrap must declare one rollout")
    if experiment.rollout_contract(training_budget) != experiment.rollout_contract(collection_budget):
        raise ValueError("Bootstrap/training rollout token and turn budgets differ")
    teacher = config.get("teacher", {})
    experiment.teacher_options(teacher)
    if teacher.get("value_contracts", True) is not True:
        raise ValueError("This bootstrap collects with value contracts enabled")
    for key in ("model", "encoder", "official_root", "isolation_config", "isolation_receipt", "datasets"):
        if key not in config or config[key] != bootstrap["experiment"].get(key):
            raise ValueError(f"Bootstrap/training must declare identical {key}")
    if teacher.get("encoder_id") != bootstrap.get("encoder_id"):
        raise ValueError("Bootstrap/training encoder identities differ")
    for key, default in (("binding_strategy", "reject"), ("binding_credit", "native_only"),
                         ("guidance_mode", "legacy"), ("trusted_alias_tools", []), ("reuse_boundary_mode", "off")):
        if teacher.get(key, default) != bootstrap.get(key, default):
            raise ValueError(f"Bootstrap/training must use identical {key}")
    sparse = policy({"mode": "sparse", "max_incident_edges": max_edges, "min_matching_tasks": min_tasks,
                     "min_abs_score": 0.0, "selection": "score", "seed": seed})
    views = {"dense": {"mode": "dense"}, "sparse_score": sparse,
             "sparse_hash": {**sparse, "selection": "hash"}, "no_edges": {"mode": "none"}}
    result = {}
    for construction in ("legacy", "portable"):
        for view, spec in views.items():
            name = construction+"_"+view
            folder = destination/name
            value, collection = deepcopy(config), deepcopy(bootstrap)
            value["run_id"] = config["run_id"]+"_"+name
            collection["experiment"]["run_id"] = value["run_id"]+"_fit"
            options = {"tool_construction": construction, "structure_policy": spec}
            value["teacher"].update(deepcopy(options))
            collection.update(deepcopy(options))
            value["teacher"].update(teacher_path=str(folder/"bootstrap/teacher/teacher.json"),
                                     alignment_path=str(folder/"bootstrap/alignment/alignment.json"))
            result[name] = {"experiment": value, "bootstrap": collection,
                            "bootstrap_root": str(folder/"bootstrap"), "experiment_root": str(folder/"experiment")}
    return result
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bootstrap-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-edges", type=int, default=2)
    parser.add_argument("--min-tasks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    values = variants(experiment.read(args.config), experiment.read(args.bootstrap_config), args.output.resolve(),
                      max_edges=args.max_edges, min_tasks=args.min_tasks, seed=args.seed)
    if args.run:
        args.output.mkdir(parents=True, exist_ok=False)
        for name, condition in values.items():
            folder = args.output/name
            experiment.write_new(folder/"experiment.json", condition["experiment"])
            experiment.write_new(folder/"bootstrap.json", condition["bootstrap"])
        experiment.write_new(args.output/"conditions.json", values)
    print(json.dumps({"run": args.run, "gpu_started": False, "conditions": list(values),
        "claim": "independent online-training configs only; no same-prefix causal comparison",
        "next": "Inspect configs, collect/fit each matching bootstrap, then freeze new roots. Do not automatically launch all eight."}, indent=2))
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
