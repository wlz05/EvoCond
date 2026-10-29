# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Run paired EVOLT inference arms on a SEARL task file without training."""
import argparse
import hashlib
import json
from pathlib import Path
from core.agent import Agent
from core.data import Corpus, Source, TEST
from learn.infer import Batch, Qwen
from learn.metric import measure, prediction
from main import local
def digest(path: Path) -> str:
    """Hash one dataset or model config without reading it all into memory."""
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            state.update(chunk)
    return state.hexdigest()
def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Paired EVOLT evaluation using SEARL benchmark resources; not a SEARL RL reproduction")
    root.add_argument("data", help="Existing JSONL, JSON, CSV, or Parquet benchmark file under the project root")
    root.add_argument("output", help="New result directory under the project root")
    root.add_argument("--name", choices=TEST, required=True)
    root.add_argument("--split", choices=("valid", "test"), default="test")
    root.add_argument("--model", default="autodl-tmp/model/qwen4b")
    root.add_argument("--state", default="memoryone.json")
    root.add_argument("--arms", default="plain,tool")
    root.add_argument("--backend", choices=("native", "searl"), default="searl")
    root.add_argument("--search-url", default="http://127.0.0.1:5003/retrieve")
    root.add_argument("--sandbox-url", default="http://127.0.0.1:8080")
    root.add_argument("--limit", type=int, default=2)
    root.add_argument("--offset", type=int, default=0)
    root.add_argument("--tokens", type=int, default=2048)
    root.add_argument("--tool-rounds", type=int, default=5)
    root.add_argument("--fusion", choices=("teacher", "semantic", "structure", "equal"), default="teacher")
    root.add_argument("--retrieve-limit", type=int, default=5)
    root.add_argument("--restrict-tools", action="store_true")
    root.add_argument("--run", action="store_true", help="Actually load the model and generate outputs; default is a read-only preflight")
    root.add_argument("--restart", action="store_true", help="Replace existing prediction JSONL files for the selected arms")
    return root
def main() -> None:
    args = parser().parse_args()
    arms = [value.strip() for value in args.arms.split(",") if value.strip()]
    if not arms or len(arms) != len(set(arms)) or set(arms) - {"plain", "tool", "evolt"}:
        raise ValueError("arms must be a unique subset of plain,tool,evolt")
    if args.limit < 1 or args.offset < 0 or args.tokens < 1 or args.tool_rounds < 1 or args.retrieve_limit < 1:
        raise ValueError("limits and tool rounds must be positive; offset must be nonnegative")
    source = Source()
    data_path = source.path(args.data)
    model_path = source.path(args.model)
    output_path = source.path(args.output)
    summary_path = output_path / "summary.json"
    if args.run and summary_path.exists() and not args.restart:
        raise FileExistsError(f"summary exists; use a new output directory or --restart: {summary_path}")
    corpus = Corpus()
    source_rows = corpus.source.read(args.data)
    if any(corpus.name(row) not in ("", args.name) for row in source_rows):
        raise ValueError("benchmark file contains a recognized dataset label that conflicts with --name")
    records = corpus.load(args.data, args.name, args.split)
    if any(row.dataset != args.name or row.split != args.split for row in records):
        raise ValueError("benchmark file contains another dataset or split")
    if len({row.key for row in records}) != len(records):
        raise ValueError("benchmark file contains duplicate stable sample keys")
    selected = records[args.offset:args.offset + args.limit]
    if len(selected) != args.limit:
        raise ValueError("requested evaluation range exceeds the dataset")
    config_path = model_path / "config.json"
    if not model_path.is_dir() or not config_path.is_file():
        raise FileNotFoundError(f"the baseline requires a full Qwen3-4B model directory with config.json: {model_path}")
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    if model_config.get("model_type") != "qwen3" or model_config.get("hidden_size") != 2560 or model_config.get("num_hidden_layers") != 36:
        raise ValueError("model config does not match the published Qwen3-4B backbone")
    state_path = local(args.state)
    if "evolt" in arms and not state_path.is_file():
        raise FileNotFoundError(f"EVOLT arm requires an existing trained memory: {state_path}")
    agent = Agent.open(state_path) if "evolt" in arms else None
    model_hash = digest(config_path)
    protocol = {"benchmark": "SEARL task resources, EVOLT inference protocol", "data": args.data, "data_sha256": digest(data_path), "dataset": args.name, "split": args.split, "model": args.model, "model_config_sha256": model_hash, "state": args.state if agent else None, "state_sha256": digest(state_path) if agent else None, "backend": args.backend, "search_url": args.search_url if args.backend == "searl" else None, "sandbox_url": args.sandbox_url if args.backend == "searl" else None, "tokens": args.tokens, "tool_rounds": args.tool_rounds, "fusion": args.fusion if agent else None, "retrieve_limit": args.retrieve_limit if agent else None, "restrict_tools": args.restrict_tools if agent else None}
    preview = {"data_rows": len(records), "selected": len(selected), "first_key": selected[0].key, "last_key": selected[-1].key, "arms": arms, "data_sha256": protocol["data_sha256"], "model_config_sha256": model_hash, "memory_tools": sorted(agent.graph.tools) if agent else None, "memory_edges": len(agent.graph.edges) if agent else None, "teacher_updates": agent.teacher.seen if agent else None, "score_warning": "The local deterministic score is not SEARL's Qwen3-32B judge score."}
    print(json.dumps(preview, ensure_ascii=True, indent=2))
    if not args.run:
        return
    if any(arm != "plain" for arm in arms) and args.backend == "searl":
        check = Qwen(args.model, args.tokens, True, args.tool_rounds, backend="searl", search_url=args.search_url, sandbox_url=args.sandbox_url).check()
        print(json.dumps({"tool_preflight": check}, ensure_ascii=True))
    results = {}
    for arm in arms:
        policy = Qwen(args.model, args.tokens, arm != "plain", args.tool_rounds, guided=args.restrict_tools if arm == "evolt" else False, backend=args.backend, search_url=args.search_url, sandbox_url=args.sandbox_url)
        retrieve = (lambda row: agent.recall(row.question, limit=args.retrieve_limit, task=row.key, fusion=args.fusion)) if arm == "evolt" else None
        destination = str(Path(args.output) / f"{arm}.jsonl")
        Batch().run(records, destination, policy.answer, args.limit, args.offset, args.restart, retrieve, {"protocol": {**protocol, "arm": arm}})
        rows = prediction(Source().read(destination))
        summary = measure(selected, rows)
        results[arm] = {"output": destination, "scores": summary}
        print(json.dumps({"arm": arm, "correct": summary["overall"]["correct"], "samples": summary["overall"]["samples"], "pass@1_local": summary["overall"]["pass@1"]}, ensure_ascii=True))
        del policy
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"protocol": protocol, "selected_keys": [row.key for row in selected], "arms": results}, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path)}, ensure_ascii=True))
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
