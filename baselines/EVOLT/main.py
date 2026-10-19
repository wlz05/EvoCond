# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Provide the EVOLT command line interface."""
import argparse
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from core.agent import Agent
from core.data import Corpus, Source, baseline
from core.searl import SearlTools
from learn.bench import evaluate
from learn.infer import Batch, Qwen
from learn.metric import measure, prediction
from learn.tune import tune
ROOT = Path(__file__).resolve().parent
STATE = ROOT.parent / "state"
def local(value: str) -> Path:
    """Resolve one safe relative state path."""
    text = str(value)
    path = Path(text)
    windows = PureWindowsPath(text)
    posix = PurePosixPath(text)
    if path.is_absolute() or windows.is_absolute() or windows.drive or posix.is_absolute() or ".." in windows.parts or ".." in posix.parts:
        raise ValueError("state path must be relative")
    target = (STATE / path).resolve()
    if target == STATE or STATE not in target.parents:
        raise ValueError("state path must stay within the project")
    return target
def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="evolt")
    root.add_argument("--state", default="memory.json")
    commands = root.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add")
    add.add_argument("name")
    add.add_argument("text")
    add.add_argument("--code", default="")
    add.add_argument("--schema", default="")
    add.add_argument("--reward", type=float, default=0.0)
    add.add_argument("--unverified", action="store_true")
    resolve = commands.add_parser("resolve")
    resolve.add_argument("token")
    resolve.add_argument("reward", type=float)
    rollback = commands.add_parser("rollback")
    rollback.add_argument("token")
    find = commands.add_parser("find")
    find.add_argument("query")
    find.add_argument("--context", default="")
    find.add_argument("--parts", default="")
    find.add_argument("--task", default="")
    find.add_argument("--limit", type=int, default=5)
    find.add_argument("--fusion", choices=("teacher", "semantic", "structure", "equal"), default="teacher")
    use = commands.add_parser("use")
    use.add_argument("query")
    use.add_argument("chosen")
    use.add_argument("reward", type=float)
    use.add_argument("--parts", default="")
    use.add_argument("--task", default="")
    commands.add_parser("show").add_argument("--compact", action="store_true")
    prune = commands.add_parser("prune")
    prune.add_argument("age", type=float)
    prune.add_argument("floor", type=float)
    commands.add_parser("bench")
    data = commands.add_parser("data")
    data.add_argument("file", nargs="?", default="")
    data.add_argument("--name", default="")
    data.add_argument("--split", default="")
    infer = commands.add_parser("infer")
    infer.add_argument("data")
    infer.add_argument("output")
    infer.add_argument("--model", default="autodl-tmp/model/qwen4b")
    infer.add_argument("--name", default="")
    infer.add_argument("--split", default="")
    infer.add_argument("--limit", type=int, default=10)
    infer.add_argument("--offset", type=int, default=0)
    infer.add_argument("--tokens", type=int, default=512)
    infer.add_argument("--mode", choices=("plain", "tool", "evolt"), default="plain")
    infer.add_argument("--fusion", choices=("teacher", "semantic", "structure", "equal"), default="teacher")
    infer.add_argument("--retrieve-limit", type=int, default=5)
    infer.add_argument("--restrict-tools", action="store_true")
    infer.add_argument("--restart", action="store_true")
    infer.add_argument("--tool-backend", choices=("native", "searl"), default="native")
    infer.add_argument("--tool-rounds", type=int, default=None)
    infer.add_argument("--search-url", default="http://127.0.0.1:5003/retrieve")
    infer.add_argument("--sandbox-url", default="http://127.0.0.1:8080")
    check = commands.add_parser("tool-check")
    check.add_argument("--search-url", default="http://127.0.0.1:5003/retrieve")
    check.add_argument("--sandbox-url", default="http://127.0.0.1:8080")
    score = commands.add_parser("score")
    score.add_argument("data")
    score.add_argument("predictions")
    score.add_argument("--name", default="")
    score.add_argument("--split", default="")
    score.add_argument("--partial", action="store_true")
    train = commands.add_parser("train")
    train.add_argument("data", nargs="?", default="data/toolstar.jsonl")
    train.add_argument("--model", default="autodl-tmp/model/qwen4b")
    train.add_argument("--output", default="result/pilot")
    train.add_argument("--limit", type=int, default=8)
    train.add_argument("--steps", type=int, default=1)
    train.add_argument("--tokens", type=int, default=768)
    train.add_argument("--generations", type=int, default=4)
    train.add_argument("--batch", type=int, default=1)
    train.add_argument("--accum", type=int, default=4)
    train.add_argument("--seed", type=int, default=2)
    train.add_argument("--save", type=int, default=25)
    train.add_argument("--rate", type=float, default=1e-6)
    train.add_argument("--ability", choices=("all", "math", "qa"), default="all")
    train.add_argument("--tool-backend", choices=("native", "searl"), default="native")
    train.add_argument("--search-url", default="http://127.0.0.1:5003/retrieve")
    train.add_argument("--sandbox-url", default="http://127.0.0.1:8080")
    return root
def main() -> None:
    args = parser().parse_args()
    if args.command == "bench":
        print(json.dumps(evaluate(), ensure_ascii=True, indent=2, allow_nan=False))
        return
    if args.command == "tool-check":
        print(json.dumps(SearlTools(args.search_url, args.sandbox_url).check(), ensure_ascii=True, indent=2))
        return
    if args.command == "data":
        output = baseline() if not args.file else Corpus().summary(Corpus().load(args.file, args.name, args.split))
        print(json.dumps(output, ensure_ascii=True, indent=2, allow_nan=False))
        return
    if args.command == "score":
        records = Corpus().load(args.data, args.name, args.split)
        values = prediction(Source().read(args.predictions))
        records = [item for item in records if item.key in values] if args.partial else records
        output = measure(records, values)
        print(json.dumps(output, ensure_ascii=True, indent=2, allow_nan=False))
        return
    if args.command == "infer":
        if args.retrieve_limit < 1:
            raise ValueError("retrieval limit must be positive")
        if args.mode != "evolt" and (args.fusion != "teacher" or args.retrieve_limit != 5 or args.restrict_tools):
            raise ValueError("fusion ablations require evolt mode")
        if args.mode == "evolt" and args.tool_backend == "searl" and not local(args.state).is_file():
            raise FileNotFoundError(f"SEARL-resource EVOLT inference requires trained memory: {local(args.state)}")
        records = Corpus().load(args.data, args.name, args.split)
        rounds = args.tool_rounds if args.tool_rounds is not None else 5 if args.tool_backend == "searl" else 2
        policy = Qwen(args.model, args.tokens, args.mode != "plain", rounds=rounds, guided=args.restrict_tools, backend=args.tool_backend, search_url=args.search_url, sandbox_url=args.sandbox_url)
        if args.mode != "plain" and args.tool_backend == "searl":
            policy.check()
        agent = Agent.open(local(args.state)) if args.mode == "evolt" and local(args.state).exists() else Agent()
        retrieve = (lambda item: agent.recall(item.question, limit=args.retrieve_limit, task=item.key, fusion=args.fusion)) if args.mode == "evolt" else None
        config = {"protocol": {"mode": args.mode, "model": args.model, "state": args.state, "fusion": args.fusion, "budget": args.retrieve_limit, "restricted": args.restrict_tools, "tokens": args.tokens}} if args.mode == "evolt" else None
        if args.tool_backend == "searl":
            config = {"protocol": {"mode": args.mode, "model": args.model, "state": args.state if args.mode == "evolt" else None, "fusion": args.fusion if args.mode == "evolt" else None, "budget": args.retrieve_limit if args.mode == "evolt" else None, "restricted": args.restrict_tools, "tokens": args.tokens, "tool_backend": "searl", "tool_rounds": rounds, "search_url": args.search_url, "sandbox_url": args.sandbox_url}}
        output = Batch().run(records, args.output, policy.answer, args.limit, args.offset, args.restart, retrieve, config)
        print(json.dumps(output, ensure_ascii=True, indent=2, allow_nan=False))
        return
    if args.command == "train":
        output = tune(args.data, local(args.state), args.model, args.output, args.limit, args.steps, args.tokens, args.generations, args.batch, args.accum, args.seed, "" if args.ability == "all" else args.ability, args.save, args.rate, backend=args.tool_backend, search_url=args.search_url, sandbox_url=args.sandbox_url)
        print(json.dumps(output, ensure_ascii=True, indent=2, allow_nan=False))
        return
    path = local(args.state)
    agent = Agent.open(path) if path.exists() else Agent()
    if args.command == "add":
        proposal = agent.stage(args.name, args.text, args.code, args.schema)
        if args.unverified:
            key, decision = None, proposal.decision
        else:
            key, decision = agent.resolve(proposal, True, args.reward)
        agent.save(path)
        output = {"key": key, "token": proposal.token if args.unverified else None, "decision": decision.dump()}
    elif args.command == "resolve":
        proposal = agent.pending.get(args.token)
        if proposal is None:
            raise ValueError("proposal token is unknown")
        key, decision = agent.resolve(proposal, True, args.reward)
        agent.save(path)
        output = {"key": key, "decision": decision.dump()}
    elif args.command == "rollback":
        decision = agent.rollback(args.token)
        agent.save(path)
        output = {"key": None, "decision": decision.dump()}
    elif args.command == "find":
        context = [item for item in args.context.split(",") if item]
        parts = [item for item in args.parts.split("|") if item]
        output = agent.recall(args.query, context, args.limit, parts or None, args.task, args.fusion).dump()
    elif args.command == "use":
        parts = [item for item in args.parts.split("|") if item]
        trace = agent.recall(args.query, subtasks=parts or None, task=args.task)
        chosen = [item for item in args.chosen.split(",") if item]
        agent.commit(trace, chosen, args.reward)
        agent.save(path)
        output = agent.teacher.state(trace.features, trace.cold)
    elif args.command == "prune":
        output = {"removed": agent.prune(args.age, args.floor)}
        agent.save(path)
    else:
        output = agent.dump() if not args.compact else agent.teacher.state()
    print(json.dumps(output, ensure_ascii=True, indent=2, allow_nan=False))
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
