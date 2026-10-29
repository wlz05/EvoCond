# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Paired SEARL-resource evaluation of EvoCond, without modifying EVOLT.'
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from fractions import Fraction
from hashlib import blake2b, sha256
import json
from pathlib import Path
import re
import string
import unicodedata
from typing import Any, Mapping
from control.joint import Controller, ToolMemory
from control.tools import Tools
NAMES = ("gsm8k", "math500", "aime24", "hotpotqa", "2wiki", "musique", "bamboogle")
CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
ALIASES = {"gsm8k": "gsm8k", "gsm": "gsm8k", "math500": "math500", "aime24": "aime24", "aime2024": "aime24", "hotpotqa": "hotpotqa", "hotpot": "hotpotqa", "2wiki": "2wiki", "2wikimultihopqa": "2wiki", "twowiki": "2wiki", "musique": "musique", "bamboogle": "bamboogle"}
@dataclass(frozen=True)
class Record:
    key: str
    question: str
    answer: object
    dataset: str
def digest(path: Path) -> str:
    state = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            state.update(chunk)
    return state.hexdigest()
def rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        source = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    elif path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        source = raw if isinstance(raw, list) else raw.get("data", raw.get("rows", [raw])) if isinstance(raw, dict) else []
    elif path.suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            source = list(csv.DictReader(stream))
    elif path.suffix == ".parquet":
        try:
            import pandas
        except ImportError as error:
            raise RuntimeError("Parquet input requires pandas and pyarrow") from error
        source = pandas.read_parquet(path).to_dict("records")
    else:
        raise ValueError("Data must be JSONL, JSON, CSV, or Parquet")
    if not isinstance(source, list) or any(not isinstance(item, dict) for item in source):
        raise ValueError("Dataset rows must be objects")
    return source
def nested(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", []):
            return value
    for parent in ("extra_info", "metadata", "reward_model"):
        child = row.get(parent)
        if isinstance(child, Mapping):
            for key in keys:
                value = child.get(key)
                if value not in (None, "", []):
                    return value
    return None
def corpus(path: Path, name: str, split: str) -> list[Record]:
    if name not in NAMES or split not in {"valid", "test"}:
        raise ValueError("Unsupported benchmark or split")
    output: list[Record] = []
    keys: set[str] = set()
    for index, row in enumerate(rows(path)):
        declared = nested(row, "dataset", "dataset_name", "benchmark", "data_source")
        label = re.sub(r"[^a-z0-9]", "", str(declared).casefold()) if declared is not None else ""
        if label in ALIASES and ALIASES[label] != name:
            raise ValueError(f"Dataset label differs from --name at row {index}")
        question = nested(row, "question", "input", "query", "problem")
        if question is None and isinstance(row.get("prompt"), list):
            messages = [m for m in row["prompt"] if isinstance(m, dict) and m.get("content")]
            users = [m for m in messages if m.get("role") == "user"]
            question = (users or messages)[-1].get("content") if messages else None
        answer = nested(row, "ground_truth", "answer", "answers", "target", "output", "response")
        if isinstance(answer, Mapping):
            answer = nested(answer, "target", "answer", "value", "ground_truth")
        if name == "gsm8k" and isinstance(answer, str) and "####" in answer:
            answer = answer.rsplit("####", 1)[-1].strip()
        if not str(question or "").strip() or answer in (None, "", []):
            raise ValueError(f"Missing question or reference at row {index}")
        rawkey = nested(row, "id", "index", "key")
        key = blake2b(f"{name}|{split}|{rawkey}|{str(question).strip()}".encode("utf-8"), digest_size=10).hexdigest()
        if key in keys:
            raise ValueError(f"Duplicate sample key at row {index}")
        keys.add(key)
        output.append(Record(key, str(question).strip(), answer, name))
    return output
def calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    output: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for match in CALL.finditer(text):
        try:
            raw = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str) or not isinstance(raw.get("arguments"), dict):
            continue
        name = raw["name"].casefold()
        if name not in seen:
            output.append((name, raw["arguments"]))
            seen.add(name)
    return output
def boxed(text: str) -> str:
    index = text.rfind("\\boxed")
    if index < 0:
        return ""
    start = text.find("{", index)
    if start < 0:
        return ""
    depth = 0
    for pos in range(start, len(text)):
        depth += (text[pos] == "{") - (text[pos] == "}")
        if depth == 0:
            return text[start + 1:pos].strip()
    return ""
def answer(text: str) -> str:
    final_message = text.rsplit("</tool_response>", 1)[-1]
    matches = re.findall(r"<answer>(.*?)</answer>", final_message, flags=re.DOTALL | re.IGNORECASE)
    return boxed(matches[-1]) if matches else ""
def canonical(text: str) -> str:
    if answer(text):
        return text
    final_message = text.rsplit("</tool_response>", 1)[-1]
    matches = re.findall(r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*\$?([^\n$]+)", final_message, flags=re.IGNORECASE)
    final = boxed(final_message) or (matches[-1].strip().strip("*` ") if matches else "")
    return f"{text}\n<answer>\\boxed{{{final}}}</answer>" if final else text
def normalized(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    # TeX delimiter sizing is presentation-only; keep this narrow so that commands such as.
    text = re.sub(r"\\(?:left|right)(?=\W|$)", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join("".join(ch for ch in text if ch not in string.punctuation).split())
def local_correct(prediction: str, reference: object) -> bool:
    values = reference if isinstance(reference, list) else [reference]
    for value in values:
        try:
            left = float(Fraction(prediction.replace(",", "")))
            right = float(Fraction(str(value).replace(",", "")))
            if abs(left - right) <= 1e-4 * max(1.0, abs(right)):
                return True
        except (ValueError, ZeroDivisionError):
            pass
        if normalized(prediction) == normalized(value):
            return True
    return False
class Qwen:
    'One frozen backbone shared across arms; greedy evaluation as in EVOLT.'
    def __init__(self, path: Path, tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-4B evaluation requires CUDA")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
        self.model.eval()
        self.tokens = tokens
    def generate(self, messages: list[dict[str, str]], tools: list[Any]) -> str:
        options: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        if tools:
            options["tools"] = tools
        prompt = self.tokenizer.apply_chat_template(messages, **options)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.tokens, do_sample=False)
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
def instruction(record: Record, tools: bool) -> list[dict[str, str]]:
    if tools:
        system = "Solve the task accurately. Use tools when helpful, and continue reasoning after their results. Do not repeat an identical failed call. End with exactly <answer>\\boxed{final answer}</answer>."
    else:
        system = "Solve the task accurately. End with exactly <answer>\\boxed{final answer}</answer>."
    return [{"role": "system", "content": system}, {"role": "user", "content": f"{record.question}\nEnd with exactly <answer>\\boxed{{final answer}}</answer>."}]
def solve(record: Record, arm: str, model: Qwen, tools: Tools | None, memory: ToolMemory | None, rounds: int) -> dict[str, Any]:
    active = arm != "plain"
    messages = instruction(record, active)
    controller = Controller(memory if arm == "joint" else None) if arm in {"evocond", "joint"} else None
    catalog = {"python": tools.python, "search": tools.search} if tools else {}
    history: list[str] = []
    names: list[str] = []
    events: list[dict[str, Any]] = []
    failed_calls: set[tuple[str, str]] = set()
    failures = 0
    generations = 0
    for turn in range(rounds + 1):
        available = list(catalog) if active and turn < rounds else []
        proposal = list(messages)
        if controller and available:
            decision = controller.decide(record.question, messages, available)
            available = list(decision.available)
            proposal.append({"role": "system", "content": decision.note})
            events.append({"scale": decision.scale, "action": decision.action, "reason": decision.reason, "suggested": decision.suggested, "memory_used": decision.memory_used, "verified_evidence": len(decision.valid_evidence)})
        raw = model.generate(proposal, [catalog[name] for name in available])
        generations += 1
        requested = calls(raw) if available else []
        if not requested:
            history.append(raw)
            break
        rendered = "\n".join(f"<tool_call>\n{json.dumps({'name': name, 'arguments': args}, ensure_ascii=True)}\n</tool_call>" for name, args in requested)
        history.append(rendered)
        messages.append({"role": "assistant", "content": rendered})
        for name, arguments in requested:
            fingerprint = (name, json.dumps(arguments, sort_keys=True, ensure_ascii=True))
            if name not in available:
                result = "Error: unavailable tool."
            elif controller and fingerprint in failed_calls:
                result = "Error: identical failed tool call was already observed."
            else:
                try:
                    result = str(catalog[name](**arguments))
                except (TypeError, ValueError) as error:
                    result = f"Error: {type(error).__name__}: {error}"
            if result.startswith("Error:"):
                failures += 1
                failed_calls.add(fingerprint)
            names.append(name)
            messages.append({"role": "tool", "name": name, "content": f"{result}\nUse this result and continue toward a final answer."})
            history.append(f'<tool_response name="{name}">\n{result}\n</tool_response>')
    output = canonical("\n".join(history))
    final = answer(output)
    return {"output": output, "answer": final, "correct_local": bool(final) and local_correct(final, record.answer), "tools": names, "tool_failures": failures, "turns": generations, "events": events}
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen Qwen3-4B paired evaluation; never trains or updates EVOLT memory")
    parser.add_argument("--data", required=True)
    parser.add_argument("--name", required=True, choices=NAMES)
    parser.add_argument("--split", default="test", choices=("valid", "test"))
    parser.add_argument("--model", default='assets/model')
    parser.add_argument("--memory", help="Frozen EVOLT memory JSON; required for joint arm")
    parser.add_argument("--arms", default="plain,tool,evocond")
    parser.add_argument("--output", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--search-url", default="http://127.0.0.1:5003/retrieve")
    parser.add_argument("--sandbox-url", default="http://127.0.0.1:8080")
    parser.add_argument("--run", action="store_true", help="Preflight is read-only unless --run is given")
    return parser.parse_args()
def main() -> None:
    args = arguments()
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    if not arms or len(arms) != len(set(arms)) or set(arms) - {"plain", "tool", "evocond", "joint"}:
        raise ValueError("Arms must be a unique subset of plain,tool,evocond,joint")
    if min(args.offset, args.limit, args.tokens, args.rounds) < 0 or args.limit < 1 or args.tokens < 1 or args.rounds < 1:
        raise ValueError("Invalid evaluation bounds")
    data_path = Path(args.data).resolve()
    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()
    records = corpus(data_path, args.name, args.split)
    selected = records[args.offset:args.offset + args.limit]
    if len(selected) != args.limit:
        raise ValueError("Requested sample range exceeds dataset")
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config.get("model_type"), config.get("hidden_size"), config.get("num_hidden_layers")) != ("qwen3", 2560, 36):
        raise ValueError("Model config does not identify the Qwen3-4B backbone")
    memory_path = Path(args.memory).resolve() if args.memory else None
    if "joint" in arms and (memory_path is None or not memory_path.is_file()):
        raise FileNotFoundError("joint arm requires a frozen EVOLT memory snapshot")
    memory = ToolMemory(json.loads(memory_path.read_text(encoding="utf-8"))) if "joint" in arms and memory_path else None
    if memory and any(record.key in memory.bridge.tasks for record in selected):
        raise ValueError("EVOLT memory contains a held-out evaluation sample key")
    memory_ready = None
    if memory:
        reasons = [memory.rank(record.question, ("python", "search"))[1] for record in selected]
        memory_ready = sum(reason == "frozen_memory_prior" for reason in reasons)
        if args.run and not memory_ready:
            raise ValueError("Frozen EVOLT memory cannot rank either tool on any selected sample; joint would collapse to EvoCond-only")
    protocol = {"dataset": args.name, "split": args.split, "data_sha256": digest(data_path), "model_config_sha256": digest(config_path), "memory_sha256": digest(memory_path) if memory_path else None, "offset": args.offset, "limit": args.limit, "tokens": args.tokens, "rounds": args.rounds, "search_url": args.search_url, "sandbox_url": args.sandbox_url, "score": "local_diagnostic_not_searl_judge"}
    preview = {"samples": len(selected), "first_key": selected[0].key, "last_key": selected[-1].key, "arms": arms, "memory_tasks": len(memory.bridge.tasks) if memory else None, "memory_tools": sorted(memory.bridge.by_name) if memory else None, "memory_ready_samples": memory_ready, "protocol": protocol}
    print(json.dumps(preview, ensure_ascii=True, indent=2))
    if not args.run:
        return
    if output_path.exists():
        raise FileExistsError(f"Use a new output directory; refusing to overwrite {output_path}")
    tools = Tools(args.search_url, args.sandbox_url) if any(arm != "plain" for arm in arms) else None
    if tools:
        print(json.dumps({"tool_preflight": tools.check()}, ensure_ascii=True))
    actor = Qwen(model_path, args.tokens)
    output_path.mkdir(parents=True)
    summary: dict[str, Any] = {"protocol": protocol, "selected_keys": [record.key for record in selected], "arms": {}}
    for arm in arms:
        path = output_path / f"{arm}.jsonl"
        correct = 0
        total_tools = 0
        total_failures = 0
        memory_events = 0
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for record in selected:
                result = solve(record, arm, actor, tools, memory, args.rounds)
                correct += int(result["correct_local"])
                total_tools += len(result["tools"])
                total_failures += result["tool_failures"]
                memory_events += sum(event["memory_used"] for event in result["events"])
                row = {"key": record.key, "dataset": record.dataset, "arm": arm, **result, "protocol": protocol}
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                stream.flush()
        summary["arms"][arm] = {"correct_local": correct, "samples": len(selected), "pass@1_local": correct / len(selected), "tool_calls": total_tools, "tool_failures": total_failures, "memory_suggestions": memory_events, "output": str(path)}
        print(json.dumps({"arm": arm, **summary["arms"][arm]}, ensure_ascii=True))
    (output_path / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
