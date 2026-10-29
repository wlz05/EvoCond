# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Run resumable local model inference."""
from pathlib import Path
import json
import re
from core.data import BASE, Record, Source
from core.model import Hit
from core.tool import python, search
from core.searl import SearlTools
from learn.metric import block, boxed
TOOLS = {"python": python, "search": search}
def adapter(path: str | Path) -> bool:
    """Detect one saved PEFT adapter."""
    return (Path(path) / "adapter_config.json").is_file()
def canonical(text: object) -> str:
    """Append one canonical answer without changing its value."""
    value = str(text or "").strip()
    answer = block(value)
    if answer and boxed(answer):
        return value
    final = boxed(value)
    if not final and answer and "\n" not in answer and len(answer) <= 128:
        final = answer.strip().strip("$").strip()
    if not final:
        matches = re.findall(r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*\$?([^\n$]+)", value, flags=re.IGNORECASE)
        final = matches[-1].strip().strip("*` ") if matches else ""
    return f"{value}\n<answer>\\boxed{{{final}}}</answer>" if final else value
def instruction(record: Record, hits: list[Hit] | None = None, tools: bool = False, available: list[str] | None = None, multi_tool: bool = False) -> list[dict]:
    """Build one aligned policy prompt."""
    guide = "\n".join(f"{item.name}: {item.text}" for item in (hits or []))
    context = f"\nRetrieved tool guidance:\n{guide}" if guide else ""
    system = "Solve the task accurately and concisely. Keep reasoning under 250 words. Python includes safe math names, loops, lists, sets, and short pure functions; do not use scipy or sympy. Call each useful tool at most once. After a tool response, give the final answer. If a tool fails, do not retry it. End with exactly <answer>\\boxed{final answer}</answer>." if tools else "Solve the task accurately. End with exactly <answer>\\boxed{final answer}</answer>."
    if tools and available is not None:
        system = f"Solve the task accurately and concisely. Keep reasoning under 250 words. Available tools: {', '.join(available)}. Use only available tools, at most once each. After a tool response, give the final answer. If a tool fails, do not retry it. End with exactly <answer>\\boxed{{final answer}}</answer>."
    if tools and multi_tool:
        scope = f"Available tools: {', '.join(available)}. " if available is not None else ""
        system = f"Solve the task accurately. {scope}Use tools when helpful, and continue reasoning after their results. Do not repeat an identical failed call. End with exactly <answer>\\boxed{{final answer}}</answer>."
    user = f"{record.question}{context}\nEnd with exactly <answer>\\boxed{{final answer}}</answer>."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
def requests(text: str) -> list[tuple[str, dict]]:
    """Parse valid model tool requests."""
    output = []
    for value in re.findall(r"<tool_call>(.*?)</tool_call>", str(text or ""), flags=re.DOTALL | re.IGNORECASE):
        try:
            row = json.loads(value.strip())
        except json.JSONDecodeError:
            continue
        name = str(row.get("name", "")).casefold() if isinstance(row, dict) else ""
        arguments = row.get("arguments", {}) if isinstance(row, dict) else {}
        if name and isinstance(arguments, dict):
            output.append((name, arguments))
    return output
def distinct(calls: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Keep the first request for each tool."""
    output = []
    seen = set()
    for name, arguments in calls:
        if name not in seen:
            seen.add(name)
            output.append((name, arguments))
    return output
def invoke(name: str, arguments: dict, used: set[str], allowed: set[str] | None = None, catalog: dict | None = None, repeat: bool = False) -> str:
    """Execute one bounded tool request."""
    key = str(name or "").casefold()
    catalog = TOOLS if catalog is None else catalog
    if key not in catalog:
        return "Error: tool is unavailable."
    if allowed is not None and key not in allowed:
        return "Error: tool was not retrieved."
    if key in used and not repeat:
        return "Error: tool was already called."
    used.add(key)
    try:
        return str(catalog[key](**arguments))
    except (TypeError, ValueError) as error:
        return f"Error: {type(error).__name__}: {error}"[:4000]
class Qwen:
    """Load one Qwen policy lazily."""
    def __init__(self, path: str | Path, tokens: int = 512, tools: bool = False, rounds: int = 2, root: str | Path = BASE, guided: bool = False, backend: str = "native", search_url: str = "http://127.0.0.1:5003/retrieve", sandbox_url: str = "http://127.0.0.1:8080"):
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
            raise ValueError("token limit must be positive")
        if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 0:
            raise ValueError("tool rounds must not be negative")
        self.path = Source(root).path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(self.path)
        self.tokens = tokens
        self.tools = bool(tools)
        self.guided = bool(guided)
        self.rounds = rounds
        if backend not in ("native", "searl"):
            raise ValueError("tool backend must be native or searl")
        self.backend = backend
        self.external = SearlTools(search_url, sandbox_url) if backend == "searl" and tools else None
        self.catalog = {"python": self.external.python, "search": self.external.search} if self.external else TOOLS
        self.tokenizer = None
        self.model = None
        self.torch = None
    def load(self) -> None:
        """Load the model on the available accelerator."""
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("inference requires torch and transformers") from error
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen inference requires CUDA")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        loader = AutoModelForCausalLM
        if adapter(self.path):
            try:
                from peft import AutoPeftModelForCausalLM
            except ImportError as error:
                raise RuntimeError("adapter inference requires peft") from error
            loader = AutoPeftModelForCausalLM
        self.model = loader.from_pretrained(self.path, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
        self.model.eval()
    def available(self, hits: list[Hit] | None = None) -> list[str]:
        """List tools exposed to one inference request."""
        if not self.tools:
            return []
        if not self.guided:
            return list(self.catalog)
        return list(dict.fromkeys(hit.name.casefold() for hit in (hits or []) if hit.name.casefold() in self.catalog))
    def check(self) -> dict:
        """Check SEARL tool services before allocating the model."""
        return self.external.check() if self.external else {"backend": self.backend, "tools": list(self.catalog) if self.tools else []}
    def answer(self, record: Record, hits: list[Hit] | None = None) -> str:
        """Generate one deterministic formatted answer."""
        self.load()
        available = self.available(hits)
        multi_tool = self.backend == "searl" and self.tools
        messages = instruction(record, hits, self.tools and bool(available), available if self.guided else None, multi_tool)
        used = set()
        history = []
        closed = False
        raw = ""
        for turn in range(self.rounds + 1):
            active = self.tools and bool(available) and not closed and turn < self.rounds
            options = {"tools": [self.catalog[name] for name in available]} if active else {}
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False, **options)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with self.torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=self.tokens, do_sample=False)
            raw = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            calls = requests(raw) if active else []
            if not calls:
                history.append(raw)
                return canonical("\n".join(history))
            unique = distinct(calls)
            calltext = "\n".join(f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments}, ensure_ascii=True)}\n</tool_call>" for name, arguments in unique)
            history.append(calltext)
            messages.append({"role": "assistant", "content": calltext})
            for name, arguments in unique:
                result = invoke(name, arguments, used, set(available) if self.guided else None, self.catalog, multi_tool)
                continuation = "Use this result and continue toward a final answer." if multi_tool else "Use this result and return the final answer without another tool call."
                messages.append({"role": "tool", "name": name, "content": f"{result}\n{continuation}"})
                history.append(f"<tool_response name=\"{name}\">\n{result}\n</tool_response>")
            closed = not multi_tool
        return canonical("\n".join(history))
class Batch:
    """Write restart-safe prediction batches."""
    def __init__(self, root: str | Path = BASE):
        self.source = Source(root)
    def saved(self, path: Path, config: dict | None = None) -> set[str]:
        """Read completed prediction keys."""
        if not path.exists():
            return set()
        keys = set()
        for pos, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"prediction line {pos + 1} is invalid") from error
            key = str(row.get("key", "")) if isinstance(row, dict) else ""
            if not key or key in keys:
                raise ValueError("saved prediction keys must be unique")
            if config is not None and any(row.get(name) != value for name, value in config.items()):
                raise ValueError("saved prediction protocol differs")
            keys.add(key)
        return keys
    def run(self, records: list[Record], output: str | Path, solve, limit: int = 0, offset: int = 0, restart: bool = False, retrieve=None, config: dict | None = None) -> dict:
        """Generate one bounded resumable batch."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit must not be negative")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must not be negative")
        if config is not None and (not isinstance(config, dict) or set(config) & {"key", "dataset", "output", "tools", "retrieved", "weights", "cold"}):
            raise ValueError("prediction protocol is invalid")
        path = self.source.path(output)
        if path.suffix.casefold() != ".jsonl":
            raise ValueError("prediction output must be jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        done = set() if restart else self.saved(path, config)
        selected = records[offset:offset + limit if limit else None]
        pending = [item for item in selected if item.key not in done]
        mode = "w" if restart else "a"
        with path.open(mode, encoding="utf-8") as stream:
            for item in pending:
                trace = retrieve(item) if retrieve is not None else None
                hits = trace.hits if trace is not None else []
                answer = solve(item, hits)
                row = {"key": item.key, "dataset": item.dataset, "output": answer, "tools": [name for name, arguments in requests(answer)], "retrieved": [hit.key for hit in hits]}
                if trace is not None:
                    row.update({"weights": trace.weights, "cold": trace.cold})
                if config is not None:
                    row.update(config)
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                stream.flush()
        return {"data": len(records), "selected": len(selected), "generated": len(pending), "skipped": len(selected) - len(pending), "output": str(output)}

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
