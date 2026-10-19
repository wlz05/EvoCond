# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Train the EVOLT policy with tool-aware GRPO."""
from inspect import signature
from math import exp, isfinite
from pathlib import Path
import json
import re
from core.agent import Agent
from core.data import BASE, Corpus, Source
from core.tool import python, search
from core.searl import SearlTools
from learn.infer import canonical, instruction
from learn.metric import block, boxed, local, normalize, number, references
MEMORY: Agent | None = None
STATE: Path | None = None
TRACES = {}
def content(value: object) -> str:
    """Flatten one generated conversation."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        item = value.get("content", "")
        if isinstance(item, str):
            return item
        return content(item)
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := content(item)))
    return str(value or "")
def calls(value: object) -> list[str]:
    """Extract tool calls in execution order."""
    rows = value if isinstance(value, list) else [value]
    names = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("role", "")).casefold() == "tool" and row.get("name"):
            names.append(str(row["name"]).casefold())
        for call in row.get("tool_calls", []) if isinstance(row.get("tool_calls"), list) else []:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            name = function.get("name", call.get("name")) if isinstance(function, dict) else None
            if name:
                names.append(str(name).casefold())
    return list(dict.fromkeys(names))
def candidate(text: object) -> str:
    """Extract one answer from native model text."""
    return boxed(block(canonical(content(text))))
def similar(prediction: str, reference: object, question: str = "") -> float:
    """Measure bounded partial answer agreement."""
    if not prediction:
        return 0.0
    best = 0.0
    for value in references(reference):
        if local(prediction, value, question):
            return 1.0
        left, right = number(prediction), number(value)
        if left is not None and right is not None:
            best = max(best, exp(-4.0 * abs(left - right) / max(1.0, abs(right))))
            continue
        first, second = set(normalize(prediction).split()), set(normalize(value).split())
        if first and second:
            common = len(first & second)
            best = max(best, 2.0 * common / (len(first) + len(second)))
    return min(1.0, best)
def outcome(text: object, reference: object, question: str = "") -> float:
    """Return one dense correctness reward."""
    final = candidate(text)
    if not final:
        return -0.5
    if local(final, reference, question):
        return 1.0
    return -0.25 + 0.25 * similar(final, reference, question)
def shape(text: object) -> float:
    """Score progressive answer formatting."""
    value = content(text)
    lower = value.casefold()
    inner = block(value)
    score = 0.03 if "<answer>" in lower and "</answer>" in lower else 0.0
    score += 0.03 if boxed(value) else 0.0
    score += 0.04 if inner and boxed(inner) else 0.0
    return round(score, 2)
def failures(value: object) -> int:
    """Count failed tool responses."""
    rows = value if isinstance(value, list) else [value]
    structured = [content(row).strip() for row in rows if isinstance(row, dict) and str(row.get("role", "")).casefold() == "tool"]
    if structured:
        return sum(item.casefold().startswith("error:") for item in structured)
    text = content(value)
    return sum(item.strip().casefold().startswith("error:") for item in re.findall(r"<tool_response[^>]*>(.*?)</tool_response>", text, flags=re.DOTALL | re.IGNORECASE))
def reliability(completions, **kwargs) -> list[float]:
    """Penalize failed tool execution."""
    return [-0.1 * min(2, failures(item)) for item in completions]
def reward(completions, answer, key, question, **kwargs) -> list[float]:
    """Score answers and update task-aware memory."""
    if MEMORY is None or STATE is None:
        raise RuntimeError("training memory is not initialized")
    scores = []
    for completion, target, task, query in zip(completions, answer, key, question):
        reference = json.loads(target)
        score = outcome(completion, reference, query)
        adjusted = score - 0.1 * min(2, failures(completion))
        trace = TRACES.get(task) or MEMORY.recall(query, task=task)
        MEMORY.commit(trace, calls(completion), adjusted)
        scores.append(score)
    MEMORY.save(STATE)
    return scores
def format(completions, **kwargs) -> list[float]:
    """Reward the required answer envelope."""
    return [shape(item) for item in completions]
def select(kind, values: dict) -> dict:
    """Keep supported library arguments."""
    keys = signature(kind).parameters
    return {key: value for key, value in values.items() if key in keys}
def prepare(agent: Agent) -> None:
    """Register the official policy tools."""
    if "python" not in agent.graph.tools:
        agent.register("python", "evaluate bounded arithmetic with safe math", schema="code:string", reward=1.0)
    if "search" not in agent.graph.tools:
        agent.register("search", "search wikipedia for grounded evidence", schema="query:string", reward=1.0)
def prompt(agent: Agent, record) -> list[dict]:
    """Build one retrieval-guided training prompt."""
    trace = agent.recall(record.question, task=record.key)
    TRACES[record.key] = trace
    return instruction(record, trace.hits, True)
def tune(data: str, state: Path, model: str = "autodl-tmp/model/qwen4b", output: str = "result/pilot", limit: int = 8, steps: int = 1, tokens: int = 768, generations: int = 4, batch: int = 1, accum: int = 4, seed: int = 2, ability: str = "", save: int = 25, rate: float = 1e-6, root: str | Path = BASE, backend: str = "native", search_url: str = "http://127.0.0.1:5003/retrieve", sandbox_url: str = "http://127.0.0.1:8080") -> dict:
    """Run one bounded single-GPU QLoRA GRPO job."""
    values = (limit, steps, tokens, generations, batch, accum, save)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values):
        raise ValueError("training limits must be positive integers")
    if batch * accum % generations:
        raise ValueError("effective batch must be divisible by generations")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must not be negative")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not isfinite(float(rate)) or not 0.0 < float(rate) <= 1e-3:
        raise ValueError("learning rate is invalid")
    skill = str(ability or "").casefold().strip()
    if skill not in ("", "math", "qa"):
        raise ValueError("ability must be math or qa")
    if backend not in ("native", "searl"):
        raise ValueError("tool backend must be native or searl")
    source = Source(root)
    modelpath = source.path(model)
    outpath = source.path(output)
    if not modelpath.is_dir():
        raise FileNotFoundError(modelpath)
    records = [item for item in Corpus(root).load(data) if not skill or item.ability == skill][:limit]
    if len(records) < generations:
        raise ValueError("matching training data is smaller than the rollout group")
    services = SearlTools(search_url, sandbox_url) if backend == "searl" else None
    if services is not None:
        services.check()
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as error:
        raise RuntimeError("training requires torch, datasets, bitsandbytes, peft, transformers, and trl") from error
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    global MEMORY, STATE
    MEMORY = Agent.open(state) if state.exists() else Agent()
    STATE = state
    prepare(MEMORY)
    MEMORY.save(STATE)
    TRACES.clear()
    lookup = {item.key: item for item in records}
    rows = [{"prompt": [{"role": "user", "content": ""}], "answer": json.dumps(item.answer, ensure_ascii=True), "key": item.key, "question": item.question} for item in records]
    def render(batch):
        """Refresh retrieval after each teacher update."""
        batch["prompt"] = [prompt(MEMORY, lookup[key]) for key in batch["key"]]
        return batch
    dataset = Dataset.from_list(rows).with_transform(render)
    tokenizer = AutoTokenizer.from_pretrained(modelpath, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    adapter = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules="all-linear")
    options = {"output_dir": str(outpath), "max_steps": steps, "learning_rate": float(rate), "per_device_train_batch_size": batch, "gradient_accumulation_steps": accum, "num_generations": generations, "max_completion_length": tokens, "temperature": 0.7 if services else 0.9, "top_p": 0.98 if services else 0.95, "repetition_penalty": 1.05, "beta": 0.0, "bf16": True, "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False}, "logging_steps": 1, "logging_first_step": True, "log_completions": True, "num_completions_to_print": 1, "save_steps": min(save, steps), "save_total_limit": 4, "save_strategy": "steps", "report_to": "none", "remove_unused_columns": False, "max_tool_calling_iterations": 6 if services else 2, "chat_template_kwargs": {"enable_thinking": False}, "loss_type": "grpo", "scale_rewards": "group", "use_vllm": False, "model_init_kwargs": {"dtype": "bfloat16"}, "seed": seed}
    config = GRPOConfig(**select(GRPOConfig, options))
    arguments = {"model": str(modelpath), "reward_funcs": [reward, format, reliability], "args": config, "train_dataset": dataset, "quantization_config": quant, "peft_config": adapter, "processing_class": tokenizer, "tools": [services.python, services.search] if services else [python, search]}
    if "tools" not in signature(GRPOTrainer).parameters:
        raise RuntimeError("installed trl does not support tool-aware GRPO")
    trainer = GRPOTrainer(**select(GRPOTrainer, arguments))
    result = trainer.train()
    final = outpath / "final"
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    MEMORY.save(STATE)
    metrics = {key: value for key, value in result.metrics.items() if isinstance(value, (int, float, str, bool))}
    saved = state.resolve().relative_to(Path(root).resolve())
    return {"records": len(records), "steps": steps, "save": save, "rate": float(rate), "backend": backend, "adapter": str(Path(output) / "final"), "state": str(saved), "metrics": metrics}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
