# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from threading import Lock
import time
import uuid
from typing import Any, Mapping
from control.bridge import EvoltBridge
from control.guard import canonical, intervene
from control.gate import decide, initial_user, settings, validate_model
from state.record import Journal, local
CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
IDENTIFIER = re.compile(r"<?(?:[A-Za-z]+)?Candidate\{?(?:(?:[A-Za-z]+)?Candidate)?(\d+)\}?>?", re.IGNORECASE)
REVIEW = "Act as an isolated GL challenger. Reconstruct every explicit user constraint, compare every surviving candidate against only the observed tool evidence, and return compact JSON with keys requirements, candidates, recommended, unsupported, and confidence. Candidate entries must include id, supported, missing, and conflicts. Never use hidden answers, invent evidence, or call tools."
def cleanvalue(value: Any) -> Any:
    if isinstance(value, str):
        match = IDENTIFIER.fullmatch(value.strip())
        return match.group(1) if match else value
    if isinstance(value, list):
        return [cleanvalue(item) for item in value]
    if isinstance(value, dict):
        return {key: cleanvalue(item) for key, item in value.items()}
    return value
def toolcalls(text: str, tools: list[Mapping[str, Any]] | None = None) -> tuple[str | None, list[dict[str, Any]], list[str]]:
    calls: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    rejected: list[str] = []
    allowed = None if tools is None else {str(tool.get("function", {}).get("name", "")) for tool in tools if isinstance(tool.get("function"), Mapping)}
    for match in CALL.finditer(text):
        try:
            raw = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, Mapping):
            continue
        name = raw.get("name")
        arguments = raw.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            continue
        spans.append(match.span())
        if allowed is not None and name.strip() not in allowed:
            rejected.append(name.strip())
            continue
        if canonical(name).startswith("selectfinal"):
            arguments = cleanvalue(arguments)
        if isinstance(arguments, str):
            encoded = arguments
        else:
            encoded = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        calls.append({"id": f"call_{uuid.uuid4().hex}", "type": "function", "function": {"name": name.strip(), "arguments": encoded}})
    content = text
    for start, end in reversed(spans):
        content = content[:start] + content[end:]
    content = content.strip()
    return (content or None, calls, rejected)
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="serve")
    parser.add_argument("--model", default="assets/model")
    parser.add_argument("--adapter", help="Optional separately trained Qwen LoRA adapter")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--evocond", action="store_true")
    parser.add_argument("--noaudit", action="store_true")
    parser.add_argument("--noplan", action="store_true")
    parser.add_argument("--levels", choices=("filter", "ll", "llgl", "full"), default="full")
    parser.add_argument("--trace", default="runs/interventions.jsonl")
    parser.add_argument("--memory", help="Frozen, disjoint CostBench trace-memory snapshot for an optional joint ablation")
    parser.add_argument("--memory-extra-cost", type=float, default=0.0, help="Maximum route cost above the cost-only plan; default preserves minimum cost")
    parser.add_argument("--memory-backoff", choices=("exact", "family"), default="exact", help="Optional operator-family outcome backoff; exact remains the default")
    parser.add_argument("--gate", help="Frozen initial-user policy gate; requires full/noaudit and its paired memory snapshot")
    parser.add_argument("--state-controller", help="Frozen train-only dual-view state controller")
    parser.add_argument("--state-policy", choices=("fixed", "learned"), default="learned")
    parser.add_argument("--state-experiment", choices=("legacy", "baseline", "prompt_only", "selective"),
                        default="legacy", help="Explicit prompt/ranking ablation; legacy preserves old runs")
    parser.add_argument("--state-value-audit", action="store_true", help="Audit proposed preference values against the latest visible user request")
    parser.add_argument("--trace-context", action="store_true", help="Record visible request schemas/prefixes for TRAIN-only supervision")
    parser.add_argument("--training-temperature", type=float, default=0.0,
                        help="Sample policy rollouts only with --trace-context")
    return parser.parse_args()
def check_state_experiment(args: argparse.Namespace) -> None:
    if args.state_experiment == "legacy":
        return
    if not args.state_controller or not args.evocond or not args.noaudit or args.noplan or args.levels != "full":
        raise ValueError("State experiments require --state-controller --evocond --noaudit --levels full with planning")
    if args.gate or args.memory or args.state_value_audit or args.training_temperature:
        raise ValueError("Prompt/ranking ablations cannot mix memory, old gates, value audit or rollout sampling")
def experiment_source_hashes() -> dict[str, str]:
    project = Path(__file__).resolve().parent
    names = ("serve.py", 'control/controller.py', "control/guard.py", 'control/bridge.py',
             'control/family.py', 'control/gate.py')
    return {name: sha256((project / name).read_bytes()).hexdigest() for name in names}
def main() -> None:
    args = arguments()
    check_state_experiment(args)
    experiment_hashes = experiment_source_hashes() if args.state_experiment != "legacy" else None
    if not isfinite(args.memory_extra_cost) or args.memory_extra_cost < 0.0:
        raise ValueError("Memory cost allowance must be finite and nonnegative")
    if args.memory_extra_cost and not args.memory:
        raise ValueError("--memory-extra-cost requires --memory")
    if args.memory_backoff != "exact" and not args.memory:
        raise ValueError("--memory-backoff family requires --memory")
    if not isfinite(args.training_temperature) or not 0.0 <= args.training_temperature <= 2.0:
        raise ValueError("Training temperature must be finite and within [0, 2]")
    if args.training_temperature and not args.trace_context:
        raise ValueError("Sampling policy rollouts requires --trace-context")
    memory = None
    memory_hash = None
    gate = None
    gate_hash = None
    state_controller = None
    state_hash = None
    if args.state_controller:
        if args.gate or args.memory or not args.evocond or not args.noaudit or args.noplan or args.levels != "full":
            raise ValueError("State controller requires full/noaudit without old --gate or --memory")
        from control.controller import validate
        raw = Path(args.state_controller).read_bytes()
        state_controller = json.loads(raw)
        validate(state_controller)
        provenance = state_controller.get("provenance") or {}
        if provenance.get("model_config_sha256") != sha256((local(args.model) / "config.json").read_bytes()).hexdigest():
            raise ValueError("State controller model configuration differs")
        state_hash = sha256(raw).hexdigest()
    if args.state_value_audit and state_controller is None:
        raise ValueError("Value audit requires a frozen state controller")
    if args.memory:
        if not args.evocond or args.levels != "full":
            raise ValueError("Frozen memory is only supported by --evocond --levels full")
        path = Path(args.memory).resolve(strict=True)
        raw = path.read_bytes()
        snapshot = json.loads(raw)
        metadata = snapshot.get("metadata") if isinstance(snapshot, Mapping) else None
        train_indices = metadata.get("train_indices") if isinstance(metadata, Mapping) else None
        if not isinstance(metadata, Mapping) or metadata.get("source") != "costbench_traces_v1" or not isinstance(metadata.get("query_sha256"), str) or not isinstance(train_indices, list) or not train_indices or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in train_indices) or len(set(train_indices)) != len(train_indices):
            raise ValueError("Memory needs CostBench trace provenance and nonempty training indices")
        memory = EvoltBridge.from_state(snapshot)
        memory_hash = sha256(raw).hexdigest()
    if args.gate:
        if memory is None or not args.noaudit or args.noplan or args.levels != "full" or args.memory_backoff != "family" or args.memory_extra_cost != 0.5:
            raise ValueError("Task gate requires --evocond --noaudit --levels full --memory ... --memory-backoff family --memory-extra-cost 0.5")
        raw = Path(args.gate).read_bytes()
        gate = json.loads(raw)
        validate_model(gate)
        provenance = gate.get("provenance") or {}
        if provenance.get("memory_sha256") != memory_hash or provenance.get("model_config_sha256") != sha256((local(args.model) / "config.json").read_bytes()).hexdigest():
            raise ValueError("Task gate was trained with another memory/model configuration")
        gate_hash = sha256(raw).hexdigest()
    import torch
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from transformers import AutoModelForCausalLM, AutoTokenizer
    modelpath = str(local(args.model))
    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    model = AutoModelForCausalLM.from_pretrained(modelpath, dtype=torch.bfloat16, device_map="auto")
    adapter_hash = None
    if args.adapter:
        from peft import PeftModel
        from control.trajectory import adapter_digest
        adapter_path = Path(args.adapter).resolve(strict=True)
        config_path = adapter_path / "adapter_config.json"
        report_path = adapter_path / "training_report.json"
        if not config_path.is_file() or not report_path.is_file():
            raise FileNotFoundError("Adapter config and training report are required")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        adapter_protocol = report.get("protocol") or {}
        if adapter_protocol.get("model_config_sha256") != sha256((local(args.model) / "config.json").read_bytes()).hexdigest():
            raise ValueError("Adapter was trained from another Qwen model configuration")
        if adapter_protocol.get("tokenizer_config_sha256") != sha256((local(args.model) / "tokenizer_config.json").read_bytes()).hexdigest():
            raise ValueError("Adapter was trained from another tokenizer configuration")
        model = PeftModel.from_pretrained(model, adapter_path)
        adapter_hash = adapter_digest(adapter_path)
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.eval()
    journal = Journal(args.trace)
    lock = Lock()
    app = FastAPI()
    def infer(history: list[dict[str, Any]], available: list[dict[str, Any]], maximum: int, temperature: float) -> tuple[str, int, int, list[int], list[int]]:
        options = {"add_generation_prompt": True, "tokenize": True, "return_dict": True, "return_tensors": "pt", "enable_thinking": False}
        if available:
            options["tools"] = available
        try:
            inputs = tokenizer.apply_chat_template(history, **options)
        except TypeError:
            options.pop("enable_thinking")
            inputs = tokenizer.apply_chat_template(history, **options)
        inputs = inputs.to(model.device)
        generate = {"max_new_tokens": maximum, "do_sample": temperature > 0.0}
        if temperature > 0.0:
            generate["temperature"] = temperature
        with torch.inference_mode():
            output = model.generate(**inputs, **generate)
        prompt = inputs["input_ids"].shape[-1]
        generated = output[0][prompt:]
        return (tokenizer.decode(generated, skip_special_tokens=True), int(prompt),
                int(generated.shape[-1]), inputs["input_ids"][0].tolist() if args.trace_context else [],
                generated.tolist() if args.trace_context else [])
    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": args.model, "adapter_sha256": adapter_hash,
                "training_temperature": args.training_temperature, "evocond": args.evocond,
                "levels": args.levels if args.evocond else None, "noaudit": args.noaudit, "memory_sha256": memory_hash, "memory_extra_cost": args.memory_extra_cost if memory is not None else None, "memory_backoff": args.memory_backoff if memory is not None else None, "gate_sha256": gate_hash, "gate_scope": gate.get("scope") if gate else None,
                "state_controller_sha256": state_hash, "state_policy": args.state_policy if state_controller else None,
                "state_experiment": args.state_experiment if state_controller else None,
                "state_experiment_source_sha256": experiment_hashes,
                "state_value_audit": args.state_value_audit,
                "trace_context": args.trace_context}
    @app.post("/v1/chat/completions")
    def complete(body: dict[str, Any]) -> dict[str, Any]:
        request_started = time.perf_counter()
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, "Messages must be a nonempty list")
        tools = body.get("tools")
        if tools is not None and not isinstance(tools, list):
            raise HTTPException(400, "Tools must be a list")
        choice = body.get("tool_choice", "auto")
        event = None
        original = [dict(message) for message in messages]
        original_tools = list(tools or [])
        controller_started = time.perf_counter()
        selected_levels, selected_memory, selected_backoff = args.levels, memory, args.memory_backoff
        gate_event = None
        if gate is not None:
            # Recomputing from the FIRST user message makes this policy sticky through tool replies and.
            gate_event = decide(gate, initial_user(original))
            selected_levels, selected_memory, selected_backoff = settings(gate_event["arm"], memory)
            with lock:
                journal.write({"action": "task_gate", **gate_event, "gate_sha256": gate_hash})
        if choice == "none":
            tools = []
        elif state_controller is not None and tools:
            from control.controller import intervene as state_intervene
            messages, tools, event = state_intervene(messages, tools, state_controller, args.state_policy,
                                                    experiment=args.state_experiment)
        elif args.evocond and tools:
            messages, tools, event = intervene(messages, tools, not args.noaudit, selected_levels, not args.noplan, selected_memory, args.memory_extra_cost if selected_memory is not None else 0.0, selected_backoff)
        controller_seconds = time.perf_counter() - controller_started
        if choice == "required" and tools:
            messages = [*messages, {"role": "system", "content": "Make exactly one valid tool call before answering."}]
        requested = int(body.get("max_tokens", 512))
        temperature = float(body.get("temperature", 0.0))
        if requested < 1 or not isfinite(temperature) or temperature < 0.0 or temperature > 2.0:
            raise HTTPException(400, "Generation parameters are invalid")
        maximum = min(requested, 8192)
        if args.training_temperature:
            temperature = args.training_temperature
        if event is not None and event.get("action") == "audit":
            review, _, _, _, _ = infer([*original, {"role": "system", "content": REVIEW}], [], min(maximum, 768), 0.0)
            review = review.strip()[:4000]
            event = {**event, "review": review}
            messages = [*messages, {"role": "system", "content": f"EvoCond isolated challenger report: {review} Use it as advisory evidence, verify it against the original observations, and make at most one valid tool call."}]
        if event is not None:
            if gate_event is not None:
                event = {**event, "task_gate": gate_event}
            with lock:
                journal.write(event)
            print(json.dumps({"evocond": event}, ensure_ascii=True, separators=(",", ":")), flush=True)
        prompt = 0
        completion = 0
        audit_seconds = 0.0
        audit_count = 0
        audit_rejections = 0
        policy_attempts: list[dict[str, Any]] = []
        for attempt in range(3):
            policy_messages = deepcopy(messages) if args.trace_context else []
            policy_tools = deepcopy(tools or []) if args.trace_context else []
            generated, usedprompt, usedcompletion, prompt_ids, generated_ids = infer(messages, tools or [], maximum, temperature if attempt == 0 else 0.0)
            if args.trace_context:
                policy_attempts.append({"messages": policy_messages, "tools": policy_tools,
                                        "prompt_ids": prompt_ids, "completion_ids": generated_ids,
                                        "generation": generated,
                                        "temperature": temperature if attempt == 0 else 0.0,
                                        "accepted": False})
            prompt += usedprompt
            completion += usedcompletion
            content, calls, rejected = toolcalls(generated, tools or [])
            if rejected:
                with lock:
                    journal.write({"action": "reject", "mode": selected_levels if args.evocond else "baseline", "attempt": attempt + 1, "unavailable": rejected})
                if not tools or attempt == 2:
                    content, calls = "No valid available tool call was generated.", []
                    break
                names = ", ".join(str(tool["function"]["name"]) for tool in tools if isinstance(tool.get("function"), Mapping) and tool["function"].get("name"))
                messages = [*messages, {"role": "system", "content": f"The previous output called unavailable tools: {', '.join(rejected)}. Call only one tool from this list: {names}. Do not call unavailable tools."}]
                continue
            if args.state_value_audit and calls:
                from control.value import interpret, latest_user, proposal, review_prompt
                item = proposal(calls[0], tools or [])
                user = latest_user(original)
                if item is not None and user:
                    audit_started = time.perf_counter()
                    review, review_prompt_tokens, review_completion_tokens, _, _ = infer(
                        review_prompt(user, item), [], min(maximum, 384), 0.0)
                    audit_seconds += time.perf_counter() - audit_started
                    audit_count += 1
                    prompt += review_prompt_tokens
                    completion += review_completion_tokens
                    verdict = interpret(review, user, item)
                    with lock:
                        journal.write({"action": "value_audit", "attempt": attempt + 1,
                                       "tool": item["tool"], **verdict,
                                       "prompt_tokens": review_prompt_tokens,
                                       "completion_tokens": review_completion_tokens})
                    if verdict["decision"] == "reject":
                        audit_rejections += 1
                        if attempt == 2:
                            content, calls = "A proposed preference value contradicted the latest user request; no verified tool call was submitted.", []
                            break
                        conflicts = "; ".join(f"{v['field']}={v['proposed']}: user wrote {v['evidence']!r}"
                                              for v in verdict["mismatches"])
                        messages = [*messages, {"role": "system", "content":
                                    "A separate check found a conflict with the latest user requirements: "
                                    + conflicts + ". Re-read the latest user request and visible tool schema. "
                                    "Choose the correct preference values and make exactly one valid tool call. "
                                    "Do not reuse stale values or invent object IDs."}]
                        continue
            if policy_attempts:
                policy_attempts[-1]["accepted"] = True
            break
        usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
        message = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = calls
        with lock:
            journal.write({"action": "request_usage", **usage, "controller_seconds": controller_seconds,
                           "value_audit_seconds": audit_seconds, "value_audits": audit_count,
                           "value_rejections": audit_rejections,
                           "wall_seconds": time.perf_counter() - request_started})
            if args.trace_context:
                journal.write({"action": "state_training_observation", "messages": original,
                               "tools": original_tools, "response": message,
                               "policy_attempts": policy_attempts})
        finish = "tool_calls" if calls else "stop"
        return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()), "model": str(body.get("model", args.model)), "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}
    uvicorn.run(app, host="127.0.0.1", port=args.port)
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
