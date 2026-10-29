# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Train-only dual-view alignment and demonstration-conditioned fusion gates.'
from __future__ import annotations
from collections import Counter, defaultdict
import math
from control import guard
from control.controller import (CONTEXT, DIM, EDGE_DIM, FORMAT, TYPE, catalog,
                                observed_state, semantic, structural, validate,
                                view_features)
def build_examples(episodes: list[dict]) -> tuple[dict, list[dict], dict]:
    nodes, examples = {}, []
    edges = defaultdict(lambda: defaultdict(list))
    counters = Counter()
    for episode in episodes:
        row = episode["result"]
        replies = [m for m in row.get("conversation_history", []) if m.get("role") == "tool"]
        for obs in episode["observations"]:
            visible = catalog(obs["tools"])
            nodes.update({node["version"]: node for node in visible.values()})
            calls = obs["response"].get("tool_calls") or []
            if len(calls) != 1:
                counters["non_single_call"] += 1
                continue
            name = guard.callname({"tool_calls": calls})
            reply_index = sum(m.get("role") == "tool" for m in obs["messages"])
            if name not in visible or reply_index >= len(replies):
                counters["unmatched_response"] += 1
                continue
            # Verify this ordinal refers to the same next call in the saved environment history, not.
            pending, ordinal, matched = "", 0, False
            for msg in row.get("conversation_history", []):
                if msg.get("role") == "assistant":
                    pending = guard.callname(msg)
                elif msg.get("role") == "tool":
                    if ordinal == reply_index:
                        matched = pending == name
                        break
                    ordinal += 1
            if not matched:
                counters["unmatched_response"] += 1
                continue
            reply = replies[reply_index]
            failed = guard.failed(reply)
            witnessed = visible[name]["produces"] in {
                guard.canonical(v) for v in TYPE.findall(str(reply.get("content") or ""))}
            state = observed_state(obs["messages"], visible)
            previous = state["last_valid"]
            if previous in visible and (failed or witnessed):
                pair = (visible[previous]["version"], visible[name]["version"])
                edges[pair][str(episode["index"])].append(float(not failed))
            counters["verified_steps" if witnessed and not failed else "failed_or_unverified_steps"] += 1
            # Final labels are permitted only here, on the TRAIN partition.
            if not (row.get("evaluation") or {}).get("success") or failed or not witnessed:
                continue
            candidates = sorted(n for n, node in visible.items()
                                if set(node["requires"]).issubset(state["facts"])
                                and (node["produces"] not in state["facts"] or guard.canonical(n).startswith("selectfinal")))
            if name not in candidates or len(candidates) < 2:
                counters["no_feasible_ranking_example"] += 1
                continue
            examples.append({"index": episode["index"], "nodes": visible,
                             "state": state, "names": candidates, "target": candidates.index(name)})
    if not nodes or not examples:
        raise ValueError("No verified successful training steps with visible schemas; refusing to create an untrained controller")
    model = {"format": FORMAT, "nodes": nodes, "edges": [
        {"source": source, "target": target,
         "outcomes": {key: sum(values)/len(values) for key, values in sorted(by_query.items())}}
        for (source, target), by_query in sorted(edges.items())],
        "alignment": [[float(i == j) for j in range(DIM)] for i in range(DIM)],
        "fusion": [0.0]*CONTEXT, "edge_gate": [0.0]*EDGE_DIM,
        "scope": "current_observed_state_typed_subgoal_dual_view",
        "supervision": "verified_step_imitation_not_causal_edge_credit",
        "semantic_encoder": "deterministic_text_hash_32_not_pretrained_LM",
        "test_memory_updates": False}
    return model, examples, dict(counters)
def fit(episodes: list[dict], seed: int = 20260927) -> tuple[dict, dict]:
    import numpy as np
    model, examples, counters = build_examples(episodes)
    rng = np.random.default_rng(seed)
    node_list = list(model["nodes"].values())
    sem = np.array([semantic(n) for n in node_list], dtype=float)
    struct = np.array([structural(n) for n in node_list], dtype=float)
    w = np.eye(DIM) + rng.normal(0, 0.005, (DIM, DIM))
    def alignment_step(matrix):
        raw = struct @ matrix
        norms = np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-8)
        z = raw / norms
        logits = z @ sem.T / 0.25
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        loss = float(-np.log(np.maximum(p.diagonal(), 1e-12)).mean())
        dz = ((p - np.eye(len(p))) @ sem) / (len(p)*0.25)
        draw = (dz - (dz*z).sum(axis=1, keepdims=True)*z) / norms
        return loss, struct.T @ draw
    before, _ = alignment_step(w)
    for _ in range(180):
        _, gradient = alignment_step(w)
        w -= 0.15 * (gradient + 0.001*w)
    after, _ = alignment_step(w)
    model["alignment"] = w.tolist()
    prepared = []
    count = Counter(e["index"] for e in examples)
    for e in examples:
        v = view_features(model, e["state"], e["names"], e["nodes"])
        prepared.append((np.array(v["context"]), np.array(v["edge_context"]),
                         np.array(v["semantic"]), np.array(v["aligned"]),
                         np.array(v["edge"]), e["target"], 1/count[e["index"]]))
    a, b = np.zeros(CONTEXT), np.zeros(EDGE_DIM)
    def loss_and_grad(item):
        ctx, eg, se, st, edge, target, weight = item
        alpha = 1/(1+np.exp(-np.clip(ctx @ a, -30, 30)))
        gates = 1/(1+np.exp(-np.clip(eg @ b, -30, 30)))
        structure = 0.5*st + 0.5*gates*edge
        logits = 6*((1-alpha)*se + alpha*structure)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        loss = -math.log(max(float(probs[target]), 1e-12))
        delta = probs.copy()
        delta[target] -= 1
        delta *= 6
        da = float(delta @ (structure-se))*alpha*(1-alpha)*ctx
        db = (delta*alpha*0.5*edge*gates*(1-gates)) @ eg
        return loss, weight*da, weight*db
    initial = sum(loss_and_grad(item)[0] for item in prepared)/len(prepared)
    order = np.arange(len(prepared))
    for epoch in range(160):
        rng.shuffle(order)
        rate = 0.4/math.sqrt(1+epoch/20)
        for i in order:
            _, da, db = loss_and_grad(prepared[i])
            a -= rate*(da + 0.0001*a/len(prepared))
            b -= rate*(db + 0.0001*b/len(prepared))
    final = sum(loss_and_grad(item)[0] for item in prepared)/len(prepared)
    model.update(fusion=a.tolist(), edge_gate=b.tolist(), seed=seed)
    validate(model)
    return model, {"qwen_updated": False, "trained_parameters": DIM*DIM + CONTEXT + EDGE_DIM,
                   "training_queries": len(count), "ranking_examples": len(examples),
                   "nodes": len(model["nodes"]), "edges": len(model["edges"]),
                   "alignment_loss_before": before, "alignment_loss_after": after,
                   "imitation_loss_before": initial, "imitation_loss_after": final,
                   "fusion_parameter_norm": float(np.linalg.norm(a)),
                   "edge_gate_parameter_norm": float(np.linalg.norm(b)),
                   "diagnostics": counters,
                   "warning": "Training loss is not held-out performance; gates do not estimate causal edge utility."}

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
