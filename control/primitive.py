# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Prompt-level portable-tool intervention and non-executing source diagnostics.'
from __future__ import annotations
import ast
from hashlib import sha256
FORMAT = "evocond_tool_abstraction_profile_v1"
PORTABLE_NOTE = (
    "Tool construction policy: portable. Build a reusable operation for a subtask, not a solver tied to this question. "
    "Expose task-specific entities, values, thresholds and preferences as arguments; do not bake this question or its answer into reusable code. "
    "Describe what the tool computes, its input/output contract and applicability limits. "
    "Prefer composing existing operations when their contracts suffice. Keep any composition's genuine input dependencies explicit; "
    "do not encode an observed invocation order as a universal requirement. A longer algorithm is allowed when required for correctness. "
    "Rebind inputs to the current question; examples, execution success and short code do not prove transferability."
)
def validate_mode(mode):
    if mode not in {"legacy", "portable"}:
        raise ValueError("Unknown tool construction policy")
    return mode
def creation_note(mode="legacy"):
    return PORTABLE_NOTE if validate_mode(mode) == "portable" else ""
def profile(spec):
    'Describe complexity separately from input-use warnings; never execute AST.'
    source, name = spec["code"], spec["name"]
    tree = ast.parse(source)
    function = next((node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name == name), None)
    if function is None:
        raise ValueError("Tool profile needs its declared function")
    args = function.args
    parameters = [item.arg for item in args.posonlyargs+args.args+args.kwonlyargs]
    parameters += [item.arg for item in (args.vararg, args.kwarg) if item is not None]
    loaded = {node.id for statement in function.body for node in ast.walk(statement)
              if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    nodes = list(ast.walk(function))
    body = [node for statement in function.body for node in ast.walk(statement)]
    # Constants remain diagnostics: mathematical constants and API identifiers can be legitimate.
    strings = [node.value for node in body if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    return {"format": FORMAT, "source_sha256": sha256(source.encode("utf-8")).hexdigest(),
            "complexity": {"source_characters": len(source), "ast_nodes": len(nodes),
                           "calls": sum(isinstance(n, ast.Call) for n in nodes),
                           "branches": sum(isinstance(n, (ast.If, ast.IfExp, ast.Match)) for n in nodes),
                           "loops": sum(isinstance(n, (ast.For, ast.AsyncFor, ast.While, ast.comprehension)) for n in nodes)},
            "input_diagnostics": {"declared_parameters": parameters,
                                  "lexically_unused_parameters": sorted(set(parameters)-loaded),
                                  "string_literal_characters": sum(map(len, strings)),
                                  "numeric_literals": sum(isinstance(n, ast.Constant) and type(n.value) in (int, float) for n in body)},
            "generality_certified": False, "behavioral_cross_task_tests": 0,
            "warning": "lexical diagnostics only; length, constants and unused parameters are not a generality score"}

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
