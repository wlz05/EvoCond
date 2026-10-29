# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Provide bounded tools for policy training."""
import ast
import html
import io
import json
import math
import re
import sys
from textwrap import dedent
from contextlib import redirect_stdout
from threading import Lock
from urllib.parse import urlencode
from urllib.request import Request, urlopen
SAFE = {"abs": abs, "all": all, "any": any, "complex": complex, "enumerate": enumerate, "float": float, "int": int, "len": len, "list": list, "max": max, "min": min, "print": print, "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip}
MATH = {"acos", "asin", "atan", "atan2", "ceil", "comb", "cos", "degrees", "dist", "e", "exp", "fabs", "factorial", "floor", "fsum", "gcd", "hypot", "isclose", "isqrt", "lcm", "log", "log2", "log10", "perm", "pi", "prod", "radians", "sin", "sqrt", "tan", "tau"}
METHODS = {"append", "count", "endswith", "index", "startswith"}
CACHE: dict[str, str] = {}
LOCK = Lock()
BLOCKED = (ast.AsyncFor, ast.AsyncFunctionDef, ast.Await, ast.ClassDef, ast.Delete, ast.Global, ast.Import, ast.ImportFrom, ast.Lambda, ast.Nonlocal, ast.Raise, ast.Try, ast.While, ast.With, ast.Yield, ast.YieldFrom)
class Guard(ast.NodeVisitor):
    """Reject unsafe or unbounded syntax."""
    def __init__(self, calls: set[str] | None = None, modules: set[str] | None = None, functions: set[str] | None = None):
        self.nodes = 0
        self.calls = calls or set()
        self.modules = modules or {"math"}
        self.functions = functions or set()
    def generic_visit(self, node):
        self.nodes += 1
        if self.nodes > 256 or isinstance(node, BLOCKED):
            raise ValueError("code is unsafe or too complex")
        super().generic_visit(node)
    def visit_Attribute(self, node):
        mathattr = isinstance(node.value, ast.Name) and node.value.id in self.modules and node.attr in MATH
        if not mathattr and node.attr not in METHODS:
            raise ValueError("attribute access is restricted")
        self.generic_visit(node)
    def visit_Call(self, node):
        valid = isinstance(node.func, ast.Name) and (node.func.id in SAFE or node.func.id in self.calls or node.func.id in ("pow", "range")) or isinstance(node.func, ast.Attribute) and (node.func.attr in METHODS or isinstance(node.func.value, ast.Name) and node.func.value.id in self.modules and node.func.attr in MATH)
        if not valid:
            raise ValueError("function call is restricted")
        self.generic_visit(node)
    def visit_FunctionDef(self, node):
        arguments = node.args
        statements = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return)
        invalid = node.name not in self.functions or node.name.startswith("_") or node.name in SAFE or node.decorator_list or not 1 <= len(node.body) <= 8 or not isinstance(node.body[-1], ast.Return) or any(not isinstance(item, statements) or isinstance(item, ast.Return) and item is not node.body[-1] for item in node.body) or arguments.vararg is not None or arguments.kwarg is not None or arguments.kwonlyargs or arguments.defaults or arguments.kw_defaults or any(item.arg.startswith("_") for item in arguments.posonlyargs + arguments.args)
        recursive = any(isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id in self.functions for item in ast.walk(node))
        if invalid or recursive:
            raise ValueError("function definition is restricted")
        self.generic_visit(node)
    def visit_Name(self, node):
        if node.id.startswith("_"):
            raise ValueError("private names are restricted")
        self.generic_visit(node)
    def visit_Constant(self, node):
        if isinstance(node.value, str) and len(node.value) > 2000 or isinstance(node.value, (int, float)) and abs(node.value) > 1e15:
            raise ValueError("constant is too large")
        self.generic_visit(node)
class Bound(ast.NodeTransformer):
    """Route exponentiation through one bounded function."""
    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Pow):
            return ast.copy_location(ast.Call(ast.Name("pow", ast.Load()), [node.left, node.right], []), node)
        return node
def span(*args: int) -> range:
    """Create one bounded range."""
    values = [int(value) for value in args]
    result = range(*values)
    if len(result) > 1000000:
        raise ValueError("range is too large")
    return result
def power(base: float, exponent: float) -> float:
    """Compute one bounded power."""
    if abs(exponent) > 1024:
        raise ValueError("exponent is too large")
    return pow(base, exponent)
def modules(tree: ast.Module) -> dict:
    """Resolve safe math imports."""
    values = {}
    body = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            if len(node.names) != 1 or node.names[0].name != "math":
                raise ValueError("only math imports are allowed")
            name = node.names[0].asname or "math"
            if name.startswith("_"):
                raise ValueError("private names are restricted")
            values[name] = math
        elif isinstance(node, ast.ImportFrom):
            if node.module != "math" or node.level or any(item.name not in MATH or (item.asname or item.name).startswith("_") for item in node.names):
                raise ValueError("only safe math imports are allowed")
            values.update({item.asname or item.name: getattr(math, item.name) for item in node.names})
        else:
            body.append(node)
    tree.body = body
    return values
def python(code: str) -> str:
    'Evaluate bounded standard-library arithmetic code.\n    Args:\n        code: Arithmetic code.\n    Returns:\n        Captured output.\n    '
    if not isinstance(code, str) or not code.strip() or len(code) > 4000:
        return "Error: code must contain at most 4000 characters."
    try:
        clean = dedent(code).strip()
        try:
            tree = ast.parse(clean, mode="exec")
        except IndentationError as error:
            if "unexpected indent" not in str(error):
                raise
            tree = ast.parse(re.sub(r"(?m)^ {1,3}(?=\S)", "", clean), mode="exec")
        tree = Bound().visit(tree)
        imports = modules(tree)
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        if len(functions) > 4:
            raise ValueError("too many functions")
        mathcalls = {name for name in MATH if callable(getattr(math, name))}
        Guard({name for name, value in imports.items() if callable(value)} | mathcalls | functions, {name for name, value in imports.items() if value is math} | {"math"}, functions).visit(tree)
        final = tree.body.pop().value if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        scope = {"__builtins__": {}, "math": math, **{name: getattr(math, name) for name in MATH}, **SAFE, **imports, "pow": power, "range": span}
        stream = io.StringIO()
        steps = 0
        def budget(frame, event, arg):
            """Stop excessive interpreted work."""
            nonlocal steps
            if event in ("call", "line"):
                steps += 1
                if steps > 100000:
                    raise RuntimeError("execution budget exceeded")
            return budget
        with redirect_stdout(stream):
            previous = sys.gettrace()
            try:
                sys.settrace(budget)
                exec(compile(ast.fix_missing_locations(tree), "<tool>", "exec"), scope, scope)
                value = eval(compile(ast.fix_missing_locations(ast.Expression(final)), "<tool>", "eval"), scope, scope) if final is not None else None
            finally:
                sys.settrace(previous)
        output = stream.getvalue().strip()
        if value is not None:
            output = f"{output}\n{value}".strip()
        return output[:4000] or "None"
    except Exception as error:
        return f"Error: {type(error).__name__}: {error}"[:4000]
def search(query: str) -> str:
    'Search Wikipedia summaries.\n    Args:\n        query: Search query.\n    Returns:\n        Ranked summaries.\n    '
    if not isinstance(query, str) or not query.strip() or len(query) > 300:
        return "Error: query must contain at most 300 characters."
    key = " ".join(query.casefold().split())
    with LOCK:
        if key in CACHE:
            return CACHE[key]
        try:
            params = urlencode({"action": "query", "generator": "search", "gsrsearch": query.strip(), "gsrlimit": 5, "prop": "extracts", "exintro": 1, "explaintext": 1, "exsentences": 3, "format": "json", "formatversion": 2})
            request = Request(f"https://en.wikipedia.org/w/api.php?{params}", headers={"User-Agent": "EVOLT/1.0"})
            with urlopen(request, timeout=4) as response:
                data = json.loads(response.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", [])
            rows = [f"{pos}. {html.unescape(str(page.get('title', '')))}: {' '.join(str(page.get('extract', '')).split())}" for pos, page in enumerate(pages, 1)]
            result = "\n".join(rows)[:8000] or "No results."
        except Exception as error:
            result = f"Search unavailable: {type(error).__name__}. Do not call search again; answer directly."
        if len(CACHE) >= 128:
            CACHE.pop(next(iter(CACHE)))
        CACHE[key] = result
        return result

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
