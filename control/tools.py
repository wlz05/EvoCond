# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'SEARL-compatible local tools for EvoCond evaluation.'
from __future__ import annotations
import ast
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener
class Tools:
    def __init__(self, search_url: str, sandbox_url: str) -> None:
        search = urlparse(search_url)
        sandbox = urlparse(sandbox_url)
        if search.scheme != "http" or not search.hostname or search.path != "/retrieve" or search.username or search.password:
            raise ValueError("search_url must be an explicit local /retrieve HTTP endpoint")
        if sandbox.scheme != "http" or not sandbox.hostname or sandbox.username or sandbox.password:
            raise ValueError("sandbox_url must be an explicit HTTP endpoint")
        self.search_url = search_url
        self.sandbox_url = sandbox_url
    def search(self, query: str) -> str:
        "Search the local Wikipedia index, returning SEARL's top-three format.\n        Args:\n            query: Search query.\n        Returns:\n            Ranked local Wikipedia passages.\n        "
        if not isinstance(query, str) or not query.strip() or len(query) > 300:
            return "Error: search query must contain at most 300 characters."
        body = json.dumps({"queries": [query.strip()], "topk": 3, "return_scores": True}).encode("utf-8")
        request = Request(self.search_url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with build_opener(ProxyHandler({})).open(request, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
            passages = data["result"][0]
            if not isinstance(passages, list) or not passages:
                return "No relevant documents found"
            output = "Following are the search results given by the wiki search engine:"
            for item in passages:
                contents = str(item["document"]["contents"])
                title, _, text = contents.partition("\n")
                output += f'\nsearch results: "{title.strip(chr(34))}"\n{text}'
            return output
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, TypeError, IndexError) as error:
            return f"Error: local SEARL retrieval failed: {type(error).__name__}: {error}"
    def python(self, code: str) -> str:
        'Execute Python through SandboxFusion, never in the EVOLT process.\n        Args:\n            code: Python code.\n        Returns:\n            SandboxFusion execution result.\n        '
        if not isinstance(code, str) or not code.strip() or len(code) > 4000:
            return "Error: code must contain at most 4000 characters."
        try:
            from sandbox_fusion import RunCodeRequest, run_code, set_endpoint
        except ImportError:
            return "Error: sandbox-fusion==0.3.7 is required for the SEARL Python backend."
        try:
            tree = ast.parse(code)
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = tree.body[-1].value
                if not isinstance(last, ast.Call) or not isinstance(last.func, ast.Name) or last.func.id != "print":
                    tree.body[-1] = ast.Expr(ast.Call(ast.Name("print", ast.Load()), [last], []))
                    code = ast.unparse(ast.fix_missing_locations(tree))
        except (SyntaxError, ValueError):
            pass
        try:
            set_endpoint(self.sandbox_url)
            response = run_code(RunCodeRequest(code=code, language="python"))
            data = json.loads(response.json())
            result = data.get("run_result") or {}
            status = str(result.get("status") or data.get("status") or "unknown")
            stdout = str(result.get("stdout") or "")
            stderr = str(result.get("stderr") or "")
            if status == "Finished" and not stderr.strip():
                return "Tool call success" + (f"\n\nSTDOUT:\n{stdout}" if stdout else "")
            message = stderr.strip() or f"Execution failed with status: {status}"
            return f"Error: {message}" + (f"\n\nSTDOUT:\n{stdout}" if stdout else "")
        except Exception as error:
            return f"Error: SandboxFusion execution failed: {type(error).__name__}: {error}"
    def check(self) -> dict[str, str]:
        calculation = self.python("print(2 + 3)")
        retrieval = self.search("Albert Einstein")
        if not calculation.startswith("Tool call success") or "5" not in calculation:
            raise RuntimeError(f"SandboxFusion preflight failed: {calculation[:300]}")
        if not retrieval.startswith("Following are the search results"):
            raise RuntimeError(f"local Wikipedia preflight failed: {retrieval[:300]}")
        return {"python": "ok", "search": "ok"}

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
