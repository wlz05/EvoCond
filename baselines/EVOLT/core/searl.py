# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Use SEARL-style local retrieval and SandboxFusion without local code execution."""
import ast
import json
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
from urllib.parse import urlparse
def endpoint(value: str, path: str = "") -> str:
    """Require an explicit HTTP service URL rather than a public web fallback."""
    parsed = urlparse(str(value or ""))
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("SEARL service URL must be an explicit http://host:port URL")
    if path and parsed.path != path:
        raise ValueError(f"SEARL retrieval URL must end in {path}")
    return value.rstrip("/")
def printable(code: str) -> str:
    """Match SEARL's last-expression printing behavior."""
    try:
        tree = ast.parse(code)
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = tree.body[-1]
            if not isinstance(last.value, ast.Call) or not isinstance(last.value.func, ast.Name) or last.value.func.id != "print":
                tree.body[-1] = ast.Expr(ast.Call(ast.Name("print", ast.Load()), [last.value], []))
                return ast.unparse(ast.fix_missing_locations(tree))
    except (SyntaxError, ValueError):
        pass
    return code
class SearlTools:
    """Talk to the same service types used by SEARL's published tool wrappers."""
    def __init__(self, search_url: str = "http://127.0.0.1:5003/retrieve", sandbox_url: str = "http://127.0.0.1:8080"):
        self.search_url = endpoint(search_url, "/retrieve")
        self.sandbox_url = endpoint(sandbox_url)
    def search(self, query: str) -> str:
        "Search the local Wikipedia index, returning SEARL's top-three format.\n        Args:\n            query: Search query.\n        Returns:\n            Ranked local Wikipedia passages.\n        "
        if not isinstance(query, str) or not query.strip() or len(query) > 300:
            return "Error: search query must contain at most 300 characters."
        payload = json.dumps({"queries": [query.strip()], "topk": 3, "return_scores": True}).encode("utf-8")
        request = Request(self.search_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with build_opener(ProxyHandler({})).open(request, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
            results = data.get("result") if isinstance(data, dict) else None
            if not isinstance(results, list):
                return "Error: local retrieval response has no result list."
            documents = results[0] if results else []
            if not isinstance(documents, list) or not documents:
                return "No relevant documents found"
            output = "Following are the search results given by the wiki search engine:"
            for item in documents:
                contents = item["document"]["contents"]
                title, _, body = str(contents).partition("\n")
                output += f'\nsearch results: "{title.strip(chr(34))}"\n{body}'
            return output
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as error:
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
            set_endpoint(self.sandbox_url)
            response = run_code(RunCodeRequest(code=printable(code), language="python"))
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
    def check(self) -> dict:
        """Fail before GPU loading if either required service is unusable."""
        calculation = self.python("print(2 + 3)")
        retrieval = self.search("Albert Einstein")
        if not calculation.startswith("Tool call success") or "5" not in calculation:
            raise RuntimeError(f"SEARL SandboxFusion preflight failed: {calculation[:300]}")
        if not retrieval.startswith("Following are the search results"):
            raise RuntimeError(f"SEARL local retrieval preflight failed: {retrieval[:300]}")
        return {"python": "ok", "search": "ok", "search_url": self.search_url, "sandbox_url": self.sandbox_url}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
