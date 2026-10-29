# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Python execution through the same inspected backend as generated tools."""
from __future__ import annotations

import asyncio
import os
from mcp.server.fastmcp import FastMCP
from control.isolation import ModelToolInputError, load_verified_backend, prepare_python_source

mcp = FastMCP("LocalServer")

def _preprocess_code(code):
    return prepare_python_source(code)

async def run_code_async(code, language="python", timeout=15.0):
    try:
        if language != "python":
            raise ModelToolInputError("Only Python is supported")
        backend = load_verified_backend(os.environ["SEARL_ISOLATION_CONFIG"],
                                        os.environ["SEARL_ISOLATION_RECEIPT"])
        result = await asyncio.to_thread(backend.run_source, _preprocess_code(code), timeout)
        return {**result, "success": result["run_success"],
                "error_type": "" if result["run_success"] else "isolated_execution_failed",
                "error_message": result.get("error", "")}
    except ModelToolInputError as error:
        return {"success": False, "run_success": False, "stdout": "", "stderr": "",
                "status": "isolated_execution_refused", "error_type": type(error).__name__,
                "error_message": str(error)[:2000]}

@mcp.tool()
async def execute_python_code(code: str, timeout: float = 15.0):
    """Execute Python in an inspected sandbox; print the last expression."""
    result = await run_code_async(code, "python", timeout)
    parts = ["Tool call success" if result["success"] else
             result["error_type"] + ": " + result["error_message"]]
    if result["stdout"]:
        parts.append("STDOUT:\n" + result["stdout"])
    if result["stderr"]:
        parts.append("STDERR:\n" + result["stderr"])
    return "\n\n".join(parts)

if __name__ == "__main__":
    mcp.run(transport="stdio")

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
