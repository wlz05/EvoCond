# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Create and register virtual tools through an inspected isolated executor."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict

# Only trusted repository source is imported; generated source stays in JSON.
_dispatcher_modules = str(Path(__file__).resolve().parents[1] / "tool_manager" / "centralized")
if _dispatcher_modules not in sys.path:
    sys.path.insert(0, _dispatcher_modules)
from mcp.server.fastmcp import FastMCP
from dispatch import IsolatedToolDispatcher
from control.isolation import ModelToolInputError

mcp = FastMCP("LocalServer")

@mcp.tool()
def create_and_execute_mcp(name: str, description: str, arguments: str,
                           returns: str, code: str, inputs: Dict[str, Any],
                           timeout: float = 15.0) -> str:
    'Probe a Python function in isolation and persist its callable version.\n\n    The actor invokes registered tools through the virtual dispatcher. This\n    service never writes or imports generated Python modules on the host.\n    '
    try:
        task_id = os.environ.get("SEARL_TASK_ID")
        if not task_id:
            raise RuntimeError("SEARL_TASK_ID is required for standalone admission")
        dispatcher = IsolatedToolDispatcher.from_environment()
        spec = {"name": name, "description": description, "arguments": arguments,
                "returns": returns, "code": code, "side_effects": "unspecified; confined to inspected sandbox"}
        result = dispatcher.create(spec, inputs, task_id=task_id, timeout=timeout)
    except ModelToolInputError as error:
        result = {"creation_success": False, "run_success": False, "registered": False,
                  "execution_result": None, "stdout": "", "stderr": "",
                  "reason": "isolated_admission_failed", "error_type": type(error).__name__,
                  "error_message": str(error)[:2000], "error": str(error)[:2000]}
    return json.dumps(result, ensure_ascii=False, allow_nan=False)

if __name__ == "__main__":
    mcp.run(transport="stdio")

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
