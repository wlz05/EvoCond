# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Entrypoint contract tests; FastMCP and sandbox calls are CPU mocks."""
import asyncio
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_root))
from control.isolation import ModelToolInputError, SandboxInfrastructureError

class MockMCP:
    def __init__(self, *args):
        pass

    def tool(self):
        return lambda function: function

def load(name, filename):
    path = Path(__file__).resolve().parents[1] / "envs" / "tools" / filename
    specification = spec_from_file_location(name, path)
    module = module_from_spec(specification)
    with patch.dict(sys.modules, {"mcp.server.fastmcp": types.SimpleNamespace(FastMCP=MockMCP)}):
        specification.loader.exec_module(module)
    return module

creation = load("mcp_creation_test_entrypoint", "creation.py")
python_tool = load("python_test_entrypoint", "python.py")

class EntrypointTests(unittest.TestCase):
    def test_creation_registers_probe_instead_of_writing_python(self):
        dispatch = MagicMock()
        dispatch.create.return_value = {"creation_success": True, "run_success": True,
                                        "registered": True, "tool_version": "version", "execution_result": 8}
        with patch.object(creation.IsolatedToolDispatcher, "from_environment", return_value=dispatch), \
                patch.dict(os.environ, {"SEARL_TASK_ID": "task1"}):
            result = json.loads(creation.create_and_execute_mcp(
                "f", "double", "x:int", "int", "def f(x): return x*2", {"x": 4}, 7))
        self.assertTrue(result["registered"])
        self.assertEqual(dispatch.create.call_args.kwargs, {"task_id": "task1", "timeout": 7})

    def test_creation_requires_task_context(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(creation.IsolatedToolDispatcher, "from_environment") as load_backend, \
                self.assertRaises(RuntimeError):
            creation.create_and_execute_mcp("f", "double", "x", "int", "def f(x): return x*2", {"x": 4})
        load_backend.assert_not_called()

    def test_creation_refuses_missing_sandbox_without_legacy_fallback(self):
        with patch.dict(os.environ, {"SEARL_TASK_ID": "task1"}, clear=True), self.assertRaises(RuntimeError):
            creation.create_and_execute_mcp("f", "double", "x", "int", "def f(x): return x*2", {"x": 4})

    def test_python_entrypoint_uses_same_backend_with_timeout(self):
        backend = MagicMock()
        backend.run_source.return_value = {"run_success": True, "stdout": "3", "stderr": "", "status": "Finished"}
        with patch.object(python_tool, "load_verified_backend", return_value=backend), \
                patch.dict(os.environ, {"SEARL_ISOLATION_CONFIG": "c.json", "SEARL_ISOLATION_RECEIPT": "r.json"}):
            result = asyncio.run(python_tool.run_code_async("1 + 2", timeout=4))
        self.assertTrue(result["success"])
        self.assertEqual(backend.run_source.call_args.args, ("print(1 + 2)", 4))

    def test_python_missing_receipt_fails_closed(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(KeyError):
            asyncio.run(python_tool.run_code_async("raise RuntimeError('not on host')"))

    def test_invalid_model_request_is_recoverable_but_infrastructure_is_not(self):
        from control.isolation import ModelToolInputError, SandboxInfrastructureError
        dispatch = MagicMock()
        dispatch.create.side_effect = ModelToolInputError("Wrong argument type")
        with patch.dict(os.environ, {"SEARL_TASK_ID": "task1"}), \
                patch.object(creation.IsolatedToolDispatcher, "from_environment", return_value=dispatch):
            result = json.loads(creation.create_and_execute_mcp("f", "double", "x", "int", "def f(x): return x*2", {}))
            self.assertFalse(result["registered"])
            dispatch.create.side_effect = SandboxInfrastructureError("Executor unavailable")
            with self.assertRaises(SandboxInfrastructureError):
                creation.create_and_execute_mcp("f", "double", "x", "int", "def f(x): return x*2", {})

    def test_python_failure_is_not_hidden_by_mcp_string(self):
        async def failure(*args):
            return {"success": False, "stdout": "", "stderr": "trace", "error_type": "timeout",
                    "error_message": "time limit"}
        with patch.object(python_tool, "run_code_async", side_effect=failure):
            text = asyncio.run(python_tool.execute_python_code("pass"))
        self.assertTrue(text.startswith("timeout:"))
        self.assertNotIn("Tool call success", text)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
