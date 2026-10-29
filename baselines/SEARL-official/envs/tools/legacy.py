# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import json
import re
import traceback
from typing import Tuple, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP
from sandbox_fusion import run_code, RunCodeRequest, set_endpoint
from config import SANDBOX_HOST
from pathlib import Path
import os
mcp = FastMCP("LocalServer")
set_endpoint(f"http://{SANDBOX_HOST}:8080")

def _get_project_name() -> str:
    'Resolve the project from environment or default_project.'
    return os.environ.get("PROJECT_NAME") or "grpo_baseline_4gpu_reasoning_mcp"

BASE_DIR = Path(__file__).parent

def _get_invented_paths():
    'Resolve invented-tool paths on each call to honor the current project.'
    project_name = _get_project_name()
    invented_dir = BASE_DIR / "invented_tools" / project_name
    invented_file = invented_dir / "temp_tool_list.py"
    official_file = invented_dir / "official_tool_list.py"
    return invented_dir, invented_file, official_file

def run_code_sync(code: str, language: str = "python", timeout: float = 15.0) -> Tuple[str, str, bool]:
    'Execute Python through sandbox_fusion with a timeout in seconds; return stdout, stderr and the final-answer flag.'
    try:
        if language != "python":
            return "", f"Unsupported language: {language}", False

        r = run_code(RunCodeRequest(code=code, language=language))
        try:
            data = json.loads(r.json())
            status = data.get('run_result', {}).get('status', 'unknown')
            if status == 'Finished':
                stdout = data.get('run_result', {}).get('stdout', '')
                stderr = data.get('run_result', {}).get('stderr', '')
            else:
                stdout = ''
                stderr = data.get('run_result', {}).get('stderr', f'Execution failed with status: {status}')
        except Exception:

            try:
                data = json.loads(r.json())
                stdout = ''
                stderr = data.get('run_result', {}).get('stderr', f'Execution failed: {data.get("status", "unknown")}')
            except Exception as e:
                return "", f"Failed to parse response: {str(e)}", False

        final_answer = "FINAL ANSWER" in stdout.upper() or "FINAL ANSWER" in stderr.upper()

        return stdout, stderr, final_answer

    except Exception as e:
        return "", f"Code execution failed: {str(e)}", False

@mcp.tool()
def create_and_execute_mcp(
    name: str,
    description: str,
    arguments: str,
    returns: str,
    code: str,
    inputs: Dict[str, Any],
    timeout: float = 15.0
) -> str:
    '\n    MCP Creation and Execution Tool\n\n    Create and immediately execute an MCP tool function.\n\n    Args:\n        name: MCP tool name (Function name)\n        description: Tool description\n        arguments: Argument description string (e.g., "a, b (int)")\n        returns: Return value description\n        code: Complete Python function implementation code\n        inputs: Input arguments dictionary required for this function call (e.g., {"a": 1, "b": 2})\n        timeout: Execution timeout in seconds (default: 15.0)\n\n    Returns:\n        str: JSON formatted string containing creation status and execution result\n             {\n                 "creation_success": bool,\n                 "execution_result": any,\n                 "stdout": str,\n                 "stderr": str,\n                 "error": str (optional)\n             }\n    '
    try:
        print(f"Creating and executing MCP: {name}...")

        INVENTED_DIR, INVENTED_FILE, OFFICIAL_FILE = _get_invented_paths()

        imports = []

        import_block = "\n".join(imports)

        inputs_json = json.dumps(inputs)

        execution_script = f"""\nimport json\nimport sys\nimport traceback\nimport math\nfrom math import *  \nimport sympy\nimport scipy as sp\nimport cmath\nfrom fractions import Fraction\nimport numpy as np\nimport re\nfrom itertools import *\nimport os\nimport json \nimport datetime\nimport re\nfrom collections import Counter, defaultdict, deque\n# Imports\n{import_block}\n\n# User Defined Function\n{code}\n\nif __name__ == "__main__":\n    try:\n        # Load inputs\n        inputs = json.loads({repr(inputs_json)})\n        \n        # Verify function exists\n        if '{name}' not in globals():\n            raise NameError(f"Function '{name}' not found in code definition.")\n        \n        target_func = globals()['{name}\']\n        \n        # Execute the function with keyword arguments when available.\n        if isinstance(inputs, dict):\n            result = target_func(**inputs)\n        else:\n            # Pass a non-dictionary input as one argument.\n            result = target_func(inputs)\n            \n        # Mark the result for extraction.\n        print(f\"__MCP_RESULT_START__{{json.dumps(result)}}__MCP_RESULT_END__\")\n        print(\"__MCP_CREATION_SUCCESS__\")\n        \n    except Exception as e:\n        traceback.print_exc()\n        print(f\"__MCP_EXECUTION_ERROR__{{str(e)}}\")\n"""

        stdout, stderr, _ = run_code_sync(execution_script, "python", timeout)

        creation_success = "__MCP_CREATION_SUCCESS__" in stdout
        execution_result = None
        error_msg = None

        result_match = re.search(r"__MCP_RESULT_START__(.*?)__MCP_RESULT_END__", stdout, re.DOTALL)
        if result_match:
            try:
                execution_result = json.loads(result_match.group(1))
            except:
                execution_result = result_match.group(1) # Fallback to raw string

        if not creation_success:

            if stderr:
                error_msg = stderr
            else:
                error_match = re.search(r"__MCP_EXECUTION_ERROR__(.*)", stdout)
                if error_match:
                    error_msg = error_match.group(1)
                else:
                    error_msg = "Unknown execution error"

        response = {
            "creation_success": creation_success,
            "execution_result": execution_result,
            "stdout": stdout.replace("__MCP_CREATION_SUCCESS__", "").replace(f"__MCP_RESULT_START__{json.dumps(execution_result) if execution_result is not None else ''}__MCP_RESULT_END__", "").strip(),
            "stderr": stderr
        }

        if error_msg:
            response["error"] = error_msg

        if creation_success and error_msg is None:
            try:
                INVENTED_DIR.mkdir(parents=True, exist_ok=True)

                common_imports = (
                    "import json\n"
                    "import sys\n"
                    "import traceback\n"
                    "import math\n"
                    "import numpy as np\n"
                    "import re\n"
                    "import os\n"
                    "import datetime\n"
                    "from collections import Counter, defaultdict, deque\n"
                    "from mcp.server.fastmcp import FastMCP\n\n"
                )

                if not INVENTED_FILE.exists() or INVENTED_FILE.stat().st_size == 0:
                    INVENTED_FILE.write_text(
                        f"{common_imports}"
                        "mcp = FastMCP('InventedServer')\n\n",
                        encoding="utf-8"
                    )
                if not OFFICIAL_FILE.exists() or OFFICIAL_FILE.stat().st_size == 0:
                    OFFICIAL_FILE.write_text(
                        f"{common_imports}"
                        "mcp = FastMCP('Official_Server')\n\n",
                        encoding="utf-8"
                    )

                def build_tool_block():
                    sig_args = list(inputs.keys()) if isinstance(inputs, dict) else []
                    def doc_lines(base_indent: str = "    "):

                        return "\n".join(
                            [
                                f'{base_indent}"""{description}',
                                f"{base_indent}{returns}",
                                f"{base_indent}Args:",
                                f"{base_indent}    {arguments}",
                                f"{base_indent}",
                                f"{base_indent}Returns:",
                                f"{base_indent}    {returns}",
                                f'{base_indent}"""',
                            ]
                        )

                    if code.lstrip().startswith("def "):
                        code_lines = code.splitlines()

                        doc_start = doc_end = -1
                        for idx, ln in enumerate(code_lines[1:], start=1):
                            if '"""' in ln or "'''" in ln:
                                doc_start = idx
                                quote = '"""' if '"""' in ln else "'''"

                                for j in range(idx, len(code_lines)):
                                    if quote in code_lines[j] and j != idx:
                                        doc_end = j
                                        break
                                break

                        if doc_start != -1 and doc_end != -1:

                            inner = "\n".join(code_lines[doc_start:doc_end + 1])
                            inner_content = inner.strip().strip('"\'')

                            base_indent = re.match(r"\s*", code_lines[0]).group(0) + "    "
                            new_doc = [
                                f'{base_indent}"""{description}',
                                f"{base_indent}{inner_content}",
                                f"{base_indent}Args:",
                                f"{base_indent}    {arguments}",
                                f"{base_indent}",
                                f"{base_indent}Returns:",
                                f"{base_indent}    {returns}",
                                f'{base_indent}"""',
                            ]
                            code_lines = code_lines[:doc_start] + new_doc + code_lines[doc_end + 1 :]
                        else:

                            indent = re.match(r"\s*", code_lines[0]).group(0) + "    "
                            insertion = doc_lines(indent)
                            code_lines = [code_lines[0], insertion, *code_lines[1:]]

                        merged = "\n".join(code_lines)

                        return f"\n\n@mcp.tool(name='{name}')\n{merged.lstrip()}\n"

                    if sig_args:
                        signature = ", ".join(sig_args)
                    else:
                        signature = "**kwargs"
                    indented_body = "\n".join(f"        {line}" if line.strip() else "        " for line in code.splitlines())
                    wrapped_body = (
                        "    try:\n"
                        f"{indented_body}\n"
                        "    except Exception as e:\n"
                        "        return f\"tool execution failed: {str(e)}\"\n"
                    )
                    return (
                        f"\n\n@mcp.tool(name='{name}')\n"
                        f"def {name}({signature}):\n"
                        f"{doc_lines()}\n"
                        f"{wrapped_body}\n"
                    )

                tool_block = build_tool_block()

                content = ""
                official_content = ""
                if INVENTED_FILE.exists():
                    content = INVENTED_FILE.read_text(encoding="utf-8")

                main_block_start = "\n\nif __name__ == \"__main__\":"
                if main_block_start in content:
                    content = content.split(main_block_start)[0]

                header = f"{common_imports}mcp = FastMCP('InventedServer')\n\n"

                if not content.strip():
                    content = header
                elif "FastMCP" not in content:
                    content = header + content

                content += tool_block

                main_block = (
                    "\n\nif __name__ == \"__main__\":\n"
                    "    print(\"\\nStart MCP service:\")\n"
                    "    mcp.run(transport='stdio')\n"
                )
                if main_block not in content:
                    content += main_block
                if main_block not in official_content:
                    official_content += main_block

                INVENTED_FILE.write_text(content, encoding="utf-8")
                OFFICIAL_FILE.write_text(official_content, encoding="utf-8")
            except Exception as e:

                response.setdefault("persist_errors", [])
                response["persist_errors"].append(f"write tool to {INVENTED_FILE} failed: {e}")

        return json.dumps(response, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "creation_success": False,
            "error": f"Tool internal error: {str(e)}"
        }, ensure_ascii=False)

def test_mcp_creation():
    'Test MCP creation and failure rollback while preserving the temporary tool list.'
    import json
    import uuid
    from pathlib import Path

    INVENTED_DIR, INVENTED_FILE, _ = _get_invented_paths()

    backup = INVENTED_FILE.read_text(encoding="utf-8") if INVENTED_FILE.exists() else ""

    ok_name = f"test_ok_{uuid.uuid4().hex[:8]}"
    bad_name = f"test_bad_{uuid.uuid4().hex[:8]}"

    try:

        ok_code = f"def {ok_name}(a:int,b:int):\n    return a + b\n"
        resp_ok = json.loads(
            create_and_execute_mcp(
                name=ok_name,
                description="add two numbers",
                arguments="a,b",
                returns="sum",
                code=ok_code,
                inputs={"a": 1, "b": 2},
                timeout=5.0,
            )
        )
        assert resp_ok["creation_success"] is True, resp_ok
        assert resp_ok["execution_result"] == 3, resp_ok

        content = INVENTED_FILE.read_text(encoding="utf-8")
        assert ok_name in content, 'Created tool was not written to disk'

        bad_code = "def broken(x):\n    raise RuntimeError('boom')\n"
        resp_bad = json.loads(
            create_and_execute_mcp(
                name=bad_name,
                description="bad tool",
                arguments="x",
                returns="y",
                code=bad_code,
                inputs={},
                timeout=5.0,
            )
        )
        assert resp_bad["creation_success"] is False, 'The failure case must not succeed'
        content_after = INVENTED_FILE.read_text(encoding="utf-8")
        assert bad_name not in content_after, 'A failed tool must not be written to disk'

        print('PASS test_mcp_creation passed')
        return True
    finally:

        INVENTED_FILE.write_text(backup, encoding="utf-8")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_mcp_creation()
    else:
        print("\nStart MCP service:")
        mcp.run(transport='stdio')

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
