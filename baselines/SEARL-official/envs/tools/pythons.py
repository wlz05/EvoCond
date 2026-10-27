# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Check the local search, Python execution and MCP creation tools.'

import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_local_search():
    'Test the local search function.'
    print("=" * 60)
    print('Testing local.py')
    print("=" * 60)

    try:
        from local import search

        print("\n[Test 1] Basic search for 'Python programming'")
        result = search("Python programming is what", topk=3)
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result) if isinstance(result, str) else 'N/A'}")
        if isinstance(result, str):
            preview = result[:300] if len(result) > 300 else result
            print(f"Result preview:\n{preview}")
        else:
            print(f"Result: {result}")

        print('\n[Test 2] Search with topk=1')
        result = search("machine learning", topk=1)
        if isinstance(result, str):
            preview = result[:200] if len(result) > 200 else result
            print(f"Result preview:\n{preview}")
        else:
            print(f"Result: {result}")

        print('\n[Test 3] Empty query')
        result = search("", topk=3)
        print(f"Result: {result}")

        print('\nlocal.py tests completed.')
        return True

    except ImportError as e:
        print(f"Import error: {e}")
        print('Install the required dependencies: requests, mcp.server.fastmcp, config')
        return False
    except Exception as e:
        print(f"Test error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

async def test_python_execution():
    'Test execute_python_code.'
    print("\n" + "=" * 60)
    print('Testing python.py')
    print("=" * 60)

    try:
        from python import execute_python_code

        print('\n[Test 1] Simple arithmetic: 2 + 2')
        code = "result = 2 + 2\nprint(result)"
        result = await execute_python_code(code, timeout=15.0)
        print(f"Result:\n{result}")

        print('\n[Test 2] String operations')
        code = "text = 'Hello, World!'\nprint(text.upper())"
        result = await execute_python_code(code, timeout=15.0)
        print(f"Result:\n{result}")

        print('\n[Test 3] List operations')
        code = "numbers = [1, 2, 3, 4, 5]\nprint(sum(numbers))"
        result = await execute_python_code(code, timeout=15.0)
        print(f"Result:\n{result}")

        print('\n[Test 4] Division-by-zero handling')
        code = "result = 1 / 0"
        result = await execute_python_code(code, timeout=15.0)
        print(f"Result:\n{result}")

        print('\n[Test 5] Importing math')
        code = "import math\nprint(math.pi)"
        result = await execute_python_code(code, timeout=15.0)
        print(f"Result:\n{result}")

        print('\n[Test 6] Multiline code')
        code = '\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nresult = [fibonacci(i) for i in range(10)]\nprint(result)\n'
        result = await execute_python_code(code, timeout=15.0)
        print(f"Result:\n{result}")

        print('\npython.py tests completed.')
        return True

    except ImportError as e:
        print(f"Import error: {e}")
        print('Install the required dependencies: mcp.server.fastmcp, sandbox_fusion, config')
        return False
    except Exception as e:
        print(f"Test error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    'Check all required imports.'
    print("=" * 60)
    print('Testing dependency imports')
    print("=" * 60)

    modules_to_test = [
        ("requests", 'local.py dependencies'),
        ("mcp.server.fastmcp", 'MCP framework'),
        ("sandbox_fusion", 'python.py dependencies'),
        ("config", 'Configuration module'),
        ("ast", 'Python standard library'),
        ("json", 'Python standard library'),
    ]

    results = {}
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"PASS {module_name:30s} - {description}")
            results[module_name] = True
        except ImportError as e:
            print(f"FAIL {module_name:30s} - {description} (error: {e})")
            results[module_name] = False

    return all(results.values())

def test_mcp_creation():
    'Test create_and_execute_mcp.'
    print("\n" + "=" * 60)
    print('Testing creation.py')
    print("=" * 60)

    try:
        from creation import create_and_execute_mcp

        print('\n[Test 1] Create and execute an addition function')
        result = create_and_execute_mcp(
            name="add_numbers",
            description="Add two numbers together",
            arguments="a, b (int)",
            returns="int: sum of a and b",
            code="def add_numbers(a: int, b: int):\n    return a + b",
            inputs={"a": 5, "b": 3},
            timeout=10.0
        )

        result_dict = json.loads(result)
        print(f"Created: {result_dict.get('creation_success')}")
        print(f"Execution result: {result_dict.get('execution_result')}")
        if result_dict.get('error'):
            print(f"Error message: {result_dict.get('error')}")

        success_1 = result_dict.get('creation_success') and result_dict.get('execution_result') == 8

        print('\n[Test 2] Create and execute a function using math')
        result = create_and_execute_mcp(
            name="calculate_circle_area",
            description="Calculate the area of a circle",
            arguments="radius (float)",
            returns="float: area of the circle",
            code="def calculate_circle_area(radius: float):\n    import math\n    return math.pi * radius ** 2",
            inputs={"radius": 5.0},
            timeout=10.0
        )

        result_dict = json.loads(result)
        print(f"Created: {result_dict.get('creation_success')}")
        print(f"Execution result: {result_dict.get('execution_result')}")
        if result_dict.get('error'):
            print(f"Error message: {result_dict.get('error')}")

        success_2 = result_dict.get('creation_success') and result_dict.get('execution_result') is not None

        print('\n[Test 3] Create and execute a list-processing function')
        result = create_and_execute_mcp(
            name="sum_list",
            description="Sum all numbers in a list",
            arguments="numbers (list)",
            returns="int: sum of all numbers",
            code="def sum_list(numbers: list):\n    return sum(numbers)",
            inputs={"numbers": [1, 2, 3, 4, 5]},
            timeout=10.0
        )

        result_dict = json.loads(result)
        print(f"Created: {result_dict.get('creation_success')}")
        print(f"Execution result: {result_dict.get('execution_result')}")
        if result_dict.get('error'):
            print(f"Error message: {result_dict.get('error')}")

        success_3 = result_dict.get('creation_success') and result_dict.get('execution_result') == 15

        print('\n[Test 4] Division-by-zero handling')
        result = create_and_execute_mcp(
            name="divide_by_zero",
            description="Test error handling",
            arguments="a (int)",
            returns="int",
            code="def divide_by_zero(a: int):\n    return a / 0",
            inputs={"a": 10},
            timeout=10.0
        )

        result_dict = json.loads(result)
        print(f"Created: {result_dict.get('creation_success')}")
        print(f"Execution result: {result_dict.get('execution_result')}")
        if result_dict.get('error'):
            print(f"Error message: {result_dict.get('error')[:100]}...")

        success_4 = not result_dict.get('creation_success')

        print('\n[Test 5] Create and execute a Fibonacci function')
        result = create_and_execute_mcp(
            name="fibonacci",
            description="Calculate nth Fibonacci number",
            arguments="n (int)",
            returns="int: nth Fibonacci number",
            code='def fibonacci(n: int):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b',
            inputs={"n": 10},
            timeout=10.0
        )

        result_dict = json.loads(result)
        print(f"Created: {result_dict.get('creation_success')}")
        print(f"Execution result: {result_dict.get('execution_result')}")
        if result_dict.get('error'):
            print(f"Error message: {result_dict.get('error')}")

        success_5 = result_dict.get('creation_success') and result_dict.get('execution_result') == 55

        print('\ncreation.py tests completed.')

        all_success = success_1 and success_2 and success_3 and success_4 and success_5
        if all_success:
            print('All mcp_creation tests passed.')
        else:
            print(f"Some tests failed (passed: {sum([success_1, success_2, success_3, success_4, success_5])}/5)")

        return all_success

    except ImportError as e:
        print(f"Import error: {e}")
        print('Install the required dependencies: mcp.server.fastmcp, sandbox_fusion, config')
        return False
    except Exception as e:
        print(f"Test error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def check_config():
    'Check the configuration.'
    print("\n" + "=" * 60)
    print('Checking configuration')
    print("=" * 60)

    try:
        from config import LOCAL_SEARCH_HOST, SANDBOX_HOST
        print(f"PASS LOCAL_SEARCH_HOST: {LOCAL_SEARCH_HOST}")
        print(f"PASS SANDBOX_HOST: {SANDBOX_HOST}")
        return True
    except ImportError as e:
        print(f"Cannot import configuration: {e}")
        return False
    except AttributeError as e:
        print(f"Configuration is missing required attributes: {e}")
        return False

async def main():
    'Run the tool test suite.'
    print("\n" + "=" * 60)
    print('Tool test suite')
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Test file location: {Path(__file__).absolute()}")
    print()

    imports_ok = test_imports()

    if not imports_ok:
        print('\nWARNING: Some imports failed; tests may not run correctly.')
        print('Install the missing dependencies before testing.')

    config_ok = check_config()

    search_ok = test_local_search()

    python_ok = await test_python_execution()

    mcp_ok = test_mcp_creation()

    print("\n" + "=" * 60)
    print('Test summary')
    print("=" * 60)
    print(f"Dependency imports: {'PASS' if imports_ok else 'FAIL'}")
    print(f"Configuration check: {'PASS' if config_ok else 'FAIL'}")
    print(f"local_search:  {'PASS' if search_ok else 'FAIL'}")
    print(f"python:        {'PASS' if python_ok else 'FAIL'}")
    print(f"mcp_creation:  {'PASS' if mcp_ok else 'FAIL'}")
    print("=" * 60)

    if imports_ok and config_ok and search_ok and python_ok and mcp_ok:
        print('\nAll tests passed.')
        return 0
    else:
        print('\nSome tests failed; check the output above.')
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
