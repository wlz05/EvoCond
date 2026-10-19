# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'\n\u6d4b\u8bd5 local.py\u3001python.py \u548c creation.py \u7684\u811a\u672c\n\u7528\u4e8e\u9a8c\u8bc1\u8fd9\u4e09\u4e2a\u5de5\u5177\u662f\u5426\u80fd\u591f\u6b63\u5e38\u8fd0\u884c\n'

import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_local_search():
    '\u6d4b\u8bd5 local.py \u7684 search \u51fd\u6570'
    print("=" * 60)
    print('\u6d4b\u8bd5 local.py')
    print("=" * 60)

    try:
        from local import search

        print("\n[\u6d4b\u8bd5 1] \u57fa\u672c\u641c\u7d22 - \u67e5\u8be2 'Python programming'")
        result = search("Python programming is what", topk=3)
        print(f"\u7ed3\u679c\u7c7b\u578b: {type(result)}")
        print(f"\u7ed3\u679c\u957f\u5ea6: {len(result) if isinstance(result, str) else 'N/A'}")
        if isinstance(result, str):
            preview = result[:300] if len(result) > 300 else result
            print(f"\u7ed3\u679c\u9884\u89c8:\n{preview}")
        else:
            print(f"\u7ed3\u679c: {result}")

        print('\n[\u6d4b\u8bd5 2] \u641c\u7d22 - topk=1')
        result = search("machine learning", topk=1)
        if isinstance(result, str):
            preview = result[:200] if len(result) > 200 else result
            print(f"\u7ed3\u679c\u9884\u89c8:\n{preview}")
        else:
            print(f"\u7ed3\u679c: {result}")

        print('\n[\u6d4b\u8bd5 3] \u7a7a\u67e5\u8be2\u6d4b\u8bd5')
        result = search("", topk=3)
        print(f"\u7ed3\u679c: {result}")

        print('\n\u2713 local.py \u6d4b\u8bd5\u5b8c\u6210')
        return True

    except ImportError as e:
        print(f"\u2717 \u5bfc\u5165\u9519\u8bef: {e}")
        print('\u8bf7\u786e\u4fdd\u5df2\u5b89\u88c5\u6240\u6709\u4f9d\u8d56 (requests, mcp.server.fastmcp, config)')
        return False
    except Exception as e:
        print(f"\u2717 \u6d4b\u8bd5\u8fc7\u7a0b\u4e2d\u51fa\u9519: {e}")
        print(f"\u9519\u8bef\u7c7b\u578b: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

async def test_python_execution():
    '\u6d4b\u8bd5 python.py \u7684 execute_python_code \u51fd\u6570'
    print("\n" + "=" * 60)
    print('\u6d4b\u8bd5 python.py')
    print("=" * 60)

    try:
        from python import execute_python_code

        print('\n[\u6d4b\u8bd5 1] \u7b80\u5355\u7b97\u672f: 2 + 2')
        code = "result = 2 + 2\nprint(result)"
        result = await execute_python_code(code, timeout=15.0)
        print(f"\u7ed3\u679c:\n{result}")

        print('\n[\u6d4b\u8bd5 2] \u5b57\u7b26\u4e32\u64cd\u4f5c')
        code = "text = 'Hello, World!'\nprint(text.upper())"
        result = await execute_python_code(code, timeout=15.0)
        print(f"\u7ed3\u679c:\n{result}")

        print('\n[\u6d4b\u8bd5 3] \u5217\u8868\u64cd\u4f5c')
        code = "numbers = [1, 2, 3, 4, 5]\nprint(sum(numbers))"
        result = await execute_python_code(code, timeout=15.0)
        print(f"\u7ed3\u679c:\n{result}")

        print('\n[\u6d4b\u8bd5 4] \u9519\u8bef\u5904\u7406 - \u9664\u4ee5\u96f6')
        code = "result = 1 / 0"
        result = await execute_python_code(code, timeout=15.0)
        print(f"\u7ed3\u679c:\n{result}")

        print('\n[\u6d4b\u8bd5 5] \u5bfc\u5165\u6d4b\u8bd5 - \u4f7f\u7528 math \u6a21\u5757')
        code = "import math\nprint(math.pi)"
        result = await execute_python_code(code, timeout=15.0)
        print(f"\u7ed3\u679c:\n{result}")

        print('\n[\u6d4b\u8bd5 6] \u591a\u884c\u4ee3\u7801\u6d4b\u8bd5')
        code = '\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nresult = [fibonacci(i) for i in range(10)]\nprint(result)\n'
        result = await execute_python_code(code, timeout=15.0)
        print(f"\u7ed3\u679c:\n{result}")

        print('\n\u2713 python.py \u6d4b\u8bd5\u5b8c\u6210')
        return True

    except ImportError as e:
        print(f"\u2717 \u5bfc\u5165\u9519\u8bef: {e}")
        print('\u8bf7\u786e\u4fdd\u5df2\u5b89\u88c5\u6240\u6709\u4f9d\u8d56 (mcp.server.fastmcp, sandbox_fusion, config)')
        return False
    except Exception as e:
        print(f"\u2717 \u6d4b\u8bd5\u8fc7\u7a0b\u4e2d\u51fa\u9519: {e}")
        print(f"\u9519\u8bef\u7c7b\u578b: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    '\u6d4b\u8bd5\u662f\u5426\u80fd\u5bfc\u5165\u6240\u6709\u5fc5\u9700\u7684\u6a21\u5757'
    print("=" * 60)
    print('\u6d4b\u8bd5\u4f9d\u8d56\u5bfc\u5165')
    print("=" * 60)

    modules_to_test = [
        ("requests", 'local.py \u4f9d\u8d56'),
        ("mcp.server.fastmcp", 'MCP \u6846\u67b6'),
        ("sandbox_fusion", 'python.py \u4f9d\u8d56'),
        ("config", '\u914d\u7f6e\u6a21\u5757'),
        ("ast", 'Python \u6807\u51c6\u5e93'),
        ("json", 'Python \u6807\u51c6\u5e93'),
    ]

    results = {}
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"\u2713 {module_name:30s} - {description}")
            results[module_name] = True
        except ImportError as e:
            print(f"\u2717 {module_name:30s} - {description} (\u9519\u8bef: {e})")
            results[module_name] = False

    return all(results.values())

def test_mcp_creation():
    '\u6d4b\u8bd5 creation.py \u7684 create_and_execute_mcp \u51fd\u6570'
    print("\n" + "=" * 60)
    print('\u6d4b\u8bd5 creation.py')
    print("=" * 60)

    try:
        from creation import create_and_execute_mcp

        print('\n[\u6d4b\u8bd5 1] \u521b\u5efa\u5e76\u6267\u884c\u7b80\u5355\u52a0\u6cd5\u51fd\u6570')
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
        print(f"\u521b\u5efa\u6210\u529f: {result_dict.get('creation_success')}")
        print(f"\u6267\u884c\u7ed3\u679c: {result_dict.get('execution_result')}")
        if result_dict.get('error'):
            print(f"\u9519\u8bef\u4fe1\u606f: {result_dict.get('error')}")

        success_1 = result_dict.get('creation_success') and result_dict.get('execution_result') == 8

        print('\n[\u6d4b\u8bd5 2] \u521b\u5efa\u5e76\u6267\u884c\u4f7f\u7528 math \u5e93\u7684\u51fd\u6570')
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
        print(f"\u521b\u5efa\u6210\u529f: {result_dict.get('creation_success')}")
        print(f"\u6267\u884c\u7ed3\u679c: {result_dict.get('execution_result')}")
        if result_dict.get('error'):
            print(f"\u9519\u8bef\u4fe1\u606f: {result_dict.get('error')}")

        success_2 = result_dict.get('creation_success') and result_dict.get('execution_result') is not None

        print('\n[\u6d4b\u8bd5 3] \u521b\u5efa\u5e76\u6267\u884c\u5217\u8868\u5904\u7406\u51fd\u6570')
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
        print(f"\u521b\u5efa\u6210\u529f: {result_dict.get('creation_success')}")
        print(f"\u6267\u884c\u7ed3\u679c: {result_dict.get('execution_result')}")
        if result_dict.get('error'):
            print(f"\u9519\u8bef\u4fe1\u606f: {result_dict.get('error')}")

        success_3 = result_dict.get('creation_success') and result_dict.get('execution_result') == 15

        print('\n[\u6d4b\u8bd5 4] \u9519\u8bef\u5904\u7406\u6d4b\u8bd5 - \u9664\u4ee5\u96f6')
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
        print(f"\u521b\u5efa\u6210\u529f: {result_dict.get('creation_success')}")
        print(f"\u6267\u884c\u7ed3\u679c: {result_dict.get('execution_result')}")
        if result_dict.get('error'):
            print(f"\u9519\u8bef\u4fe1\u606f: {result_dict.get('error')[:100]}...")

        success_4 = not result_dict.get('creation_success')

        print('\n[\u6d4b\u8bd5 5] \u521b\u5efa\u5e76\u6267\u884c\u6590\u6ce2\u90a3\u5951\u51fd\u6570')
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
        print(f"\u521b\u5efa\u6210\u529f: {result_dict.get('creation_success')}")
        print(f"\u6267\u884c\u7ed3\u679c: {result_dict.get('execution_result')}")
        if result_dict.get('error'):
            print(f"\u9519\u8bef\u4fe1\u606f: {result_dict.get('error')}")

        success_5 = result_dict.get('creation_success') and result_dict.get('execution_result') == 55

        print('\n\u2713 creation.py \u6d4b\u8bd5\u5b8c\u6210')

        all_success = success_1 and success_2 and success_3 and success_4 and success_5
        if all_success:
            print('\u2713 \u6240\u6709 mcp_creation \u6d4b\u8bd5\u901a\u8fc7')
        else:
            print(f"\u2717 \u90e8\u5206\u6d4b\u8bd5\u5931\u8d25 (\u6210\u529f: {sum([success_1, success_2, success_3, success_4, success_5])}/5)")

        return all_success

    except ImportError as e:
        print(f"\u2717 \u5bfc\u5165\u9519\u8bef: {e}")
        print('\u8bf7\u786e\u4fdd\u5df2\u5b89\u88c5\u6240\u6709\u4f9d\u8d56 (mcp.server.fastmcp, sandbox_fusion, config)')
        return False
    except Exception as e:
        print(f"\u2717 \u6d4b\u8bd5\u8fc7\u7a0b\u4e2d\u51fa\u9519: {e}")
        print(f"\u9519\u8bef\u7c7b\u578b: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def check_config():
    '\u68c0\u67e5\u914d\u7f6e\u6587\u4ef6'
    print("\n" + "=" * 60)
    print('\u68c0\u67e5\u914d\u7f6e')
    print("=" * 60)

    try:
        from config import LOCAL_SEARCH_HOST, SANDBOX_HOST
        print(f"\u2713 LOCAL_SEARCH_HOST: {LOCAL_SEARCH_HOST}")
        print(f"\u2713 SANDBOX_HOST: {SANDBOX_HOST}")
        return True
    except ImportError as e:
        print(f"\u2717 \u65e0\u6cd5\u5bfc\u5165\u914d\u7f6e: {e}")
        return False
    except AttributeError as e:
        print(f"\u2717 \u914d\u7f6e\u7f3a\u5c11\u5fc5\u8981\u7684\u5c5e\u6027: {e}")
        return False

async def main():
    '\u4e3b\u6d4b\u8bd5\u8fd0\u884c\u5668'
    print("\n" + "=" * 60)
    print('\u5de5\u5177\u6d4b\u8bd5\u5957\u4ef6')
    print("=" * 60)
    print(f"Python \u7248\u672c: {sys.version}")
    print(f"\u6d4b\u8bd5\u6587\u4ef6\u4f4d\u7f6e: {Path(__file__).absolute()}")
    print()

    imports_ok = test_imports()

    if not imports_ok:
        print('\n\u26a0 \u8b66\u544a: \u67d0\u4e9b\u5bfc\u5165\u5931\u8d25\u3002\u6d4b\u8bd5\u53ef\u80fd\u65e0\u6cd5\u6b63\u5e38\u8fd0\u884c\u3002')
        print('\u8bf7\u5728\u8fd0\u884c\u6d4b\u8bd5\u524d\u5b89\u88c5\u7f3a\u5c11\u7684\u4f9d\u8d56\u3002')

    config_ok = check_config()

    search_ok = test_local_search()

    python_ok = await test_python_execution()

    mcp_ok = test_mcp_creation()

    print("\n" + "=" * 60)
    print('\u6d4b\u8bd5\u6c47\u603b')
    print("=" * 60)
    print(f"\u4f9d\u8d56\u5bfc\u5165:      {'\u2713 \u901a\u8fc7' if imports_ok else '\u2717 \u5931\u8d25'}")
    print(f"\u914d\u7f6e\u68c0\u67e5:      {'\u2713 \u901a\u8fc7' if config_ok else '\u2717 \u5931\u8d25'}")
    print(f"local_search:  {'\u2713 \u901a\u8fc7' if search_ok else '\u2717 \u5931\u8d25'}")
    print(f"python:        {'\u2713 \u901a\u8fc7' if python_ok else '\u2717 \u5931\u8d25'}")
    print(f"mcp_creation:  {'\u2713 \u901a\u8fc7' if mcp_ok else '\u2717 \u5931\u8d25'}")
    print("=" * 60)

    if imports_ok and config_ok and search_ok and python_ok and mcp_ok:
        print('\n\u2713 \u6240\u6709\u6d4b\u8bd5\u901a\u8fc7\uff01')
        return 0
    else:
        print('\n\u2717 \u67d0\u4e9b\u6d4b\u8bd5\u5931\u8d25\u3002\u8bf7\u68c0\u67e5\u4e0a\u9762\u7684\u8f93\u51fa\u3002')
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
