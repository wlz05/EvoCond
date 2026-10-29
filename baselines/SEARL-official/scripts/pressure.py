# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import asyncio
import time
import json
import os
import sys
import numpy as np
from tqdm.asyncio import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

from envs.tool_manager.baseline import QwenManagerBaseline

CONFIG_PATH = os.environ.get("RL_FACTORY_MCP_CONFIG", "envs/configs/mcp_tools_local_graph.pydata")
TOOL_TIMEOUT = 60
TEST_ROUNDS = [10, 50, 100, 200]

TOOLS_TO_TEST = {
    "search": {
        "name": "search-search",
        "args": json.dumps({"query": "current who is jij", "topk": 3})
    },
    "python": {
        "name": "execute_python_code-execute_python_code",
        "args": json.dumps({"code": "print(123 + 456)", "timeout": 5})
    }
}
# ===========================================

class MockConfig:
    def __init__(self):
        self.config_path = CONFIG_PATH
        self.mcp_mode = 'stdio'
        self.enable_limiter = False
        self.tool_timeout = TOOL_TIMEOUT
        self.tool_name_selected = []
        self.use_storage_manager = False
        self.enable_thinking = False
        self.local_search = True
        self.max_concurrency = 1000

    def get(self, key, default=None):
        return getattr(self, key, default)

async def run_batch(manager, tool_name, payload, concurrency):
    print(f"\n>>> Testing tool [{tool_name}] concurrency: {concurrency}")

    start_time = time.time()

    async def single_request(idx):
        try:

            current_payload = payload.copy()
            if tool_name == "search":
                args = json.loads(current_payload["args"])
                args["query"] = f"test query {idx} timestamp {time.time()}"
                current_payload["args"] = json.dumps(args)

            req_start = time.time()

            result = await asyncio.to_thread(
                manager._call_tool,
                current_payload["name"],
                current_payload["args"]
            )
            duration = time.time() - req_start

            success = True
            if '# Tool call failed' in str(result) or "# Execute the tool" in str(result) and "failed" in str(result):
                success = False
            if "timeout" in str(result).lower():
                success = False

            return {"success": success, "duration": duration, "error": None if success else str(result)[:100]}
        except Exception as e:
            return {"success": False, "duration": time.time() - req_start, "error": str(e)}

    tasks = [single_request(i) for i in range(concurrency)]

    results = await tqdm.gather(*tasks, desc=f"Testing {tool_name}")

    total_time = time.time() - start_time

    success_count = sum(1 for r in results if r['success'])
    durations = [r['duration'] for r in results]
    errors = [r['error'] for r in results if not r['success']]

    print(f"--- Statistics (concurrency {concurrency}) ---")
    print(f"Total duration: {total_time:.2f}s")
    print(f"QPS: {concurrency / total_time:.2f}")
    print(f"Success rate: {success_count}/{concurrency} ({success_count/concurrency*100:.1f}%)")
    if durations:
        print(f"Mean duration: {np.mean(durations):.4f}s")
        print(f"P50 duration: {np.percentile(durations, 50):.4f}s")
        print(f"P95 duration: {np.percentile(durations, 95):.4f}s")
        print(f"P99 duration: {np.percentile(durations, 99):.4f}s")

    if errors:
        print('Most frequent errors (top 3):')
        from collections import Counter
        for err, count in Counter(errors).most_common(3):
            print(f"  - {count}x: {err}")

    return success_count == concurrency

async def main():
    print('Initializing the tool manager...')
    config = MockConfig()
    manager = QwenManagerBaseline(config)

    for tool_key, payload in TOOLS_TO_TEST.items():
        print(f"\n{'='*40}")
        print(f"Test target: {tool_key}")
        print(f"{'='*40}")

        for concurrency in TEST_ROUNDS:
            await run_batch(manager, tool_key, payload, concurrency)

            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
