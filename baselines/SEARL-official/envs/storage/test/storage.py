# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import asyncio
import argparse
from storage import create_storage_manager, parse_args

async def test_storage_manager():

    args = parse_args()

    storage = create_storage_manager(args)

    async def mock_tool_call(method_name: str, params: dict, ttl: Optional[int] = None):

        result = await storage.get(method_name, params)

        if result is not None:
            print(f"Cache hit for {method_name}")
            return result

        print(f"Cache miss for {method_name}, executing...")
        result = {"result": f"Result for {method_name} with params {params}"}

        await storage.set(method_name, params, result, ttl)

        return result

    test_cases = [
        ("search", {"query": "test query", "page": 1}, None),
        ("search", {"query": "test query", "page": 1}, None),
        ("search", {"query": "different query", "page": 1}, 60),
    ]

    for method_name, params, ttl in test_cases:
        result = await mock_tool_call(method_name, params, ttl)
        print(f"Result: {result}\n")

    print("Storage Stats:")
    print(storage.get_stats())

if __name__ == "__main__":
    asyncio.run(test_storage_manager())

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
