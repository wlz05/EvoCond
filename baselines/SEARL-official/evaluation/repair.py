# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import asyncio
import aiohttp
import os
import time
import random
import argparse
from typing import List

# API_KEY = os.getenv('BRIGHTDATA_API_KEY')
API_KEY = "f824481d-3ee0-4d05-8ac3-8f208b156f31"
ZONE = os.getenv('BRIGHTDATA_ZONE', 'sxh_bing_search')

from urllib.parse import urlencode

async def single_search(session: aiohttp.ClientSession, query: str, task_id: int):
    'Simulate one search request.'

    search_params = {
        "q": query,
        "brd_json": "1"
    }
    target_url = f"https://www.bing.com/search?{urlencode(search_params)}"

    payload = {
        "zone": ZONE,
        "url": target_url,
        "format": "raw"
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    start_time = time.time()
    try:

        async with session.post(
            "https://api.brightdata.com/request",
            json=payload,
            headers=headers,
            timeout=120
        ) as response:
            status = response.status
            text = await response.text()
            duration = time.time() - start_time

            if status == 200:
                if not text or text.strip() == "":

                    headers_str = ", ".join([f"{k}: {v}" for k, v in response.headers.items() if k.lower() in ['server', 'x-brd-error', 'x-request-id', 'content-type']])
                    return False, duration, f"Empty Body (Headers: {headers_str})"
                try:
                    import json
                    data = json.loads(text)
                    if 'organic' in data:
                        return True, duration, "Success"
                    else:
                        return False, duration, f"JSON without organic (Keys: {list(data.keys())})"
                except:
                    return False, duration, f"Invalid JSON (First 100 chars: {text[:100]})"
            else:
                return False, duration, f"HTTP {status}: {text[:100]}"
    except Exception as e:
        return False, time.time() - start_time, f"Error: {str(e)}"

async def run_stress_test(concurrency: int, total_requests: int):
    'Run a concurrent search stress test.'
    if not API_KEY:
        print('ERROR: Set the BRIGHTDATA_API_KEY environment variable.')
        return

    print(f"=== Concurrent search test ===")
    print(f"Target API: Brightdata ({ZONE})")
    print(f"Concurrency: {concurrency}, total requests: {total_requests}")
    print("---------------------------------------")

    queries = [
        f"test query {i} - {random.randint(1000, 9999)}" for i in range(total_requests)
    ]

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    async def sem_task(session, query, i):
        nonlocal completed
        async with semaphore:
            success, duration, msg = await single_search(session, query, i)
            completed += 1
            status_str = 'SUCCESS' if success else f"FAILED ({msg})"
            print(f"[{completed}/{total_requests}] {status_str} | Duration: {duration:.2f}s | Query: {query[:20]}...")
            return success, duration, msg

    timeout = aiohttp.ClientTimeout(total=45, connect=10)
    async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as session:
        tasks = [sem_task(session, queries[i], i) for i in range(total_requests)]
        results = await asyncio.gather(*tasks)

    successes = [r for r in results if r[0]]
    failures = [r for r in results if not r[0]]

    avg_lat = sum(r[1] for r in results) / len(results) if results else 0

    print('\n=== Test results ===')
    print(f"Total requests: {total_requests}")
    print(f"Succeeded: {len(successes)} ({(len(successes)/total_requests)*100:.1f}%)")
    print(f"Failed: {len(failures)} ({(len(failures)/total_requests)*100:.1f}%)")
    print(f"Mean duration: {avg_lat:.2f}s")

    if failures:
        print('\n--- Error distribution ---')
        error_counts = {}
        for f in failures:
            err_msg = f[2]
            error_counts[err_msg] = error_counts.get(err_msg, 0) + 1
        for msg, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"- {msg}: {count} times")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=int, default=10, help='Concurrency')
    parser.add_argument("--n", type=int, default=20, help='Total requests')
    args = parser.parse_args()

    asyncio.run(run_stress_test(args.c, args.n))

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
