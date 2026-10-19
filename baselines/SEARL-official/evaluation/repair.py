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
    '\u6a21\u62df\u5355\u6b21\u641c\u7d22\u8bf7\u6c42'

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
    '\u8fd0\u884c\u9ad8\u5e76\u53d1\u538b\u6d4b'
    if not API_KEY:
        print('\u9519\u8bef: \u8bf7\u8bbe\u7f6e BRIGHTDATA_API_KEY \u73af\u5883\u53d8\u91cf')
        return

    print(f"=== \u5f00\u59cb\u9ad8\u5e76\u53d1\u641c\u7d22\u6d4b\u8bd5 ===")
    print(f"\u76ee\u6807 API: Brightdata ({ZONE})")
    print(f"\u5e76\u53d1\u6570: {concurrency}, \u603b\u8bf7\u6c42\u6570: {total_requests}")
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
            status_str = '\u2705 \u6210\u529f' if success else f"\u274c \u5931\u8d25 ({msg})"
            print(f"[{completed}/{total_requests}] {status_str} | \u8017\u65f6: {duration:.2f}s | Query: {query[:20]}...")
            return success, duration, msg

    timeout = aiohttp.ClientTimeout(total=45, connect=10)
    async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as session:
        tasks = [sem_task(session, queries[i], i) for i in range(total_requests)]
        results = await asyncio.gather(*tasks)

    successes = [r for r in results if r[0]]
    failures = [r for r in results if not r[0]]

    avg_lat = sum(r[1] for r in results) / len(results) if results else 0

    print('\n=== \u6d4b\u8bd5\u7ed3\u679c\u62a5\u544a ===')
    print(f"\u603b\u8bf7\u6c42\u6570: {total_requests}")
    print(f"\u6210\u529f\u6b21\u6570: {len(successes)} ({(len(successes)/total_requests)*100:.1f}%)")
    print(f"\u5931\u8d25\u6b21\u6570: {len(failures)} ({(len(failures)/total_requests)*100:.1f}%)")
    print(f"\u5e73\u5747\u8017\u65f6: {avg_lat:.2f}s")

    if failures:
        print('\n--- \u9519\u8bef\u7c7b\u578b\u5206\u5e03 ---')
        error_counts = {}
        for f in failures:
            err_msg = f[2]
            error_counts[err_msg] = error_counts.get(err_msg, 0) + 1
        for msg, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"- {msg}: {count} \u6b21")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=int, default=10, help='\u5e76\u53d1\u6570')
    parser.add_argument("--n", type=int, default=20, help='\u603b\u8bf7\u6c42\u6570')
    args = parser.parse_args()

    asyncio.run(run_stress_test(args.c, args.n))

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
