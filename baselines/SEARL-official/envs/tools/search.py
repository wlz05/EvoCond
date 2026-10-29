# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import json
import uuid
import redis
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, LOCAL_SEARCH_HOST
from mcp.server.fastmcp import FastMCP
import socket

REQUEST_QUEUE_NAME = 'web_search_request_queue'

RESPONSE_QUEUE_PREFIX = 'web_search_response_queue:'

RESULT_TIMEOUT = 180

_redis_pool = None

def get_redis_client():
    'Get the Redis client through the shared connection pool.'
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = redis.ConnectionPool(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            max_connections=10,
            socket_keepalive=True,
            socket_keepalive_options={
                socket.TCP_KEEPIDLE: 60,
                socket.TCP_KEEPINTVL: 10,
                socket.TCP_KEEPCNT: 3,
            }
        )
    return redis.Redis(connection_pool=_redis_pool)

mcp = FastMCP("LocalServer")

@mcp.tool()
def search(query: str, topk: int = 10):
    'Search Tool for Google search\n\n    Args:\n        query: query text\n\n    Returns:\n        str: The formatted query result\n    '
    try:

        redis_client = get_redis_client()

        task_id = str(uuid.uuid4())
        response_queue = f"{RESPONSE_QUEUE_PREFIX}{task_id}"

        task_payload = {
            'task_id': task_id,
            'query': query,
            'topk': topk,
            'return_scores': True,
            'response_queue': response_queue
        }

        redis_client.lpush(REQUEST_QUEUE_NAME, json.dumps(task_payload))
        print(f"Task {task_id} submitted: query \'{query}'")

        result_data = redis_client.brpop(response_queue, timeout=RESULT_TIMEOUT)

        if result_data is None:
            return f"Search timed out after {RESULT_TIMEOUT} seconds)"

        response_payload = json.loads(result_data[1])
        print(f"Completed task {response_queue}")

        flag = response_payload.get('flag', 'error')
        content = response_payload.get('content', 'Invalid response format')

        if flag == 'error':
            return f"Search failed: {content}"
        else:
            return content

    except Exception as e:
        return f"Search query failed: {str(e)}"

if __name__ == "__main__":
    print("\nStart MCP service:")
    mcp.run(transport='stdio')

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
