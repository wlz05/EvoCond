# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Bounded requests to an explicitly configured retrieval service."""
from control.search import SearchResponseError, endpoint, fetch, session
import requests
from mcp.server.fastmcp import FastMCP
if __package__:
    from envs.tools.config import LOCAL_SEARCH_URL
else:
    from config import LOCAL_SEARCH_URL

mcp = FastMCP("LocalServer")
client = session()

@mcp.tool()
def search(query: str, topk: int = 3):
    """Retrieve up to ten relevant passages from the configured wiki service."""
    if not isinstance(query, str) or not query.strip() or len(query) > 16000:
        return "Search error: query must contain 1-16000 characters"
    if type(topk) is not int or not 1 <= topk <= 10:
        return "Search error: topk must be an integer between 1 and 10"
    try:
        return witnessed_search(query, topk)[0]
    except SearchResponseError as exc:
        return f"Search error: {exc}"
    except (requests.RequestException, ValueError, KeyError, TypeError, RecursionError) as exc:
        return f"Search error: {type(exc).__name__}; verify endpoint and service logs"

def witnessed_search(query, topk):
    return fetch(client, LOCAL_SEARCH_URL, query, topk)

if __name__ == "__main__":
    endpoint(LOCAL_SEARCH_URL)
    mcp.run(transport="stdio")

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
