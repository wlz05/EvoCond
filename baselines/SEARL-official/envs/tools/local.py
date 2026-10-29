# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Bounded requests to an explicitly configured retrieval service."""
from urllib.parse import urlsplit
import json
import requests
from requests.adapters import HTTPAdapter
from mcp.server.fastmcp import FastMCP
from config import LOCAL_SEARCH_URL

mcp = FastMCP("LocalServer")
session = requests.Session()
session.trust_env = False
session.mount("http://", HTTPAdapter(pool_connections=32, pool_maxsize=32))
session.mount("https://", HTTPAdapter(pool_connections=32, pool_maxsize=32))
MAX_BYTES = 1048576

def endpoint(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path != "/retrieve"):
        raise ValueError("Set SEARL_SEARCH_URL to an explicit HTTP(S) /retrieve endpoint")
    return value

@mcp.tool()
def search(query: str, topk: int = 3):
    """Retrieve up to ten relevant passages from the configured wiki service."""
    if not isinstance(query, str) or not query.strip() or len(query) > 16000:
        return "Search error: query must contain 1-16000 characters"
    if type(topk) is not int or not 1 <= topk <= 10:
        return "Search error: topk must be an integer between 1 and 10"
    try:
        target = endpoint(LOCAL_SEARCH_URL)
        with session.post(target, json={"queries": [query], "topk": topk, "return_scores": True},
                          timeout=(5, 10), stream=True, allow_redirects=False) as response:
            if response.status_code != 200:
                return f"Search error: retrieval HTTP status {response.status_code}"
            data = bytearray()
            for chunk in response.iter_content(16384):
                data.extend(chunk)
                if len(data) > MAX_BYTES:
                    return "Search error: response exceeded size limit"
            result = json.loads(data)
        if not isinstance(result, dict):
            raise ValueError("Retrieval response must be a JSON object")
        batches = result.get("result")
        if not batches:
            return "No relevant documents found"
        if not isinstance(batches, list) or not isinstance(batches[0], list):
            raise ValueError("Invalid retrieval result schema")
        output = "Following are the search results given by the wiki search engine:"
        for item in batches[0][:topk]:
            text = item["document"]["contents"]
            if not isinstance(text, str):
                raise ValueError("Document contents must be text")
            title, _, content = text.partition("\n")
            output += '\nsearch results: "' + title.strip('"') + '"\n' + content
        return output
    except (requests.RequestException, ValueError, KeyError, TypeError, RecursionError) as exc:
        return f"Search error: {type(exc).__name__}; verify endpoint and service logs"

if __name__ == "__main__":
    endpoint(LOCAL_SEARCH_URL)
    mcp.run(transport="stdio")

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
