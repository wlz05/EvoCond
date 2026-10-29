# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import requests
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any

mcp = FastMCP("LocalServer")

@mcp.tool()
def attraction(city: str) -> Dict[str, Any]:
    'Query attraction information\n    \n    Args:\n        city: City name\n        \n    Returns:\n        Dict: Dictionary containing attraction information\n    '
    try:
        request_data = {"city": city}

        # Set request headers and proxy
        headers = {
            "Content-Type": "application/json"
        }

        # Use local connection, bypass proxy
        proxies = {
            "http": None,
            "https": None
        }

        response = requests.post(
            "http://127.0.0.1:8004/attractions",
            json=request_data,
            headers=headers,
            proxies=proxies,
            timeout=10
        )

        response.raise_for_status()
        return response.json()

    except Exception as e:
        return {"error": f"Attraction query failed: {str(e)}"}

if __name__ == "__main__":
    mcp.run(transport='stdio')

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
