# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import requests
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any

mcp = FastMCP("LocalServer")

@mcp.tool()
def flight(origin: str, destination: str, date: str) -> Dict[str, Any]:
    'Query flight information\n    \n    Args:\n        origin: Origin city\n        destination: Destination city\n        date: Departure date (YYYY-MM-DD format)\n        \n    Returns:\n        Dict: Dictionary containing flight information\n    '
    try:
        request_data = {
            "origin": origin,
            "destination": destination,
            "date": date
        }

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
            "http://127.0.0.1:8007/flights",
            json=request_data,
            headers=headers,
            proxies=proxies,
            timeout=10
        )

        response.raise_for_status()
        return response.json()

    except Exception as e:
        return {"error": f"Flight query failed: {str(e)}"}

if __name__ == "__main__":
    mcp.run(transport='stdio')

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
