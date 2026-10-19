# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import requests
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any

mcp = FastMCP("LocalServer")

@mcp.tool()
def distance(origin: str, destination: str) -> Dict[str, Any]:
    'Query distance information between two locations\n    \n    Args:\n        origin: Origin location\n        destination: Destination location\n        \n    Returns:\n        Dict: Dictionary containing distance information\n    '
    try:
        request_data = {
            "origin": origin,
            "destination": destination,
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
            "http://127.0.0.1:8006/distance",
            json=request_data,
            headers = headers,
            proxies = proxies,
            timeout=10
        )

        response.raise_for_status()
        return response.json()

    except Exception as e:
        return {"error": f"Distance query failed: {str(e)}"}

if __name__ == "__main__":
    mcp.run(transport='stdio')

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
