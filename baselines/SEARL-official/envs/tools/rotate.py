# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from PIL import Image
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.tools.base import Tool
from io import BytesIO
import base64
import binascii
import inspect # For inspecting function signature in decorator
from typing import Any, Callable, List # For type hints

mcp = FastMCP("LocalServer")

@mcp.tool()
def rotate(degree: int, img_base64: str) -> str: # Rotate tool for MM tool use demo
    'Rotate the image by a specified angle.\n\n    Args:\n        degree (int): The angle of rotation (positive for clockwise, negative for counter-clockwise).\n\n    Returns:\n        str: The Base64 encoded string of the rotated image.\n    '
    print(f"================= Calling image_rotate tool ==================")
    print(f"Received parameters: degree={degree}, img_base64 length={len(img_base64) if img_base64 else 0}")

    if img_base64 is None:
        return "Error: Missing image data"

    if degree is None:
        return "Error: Missing degree parameter"

    try:
        img_data = base64.b64decode(img_base64)
        img = Image.open(BytesIO(img_data))

        rotated_img = img.rotate(degree, expand=True)

        buffer = BytesIO()
        rotated_img.save(buffer, format='PNG')
        img_base64_output = base64.b64encode(buffer.getvalue()).decode('utf-8')
        print(f" {'*'*50}\n\n\n\n\n use image success {'*'*50}\n\n\n")
        return img_base64_output

    except binascii.Error:
        return "Error: Invalid base64 image data"
    except Exception as e:
        return f"Image rotation failed: {str(e)}"

if __name__ == "__main__":
    print('\n\u542f\u52a8 MCP \u56fe\u50cf\u65cb\u8f6c\u670d\u52a1...')
    mcp.run(transport='stdio')

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
