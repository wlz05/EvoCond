# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Strictly recognize a successful isolated MCP creation response.\n\nUnknown wrappers, free-form success text, and partial successes fail closed.\n'
from __future__ import annotations

import json

def verified_creation_result(value):
    for _ in range(5):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError):
                return False
            continue
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
            continue
        if isinstance(value, dict):
            if "creation_success" in value or "run_success" in value:
                return value.get("creation_success") is True and value.get("run_success") is True
            if "content" in value:
                value = value["content"]
                continue
            if value.get("type") == "text" and isinstance(value.get("text"), str):
                value = value["text"]
                continue
        return False
    return False

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
