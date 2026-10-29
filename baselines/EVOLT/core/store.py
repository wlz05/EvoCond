# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Persist checked versioned state atomically."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
VERSION = 4
def migrate(data: dict) -> dict:
    if not isinstance(data, dict):
        raise TypeError("state must be a dictionary")
    state = deepcopy(data)
    saved = state.get("version", 1)
    if not isinstance(saved, int) or isinstance(saved, bool):
        raise ValueError("state version is invalid")
    version = saved
    if version < 1 or version > VERSION:
        raise ValueError("state version is unsupported")
    if version == 1:
        state["version"] = 2
        version = 2
    if version == 2:
        graph = state.get("graph", {})
        stamps = [float(item.get("used", 0.0)) for item in graph.get("tools", []) + graph.get("edges", [])]
        state["clock"] = {"value": max(stamps, default=0.0)}
        state["pending"] = []
        state["journal"] = []
        state["semantic"] = {}
        state["structural"] = {}
        state["version"] = 3
        version = 3
    if version == 3:
        graph = state.get("graph", {})
        tasks = []
        for pos, edge in enumerate(graph.get("edges", [])):
            count = max(0, int(edge.get("count", 0)))
            if not count and float(edge.get("positive", 0.0)) + float(edge.get("negative", 0.0)) > 0.0:
                count = 1
            if count:
                edge["positive"] = float(edge.get("positive", 0.0)) / count
                edge["negative"] = float(edge.get("negative", 0.0)) / count
                edge["tasks"] = [f"legacy{pos}"]
                edge["count"] = 1
                tasks.append(f"legacy{pos}")
            else:
                edge["tasks"] = []
        graph["tasks"] = tasks
        state["version"] = 4
    return state
def encode(data: dict) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
def write(path: str | Path, data: dict) -> None:
    target = Path(path)
    if target.exists() and target.is_dir():
        raise IsADirectoryError(str(target))
    state = migrate(data)
    raw = encode(state)
    envelope = {"checksum": sha256(raw.encode("utf-8")).hexdigest(), "state": state}
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False)
    try:
        # Replace only verified state files.
        with handle:
            json.dump(envelope, handle, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
def read(path: str | Path) -> dict:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(str(target))
    with target.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("state root must be a dictionary")
    if "state" not in data:
        if isinstance(data.get("version", 1), int) and data.get("version", 1) >= VERSION:
            raise ValueError("current state requires an envelope")
        return migrate(data)
    state = data.get("state")
    checksum = data.get("checksum")
    if not isinstance(state, dict) or not isinstance(checksum, str):
        raise ValueError("state envelope is invalid")
    actual = sha256(encode(state).encode("utf-8")).hexdigest()
    if actual != checksum:
        raise ValueError("state checksum does not match")
    return migrate(state)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
