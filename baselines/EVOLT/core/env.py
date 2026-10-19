# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Provide a cloneable stateful tool sandbox."""
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
from core.embed import fold, word
TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
def stable(value) -> bool:
    try:
        json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False)
        return True
    except (TypeError, ValueError):
        return False
def reference(value):
    if not isinstance(value, str):
        return deepcopy(value)
    lower = value.casefold()
    for prefix in ("$arg:", "$state:"):
        if lower.startswith(prefix):
            return prefix + fold(value[len(prefix):])
    return value
@dataclass(slots=True)
class Spec:
    name: str
    inputs: dict[str, str] = field(default_factory=dict)
    needs: dict = field(default_factory=dict)
    writes: dict = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    active: bool = True
    version: int = 1
    def key(self) -> str:
        return word(self.name)
    def schema(self) -> str:
        return fold(json.dumps(self.inputs, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    def dump(self) -> dict:
        return {"name": self.name, "inputs": self.inputs, "needs": self.needs, "writes": self.writes, "aliases": self.aliases, "active": self.active, "version": self.version}
    @classmethod
    def load(cls, data: dict) -> "Spec":
        return cls(str(data["name"]), dict(data.get("inputs", {})), dict(data.get("needs", {})), dict(data.get("writes", {})), list(data.get("aliases", [])), bool(data.get("active", True)), int(data.get("version", 1)))
class World:
    """Execute deterministic tools against copied state."""
    def __init__(self, state: dict | None = None, tools: list[Spec] | None = None):
        if state is not None and not isinstance(state, dict) or tools is not None and not isinstance(tools, list):
            raise TypeError("world inputs are invalid")
        if not stable(state or {}):
            raise ValueError("world state is invalid")
        self.state = {fold(str(key)): deepcopy(value) for key, value in (state or {}).items()}
        self.tools: dict[str, Spec] = {}
        self.aliases: dict[str, str] = {}
        self.history: list[dict] = []
        for tool in tools or []:
            self.put(tool)
    def put(self, tool: Spec) -> None:
        if not isinstance(tool, Spec) or not isinstance(tool.version, int) or isinstance(tool.version, bool) or tool.version < 1 or not isinstance(tool.active, bool):
            raise ValueError("tool spec is invalid")
        if not isinstance(tool.inputs, dict) or not isinstance(tool.needs, dict) or not isinstance(tool.writes, dict) or not isinstance(tool.aliases, list):
            raise ValueError("tool fields are invalid")
        if not stable(tool.needs) or not stable(tool.writes):
            raise ValueError("tool state is invalid")
        tool = deepcopy(tool)
        tool.name = fold(tool.name)
        tool.inputs = {fold(str(key)): fold(kind) for key, kind in tool.inputs.items()}
        tool.needs = {fold(str(key)): value for key, value in tool.needs.items()}
        tool.writes = {fold(str(key)): reference(value) for key, value in tool.writes.items()}
        tool.aliases = [fold(alias) for alias in tool.aliases]
        if any(kind not in TYPES for kind in tool.inputs.values()):
            raise ValueError("input type is invalid")
        key = tool.key()
        names = {key, *(word(alias) for alias in tool.aliases)}
        if key in self.tools or any(name in self.aliases and self.aliases[name] != key for name in names):
            raise ValueError("tool name collides")
        if any(isinstance(value, str) and value.startswith("$arg:") and value[5:] not in tool.inputs for value in tool.writes.values()):
            raise ValueError("write argument is invalid")
        self.tools[key] = tool
        self.aliases[key] = key
        for alias in tool.aliases:
            self.aliases[word(alias)] = key
    def shift(self, key: str, name: str | None = None, inputs: dict[str, str] | None = None, active: bool | None = None) -> str:
        key = self.aliases.get(word(key), word(key))
        if key not in self.tools:
            raise KeyError(key)
        savedtools = deepcopy(self.tools)
        savedaliases = dict(self.aliases)
        tool = deepcopy(self.tools[key])
        if active is not None and not isinstance(active, bool):
            raise TypeError("tool activity must be boolean")
        previous = tool.name
        tool.name = fold(name) if name is not None else tool.name
        if name is not None:
            tool.aliases = list(dict.fromkeys(tool.aliases + [previous]))
        tool.inputs = dict(inputs) if inputs is not None else tool.inputs
        tool.active = active if active is not None else tool.active
        tool.version += 1
        try:
            self.tools.pop(key)
            self.aliases = {alias: target for alias, target in self.aliases.items() if target != key}
            self.put(tool)
        except BaseException:
            self.tools = savedtools
            self.aliases = savedaliases
            raise
        return tool.key()
    def clone(self) -> "World":
        return World.load(self.dump())
    def resolve(self, value, args: dict):
        if isinstance(value, str) and value.startswith("$arg:"):
            return deepcopy(args[value[5:]])
        if isinstance(value, str) and value.startswith("$state:"):
            return deepcopy(self.state.get(value[7:]))
        return deepcopy(value)
    def execute(self, name: str, args: dict | None = None) -> dict:
        if args is not None and not isinstance(args, dict):
            raise TypeError("tool arguments must be a dictionary")
        if not stable(args or {}):
            raise ValueError("tool arguments are invalid")
        args = {fold(str(key)): value for key, value in (args or {}).items()}
        alias = word(name)
        key = self.aliases.get(alias)
        if key is None or key not in self.tools:
            result = {"ok": False, "error": "missing tool", "tool": alias}
            self.history.append(result)
            return result
        tool = self.tools[key]
        if not tool.active:
            result = {"ok": False, "error": "inactive tool", "tool": key}
            self.history.append(result)
            return result
        if any(field not in tool.inputs for field in args):
            result = {"ok": False, "error": "schema mismatch", "tool": key, "field": "extra"}
            self.history.append(result)
            return result
        for field, kind in tool.inputs.items():
            expected = TYPES[kind]
            value = args.get(field)
            valid = field in args and isinstance(value, expected) and not (kind in ("integer", "number") and isinstance(value, bool))
            valid = valid and not (kind in ("integer", "number") and not isfinite(value))
            if not valid:
                result = {"ok": False, "error": "schema mismatch", "tool": key, "field": field}
                self.history.append(result)
                return result
        if any(self.state.get(field) != value for field, value in tool.needs.items()):
            result = {"ok": False, "error": "state mismatch", "tool": key}
            self.history.append(result)
            return result
        for field, value in tool.writes.items():
            self.state[fold(field)] = self.resolve(value, args)
        result = {"ok": True, "tool": key, "version": tool.version}
        self.history.append(result)
        return result
    def verify(self, goal: dict) -> bool:
        if not isinstance(goal, dict):
            raise TypeError("world goal must be a dictionary")
        if not stable(goal):
            raise ValueError("world goal is invalid")
        return all(self.state.get(fold(key)) == value for key, value in goal.items())
    def fingerprint(self) -> str:
        raw = json.dumps(self.state, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(raw.encode("utf-8")).hexdigest()
    def dump(self) -> dict:
        return {"state": self.state, "tools": [tool.dump() for tool in self.tools.values()], "history": self.history}
    @classmethod
    def load(cls, data: dict) -> "World":
        world = cls(dict(data.get("state", {})), [Spec.load(item) for item in data.get("tools", [])])
        world.history = [dict(item) for item in data.get("history", [])]
        return world

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
