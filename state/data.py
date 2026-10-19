# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import json
from typing import Any, Mapping, TypeVar
Item = TypeVar("Item")
FIELDS = {"dataset", "datasetname", "benchmark", "benchmarkname"}
def reject(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number: {value}")
def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result
def decode(text: str) -> Any:
    return json.loads(text, parse_constant=reject, object_pairs_hook=unique)
# Canonicalize dataset labels before lookup.
def lower(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("Dataset name must be a string")
    result = value.strip().lower()
    if not result:
        raise ValueError("Dataset name must not be empty")
    return result
def isfield(value: Any) -> bool:
    token = "".join(character for character in str(value).strip().lower() if character.isalnum())
    return token in FIELDS
def canonical(value: Any) -> str:
    return "dataset" if isfield(value) else str(value).strip()
def normalize(values: Mapping[Any, Any]) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError("State values must be a mapping")
    result: dict[str, Any] = {}
    for rawkey, rawvalue in values.items():
        key = canonical(rawkey)
        if not key:
            raise ValueError("State keys must not be empty")
        if key == "dataset":
            value = lower(rawvalue)
        else:
            value = walk(rawvalue)
        if key in result and result[key] != value:
            raise ValueError(f"Conflicting state field: {key}")
        result[key] = value
    return result
def walk(value: Any) -> Any:
    if isinstance(value, Mapping):
        return normalize(value)
    if isinstance(value, list):
        return [walk(item) for item in value]
    if isinstance(value, tuple):
        return tuple(walk(item) for item in value)
    return value
def identify(value: Any, choices: Mapping[str, Item]) -> Item | None:
    target = lower(value)
    index: dict[str, Item] = {}
    for name, item in choices.items():
        key = lower(name)
        if key in index and index[key] != item:
            raise ValueError(f"Conflicting dataset alias: {key}")
        index[key] = item
    return index.get(target)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
