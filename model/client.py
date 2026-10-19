# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import json
import re
from math import isfinite
from typing import Any, Callable, Mapping, Protocol, TypeVar
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen
from state.data import canonical, decode, reject, unique
Result = TypeVar("Result")
class Model(Protocol):
    def generate(self, prompt: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        ...
def contractual(raw: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(raw)
    predictions = data.get("predictions")
    if not isinstance(predictions, Mapping):
        return data
    keys = {canonical(key) for key in predictions}
    rawprogress = data.get("progress", "")
    progress = "" if rawprogress is None or isinstance(rawprogress, Mapping) else canonical(rawprogress)
    rawinvariants = data.get("invariants", ())
    if rawinvariants is None:
        invariants: tuple[Any, ...] = ()
    elif isinstance(rawinvariants, str):
        invariants = (rawinvariants,)
    elif isinstance(rawinvariants, Mapping):
        invariants = tuple(rawinvariants)
    elif isinstance(rawinvariants, (list, tuple)):
        invariants = tuple(rawinvariants)
    else:
        return data
    data["invariants"] = list(dict.fromkeys(key for item in invariants if (key := canonical(item)) in keys and key != progress))
    return data
def validated(model: Model, prompt: str, payload: Mapping[str, Any], validator: Callable[[dict[str, Any]], Result]) -> Result:
    request = dict(payload)
    for attempt in range(2):
        raw = model.generate(prompt, request)
        if not isinstance(raw, Mapping):
            raise TypeError("Model response must be a mapping")
        data = dict(raw)
        try:
            return validator(data)
        except (TypeError, ValueError) as error:
            if attempt:
                response = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
                if len(response) > 2048:
                    response = response[:2048] + "..."
                raise ValueError(f"Model contract remained invalid after one repair: {error}; response={response}") from error
            request = dict(payload)
            request["repair"] = {"error": str(error), "previous": data, "instruction": "Return one corrected JSON object that satisfies the system schema"}
    raise RuntimeError("Model validation retry was exhausted")
def load(text: str) -> Any:
    return decode(text)
# Extract the first complete JSON object.
def parse(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder(parse_constant=reject, object_pairs_hook=unique)
    for start in (match.start() for match in re.finditer(r"\{", text)):
        try:
            data, unused = decoder.raw_decode(text[start:])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    raise ValueError("Model response does not contain a valid JSON object")
class Http:
    def __init__(self, url: str, model: str, key: str, timeout: float = 120.0):
        parts = urlsplit(url.strip())
        if parts.scheme not in {"http", "https"} or not parts.netloc or parts.fragment:
            raise ValueError("Model URL must be an HTTP endpoint")
        path = parts.path.rstrip("/")
        if not path.endswith("/chat/completions"):
            path += "/chat/completions"
        self.url = urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))
        self.model = model.strip()
        if not self.model:
            raise ValueError("Model name must not be empty")
        self.key = key
        self.timeout = float(timeout)
        if not isfinite(self.timeout) or self.timeout <= 0.0:
            raise ValueError("Model timeout must be finite and positive")
    def generate(self, prompt: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = {"model": self.model, "temperature": 0, "response_format": {"type": "json_object"}, "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(payload, ensure_ascii=True, separators=(",", ":"))}]}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        request = Request(self.url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            raw = response.read(4194305)
        if len(raw) > 4194304:
            raise ValueError("Model response is too large")
        result = load(raw.decode("utf-8"))
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("Model response has an invalid shape") from error
        if not isinstance(content, str):
            raise ValueError("Model content must be a string")
        return parse(content)
class Script:
    def __init__(self, handler: Callable[[str, Mapping[str, Any]], dict[str, Any]]):
        self.handler = handler
    def generate(self, prompt: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.handler(prompt, payload)
# Provide deterministic offline behavior.
class Rule:
    def point(self, value: Any, audit: Mapping[str, Any]) -> dict[str, Any]:
        # Apply the audited width.
        width = float(audit.get("width", 0.0) or 0.0)
        number = float(value)
        return {"kind": "interval", "lower": number - width, "upper": number + width}
    def step(self, environment: Mapping[str, Any]) -> Any:
        try:
            raw = environment.get("progress", 0)
            number = 0.0 if isinstance(raw, bool) else float(raw)
        except (TypeError, ValueError):
            number = 0.0
        number = number if isfinite(number) else 0.0
        return int(number + 1) if number.is_integer() else number + 1.0
    def generate(self, prompt: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        kind = str(payload.get("kind", "")).strip().lower()
        environment = dict(payload.get("environment", {}))
        audit = dict(payload.get("audit") or {})
        forbid = {str(item).strip().lower() for item in audit.get("forbid", [])}
        if kind == "gg":
            if "policy" in forbid:
                return {"assumptions": ["The task remains active"], "guarantees": ["The goal remains feasible"], "predictions": {"progress": self.point(self.step(environment), audit)}, "due": 6, "progress": "progress"}
            return {"assumptions": ["The task remains active"], "guarantees": ["The policy remains valid"], "predictions": {"policy": {"kind": "categorical", "shares": {"stable": 1.0}}}, "due": 6, "invariants": ["policy"]}
        if kind == "gl":
            predictions: dict[str, Any] = {}
            if "progress" in forbid:
                predictions["milestone"] = {"kind": "categorical", "shares": {"ready": 1.0}}
                progress = "milestone"
            else:
                predictions["progress"] = self.point(self.step(environment), audit)
                progress = "progress"
            if "intent" not in forbid:
                # Mark internal state as unobservable.
                predictions["intent"] = {"kind": "categorical", "shares": {"advance": 1.0}, "verifier": "none"}
            return {"assumptions": ["The global contract is current"], "guarantees": ["The next milestone is reachable"], "predictions": predictions, "due": 2, "plan": ["advance"], "progress": progress}
        if kind != "ll":
            raise ValueError("Unknown reasoning scale")
        return {"action": "advance", "arguments": {}, "assumptions": ["The action is available"], "guarantees": ["Progress increases"], "predictions": {"progress": self.point(self.step(environment), audit)}, "due": 1, "progress": "progress", "wait": False}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
