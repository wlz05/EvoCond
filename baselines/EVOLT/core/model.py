# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Define compact memory records."""
from dataclasses import dataclass, field, fields
@dataclass(slots=True)
class Tool:
    key: str
    name: str
    text: str
    code: str
    vector: list[float]
    schema: str = ""
    aliases: list[str] = field(default_factory=list)
    quality: float = 0.0
    trials: int = 0
    created: float = 0.0
    used: float = 0.0
    recent: int = 0
    confidence: float = 1.0
    state: str = "active"
    def dump(self) -> dict:
        return {item.name: getattr(self, item.name) for item in fields(self)}
    @classmethod
    def load(cls, data: dict) -> "Tool":
        names = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in names})
@dataclass(slots=True)
class Edge:
    source: str
    target: str
    kind: str = "precedes"
    positive: float = 0.0
    negative: float = 0.0
    count: int = 0
    used: float = 0.0
    tasks: list[str] = field(default_factory=list)
    @property
    def weight(self) -> float:
        return (self.positive - self.negative) / (self.positive + self.negative + 2.0)
    @property
    def confidence(self) -> float:
        total = self.positive + self.negative
        return total / (total + 2.0)
    def add(self, value: float, stamp: float, task: str) -> bool:
        if task in self.tasks:
            return False
        self.tasks.append(task)
        if value >= 0.0:
            self.positive += value
        else:
            self.negative -= value
        self.count = len(self.tasks)
        self.used = max(self.used, stamp)
        return True
    def fade(self, factor: float, stamp: float) -> None:
        self.positive *= factor
        self.negative *= factor
        self.used = stamp
    def dump(self) -> dict:
        return {"source": self.source, "target": self.target, "kind": self.kind, "positive": self.positive, "negative": self.negative, "count": self.count, "used": self.used, "tasks": self.tasks}
    @classmethod
    def load(cls, data: dict) -> "Edge":
        if "positive" in data or "negative" in data:
            saved = data.get("tasks", [])
            tasks = list(saved) if isinstance(saved, list) else []
            return cls(data["source"], data["target"], data.get("kind", "precedes"), float(data.get("positive", 0.0)), float(data.get("negative", 0.0)), len(tasks), float(data.get("used", 0.0)), tasks)
        weight = float(data.get("weight", 0.0))
        return cls(data["source"], data["target"], data.get("kind", "precedes"), max(0.0, weight), max(0.0, -weight), 1, float(data.get("used", 0.0)), ["legacy"])
@dataclass(slots=True)
class Hit:
    key: str
    name: str
    text: str
    code: str
    score: float
    semantic: float
    structure: float
    spread: float
    sources: list[str] = field(default_factory=list)
    def dump(self) -> dict:
        return {item.name: getattr(self, item.name) for item in fields(self)}
@dataclass(slots=True)
class Trace:
    query: str
    hits: list[Hit]
    context: list[str]
    features: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    cold: bool = False
    stamp: float = 0.0
    subtasks: list[str] = field(default_factory=list)
    task: str = ""
    def dump(self) -> dict:
        return {"query": self.query, "hits": [hit.dump() for hit in self.hits], "context": self.context, "features": self.features, "weights": self.weights, "cold": self.cold, "stamp": self.stamp, "subtasks": self.subtasks, "task": self.task}
@dataclass(slots=True, frozen=True)
class Decision:
    mode: str
    label: str | None
    confidence: float
    reason: str
    def dump(self) -> dict:
        return {item.name: getattr(self, item.name) for item in fields(self)}
    @classmethod
    def load(cls, data: dict) -> "Decision":
        return cls(str(data["mode"]), data.get("label"), float(data["confidence"]), str(data["reason"]))
@dataclass(slots=True, frozen=True)
class Proposal:
    token: str
    decision: Decision
    key: str
    target: str | None
    name: str
    text: str
    code: str
    schema: str
    vector: list[float]
    stamp: float
    revision: int
    def dump(self) -> dict:
        return {"token": self.token, "decision": self.decision.dump(), "key": self.key, "target": self.target, "name": self.name, "text": self.text, "code": self.code, "schema": self.schema, "vector": self.vector, "stamp": self.stamp, "revision": self.revision}
    @classmethod
    def load(cls, data: dict) -> "Proposal":
        return cls(str(data["token"]), Decision.load(data["decision"]), str(data["key"]), data.get("target"), str(data["name"]), str(data["text"]), str(data.get("code", "")), str(data.get("schema", "")), list(data["vector"]), float(data["stamp"]), int(data["revision"]))

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
