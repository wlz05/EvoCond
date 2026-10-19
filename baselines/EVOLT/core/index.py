# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Store independent retrieval views."""
from math import isfinite
from core.embed import cosine, word
from core.graph import Graph
class Semantic:
    """Index normalized semantic vectors."""
    def __init__(self, dim: int):
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise ValueError("index dim must be positive")
        self.dim = dim
        self.vectors: dict[str, list[float]] = {}
        self.version = 0
    def put(self, key: str, vector: list[float]) -> None:
        key = word(key)
        if len(vector) != self.dim or any(not isfinite(value) for value in vector):
            raise ValueError("semantic vector is invalid")
        self.vectors[key] = list(vector)
        self.version += 1
    def drop(self, key: str) -> None:
        if self.vectors.pop(word(key), None) is not None:
            self.version += 1
    def rebuild(self, graph: Graph) -> None:
        self.vectors = {key: list(tool.vector) for key, tool in graph.tools.items() if len(tool.vector) == self.dim and all(isfinite(value) for value in tool.vector)}
        self.version += 1
    def search(self, vector: list[float], limit: int | None = None) -> dict[str, float]:
        if len(vector) != self.dim or any(not isfinite(value) for value in vector):
            raise ValueError("query vector is invalid")
        scores = {key: cosine(vector, item) for key, item in self.vectors.items()}
        if limit is None:
            return scores
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("search limit is invalid")
        return {key: scores[key] for key in sorted(scores, key=scores.get, reverse=True)[:limit]}
    def dump(self) -> dict:
        return {"dim": self.dim, "version": self.version, "vectors": self.vectors}
    @classmethod
    def load(cls, data: dict, dim: int) -> "Semantic":
        index = cls(dim)
        vectors = data.get("vectors", {}) if isinstance(data, dict) else {}
        if isinstance(vectors, dict):
            for key, vector in vectors.items():
                if isinstance(vector, list) and len(vector) == dim and all(isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value) for value in vector):
                    index.put(key, vector)
        saved = data.get("version", index.version) if isinstance(data, dict) else index.version
        index.version = max(index.version, int(saved))
        return index
class Structural:
    """Index typed graph role signatures."""
    def __init__(self):
        self.vectors: dict[str, list[float]] = {}
        self.graphrev = -1
        self.version = 0
    def refresh(self, graph: Graph) -> None:
        if self.graphrev == graph.revision and set(self.vectors) == set(graph.tools):
            return
        self.vectors = {key: graph.shape(key) for key in graph.tools}
        self.graphrev = graph.revision
        self.version += 1
    def search(self, anchors: list[str], graph: Graph, limit: int | None = None) -> dict[str, float]:
        self.refresh(graph)
        keys = list(dict.fromkeys(word(key) for key in anchors if word(key) in self.vectors))
        shapes = [self.vectors[key] for key in keys]
        query = [sum(values) / len(values) for values in zip(*shapes)] if shapes else []
        scores = {key: cosine(query, vector) if query and any(query) else 0.0 for key, vector in self.vectors.items()}
        if limit is None:
            return scores
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("search limit is invalid")
        return {key: scores[key] for key in sorted(scores, key=scores.get, reverse=True)[:limit]}
    def dump(self) -> dict:
        return {"version": self.version, "graphrev": self.graphrev, "vectors": self.vectors}
    @classmethod
    def load(cls, data: dict, graph: Graph) -> "Structural":
        index = cls()
        vectors = data.get("vectors", {}) if isinstance(data, dict) else {}
        clean = isinstance(vectors, dict) and set(map(word, vectors)) == set(graph.tools)
        listed = clean and all(isinstance(value, list) for value in vectors.values())
        lengths = {len(value) for value in vectors.values()} if listed else set()
        finite = listed and all(all(isinstance(item, (int, float)) and not isinstance(item, bool) and isfinite(item) for item in value) for value in vectors.values())
        expected = len(graph.shape(next(iter(graph.tools)))) if graph.tools else 0
        shaped = not graph.tools and not vectors or lengths == {expected}
        if clean and finite and shaped:
            index.vectors = {word(key): list(value) for key, value in vectors.items()}
            index.graphrev = int(data.get("graphrev", -1))
            index.version = max(0, int(data.get("version", 0)))
        index.refresh(graph)
        return index
class Aligned:
    """Keep both indexes keyed by the same tools."""
    @staticmethod
    def sync(graph: Graph, semantic: Semantic, structural: Structural) -> float:
        keys = set(graph.tools)
        if set(semantic.vectors) != keys:
            semantic.rebuild(graph)
        structural.refresh(graph)
        paired = keys & set(semantic.vectors) & set(structural.vectors)
        return len(paired) / max(1, len(keys))
    @staticmethod
    def check(graph: Graph, semantic: Semantic, structural: Structural) -> bool:
        keys = set(graph.tools)
        return keys == set(semantic.vectors) == set(structural.vectors)
    @staticmethod
    def pairs(graph: Graph, semantic: Semantic, structural: Structural) -> dict[str, tuple[list[float], list[float]]]:
        Aligned.sync(graph, semantic, structural)
        return {key: (list(semantic.vectors[key]), list(structural.vectors[key])) for key in graph.tools if key in semantic.vectors and key in structural.vectors}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
