# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Coordinate retrieval and transactional memory."""
from hashlib import blake2b
from math import isfinite
from pathlib import Path
from core.clock import Clock
from core.embed import Embed, cosine, fold, unique, word
from core.graph import Graph
from core.index import Semantic, Structural
from core.model import Decision, Proposal, Tool, Trace
from core.novel import Novelty
from core.rank import Ranker
from core.store import migrate, read, write
from core.task import Decomposer
from learn.teach import Teacher
class Agent:
    """Expose staged and recoverable memory operations."""
    def __init__(self, dim: int = 128):
        self.embedder = Embed(dim)
        self.decomposer = Decomposer()
        self.clock = Clock()
        self.graph = Graph()
        self.novelty = Novelty(dim)
        self.teacher = Teacher()
        self.semantic = Semantic(dim)
        self.structural = Structural()
        self.pending: dict[str, Proposal] = {}
        self.journal: list[dict] = []
        self.ranker = Ranker(self.graph, self.embedder, self.teacher, self.semantic, self.structural)
    def vector(self, name: str, text: str, schema: str = "") -> list[float]:
        return self.embedder.vector(" ".join(value for value in (name, text, schema) if value))
    def record(self, token: str, status: str, stamp: float, mode: str) -> None:
        self.journal.append({"token": token, "status": status, "stamp": stamp, "mode": mode})
        if len(self.journal) > 256:
            self.journal.pop(0)
    def stage(self, name: str, text: str, code: str = "", schema: str = "", stamp: float | None = None) -> Proposal:
        cleanname = fold(name)
        cleantext = fold(text)
        cleanschema = fold(schema)
        if not cleanname or not cleantext:
            raise ValueError("name and text must not be empty")
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        now = self.clock.tick(stamp)
        vector = self.vector(cleanname, cleantext, cleanschema)
        scores = {key: cosine(vector, tool.vector) for key, tool in self.graph.tools.items()}
        decision = self.novelty.decide(vector, scores, cleanschema, self.graph.tools)
        target = decision.label
        key = target if decision.mode == "reuse" and target is not None else unique(word(cleanname), set(self.graph.tools))
        raw = "|".join((cleanname, cleantext, code, cleanschema, str(now), str(self.graph.revision)))
        token = blake2b(raw.encode("utf-8"), digest_size=12).hexdigest()
        proposal = Proposal(token, decision, key, target, cleanname, cleantext, code, cleanschema, vector, now, self.graph.revision)
        self.pending[token] = proposal
        self.record(token, "proposed", now, decision.mode)
        return proposal
    def rollback(self, token: str, reason: str = "rejected") -> Decision:
        proposal = self.pending.pop(token, None)
        if proposal is None:
            raise ValueError("proposal is not pending")
        now = self.clock.tick()
        self.record(token, "rolledback", now, proposal.decision.mode)
        return Decision("abstain", proposal.target, 1.0 - proposal.decision.confidence, reason)
    def resolve(self, proposal: Proposal, verified: bool, reward: float = 0.0, stamp: float | None = None) -> tuple[str | None, Decision]:
        if not isinstance(proposal, Proposal) or self.pending.get(proposal.token) != proposal:
            raise ValueError("proposal is not pending")
        if not isinstance(verified, bool):
            raise TypeError("verified must be a boolean")
        if not isfinite(reward):
            raise ValueError("reward must be finite")
        now = self.clock.tick(stamp)
        if proposal.revision != self.graph.revision:
            self.pending.pop(proposal.token)
            self.record(proposal.token, "stale", now, proposal.decision.mode)
            proposal = self.stage(proposal.name, proposal.text, proposal.code, proposal.schema)
        if not verified:
            return None, self.rollback(proposal.token, "execution not verified")
        if proposal.decision.mode == "abstain":
            return None, self.rollback(proposal.token, "policy abstained")
        if proposal.decision.mode == "reuse" and proposal.target in self.graph.tools:
            key = proposal.target
            tool = self.graph.tools[key]
            previous = tool.quality
            tool.trials += 1
            tool.quality += (reward - tool.quality) / tool.trials
            tool.confidence += ((1.0 if reward >= 0.0 else 0.0) - tool.confidence) / tool.trials
            tool.used = max(tool.used, now)
            tool.recent += 1
            alias = word(proposal.name)
            tool.aliases = list(dict.fromkeys(tool.aliases + [alias]))
            self.novelty.learn(key, proposal.vector, cosine(proposal.vector, tool.vector))
            tool.vector = self.novelty.clusters[key].mean()
            if reward >= previous:
                tool.text = proposal.text
                tool.code = proposal.code or tool.code
                tool.schema = proposal.schema or tool.schema
            self.semantic.put(key, tool.vector)
            self.graph.revision += 1
            decision = Decision("reuse", key, proposal.decision.confidence, proposal.decision.reason)
        else:
            key = unique(proposal.key, set(self.graph.tools))
            confidence = max(0.0, min(1.0, proposal.decision.confidence))
            tool = Tool(key, word(proposal.name), proposal.text, proposal.code, proposal.vector, proposal.schema, [word(proposal.name)], reward, 1, now, now, 1, confidence)
            self.graph.put(tool)
            self.semantic.put(key, proposal.vector)
            self.novelty.learn(key, proposal.vector)
            mode = "split" if proposal.decision.mode == "split" else "create"
            decision = Decision(mode, key, confidence, proposal.decision.reason)
        self.pending.pop(proposal.token)
        self.record(proposal.token, "committed", now, decision.mode)
        return key, decision
    def register(self, name: str, text: str, code: str = "", reward: float = 0.0, stamp: float | None = None, schema: str = "", verified: bool = True) -> tuple[str, bool]:
        if not isfinite(reward):
            raise ValueError("reward must be finite")
        proposal = self.stage(name, text, code, schema, stamp)
        key, decision = self.resolve(proposal, verified, reward, stamp)
        return key or "", decision.mode in ("create", "split")
    def recall(self, query: str, context: list[str] | None = None, limit: int = 5, subtasks: list[str] | None = None, task: str = "", fusion: str = "teacher") -> Trace:
        clean = fold(query)
        if not clean:
            raise ValueError("query must not be empty")
        if context is not None and not isinstance(context, list):
            raise TypeError("context must be a list")
        parts = self.decomposer.split(clean, subtasks)
        now = self.clock.tick()
        taskid = self.decomposer.key(task, None) if task else self.decomposer.key(clean, now)
        return self.ranker.search(clean, [word(key) for key in (context or [])], limit, parts, taskid, now, fusion)
    def commit(self, trace: Trace, chosen: list[str], reward: float, teach: bool = True, stamp: float | None = None) -> None:
        if not isinstance(trace, Trace):
            raise TypeError("trace must be valid")
        if not isinstance(chosen, list):
            raise TypeError("chosen must be a list")
        if not isfinite(reward):
            raise ValueError("reward must be finite")
        keys = list(dict.fromkeys(word(key) for key in chosen if word(key) in self.graph.tools))
        if not keys:
            return
        now = self.clock.tick(stamp)
        self.graph.link(keys, reward, now, trace.task)
        for key in keys:
            tool = self.graph.tools[key]
            tool.trials += 1
            tool.quality += (reward - tool.quality) / tool.trials
            tool.confidence += ((1.0 if reward >= 0.0 else 0.0) - tool.confidence) / tool.trials
        if teach:
            self.teacher.learn(trace, keys, reward)
    def prune(self, age: float, floor: float, stamp: float | None = None) -> list[str]:
        removed = self.graph.forget(age, floor, self.clock.tick(stamp))
        for key in removed:
            self.novelty.remove(key)
            self.semantic.drop(key)
        return removed
    def calibrate(self, view: str, scores: list[float], labels: list[int]) -> float:
        return self.ranker.fit(fold(view), scores, labels)
    def dump(self) -> dict:
        return {"version": 4, "dim": self.embedder.dim, "clock": self.clock.dump(), "graph": self.graph.dump(), "semantic": self.semantic.dump(), "structural": self.structural.dump(), "novelty": self.novelty.dump(), "teacher": self.teacher.dump(), "ranker": self.ranker.dump(), "pending": [item.dump() for item in self.pending.values()], "journal": self.journal}
    def clone(self) -> "Agent":
        return self.restore(self.dump())
    @classmethod
    def restore(cls, data: dict) -> "Agent":
        state = migrate(data)
        agent = cls(int(state["dim"]))
        agent.clock = Clock.load(state.get("clock"))
        agent.graph = Graph.load(state["graph"])
        agent.novelty = Novelty.load(state["novelty"])
        for label in list(agent.novelty.clusters):
            if label not in agent.graph.tools:
                agent.novelty.remove(label)
        agent.semantic = Semantic.load(state.get("semantic", {}), agent.embedder.dim)
        for key in list(agent.semantic.vectors):
            if key not in agent.graph.tools:
                agent.semantic.drop(key)
        for tool in agent.graph.tools.values():
            tool.name = word(tool.name)
            tool.text = fold(tool.text)
            tool.schema = fold(tool.schema)
            tool.aliases = list(dict.fromkeys(word(item) for item in tool.aliases))
            tool.vector = agent.vector(tool.name, tool.text, tool.schema)
            agent.semantic.put(tool.key, tool.vector)
            if tool.key not in agent.novelty.clusters:
                agent.novelty.learn(tool.key, tool.vector)
            agent.clock.observe(max(tool.created, tool.used))
        for edge in agent.graph.edges.values():
            agent.clock.observe(edge.used)
        agent.structural = Structural.load(state.get("structural", {}), agent.graph)
        agent.teacher = Teacher.load(state.get("teacher", {}))
        agent.pending = {}
        for item in state.get("pending", []):
            proposal = Proposal.load(item)
            valid = bool(proposal.token) and proposal.key == word(proposal.key) and (proposal.target is None or proposal.target == word(proposal.target))
            valid = valid and proposal.decision.mode in ("reuse", "split", "create", "abstain") and isfinite(proposal.decision.confidence) and 0.0 <= proposal.decision.confidence <= 1.0
            valid = valid and isfinite(proposal.stamp) and proposal.stamp >= 0.0 and proposal.revision >= 0 and len(proposal.vector) == agent.embedder.dim and all(isfinite(value) for value in proposal.vector)
            if valid:
                agent.pending[proposal.token] = proposal
                agent.clock.observe(proposal.stamp)
        agent.journal = [dict(item) for item in state.get("journal", []) if isinstance(item, dict) and isinstance(item.get("stamp"), (int, float)) and not isinstance(item.get("stamp"), bool) and isfinite(item["stamp"])][-256:]
        agent.ranker = Ranker(agent.graph, agent.embedder, agent.teacher, agent.semantic, agent.structural)
        agent.ranker.restore(state.get("ranker", {}))
        return agent
    def save(self, path: str | Path) -> None:
        write(path, self.dump())
    @classmethod
    def open(cls, path: str | Path) -> "Agent":
        return cls.restore(read(path))

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
