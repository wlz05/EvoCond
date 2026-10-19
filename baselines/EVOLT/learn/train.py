# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Train from paired outcome branches."""
from dataclasses import dataclass, field
from math import exp, isfinite
from core.agent import Agent
from core.model import Trace
from learn.credit import Episode, Step, credit, turn, utility
@dataclass(slots=True)
class Draft:
    name: str
    text: str
    code: str = ""
    schema: str = ""
@dataclass(slots=True)
class Rollout:
    steps: list[Step]
    outcome: float
    chosen: list[str] = field(default_factory=list)
    drafts: list[Draft] = field(default_factory=list)
    trace: Trace | None = None
    cost: float = 0.0
    errors: float = 0.0
    duplicates: float = 0.0
    wrong: float = 0.0
    growth: float = 0.0
    verified: bool = True
    def value(self) -> float:
        return utility(self.outcome, [step.reward for step in self.steps], self.cost, self.errors, self.duplicates, self.wrong, self.growth)
class Trainer:
    """Learn all branches and commit one branch."""
    def __init__(self, agent: Agent, gamma: float = 0.99, blend: float = 0.5):
        self.agent = agent
        self.gamma = gamma
        self.blend = blend
    def warm(self, samples: list[tuple[Trace, float, float]]) -> float:
        prepared = []
        for trace, semantic, structure in samples:
            if not isfinite(semantic) or not isfinite(structure):
                raise ValueError("route utilities must be finite")
            peak = max(semantic, structure)
            scores = [exp(semantic - peak), exp(structure - peak)]
            prepared.append((trace.features, scores))
        return self.agent.teacher.supervise(prepared)
    def learn(self, rollouts: list[Rollout]) -> dict:
        values = [item.value() for item in rollouts]
        episodes = [Episode(item.steps, item.outcome - 0.05 * item.cost - 0.2 * item.errors - 0.2 * item.duplicates - 0.5 * item.wrong - 0.05 * item.growth) for item in rollouts]
        policy = credit(episodes, self.gamma, self.blend)
        turns = turn([[step.reward for step in item.steps] for item in rollouts])
        samples = [(item.trace, item.chosen, value) for item, value in zip(rollouts, values) if item.trace is not None]
        teacher = self.agent.teacher.group(samples)
        valid = [pos for pos, item in enumerate(rollouts) if item.verified]
        best = max(valid, key=values.__getitem__) if valid else None
        if best is not None:
            item = rollouts[best]
            if item.trace is not None:
                self.agent.commit(item.trace, item.chosen, values[best], False)
            for draft in item.drafts:
                proposal = self.agent.stage(draft.name, draft.text, draft.code, draft.schema)
                self.agent.resolve(proposal, True, values[best])
        return {"policy": policy, "turn": turns, "teacher": teacher, "utility": values, "committed": best}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
