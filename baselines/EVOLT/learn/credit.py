# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Compute branch and temporal advantages."""
from dataclasses import dataclass, field
from math import isfinite, sqrt
from core.embed import fold
@dataclass(slots=True)
class Step:
    anchor: str
    reward: float
@dataclass(slots=True)
class Episode:
    steps: list[Step] = field(default_factory=list)
    outcome: float = 0.0
def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    if any(not isfinite(value) for value in values):
        raise ValueError("values must be finite")
    center = sum(values) / len(values)
    spread = sqrt(sum((value - center) ** 2 for value in values) / len(values))
    return [(value - center) / spread for value in values] if spread > 1e-12 else [0.0] * len(values)
def utility(outcome: float, rewards: list[float] | None = None, cost: float = 0.0, errors: float = 0.0, duplicates: float = 0.0, wrong: float = 0.0, growth: float = 0.0, penalties: tuple[float, float, float, float, float] = (0.05, 0.2, 0.2, 0.5, 0.05)) -> float:
    values = [outcome, cost, errors, duplicates, wrong, growth, *penalties, *(rewards or [])]
    if any(not isfinite(value) for value in values):
        raise ValueError("utility inputs must be finite")
    if any(value < 0.0 for value in (cost, errors, duplicates, wrong, growth, *penalties)):
        raise ValueError("utility costs must not be negative")
    return outcome + sum(rewards or []) - sum(weight * value for weight, value in zip(penalties, (cost, errors, duplicates, wrong, growth)))
def future(episode: Episode, gamma: float) -> list[float]:
    output = [0.0] * len(episode.steps)
    value = 0.0
    for pos in range(len(episode.steps) - 1, -1, -1):
        reward = episode.steps[pos].reward
        if not isfinite(reward):
            raise ValueError("step rewards must be finite")
        value = reward + gamma * value
        output[pos] = value
    return output
def credit(episodes: list[Episode], gamma: float = 0.99, blend: float = 0.5) -> list[list[float]]:
    if not isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be within zero and one")
    if not isfinite(blend) or blend < 0.0:
        raise ValueError("blend must not be negative")
    totals = [episode.outcome + sum(step.reward for step in episode.steps) for episode in episodes]
    broad = normalize(totals)
    returns = [future(episode, gamma) for episode in episodes]
    groups: dict[str, list[tuple[int, int, float]]] = {}
    for row, episode in enumerate(episodes):
        for col, step in enumerate(episode.steps):
            groups.setdefault(fold(step.anchor), []).append((row, col, returns[row][col]))
    narrow: dict[tuple[int, int], float] = {}
    for values in groups.values():
        scores = normalize([item[2] for item in values])
        for item, score in zip(values, scores):
            narrow[(item[0], item[1])] = score
    return [[broad[row] + blend * narrow.get((row, col), 0.0) for col in range(len(episode.steps))] for row, episode in enumerate(episodes)]
def turn(rewards: list[list[float]]) -> list[list[float]]:
    if not rewards:
        return []
    if any(not isfinite(value) for row in rewards for value in row):
        raise ValueError("rewards must be finite")
    output = [[0.0] * len(row) for row in rewards]
    for col in range(max((len(row) for row in rewards), default=0)):
        places = [row for row, values in enumerate(rewards) if col < len(values)]
        values = [rewards[row][col] for row in places]
        center = sum(values) / len(values)
        for row, value in zip(places, values):
            output[row][col] = value - center
    return output

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
