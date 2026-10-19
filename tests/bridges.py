# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Offline checks for the optional EVOLT -> EvoCond soft-signal boundary.'
from __future__ import annotations
import unittest
from control.bridge import EvoltBridge, _vector
from control.recourse import Artifact, EvidenceGraph, Scenario, ToolStep
def fixture(b_extra: float = 0.0):
    artifacts = {
        "start": Artifact("start", "request", "goal"),
        "a_mid": Artifact("a_mid", "candidate", "a", frozenset({"start"})),
        "b_mid": Artifact("b_mid", "candidate", "b", frozenset({"start"})),
        "a_final": Artifact("a_final", "answer", "valid", frozenset({"a_mid"})),
        "b_final": Artifact("b_final", "answer", "valid", frozenset({"b_mid"})),
        "secret": Artifact("secret", "private", "not_seen"),
    }
    graph = EvidenceGraph(
        artifacts,
        (
            ToolStep("a_search", frozenset({"start"}), frozenset({"a_mid"}), 1),
            ToolStep("a_select", frozenset({"a_mid"}), frozenset({"a_final"}), 1),
            ToolStep("b_search", frozenset({"start"}), frozenset({"b_mid"}), 1),
            ToolStep("b_select", frozenset({"b_mid"}), frozenset({"b_final"}), 1 + b_extra),
            ToolStep("shortcut", frozenset({"secret"}), frozenset({"b_final"}), 0),
        ),
    )
    goal = lambda state, _: bool({"a_final", "b_final"} & state)
    scenarios = (Scenario(1.0, {}, frozenset(), goal),)
    phrase = "quiet coastal museum"
    tools = []
    for name, text in (
        ("a_search", "truck machinery logistics"),
        ("a_select", "truck machinery logistics"),
        ("b_search", phrase),
        ("b_select", phrase),
        ("shortcut", phrase),
    ):
        tools.append({"key": name, "name": name, "state": "active", "vector": list(_vector(text, 32))})
    tasks = [f"train_{index}" for index in range(4)]
    snapshot = {
        "graph": {
            "tools": tools,
            "tasks": tasks,
            "edges": [
                {"source": "a_search", "target": "a_select", "kind": "precedes", "positive": 1000, "negative": 0, "tasks": tasks.copy()},
                {"source": "a_search", "target": "a_select", "kind": "supports", "positive": 0, "negative": 4, "tasks": tasks.copy()},
                {"source": "b_search", "target": "b_select", "kind": "supports", "positive": 4, "negative": 0, "tasks": tasks.copy()},
                {"source": "b_search", "target": "shortcut", "kind": "supports", "positive": 1000, "negative": 0, "tasks": tasks.copy()},
            ],
        }
    }
    return graph, goal, scenarios, phrase, snapshot
class EvoltBridgeTests(unittest.TestCase):
    def test_soft_memory_breaks_equal_cost_tie_only_after_hard_feasibility(self) -> None:
        graph, goal, scenarios, query, snapshot = fixture()
        bridge = EvoltBridge.from_state(snapshot)
        choice = bridge.choose(graph, {"start"}, {}, goal, scenarios, query)
        self.assertEqual(choice.cost_only.plan.actions, ("a_search", "a_select"))
        self.assertEqual(choice.selected.plan.actions, ("b_search", "b_select"))
        self.assertTrue(choice.used_memory)
        self.assertNotIn("shortcut", choice.selected.plan.actions)
        self.assertEqual(choice.selected.expected_total, choice.cost_only.expected_total)
        self.assertGreater(choice.signals[("b_search", "b_select")].fused, choice.signals[("a_search", "a_select")].fused)
    def test_statistical_sequence_is_not_support_or_prerequisite(self) -> None:
        graph, goal, scenarios, query, snapshot = fixture()
        snapshot["graph"]["edges"] = [edge for edge in snapshot["graph"]["edges"] if edge["kind"] == "precedes"]
        choice = EvoltBridge.from_state(snapshot).choose(graph, {"start"}, {}, goal, scenarios, query)
        self.assertFalse(choice.used_memory)
        self.assertEqual(choice.reason, "cold_or_unaligned_memory")
        self.assertEqual(choice.selected, choice.cost_only)
    def test_cost_envelope_is_explicit(self) -> None:
        graph, goal, scenarios, query, snapshot = fixture(b_extra=1.0)
        bridge = EvoltBridge.from_state(snapshot)
        strict = bridge.choose(graph, {"start"}, {}, goal, scenarios, query)
        allowed = bridge.choose(graph, {"start"}, {}, goal, scenarios, query, max_extra_cost=1.0)
        self.assertEqual(strict.selected.plan.actions, ("a_search", "a_select"))
        self.assertEqual(allowed.selected.plan.actions, ("b_search", "b_select"))
        self.assertEqual(allowed.selected.expected_total - allowed.cost_only.expected_total, 1.0)
    def test_cold_snapshot_and_unaligned_identity_abstain(self) -> None:
        graph, goal, scenarios, query, snapshot = fixture()
        snapshot["graph"]["tasks"] = ["train_0"]
        for edge in snapshot["graph"]["edges"]:
            edge["tasks"] = ["train_0"]
        cold = EvoltBridge.from_state(snapshot).choose(graph, {"start"}, {}, goal, scenarios, query)
        self.assertEqual(cold.selected, cold.cost_only)
        self.assertFalse(cold.used_memory)
        graph, goal, scenarios, query, snapshot = fixture()
        snapshot["graph"]["tools"][0]["name"] = "different"
        unaligned = EvoltBridge.from_state(snapshot).choose(graph, {"start"}, {}, goal, scenarios, query)
        self.assertEqual(unaligned.selected, unaligned.cost_only)
        mapped = EvoltBridge.from_state(snapshot, tool_map={"a_search": "a_search"}).choose(graph, {"start"}, {}, goal, scenarios, query)
        self.assertTrue(mapped.used_memory)
    def test_invalid_snapshot_is_rejected(self) -> None:
        _, _, _, _, snapshot = fixture()
        snapshot["graph"]["tools"][0]["vector"][0] = float("nan")
        with self.assertRaises(ValueError):
            EvoltBridge.from_state(snapshot)
        _, _, _, _, snapshot = fixture()
        snapshot["graph"]["edges"][0]["tasks"].append("future_heldout_task")
        with self.assertRaises(ValueError):
            EvoltBridge.from_state(snapshot)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
