# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Offline fixtures for the proposed evidence-recourse formulation.'
from __future__ import annotations
import unittest
from control.recourse import Artifact, EvidenceGraph, Scenario, ToolStep
def art(name: str, *, value: str = "", parents: tuple[str, ...] = (), versions: tuple[tuple[str, int], ...] = ()) -> Artifact:
    return Artifact(name, name.split("_")[0], value, frozenset(parents), versions)
def has(name: str):
    return lambda state, artifacts: name in state
class EvidenceRecourseTests(unittest.TestCase):
    def test_preference_change_retains_time_but_invalidates_descendants(self) -> None:
        graph = EvidenceGraph(
            {
                "time": art("time", value="TimeInfo02502", versions=(("time", 1),)),
                "pref_old": art("pref_old", value="LocationPreference00157", versions=(("preference", 1),)),
                "pref_new": art("pref_new", value="LocationPreference00082", versions=(("preference", 2),)),
                "candidate_old": art("candidate_old", parents=("time", "pref_old")),
                "candidate_new": art("candidate_new", parents=("time", "pref_new")),
            },
            (ToolStep("search_new", frozenset({"time", "pref_new"}), frozenset({"candidate_new"}), 8),),
        )
        known = {"time", "pref_old", "candidate_old", "pref_new"}
        retained = graph.valid(known, {"time": 1, "preference": 2})
        self.assertEqual(retained, {"time", "pref_new"})
        repair = graph.shortest(known, {"time": 1, "preference": 2}, has("candidate_new"))
        self.assertIsNotNone(repair)
        self.assertEqual(repair.actions, ("search_new",))
        self.assertEqual(repair.cost, 8)
    def test_goal_checks_values_not_just_types(self) -> None:
        graph = EvidenceGraph(
            {
                "request": art("request"),
                "candidate_bad": art("candidate_bad", value="cheap_no_live_music", parents=("request",)),
                "candidate_good": art("candidate_good", value="cheap_live_music", parents=("request",)),
            },
            (
                ToolStep("cheap_bad", frozenset({"request"}), frozenset({"candidate_bad"}), 1),
                ToolStep("verified_good", frozenset({"request"}), frozenset({"candidate_good"}), 3),
            ),
        )
        goal = lambda state, artifacts: any(
            name.startswith("candidate_") and artifacts[name].value == "cheap_live_music" for name in state
        )
        plan = graph.shortest({"request"}, {}, goal)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.actions, ("verified_good",))
        self.assertEqual(plan.cost, 3)
    def test_lookahead_can_prefer_reusable_slightly_costlier_plan(self) -> None:
        graph = EvidenceGraph(
            {
                "time": art("time", versions=(("time", 1),)),
                "old_pref": art("old_pref", versions=(("preference", 1),)),
                "new_pref": art("new_pref", versions=(("preference", 2),)),
                "shared": art("shared", parents=("time",)),
                "old_final": art("old_final", parents=("time", "old_pref")),
                "new_final": art("new_final", parents=("time", "new_pref")),
            },
            (
                ToolStep("direct_old", frozenset({"time", "old_pref"}), frozenset({"old_final"}), 90),
                ToolStep("direct_new", frozenset({"time", "new_pref"}), frozenset({"new_final"}), 110),
                ToolStep("make_shared", frozenset({"time"}), frozenset({"shared"}), 80),
                ToolStep("select_old", frozenset({"shared", "old_pref"}), frozenset({"old_final"}), 20),
                ToolStep("select_new", frozenset({"shared", "new_pref"}), frozenset({"new_final"}), 20),
            ),
        )
        known = {"time", "old_pref"}
        current = {"time": 1, "preference": 1}
        myopic = graph.shortest(known, current, has("old_final"))
        self.assertIsNotNone(myopic)
        self.assertEqual(myopic.actions, ("direct_old",))
        scenarios = (
            Scenario(0.8, current, frozenset(), has("old_final")),
            Scenario(0.2, {"time": 1, "preference": 2}, frozenset({"new_pref"}), has("new_final")),
        )
        forward = graph.prospective(known, current, has("old_final"), scenarios)
        self.assertIsNotNone(forward)
        self.assertEqual(forward.plan.actions, ("make_shared", "select_old"))
        self.assertEqual(forward.plan.cost, 100)
        self.assertEqual(forward.expected_repair, 4)
    def test_unreachable_and_invalid_cost(self) -> None:
        graph = EvidenceGraph({"start": art("start"), "finish": art("finish")}, ())
        self.assertIsNone(graph.shortest({"start"}, {}, has("finish")))
        with self.assertRaises(ValueError):
            EvidenceGraph(graph.artifacts, (ToolStep("bad", frozenset({"start"}), frozenset({"finish"}), -1),))
        with self.assertRaises(ValueError):
            EvidenceGraph({"a": art("a", parents=("b",)), "b": art("b", parents=("a",))}, ())
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
