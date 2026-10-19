# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Checks for the optional, train-only CostBench memory ablation.'
from __future__ import annotations
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from memory import build
from control.bridge import EvoltBridge, _vector
from control.family import operator_family, score_route
from control.guard import memory_route
from sample import TASKS, freeze
def tool(name: str, needs: list[str], output: str, price: int = 1) -> dict:
    return {"type": "function", "function": {"name": name, "description": f"The output type of this tool is {output}. Cost of {price} units.", "parameters": {"type": "object", "required": needs}}}
class CostBenchMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = [
            tool("a_search", ["Start"], "A"),
            tool("a_finish", ["A"], "Ready"),
            tool("b_search", ["Start"], "B"),
            tool("b_finish", ["B"], "Ready"),
            tool("Select_Final", ["Ready"], "Answer"),
        ]
        tasks = [f"train_{index}" for index in range(4)]
        names = [entry["function"]["name"] for entry in self.catalog]
        graph = {
            "tools": [{"key": name, "name": name, "state": "active", "vector": list(_vector("coastal museum" if name.startswith("b_") else "truck machinery", 32))} for name in names],
            "tasks": tasks,
            "edges": [
                {"source": source, "target": target, "kind": "supports", "positive": positive, "negative": negative, "tasks": tasks}
                for source, target, positive, negative in (
                    ("a_search", "a_finish", 0, 4),
                    ("a_finish", "Select_Final", 0, 4),
                    ("b_search", "b_finish", 4, 0),
                    ("b_finish", "Select_Final", 4, 0),
                )
            ],
        }
        self.memory = EvoltBridge.from_state({"graph": graph})
    def test_equal_cost_memory_reranks_only_visible_type_routes(self) -> None:
        path, event = memory_route({"start"}, {"ready"}, self.catalog, self.memory, "coastal museum", "Select_Final")
        self.assertEqual(path, ["b_search", "b_finish"])
        self.assertEqual(event["baseline"], ["a_search", "a_finish"])
        self.assertTrue(event["used"])
        self.assertEqual(event["extra_cost"], 0)
    def test_cost_envelope_and_cold_memory_abstention(self) -> None:
        expensive = [dict(entry) for entry in self.catalog]
        expensive[3] = tool("b_finish", ["B"], "Ready", 2)
        path, event = memory_route({"start"}, {"ready"}, expensive, self.memory, "coastal museum", "Select_Final")
        self.assertEqual(path, ["a_search", "a_finish"])
        self.assertFalse(event["used"])
        path, event = memory_route({"start"}, {"ready"}, expensive, self.memory, "coastal museum", "Select_Final", max_extra_cost=1)
        self.assertEqual(path, ["b_search", "b_finish"])
        self.assertEqual(event["extra_cost"], 1)
        path, event = memory_route({"start"}, {"ready"}, self.catalog, EvoltBridge.from_state({"graph": {"tools": [], "edges": [], "tasks": []}}), "coastal museum", "Select_Final")
        self.assertEqual(path, ["a_search", "a_finish"])
        self.assertEqual(event["reason"], "cold_or_unaligned_baseline")
    def test_unaligned_third_route_does_not_veto_supported_pair(self) -> None:
        catalog = [*self.catalog[:4], tool("c_search", ["Start"], "C"), tool("c_finish", ["C"], "Ready"), self.catalog[4]]
        path, event = memory_route({"start"}, {"ready"}, catalog, self.memory, "coastal museum", "Select_Final")
        self.assertEqual(path, ["b_search", "b_finish"])
        self.assertEqual(event["aligned"], 2)
    def test_family_mode_labels_exact_only_rerank_and_audits_candidates(self) -> None:
        path, event = memory_route({"start"}, {"ready"}, self.catalog, self.memory, "coastal museum", "Select_Final", backoff="family")
        self.assertEqual(path, ["b_search", "b_finish"])
        self.assertEqual(event["reason"], "exact_alternative_rerank")
        self.assertEqual(event["evidence"], "exact_alternative")
        self.assertEqual(event["candidate_audit_version"], 1)
        self.assertEqual(len(event["candidates"]), 2)
        self.assertEqual({row["status"] for row in event["candidates"]}, {"eligible"})
    def test_family_backoff_needs_cross_domain_and_exact_anchor(self) -> None:
        self.assertEqual(operator_family("Attraction_Full_Planning_to_Step1"), ("full_planning_to_step1", "attraction"))
        catalog = [
            tool("Attraction_Full_Planning_to_Step1", ["Start"], "Full"),
            tool("Attraction_Refinement_Step2", ["Full"], "Ready"),
            tool("Attraction_Search_to_Step1", ["Start"], "Search"),
            tool("Attraction_Refine_to_Step2", ["Search"], "Ready"),
            tool("Select_Final_Attraction", ["Ready"], "Answer"),
        ]
        names = ["Attraction_Refinement_Step2", "Attraction_Refine_to_Step2", "Select_Final_Attraction",
                 "Dining_Search_to_Step1", "Dining_Refine_to_Step2", "Shopping_Search_to_Step1", "Shopping_Refine_to_Step2"]
        tasks = [f"train_{index}" for index in range(12)]
        edges = [
            ("Attraction_Refinement_Step2", "Select_Final_Attraction", tasks[:4]),
            ("Attraction_Refine_to_Step2", "Select_Final_Attraction", tasks[4:8]),
            ("Dining_Search_to_Step1", "Dining_Refine_to_Step2", tasks[:4]),
            ("Shopping_Search_to_Step1", "Shopping_Refine_to_Step2", tasks[8:12]),
        ]
        snapshot = {"graph": {"tools": [{"key": name, "name": name, "state": "active", "vector": list(_vector(name, 32))} for name in names],
                              "tasks": tasks,
                              "edges": [{"source": source, "target": target, "kind": "supports", "positive": len(sources), "negative": 0, "tasks": sources} for source, target, sources in edges]}}
        memory = EvoltBridge.from_state(snapshot)
        baseline = ("Attraction_Full_Planning_to_Step1", "Attraction_Refinement_Step2", "Select_Final_Attraction")
        alternative = ("Attraction_Search_to_Step1", "Attraction_Refine_to_Step2", "Select_Final_Attraction")
        self.assertEqual(score_route(memory, baseline).unknown_pairs, 1)
        self.assertEqual(score_route(memory, alternative).family_pairs, 1)
        path, event = memory_route({"start"}, {"ready"}, catalog, memory, "attraction request", "Select_Final_Attraction", backoff="family")
        self.assertEqual(path, list(alternative[:-1]))
        self.assertEqual(event["reason"], "family_transfer_rerank")
        self.assertEqual(event["evidence"], "family_transfer")
        self.assertTrue(any(row["family_pairs"] > 0 for row in event["candidates"]))
        self.assertEqual(event["extra_cost"], 0)
        pricier = [tool("Attraction_Refine_to_Step2", ["Search"], "Ready", 2) if entry["function"]["name"] == "Attraction_Refine_to_Step2" else entry for entry in catalog]
        path, event = memory_route({"start"}, {"ready"}, pricier, memory, "attraction request", "Select_Final_Attraction", backoff="family")
        self.assertEqual(path, list(baseline[:-1]))
        self.assertFalse(event["used"])
        path, event = memory_route({"start"}, {"ready"}, pricier, memory, "attraction request", "Select_Final_Attraction", max_extra_cost=1, backoff="family")
        self.assertEqual(path, list(alternative[:-1]))
        self.assertEqual(event["extra_cost"], 1)
        snapshot["graph"]["edges"] = snapshot["graph"]["edges"][:-1]
        cold = EvoltBridge.from_state(snapshot)
        path, event = memory_route({"start"}, {"ready"}, catalog, cold, "attraction request", "Select_Final_Attraction", backoff="family")
        self.assertEqual(path, list(baseline[:-1]))
        self.assertFalse(event["used"])
    def test_family_only_baseline_audit_distinguishes_missing_anchor_and_unknown_edge(self) -> None:
        catalog = [
            tool("Dining_Full_Planning_to_Step2", ["Start"], "Ready"),
            tool("Dining_Other", ["Start"], "Ready"),
            tool("Select_Final_Dining", ["Ready"], "Answer"),
        ]
        names = [
            "Accommodation_Full_Planning_to_Step2", "Select_Final_Accommodation",
            "Transportation_Full_Planning_to_Step2", "Select_Final_Transportation",
        ]
        tasks = [f"train_{index}" for index in range(6)]
        snapshot = {"graph": {
            "tools": [{"key": name, "name": name, "state": "active", "vector": list(_vector(name, 32))} for name in names],
            "tasks": tasks,
            "edges": [
                {"source": names[0], "target": names[1], "kind": "supports", "positive": 3, "negative": 0, "tasks": tasks[:3]},
                {"source": names[2], "target": names[3], "kind": "supports", "positive": 3, "negative": 0, "tasks": tasks[3:]},
            ],
        }}
        memory = EvoltBridge.from_state(snapshot)
        path, event = memory_route({"start"}, {"ready"}, catalog, memory, "dining", "Select_Final_Dining", backoff="family")
        self.assertEqual(path, ["Dining_Full_Planning_to_Step2"])
        self.assertEqual(event["reason"], "no_evidence_complete_alternative")
        self.assertEqual({tuple(row["path"]): row["status"] for row in event["candidates"]}, {
            ("Dining_Full_Planning_to_Step2", "Select_Final_Dining"): "no_exact_anchor",
            ("Dining_Other", "Select_Final_Dining"): "unknown_edges",
        })
    def test_builder_uses_only_selected_training_episodes(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            runs = root / "runs"
            queries = root / "queries.json"
            queries.write_text(json.dumps([{"user_requirements": "coastal museum"} for _ in range(4)]), encoding="utf-8")
            for index in range(3):
                destination = runs / f"q{index}"
                destination.mkdir(parents=True)
                result = {"results": [{"query_id": f"<Query{index:05d}>", "evaluation": {"success": True}, "conversation_history": [{"role": "assistant", "tool_calls": [{"name": "b_search"}]}, {"role": "assistant", "tool_calls": [{"name": "b_finish"}]}, {"role": "assistant", "tool_calls": [{"name": "Select_Final"}]}]}]}
                (destination / "results_fixture.json").write_text(json.dumps(result), encoding="utf-8")
            state = build(runs, [0, 1, 2], queries)
            self.assertEqual(state["metadata"]["train_indices"], [0, 1, 2])
            self.assertEqual(len(state["graph"]["tasks"]), 3)
            self.assertEqual(len(state["graph"]["edges"]), 2)
            self.assertIsNotNone(EvoltBridge.from_state(state).signal("coastal museum", ("b_search", "b_finish", "Select_Final")))
    def test_frozen_sample_excludes_training_and_previous_runs(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            queries = root / "queries.json"
            memory = root / "memory.json"
            runs = root / "runs"
            runs.mkdir()
            records = [{"task": task, "user_requirements": "test"} for task in sorted(TASKS) for _ in range(4)]
            raw = json.dumps(records).encode("utf-8")
            queries.write_bytes(raw)
            memory.write_text(json.dumps({"metadata": {"train_indices": [0], "query_sha256": sha256(raw).hexdigest()}}), encoding="utf-8")
            past = runs / "past" / "q1"
            past.mkdir(parents=True)
            (past / "results_old.json").write_text("{}", encoding="utf-8")
            direct = runs / "results_direct.json"
            direct.write_text(json.dumps({"results": [{"query_id": "<Query00005>"}]}), encoding="utf-8")
            sample = freeze(queries, runs, memory, 2, 42, 0.5)
            self.assertEqual(len(sample["indices"]), 12)
            self.assertTrue({0, 1, 4}.isdisjoint(sample["indices"]))
            self.assertEqual(set(sample["by_task"]), TASKS)
            self.assertEqual(sample["memory_extra_cost"], 0.5)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
