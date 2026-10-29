# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import unittest
from control.memory import admit_visible_tool, empty_store, record_transition, validate
def tool(name: str, description: str, requires: list[str], produces: str) -> dict:
    return {"type": "function", "function": {"name": name,
            "description": f"{description} The output type of this tool is {produces}. It has a cost of 1 units.",
            "parameters": {"type": "object", "required": requires,
                           "properties": {key: {"type": "string"} for key in requires}}}}
class TeacherMemoryTests(unittest.TestCase):
    def test_separate_stores_align_by_version(self):
        store, event = admit_visible_tool(empty_store(), tool("Find", "Find hotel", ["TimeInfo"], "Hotel"))
        self.assertEqual(event["decision"], "new_versioned_node")
        self.assertEqual(set(store["semantic_store"]), set(store["structural_store"]["nodes"]))
        validate(store)
    def test_semantic_similarity_cannot_merge_different_contracts(self):
        first = tool("Search_A", "Find hotel", ["TimeInfo"], "Hotel")
        second = tool("Search_B", "Find hotel", ["Location"], "Hotel")
        store, event = admit_visible_tool(empty_store(), first)
        with self.assertRaises(ValueError):
            admit_visible_tool(store, second, verified_equivalent_to=event["version"])
        updated, admitted = admit_visible_tool(store, second)
        self.assertEqual(admitted["decision"], "new_versioned_node")
        self.assertEqual(len(updated["semantic_store"]), 2)
    def test_similar_tool_needs_behavioral_verification_to_merge(self):
        first = tool("Search_A", "Find hotel", ["TimeInfo"], "Hotel")
        second = tool("Search_B", "Find hotel", ["TimeInfo"], "Hotel")
        store, event = admit_visible_tool(empty_store(), first)
        unverified, admitted = admit_visible_tool(store, second, threshold=0)
        self.assertEqual(admitted["decision"], "new_versioned_node")
        self.assertEqual(len(unverified["semantic_store"]), 2)
        verification = {"candidate_version": event["version"],
                        "input_output_checked": True, "side_effects_checked": True,
                        "evidence_sha256": "a" * 64}
        merged, decision = admit_visible_tool(store, second, threshold=0,
                                               verified_equivalent_to=event["version"],
                                               verification=verification)
        self.assertEqual(decision["decision"], "merge_verified")
        self.assertEqual(len(merged["semantic_store"]), 1)
    def test_edge_feedback_is_query_deduplicated_and_no_blanket_failure(self):
        store = empty_store()
        for name, needs, produces in (("Preference", ["TimeInfo"], "Preference"),
                                      ("Search", ["Preference"], "Candidates")):
            store, _ = admit_visible_tool(store, tool(name, name, needs, produces))
        source, target = store["aliases"]["Preference"], store["aliases"]["Search"]
        no_update = record_transition(store, source=source, target=target, query_id="q1",
                                      feedback="verified", task_success=False)
        self.assertEqual(no_update, store)
        updated = record_transition(store, source=source, target=target, query_id="q1",
                                    feedback="verified", task_success=True)
        edge = next(iter(updated["structural_store"]["edges"].values()))
        self.assertGreater(edge["weight"], 0.5)
        repeated = record_transition(updated, source=source, target=target, query_id="q1",
                                     feedback="verified", task_success=True)
        self.assertEqual(repeated, updated)
        with self.assertRaises(ValueError):
            record_transition(updated, source=source, target=target, query_id="q1",
                              feedback="failed", task_success=False)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
