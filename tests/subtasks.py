# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import unittest
from control.subtasks import complete, make_plan, parse_proposal, ready, revise
def proposal():
    return [{"id": "find", "goal": "Find an authoritative source", "depends_on": [],
             "evidence_required": "retrieval result"},
            {"id": "answer", "goal": "Answer from evidence", "depends_on": ["find"],
             "evidence_required": "verified answer"}]
class SubtaskTests(unittest.TestCase):
    def test_dependencies_and_observed_completion(self):
        plan = make_plan("Who?", proposal())
        self.assertEqual([task["id"] for task in ready(plan)], ["find"])
        with self.assertRaises(ValueError):
            complete(plan, "answer", observed_tool_version="v1", evidence="yes", epoch=0)
        completed = complete(plan, "find", observed_tool_version="v1", evidence="source", epoch=0)
        self.assertEqual([task["id"] for task in ready(completed)], ["answer"])
        self.assertEqual(plan["tasks"]["find"]["status"], "pending")
    def test_revision_invalidates_old_evidence(self):
        original = make_plan("Old goal", proposal())
        revised = revise(original, "New goal", proposal())
        self.assertEqual(revised["epoch"], 1)
        with self.assertRaises(ValueError):
            complete(revised, "find", observed_tool_version="v1", evidence="old", epoch=0)
        self.assertIsNone(revised["tasks"]["find"]["evidence"])
    def test_cycle_and_non_json_fail_closed(self):
        cyclic = proposal()
        cyclic[0]["depends_on"] = ["answer"]
        with self.assertRaises(ValueError):
            make_plan("Goal", cyclic)
        with self.assertRaises(Exception):
            parse_proposal("## ST1: informal plan", "Goal")
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
