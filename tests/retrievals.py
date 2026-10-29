# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import unittest
from control.dynamic import admit, empty_registry, update_edge
from control.retrieval import fit_alignment, rank
from tests.generation import FakeSandbox, tool
class FixedEncoder:
    def encode(self, text):
        return [1.0] + [0.0]*31
def teacher():
    return {"format": "evocond_retrieval_teacher_v1", "weights": [1.0]+[0.0]*44,
            "margin": 0.05, "train_indices": [1], "validation_indices": [2],
            "test_indices": [3], "paired_training_hash": "a"*64}
class RetrievalTests(unittest.TestCase):
    def test_alignment_rejects_held_out_data(self):
        pair = {"query_id": "q1", "tool_version": "v1", "semantic": [1.0]*32,
                "structural": [1.0]*32}
        with self.assertRaises(ValueError):
            fit_alignment([pair], train_ids={"q2"}, validation_ids={"q1"}, test_ids=set())
        self.assertEqual(fit_alignment([pair], train_ids={"q1"},
                                       validation_ids={"q2"}, test_ids={"q3"})["dimension"], 32)
    def test_teacher_abstains_without_edge_and_fuses_with_support(self):
        backend = FakeSandbox()
        registry, _ = admit(empty_registry(), tool("first"), backend,
                            probe_inputs={"value": 1}, task_id="q1")
        registry, _ = admit(registry, tool("second"), backend,
                            probe_inputs={"value": 1}, task_id="q1")
        basis = [[1.0 if i == j else 0.0 for j in range(32)] for i in range(32)]
        alignment = {"format": "evocond_searl_alignment_v1", "dimension": 32, "matrix": basis}
        before = rank(registry, FixedEncoder(), alignment, teacher(), query="question",
                      previous_tool="first")
        self.assertTrue(all(row["teacher"]["decision"] == "abstain" for row in before["tools"]))
        updated = update_edge(registry, source_name="first", target_name="second",
                              task_id="q1", split="train", direct_feedback="verified", task_success=True)
        after = rank(updated, FixedEncoder(), alignment, teacher(), query="question",
                     previous_tool="first")
        second = next(row for row in after["tools"] if row["name"] == "second")
        self.assertEqual(second["teacher"]["decision"], "fuse")
        self.assertEqual(second["support"], 1)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
