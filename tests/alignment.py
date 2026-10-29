# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import unittest
from control.fusion import decide, fit, validate
def pair(index: int, sem: int, struct: int) -> dict:
    vector = [0.0] * 44
    vector[0] = float(index % 2)
    vector[-2] = 0.8
    return {"index": index, "mode": "none", "seed": 11,
            "features": vector, "semantic_reward": sem, "structural_reward": struct}
class TeacherFusionTests(unittest.TestCase):
    def test_paired_teacher_fits_and_abstains_without_support(self):
        pairs = [pair(0, 0, 1), pair(1, 1, 0), pair(2, 0, 1), pair(3, 1, 0)]
        model, report = fit(pairs, train_indices=list(range(4)), validation_indices=[4], test_indices=[5])
        self.assertEqual(report["informative_pairs"], 4)
        validate(model)
        x = pair(0, 0, 1)["features"]
        self.assertGreater(decide(model, x)["structure_weight"], 0.5)
        x[-2] = 0.0
        self.assertEqual(decide(model, x)["decision"], "abstain")
    def test_negative_structural_benefit_never_injects_structure(self):
        pairs = [pair(index, 1, 0) for index in range(4)]
        model, _ = fit(pairs, train_indices=list(range(4)), validation_indices=[4], test_indices=[5])
        decision = decide(model, pair(0, 1, 0)["features"])
        self.assertLess(decision["predicted_delta"], 0)
        self.assertEqual(decision["structure_weight"], 0.0)
    def test_rejects_leakage_and_weak_signal(self):
        with self.assertRaises(ValueError):
            fit([pair(0, 0, 1)] * 3, train_indices=[0], validation_indices=[0], test_indices=[])
        with self.assertRaises(ValueError):
            fit([pair(0, 1, 1), pair(1, 1, 1), pair(2, 0, 1)],
                train_indices=[0, 1, 2], validation_indices=[], test_indices=[])
        with self.assertRaises(ValueError):
            fit([pair(0, 0, 1), pair(0, 1, 0), pair(1, 0, 1)],
                train_indices=[0, 1], validation_indices=[], test_indices=[])
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
