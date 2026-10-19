# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import unittest
from fit import prepare
class TeacherFitTests(unittest.TestCase):
    def test_train_only_paired_fit(self):
        splits = {"train": ["q1", "q2", "q3"], "validation": ["q4"], "test": ["q5"]}
        pairs = [{"query_id": identity, "mode": "none", "seed": 1,
                  "features": [0.0]*44, "semantic_reward": 0.0,
                  "structural_reward": 1.0}
                 for identity in splits["train"]]
        alignment_pairs = [{"query_id": identity, "tool_version": "v1",
                            "semantic": [1.0]+[0.0]*31,
                            "structural": [1.0]+[0.0]*31}
                           for identity in splits["train"]]
        teacher, alignment, report = prepare(splits, pairs, alignment_pairs)
        self.assertEqual(report["informative_pairs"], 3)
        self.assertEqual(alignment["dimension"], 32)
        self.assertEqual(len(teacher["weights"]), 45)
        with self.assertRaises(ValueError):
            prepare(splits, [*pairs, {**pairs[0], "query_id": "q4"}], alignment_pairs)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
