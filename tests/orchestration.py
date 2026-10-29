# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
from argparse import Namespace
from pathlib import Path
import unittest
from panel import PROJECT
from orchestrate import arm_server
class ThreeArmWorkflowTests(unittest.TestCase):
    def test_each_arm_starts_its_own_controller(self) -> None:
        args = Namespace(model="assets/model", port=8000, temperature=0.7)
        state = PROJECT / "runs" / "state"
        trace = PROJECT / "runs" / "three" / "events" / "arm.jsonl"
        searl = arm_server("searl_adapted", args, state, trace)
        evocond = arm_server("evocond", args, state, trace)
        teacher = arm_server("teacher_dual", args, state, trace)
        self.assertNotIn("--evocond", searl)
        self.assertIn("--evocond", evocond)
        self.assertNotIn("--state-controller", evocond)
        self.assertIn("--state-controller", teacher)
        self.assertIn("--trace-context", searl)
        self.assertEqual(teacher[teacher.index("--state-policy") + 1], "learned")
        self.assertFalse(Path(teacher[teacher.index("--trace") + 1]).is_absolute())
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
