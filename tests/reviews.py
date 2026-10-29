# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from review import REFERENCES, summarize
def episode(success, cost):
    return ({"results": [{"evaluation": {"success": success}, "total_steps": 3}],
             "stats": {"scores": {"cost_metrics": {"total_agent_cost": cost}}}},
            {"elapsed_seconds": 2.0, "prompt_tokens": 100, "completion_tokens": 20,
             "controller_seconds": 0.1, "value_audit_seconds": 0.5,
             "value_audits": 1, "value_rejections": int(success)})
class ValueWorkflowTests(unittest.TestCase):
    def test_development_report_counts_paired_gain_and_loss(self):
        source = {"records": [
            {"mode": "none", "index": 1, "outcomes": {arm: {"success": False} for arm in REFERENCES}},
            {"mode": "preference_change", "index": 1,
             "outcomes": {arm: {"success": True} for arm in REFERENCES}}],
            "overall": {arm: {"mean_cost_all": 12.0} for arm in REFERENCES}}
        outcomes = {("none", 1): episode(True, 10.0),
                    ("preference_change", 1): episode(False, 14.0)}
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            with patch('review.checked_episode',
                       side_effect=lambda _root, _source, mode, index, _record: outcomes[mode, index]):
                report = summarize(root, root, {"validation_indices": [1]}, source)
            self.assertEqual(report["n"], 2)
            self.assertEqual(report["correct"], 1)
            self.assertEqual(report["paired"]["evocond"]["gain"], 1)
            self.assertEqual(report["paired"]["evocond"]["loss"], 1)
            self.assertEqual(report["mean_overhead"]["value_audits"], 1)
            self.assertTrue((root / "validation_report.json").is_file())
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
