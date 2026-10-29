# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Telemetry checks never replace task-quality evaluation.'
import json
from pathlib import Path
import tempfile
import unittest
import telemetry
class TelemetryTests(unittest.TestCase):
    def test_missing_logs_do_not_report_perfect_success(self):
        with tempfile.TemporaryDirectory() as folder:
            result = telemetry.summarize(Path(folder))
        self.assertIsNone(result['splits']['test']['blind_rate'])
        self.assertNotIn('success', result)
    def test_blind_action_mismatch_is_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [dict(split='validation', task_id='q', alpha=alpha, marked=False, edge_support=False, teacher_depth=0,
                abstention=dict(active=True, reason='outside_absolute_support')) for alpha in (0, .5)]
            (root/'retrievals.jsonl').write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
            result = telemetry.summarize(root)
        self.assertEqual(result['splits']['validation']['blind_rate'], 1)
        self.assertEqual(result['splits']['validation']['independent_questions'], 1)
        self.assertEqual(result['splits']['validation']['fallback_contract_violations'], 1)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
