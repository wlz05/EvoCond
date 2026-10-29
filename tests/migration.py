# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import tempfile
import unittest
from pathlib import Path
from preflight import inspect
class MigrationPreflightTests(unittest.TestCase):
    def test_missing_assets_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            report = inspect({"format": "evocond_searl_migration_v1"}, base=Path(folder))
        self.assertFalse(report["ready"])
        self.assertTrue(any("isolated generated-code" in item for item in report["blockers"]))
        self.assertTrue(any("method integrations" in item for item in report["blockers"]))
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
