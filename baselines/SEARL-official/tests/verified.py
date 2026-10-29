# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

_path = Path(__file__).resolve().parents[1] / "envs" / "tool_manager" / "centralized" / "verified.py"
_spec = spec_from_file_location("verified_creation", _path)
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)
verified_creation_result = _module.verified_creation_result

class VerifiedCreationTests(unittest.TestCase):
    def test_success_requires_both_flags(self):
        payload = {"creation_success": True, "run_success": True}
        self.assertTrue(verified_creation_result(json.dumps(payload)))
        self.assertTrue(verified_creation_result({"content": [{"type": "text", "text": json.dumps(payload)}]}))
        self.assertFalse(verified_creation_result({"creation_success": True, "run_success": False}))
        self.assertFalse(verified_creation_result("Created successfully"))
        self.assertFalse(verified_creation_result({"content": []}))

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
