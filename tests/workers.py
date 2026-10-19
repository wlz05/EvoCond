# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Verify distributed workers get only the declared experiment environment.'
import importlib.util
import os
import unittest
from unittest.mock import patch
from tests.checkout import official_root
class RayEnvironmentTests(unittest.TestCase):
    def test_fresh_arm_paths_are_forwarded_to_reused_ray_daemon(self):
        path = official_root() / "verl/trainer/constants.py"
        spec = importlib.util.spec_from_file_location("tested_searl_ray_constants", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        values = {"SEARL_DYNAMIC_REGISTRY": "/new-arm/runtime/registry.json",
                  "SEARL_ISOLATION_CONFIG": "/inspected/config.json",
                  "SEARL_ISOLATION_RECEIPT": "/inspected/receipt.json",
                  "SEARL_SEARCH_URL": "http://127.0.0.1:5003/retrieve",
                  "RL_FACTORY_EMBEDDING_MODEL": "/encoder", "PRIVATE_SECRET": "must-not-forward"}
        with patch.dict(os.environ, values, clear=True):
            env = module.get_ppo_ray_runtime_env()["env_vars"]
        for key, value in values.items():
            if key != "PRIVATE_SECRET":
                self.assertEqual(env[key], value)
        self.assertNotIn("PRIVATE_SECRET", env)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
