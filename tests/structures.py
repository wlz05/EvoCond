# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from copy import deepcopy
from pathlib import Path
import unittest
from structure import variants
class StructureControlTests(unittest.TestCase):
    def fixtures(self):
        config = {"format": "evocond_searl_experiment_v1", "run_id": "study",
                  **{key: "/fixture/"+key for key in ("model", "encoder", "official_root", "isolation_config", "isolation_receipt")},
                  "datasets": {"train": "train", "validation": "val", "test": "test"},
                  "budget": {"gpus": 4}, "teacher": {"encoder_id": "pinned", "binding_strategy": "reject"}}
        bootstrap = {"format": "evocond_teacher_bootstrap_v1", "experiment": deepcopy(config), "encoder_id": "pinned"}
        bootstrap["experiment"].pop("teacher")
        bootstrap["experiment"]["budget"]["rollouts"] = 1
        return config, bootstrap
    def test_all_eight_have_matching_private_artifacts_and_preserve_controls(self):
        config, bootstrap = self.fixtures()
        before = deepcopy((config, bootstrap))
        result = variants(config, bootstrap, Path("/new-root"))
        self.assertEqual(len(result), 8)
        self.assertEqual(len({item["experiment"]["teacher"]["teacher_path"] for item in result.values()}), 8)
        for item in result.values():
            for key in ("tool_construction", "structure_policy"):
                self.assertEqual(item["experiment"]["teacher"][key], item["bootstrap"][key])
            self.assertEqual(item["experiment"]["budget"], config["budget"])
            self.assertEqual(item["experiment"]["datasets"], config["datasets"])
        self.assertEqual((config, bootstrap), before)
    def test_refuses_mismatched_input_or_binding_semantics(self):
        for key in ("model", "datasets"):
            config, bootstrap = self.fixtures()
            bootstrap["experiment"][key] = "changed"
            with self.assertRaises(ValueError):
                variants(config, bootstrap, Path("/new-root"))
        config, bootstrap = self.fixtures()
        bootstrap["binding_strategy"] = "rebind"
        with self.assertRaises(ValueError):
            variants(config, bootstrap, Path("/new-root"))
    def test_refuses_mismatched_boundary_and_keeps_declared_teacher_policy_scope(self):
        config, bootstrap = self.fixtures()
        config['teacher']['reuse_boundary_mode'] = 'explicit'
        config['teacher']['teacher_policy_mode'] = 'semantic_fallback_on_policy_change'
        with self.assertRaisesRegex(ValueError, 'reuse_boundary_mode'):
            variants(config, bootstrap, Path('/new-root'))
        bootstrap['reuse_boundary_mode'] = 'explicit'
        for item in variants(config, bootstrap, Path('/new-root')).values():
            self.assertEqual(item['experiment']['teacher']['reuse_boundary_mode'], 'explicit')
            self.assertEqual(item['bootstrap']['reuse_boundary_mode'], 'explicit')
            self.assertEqual(item['experiment']['teacher']['teacher_policy_mode'], 'semantic_fallback_on_policy_change')
    def test_refuses_different_rollout_contracts_before_writing_configs(self):
        for key, value in (("max_prompt", 8192), ("max_response", 4096),
                           ("max_context", 65536), ("max_turns", 8)):
            config, bootstrap = self.fixtures()
            bootstrap["experiment"]["budget"][key] = value
            with self.assertRaisesRegex(ValueError, "budgets differ"):
                variants(config, bootstrap, Path("/new-root"))
    def test_refuses_fitted_teacher_bootstrap_or_non_collection_rollouts(self):
        config, bootstrap = self.fixtures()
        bootstrap["experiment"]["teacher"] = deepcopy(config["teacher"])
        with self.assertRaisesRegex(ValueError, "already fitted teacher"):
            variants(config, bootstrap, Path("/new-root"))
        config, bootstrap = self.fixtures()
        bootstrap["experiment"]["budget"]["rollouts"] = 8
        with self.assertRaisesRegex(ValueError, "one rollout"):
            variants(config, bootstrap, Path("/new-root"))
        config, bootstrap = self.fixtures()
        config.pop("teacher")
        with self.assertRaisesRegex(ValueError, "explicit teacher"):
            variants(config, bootstrap, Path("/new-root"))
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
