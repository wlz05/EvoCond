# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import bench
import serve
class StateExperimentCliTests(unittest.TestCase):
    def server(self, **changes):
        values = dict(state_experiment="selective", state_controller="frozen.json",
                      evocond=True, noaudit=True, noplan=False, levels="full",
                      gate=None, memory=None, state_value_audit=False, training_temperature=0.0)
        return Namespace(**(values | changes))
    def client(self, **changes):
        values = dict(state_experiment="selective", state_controller="frozen.json",
                      state_split="validation", state_value_audit=False)
        return Namespace(**(values | changes))
    def test_old_cli_defaults_are_preserved(self):
        with patch("sys.argv", ["serve.py"]):
            self.assertEqual(serve.arguments().state_experiment, "legacy")
        with patch("sys.argv", ["bench.py"]):
            self.assertEqual(bench.arguments().state_experiment, "legacy")
    def test_new_server_requires_unconfounded_settings(self):
        serve.check_state_experiment(self.server())
        for changes in ({"state_controller": None}, {"memory": "old.json"},
                        {"noaudit": False}, {"noplan": True}, {"gate": "gate.json"},
                        {"state_value_audit": True}, {"training_temperature": 0.7}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                serve.check_state_experiment(self.server(**changes))
    def test_service_mode_must_match(self):
        bench.check_state_experiment(self.client(), {"state_experiment": "selective"})
        for actual in (None, "legacy", "baseline", "prompt_only"):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                bench.check_state_experiment(self.client(), {"state_experiment": actual})
    def test_sealed_test_and_audit_mix_rejected(self):
        for changes in ({"state_split": "test"}, {"state_controller": None},
                        {"state_value_audit": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                bench.check_state_experiment(self.client(**changes), {"state_experiment": "selective"})
    def test_old_service_without_field_remains_legacy(self):
        bench.check_state_experiment(self.client(state_experiment="legacy"), {})
    def test_new_output_records_mode_and_refuses_reuse(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            queries = root / "queries.json"
            queries.write_text("[]", encoding="utf-8")
            args = self.client(output=str(root / "new"), state_policy="learned", start=1, end=2,
                               blocker=False, blockmode="preference_change", tokens=2048, refinement=2)
            health = {"state_experiment_source_sha256": serve.experiment_source_hashes(),
                      "state_experiment": "selective"}
            bench.prepare_experiment_output(args, health, queries)
            report = json.loads((root / "new/diagnostic_protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(report["experiment"], "selective")
            self.assertFalse(report["fresh_test"])
            with self.assertRaises(FileExistsError):
                bench.prepare_experiment_output(args, health, queries)
            self.assertEqual(report, json.loads((root / "new/diagnostic_protocol.json").read_text(encoding="utf-8")))
    def test_mismatched_code_and_existing_empty_output_refused(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.client(output=str(root / "new"))
            with self.assertRaises(ValueError):
                bench.prepare_experiment_output(args, {}, root / "absent.json")
            self.assertFalse((root / "new").exists())
            (root / "new").mkdir()
            with self.assertRaises(FileExistsError):
                bench.prepare_experiment_output(args, {"state_experiment_source_sha256": serve.experiment_source_hashes()}, root / "absent.json")
    def test_legacy_output_unchanged(self):
        bench.prepare_experiment_output(self.client(state_experiment="legacy"), {}, Path("absent.json"))
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
