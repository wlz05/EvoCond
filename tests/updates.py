# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from control.trajectory import digest, episode_record
from update import config, smoke_group
class QwenPolicyUpdateTest(unittest.TestCase):
    def prepare(self, folder: Path) -> Namespace:
        model = folder / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 2560,
                                                        "num_hidden_layers": 36}), encoding="utf-8")
        (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        rows = []
        for rollout, success in ((0, True), (1, False)):
            result = folder / f"result_{rollout}.json"
            trace = folder / f"trace_{rollout}.jsonl"
            result.write_text(json.dumps({"results": [{"status": "completed", "evaluation": {"success": success}}],
                                          "stats": {"scores": {"cost_metrics": {"total_agent_cost": 100.0}}}}), encoding="utf-8")
            trace.write_text(json.dumps({"action": "state_training_observation", "policy_attempts": [
                {"prompt_ids": [1, 2], "completion_ids": [3, 4], "accepted": True}]})+"\n", encoding="utf-8")
            rows.append(episode_record(result, trace, 0, trace.stat().st_size, "evocond", "q17/none",
                                       rollout, digest(model / "config.json"),
                                       digest(model / "tokenizer_config.json"), "a"*64))
        source = folder / "rollouts.jsonl"
        source.write_text("\n".join(json.dumps(item) for item in rows)+"\n", encoding="utf-8")
        return Namespace(arm="evocond", rollouts=str(source), base_model=str(model), output=str(folder / "adapter"),
                         seed=1, learning_rate=1e-6, lora_r=8, max_prompt=4096, max_completion=2048)
    def test_cpu_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            records, protocol = config(self.prepare(Path(folder)))
            self.assertEqual(len(records), 2)
            self.assertEqual(protocol["data"]["nonzero_advantages"], 2)
            self.assertIn("not_searl_exact", protocol["algorithm"])
            self.assertEqual(smoke_group(records), "q17/none")
    def test_overlength_turn_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            args = self.prepare(Path(folder))
            args.max_prompt = 1
            with self.assertRaisesRegex(ValueError, "token budget"):
                config(args)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
