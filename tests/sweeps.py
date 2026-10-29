# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Bounded sensitivity recipes keep budgets and sealed-test inputs unchanged.'
from copy import deepcopy
import json
from pathlib import Path
import unittest
import sweep
class SweepTests(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).resolve().parents[1]/'configs/contextual.json').read_text(encoding='utf-8'))
    def test_one_factor_sweep_keeps_budget_and_splits(self):
        base = self.config()
        rows = sweep.recipes(base)
        self.assertLess(len(rows), 30)
        self.assertEqual(len({row['config']['run_id'] for row in rows}), len(rows))
        for row in rows:
            self.assertEqual(row['config']['budget'], base['budget'])
            self.assertEqual(row['config']['datasets'], base['datasets'])
    def test_only_explicit_ablation_can_remove_residual(self):
        rows = {row['name']: row['config'] for row in sweep.recipes(self.config())}
        self.assertEqual(rows['unprotected']['contextual']['metric']['residual'], 0)
        self.assertEqual(rows['identity']['contextual']['metric']['residual'], 1)
        self.assertEqual(rows['base']['contextual']['metric']['protection'], 'adaptive')
    def test_absolute_paths_are_not_generated(self):
        config = self.config()
        config['model'] = 'C:/models/qwen'
        with self.assertRaises(ValueError):
            sweep.recipes(config)
    def test_recipes_are_deterministic_and_do_not_mutate_input(self):
        config = self.config()
        original = deepcopy(config)
        self.assertEqual(sweep.recipes(config), sweep.recipes(config))
        self.assertEqual(config, original)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
