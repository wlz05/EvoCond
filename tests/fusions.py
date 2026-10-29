# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from copy import deepcopy
import unittest
import experiment as experiment
from fusion import variants
class FusionControlTests(unittest.TestCase):
    def config(self):
        return {'format': experiment.FORMAT, 'run_id': 'test_run', 'teacher': {'value_contracts': True},
                'model': 'shared_base', 'datasets': {'train': 'same'}, 'budget': {'seed': 123}}
    def test_only_fusion_and_identity_change(self):
        original = self.config()
        for name, variant in variants(original).items():
            value = deepcopy(variant)
            value['run_id'] = original['run_id']
            value['teacher'].pop('fixed_alpha')
            self.assertEqual(value, original)
            self.assertNotEqual(variant['run_id'], original['run_id'])
            experiment.teacher_options(variant['teacher'])
        self.assertNotIn('fixed_alpha', original['teacher'])
    def test_invalid_or_ambiguous_weights_fail(self):
        for value in (True, -1, .33, float('nan'), 'semantic'):
            with self.assertRaises(ValueError):
                experiment.teacher_options({'fixed_alpha': value})
        config = self.config()
        config['teacher']['fixed_alpha'] = 0.
        with self.assertRaises(ValueError):
            variants(config)
    def test_runtime_propagates_fusion_only_to_teacher(self):
        config = variants(self.config())['semantic']
        config['encoder'] = 'encoder'
        config['budget'] = experiment.settings(config['budget'])
        from pathlib import Path
        runtime = experiment.runtimes({'config': config, 'fingerprints': {'model': {'sha256': 'a'*64},
                                      'official_source': {'sha256': 'b'*64}}}, Path('new_root'))
        self.assertEqual(runtime['teacher']['fixed_alpha'], 0.)
        self.assertNotIn('fixed_alpha', runtime['searl'])
        self.assertNotIn('fixed_alpha', runtime['evocond'])
if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
