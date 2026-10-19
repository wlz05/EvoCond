# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from namespace import Initializers, install
from run import SEEDS, recipes
from source import reviewed
from experiment import directory_digest

class PublicationTests(unittest.TestCase):
    def test_eight_gpus_and_all_four_seeds_preserve_partitions(self):
        value = dict(datasets=dict(train='assets/train.parquet'), budget=dict(gpus=4))
        before = deepcopy(value)
        plans = recipes(value)
        self.assertEqual(tuple(seed for seed, _ in plans), (42, 52, 3047, 3407))
        self.assertEqual(SEEDS, (42, 52, 3047, 3407))
        self.assertEqual(value, before)
        for seed, plan in plans:
            self.assertEqual(plan['budget'], dict(gpus=8, seed=seed))
            self.assertEqual(plan['contextual']['seed'], seed)
            self.assertEqual(plan['datasets'], before['datasets'])

    def test_initializer_preserves_exports_and_relative_imports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'fixture').mkdir()
            (root/'fixture/initialize.py').write_text('from .value import number\n', encoding='utf-8')
            (root/'fixture/value.py').write_text('number = 7\n', encoding='utf-8')
            finder = Initializers(root, [dict(module='fixture', path='fixture/initialize.py', package=True)])
            with patch.object(sys, 'path', [str(root), *sys.path]), patch.object(sys, 'meta_path', [finder, *sys.meta_path]):
                try:
                    self.assertEqual(importlib.import_module('fixture').number, 7)
                finally:
                    sys.modules.pop('fixture.value', None)
                    sys.modules.pop('fixture', None)

    def test_initializer_rejects_escape_and_missing_source(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                Initializers(folder, [dict(module='fixture', path='../missing.py', package=True)])

    def test_vendored_source_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payload = b'number = 7\n'
            (root/'value.py').write_bytes(payload)
            receipt = dict(format='evocond_vendored_source_v1', upstream_commit='pin',
                files={'value.py': sha256(payload).hexdigest()})
            (root/'source.json').write_text(json.dumps(receipt), encoding='utf-8')
            self.assertEqual(reviewed(root, 'pin'), 'pin')
            (root/'value.py').write_bytes(b'number = 8\n')
            with self.assertRaises(ValueError):
                reviewed(root, 'pin')

    def test_vendored_source_pin_must_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'source.json').write_text(json.dumps(dict(format='evocond_vendored_source_v1',
                upstream_commit='other', files={'value.py': 'x'})), encoding='utf-8')
            with self.assertRaises(ValueError):
                reviewed(root, 'pin')

    def test_package_manifest_is_frozen_with_method_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'configs').mkdir()
            manifest = root/'configs/packages.json'
            manifest.write_text('{}', encoding='utf-8')
            before = directory_digest(root)
            manifest.write_text('{"entries":[]}', encoding='utf-8')
            self.assertNotEqual(directory_digest(root), before)

if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
