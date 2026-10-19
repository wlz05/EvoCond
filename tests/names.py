# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import importlib
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile
import rename
class NamesTests(unittest.TestCase):
    def fixture(self, root):
        package = root/'pkg'
        package.mkdir()
        (package/'__init__.py').write_text('', encoding='utf-8')
        (package/'base_config.py').write_text('VALUE = 7\n', encoding='utf-8')
        (package/'use.py').write_text('from . import base_config\nfrom .base_config import VALUE\nresult = base_config.VALUE + VALUE\n', encoding='utf-8')
        (root/'main.py').write_text('import pkg.base_config\nfrom pkg import base_config\nresult = pkg.base_config.VALUE + base_config.VALUE\n', encoding='utf-8')
    def test_plan_and_import_bindings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            mapping = rename.plan(root)
            modules = rename.modules(root, mapping)
            main = root/'main.py'
            use = root/'pkg/use.py'
            changed = rename.rewrite(main, main.read_text(), root, mapping, modules)
            self.assertIn('import pkg.base', changed)
            self.assertIn('from pkg import base as base_config', changed)
            self.assertIn('pkg.base.VALUE + base_config.VALUE', changed)
            changed = rename.rewrite(use, use.read_text(), root, mapping, modules)
            self.assertIn('from . import base as base_config', changed)
            self.assertIn('from .base import VALUE', changed)
    def test_attributes_and_strings_do_not_rename_external_symbols(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            mapping = rename.plan(root)
            text = 'import torch\nstate_dict = torch.state_dict\nname = "pkg.base_config"\npath = "pkg/base_config.py"\n'
            changed = rename.rewrite(root/'main.py', text, root, mapping, rename.modules(root, mapping))
            self.assertIn('state_dict = torch.state_dict', changed)
            self.assertIn('"pkg.base"', changed)
            self.assertIn('"pkg/base.py"', changed)
    def test_bare_module_imports_preserve_the_old_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'tools').mkdir()
            old = root/'tools/isolated_dispatch.py'
            old.write_text('VALUE = 1\n')
            mapping = rename.plan(root)
            text = 'from isolated_dispatch import VALUE\nimport isolated_dispatch\nresult = isolated_dispatch.VALUE\n'
            changed = rename.rewrite(root/'main.py', text, root, mapping, rename.modules(root, mapping))
            self.assertIn('from dispatch import VALUE', changed)
            self.assertIn('import dispatch as isolated_dispatch', changed)
            self.assertIn('isolated_dispatch.VALUE', changed)
    def test_migration_has_verified_backup_and_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            receipt, backup = root/'configs/names.json', root/'configs/original.zip'
            before = (root/'pkg/base_config.py').read_bytes()
            readonly = rename.migrate(root)
            self.assertEqual(readonly['renamed'], 1)
            self.assertTrue((root/'pkg/base_config.py').is_file())
            result = rename.migrate(root, run=True, receipt=receipt, backup=backup)
            self.assertFalse((root/'pkg/base_config.py').exists())
            self.assertEqual((root/'pkg/base.py').read_bytes(), before)
            with ZipFile(backup) as archive:
                self.assertEqual(archive.read('pkg/base_config.py'), before)
            self.assertEqual(json.loads(receipt.read_text())['renamed'], 1)
            self.assertTrue(result['applied'])
            with self.assertRaises(ValueError):
                rename.migrate(root, run=True, receipt=receipt, backup=backup)
    def test_collision_and_special_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            (root/'pkg/base.py').write_text('VALUE = 1\n')
            (root/'__main__.py').write_text('pass\n')
            (root/'try.py').write_text('pass\n')
            mapping = rename.plan(root)
            self.assertEqual(mapping[root/'pkg/base_config.py'].name, 'bases.py')
            self.assertNotIn(root/'__main__.py', mapping)
            self.assertNotIn(root/'try.py', mapping)
    def test_outside_receipt_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            with self.assertRaises(ValueError):
                rename.migrate(root, run=True, receipt=root.parent/'outside.json', backup=root/'backup.zip')
            self.assertTrue((root/'pkg/base_config.py').is_file())
if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
