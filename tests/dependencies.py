# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Dependency audits must not install packages or load GPU libraries.'
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import environment
import install
class DependencyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)/'requirements.txt'
    @patch('environment.importlib.metadata.version', return_value='1.0')
    def test_local_requirements_are_not_silently_converted(self, _):
        self.path.write_text('torch==1.0\nname @ file:///build/local\n', encoding='utf-8')
        result = environment.inspect(self.path)
        self.assertEqual(result['unresolved'][0]['line'], 2)
        self.assertFalse(result['portable_exact_lock'])
        self.assertTrue(result['core']['torch']['matches'])
        with self.assertRaises(ValueError):
            environment.lock(self.path)
    def test_normalized_duplicate_distributions_are_rejected(self):
        self.path.write_text('sentence_transformers==1\nsentence-transformers==2\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            environment.inspect(self.path)
    def test_complete_portable_exact_lock_is_explicit(self):
        self.path.write_text('\ufeff'+'\r\n'.join(name+'==1.0' for name in environment.CORE), encoding='utf-8')
        result = environment.lock(self.path)
        self.assertTrue(result['portable_exact_lock'])
        self.assertFalse(result['compatibility_certified'])
        self.assertFalse(result['gpu_verified'])
    def test_default_install_is_readonly(self):
        with patch.object(sys, 'argv', ['install.py']), patch.object(install.subprocess, 'run') as process, \
                redirect_stdout(io.StringIO()):
            install.main()
        process.assert_not_called()
    def test_native_without_portable_lock_cannot_install(self):
        with patch.object(sys, 'argv', ['install.py', '--run']), patch.object(install.subprocess, 'run') as process, \
                redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'portable exact lock'):
            install.main()
        process.assert_not_called()
    def test_legacy_is_explicit_and_dry_by_default(self):
        with patch.object(sys, 'argv', ['install.py', '--profile', 'legacy']), \
                patch.object(install.subprocess, 'run') as process, redirect_stdout(io.StringIO()):
            install.main()
        process.assert_not_called()
if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
