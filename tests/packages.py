# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Source packages must be portable, complete and non-destructive.'
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile, ZipInfo
import reproduce
class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
    def archive(self, values=None, *, extra=None, duplicates=False):
        values = {'EvoCond/main.py': b'VALUE = 1\n'} if values is None else values
        manifest = dict(format='evocond_source_package_v1', gpu_verified=False,
                        files={name: sha256(content).hexdigest() for name, content in values.items()})
        path = self.root/'source.zip'
        with ZipFile(path, 'w') as bundle:
            for name, content in values.items():
                bundle.writestr(name, content)
            bundle.writestr('PACKAGE.json', json.dumps(manifest))
            if extra is not None:
                bundle.writestr(*extra)
            if duplicates:
                bundle.writestr('evocond/MAIN.py', b'VALUE = 1\n')
        return path
    def test_verify_then_unpack_into_new_directory(self):
        path, output = self.archive(), self.root/'new'
        self.assertFalse(reproduce.unpack(path, output)['extracted'])
        self.assertFalse(output.exists())
        self.assertTrue(reproduce.unpack(path, output, True)['extracted'])
        self.assertEqual((output/'EvoCond/main.py').read_text(), 'VALUE = 1\n')
        with self.assertRaises(FileExistsError):
            reproduce.unpack(path, output, True)
    def test_unlisted_content_and_portable_collisions_are_refused(self):
        for options in ({'extra': ('extra.py', b'pass')}, {'duplicates': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                reproduce.verify(self.archive(**options))
    def test_invalid_python_or_shell_format_is_refused(self):
        for values in ({'EvoCond/old_name.py': b'pass'}, {'EvoCond/main.py': b'if'},
                       {'official_overlay/main.sh': b'echo ok\r\n'}):
            with self.subTest(values=values), self.assertRaises((ValueError, SyntaxError)):
                reproduce.verify(self.archive(values))
    def test_path_escape_and_symlink_are_refused(self):
        for name in ('../outside.py', 'EvoCond//main.py', 'C:/main.py', 'EvoCond\\main.py'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                reproduce.verify(self.archive({name: b'pass'}))
        path = self.archive()
        link = ZipInfo('EvoCond/link.py')
        link.create_system = 3
        link.external_attr = (0o120777 << 16)
        with ZipFile(path, 'a') as bundle:
            bundle.writestr(link, b'../outside')
        with self.assertRaises(ValueError):
            reproduce.verify(path)
    def test_bad_checksum_never_creates_destination(self):
        path = self.archive()
        output = self.root/'new'
        with ZipFile(path, 'a') as bundle:
            bundle.writestr('untracked.py', b'pass')
        with self.assertRaises(ValueError):
            reproduce.unpack(path, output, True)
        self.assertFalse(output.exists())
if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
