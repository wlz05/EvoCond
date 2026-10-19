# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import json
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from overlay import PIN, digest, install, safe_path
class OverlayTests(unittest.TestCase):
    def fixture(self, base):
        root, overlay = base / "repo", base / "overlay"
        for p in (root / "envs", overlay / "envs"):
            p.mkdir(parents=True)
        old, new = root / "envs/a.py", overlay / "envs/a.py"
        old.write_text("old", encoding="utf-8")
        new.write_text("new", encoding="utf-8")
        (overlay / "manifest.json").write_text(json.dumps({"format": "evocond_searl_overlay_v1",
            "upstream_commit": PIN, "files": {"envs/a.py": {"base_sha256": digest(old), "sha256": digest(new)}}}))
        return root, overlay
    @patch('overlay.subprocess.check_output', return_value=PIN)
    def test_readonly_install_preserves_backup_and_idempotence(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root, overlay = self.fixture(Path(temp))
            self.assertEqual(install(root, overlay)["files_to_install"], 1)
            self.assertEqual((root / "envs/a.py").read_text(), "old")
            result = install(root, overlay, True)
            self.assertEqual((root / "envs/a.py").read_text(), "new")
            self.assertEqual((Path(result["backup"]) / "envs/a.py").read_text(), "old")
            self.assertEqual(install(root, overlay, True)["files_to_install"], 0)
    @patch('overlay.subprocess.check_output', return_value=PIN)
    def test_unknown_local_edits_are_preserved(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root, overlay = self.fixture(Path(temp))
            (root / "envs/a.py").write_text("user edit")
            with self.assertRaisesRegex(ValueError, "local edits"):
                install(root, overlay, True)
            self.assertEqual((root / "envs/a.py").read_text(), "user edit")
    @patch('overlay.subprocess.check_output', return_value=PIN)
    def test_overlay_checksum_rejected(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root, overlay = self.fixture(Path(temp))
            (overlay / "envs/a.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                install(root, overlay, True)
    def test_path_confinement(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ("../outside", "envs/../../outside", "/etc/passwd", "C:/x", ".git/config", "envs\\x.py"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    safe_path(Path(temp), name)
    @patch('overlay.subprocess.check_output', return_value=PIN)
    def test_renamed_sources_are_backed_up_and_idempotent(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root, overlay = self.fixture(Path(temp))
            old = root/'envs/old_name.py'
            old.write_text('original')
            manifest = json.loads((overlay/'manifest.json').read_text())
            manifest.update(format='evocond_searl_overlay_v2', deleted={'envs/old_name.py': {'base_sha256': digest(old)}})
            (overlay/'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(install(root, overlay)['files_to_install'], 2)
            self.assertTrue(old.exists())
            result = install(root, overlay, True)
            self.assertFalse(old.exists())
            self.assertEqual((Path(result['backup'])/'envs/old_name.py').read_text(), 'original')
            self.assertEqual(install(root, overlay, True)['files_to_install'], 0)
    @patch('overlay.subprocess.check_output', return_value=PIN)
    def test_renamed_local_edits_are_not_removed(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root, overlay = self.fixture(Path(temp))
            old = root/'envs/old_name.py'
            old.write_text('user edit')
            manifest = json.loads((overlay/'manifest.json').read_text())
            manifest.update(format='evocond_searl_overlay_v2', deleted={'envs/old_name.py': {'base_sha256': 'unknown'}})
            (overlay/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'local edits'):
                install(root, overlay, True)
            self.assertEqual(old.read_text(), 'user edit')
            self.assertEqual((root/'envs/a.py').read_text(), 'old')
    @patch('overlay.subprocess.check_output', return_value=PIN)
    def test_git_crlf_conversion_is_not_mistaken_for_user_edits(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root, overlay = self.fixture(Path(temp))
            target = root/'envs/a.py'
            original = b'first\r\nsecond\r\n'
            target.write_bytes(original)
            manifest = json.loads((overlay/'manifest.json').read_text())
            manifest['files']['envs/a.py']['base_sha256'] = sha256(original.replace(b'\r\n', b'\n')).hexdigest()
            (overlay/'manifest.json').write_text('\ufeff'+json.dumps(manifest), encoding='utf-8')
            result = install(root, overlay, True)
            self.assertEqual((Path(result['backup'])/'envs/a.py').read_bytes(), original)
            self.assertEqual(install(root, overlay, True)['files_to_install'], 0)
    @patch('overlay.subprocess.check_output', return_value=PIN)
    def test_crlf_does_not_hide_a_content_edit(self, _):
        with tempfile.TemporaryDirectory() as temp:
            root, overlay = self.fixture(Path(temp))
            (root/'envs/a.py').write_bytes(b'old user edit\r\n')
            with self.assertRaisesRegex(ValueError, 'local edits'):
                install(root, overlay, True)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
