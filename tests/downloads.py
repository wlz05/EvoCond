# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import assets as assets
class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}
class AssetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.data = b"small pinned asset\n"
        self.spec = {"key": "test", "category": "datasets", "repo": "owner/repo",
                     "revision": "a" * 40, "type": "datasets", "folder": "data/test"}
        self.entry = {"name": "train.parquet", "size": len(self.data), "hash": "sha256",
                      "checksum": hashlib.sha256(self.data).hexdigest()}
        self.spec["files"] = [self.entry]
        self.target = self.root / "data/test/train.parquet"
    def test_completed_download_is_verified_without_duplicate_copy(self):
        assets.download_one(self.root, self.spec, self.entry, reserve_gib=0,
                            opener=lambda *a, **k: Response(self.data))
        self.assertTrue(assets.verify(self.target, self.entry))
        self.assertFalse(assets.partial_path(self.root, self.spec, self.entry).exists())
    def test_resume_validates_range_and_appends(self):
        partial = assets.partial_path(self.root, self.spec, self.entry)
        partial.parent.mkdir()
        partial.write_bytes(self.data[:4])
        def opener(request, **kwargs):
            self.assertEqual(request.get_header("Range"), "bytes=4-")
            return Response(self.data[4:], 206, {"Content-Range": f"bytes 4-{len(self.data)-1}/{len(self.data)}"})
        assets.download_one(self.root, self.spec, self.entry, reserve_gib=0, opener=opener)
        self.assertEqual(self.target.read_bytes(), self.data)
    def test_ignored_range_restarts_only_partial(self):
        partial = assets.partial_path(self.root, self.spec, self.entry)
        partial.parent.mkdir()
        partial.write_bytes(self.data[:4])
        assets.download_one(self.root, self.spec, self.entry, reserve_gib=0,
                            opener=lambda *a, **k: Response(self.data))
        self.assertEqual(self.target.read_bytes(), self.data)
    def test_bad_range_preserves_partial(self):
        partial = assets.partial_path(self.root, self.spec, self.entry)
        partial.parent.mkdir()
        partial.write_bytes(self.data[:4])
        with self.assertRaisesRegex(ValueError, "Content-Range"):
            assets.download_one(self.root, self.spec, self.entry, reserve_gib=0,
                                opener=lambda *a, **k: Response(self.data[4:], 206, {"Content-Range": "bytes 0-10/11"}))
        self.assertEqual(partial.read_bytes(), self.data[:4])
    def test_incomplete_download_retains_resumable_prefix(self):
        with self.assertRaisesRegex(OSError, "Incomplete"):
            assets.download_one(self.root, self.spec, self.entry, reserve_gib=0,
                                opener=lambda *a, **k: Response(self.data[:4]))
        self.assertFalse(self.target.exists())
        self.assertEqual(assets.partial_path(self.root, self.spec, self.entry).read_bytes(), self.data[:4])
    def test_existing_mismatch_is_never_overwritten(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"x" * len(self.data))
        with self.assertRaisesRegex(ValueError, "preserved"):
            assets.download_one(self.root, self.spec, self.entry, reserve_gib=0)
        self.assertEqual(self.target.read_bytes(), b"x" * len(self.data))
    def test_checksum_failure_is_not_promoted(self):
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            assets.download_one(self.root, self.spec, self.entry, reserve_gib=0,
                                opener=lambda *a, **k: Response(b"x" * len(self.data)))
        self.assertFalse(self.target.exists())
    def test_existing_valid_file_skips_network(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(self.data)
        assets.download_one(self.root, self.spec, self.entry, reserve_gib=0,
                            opener=lambda *a, **k: self.fail("Unexpected network request"))
    def test_disk_check_happens_before_writes(self):
        with patch.object(assets, "free_bytes", return_value=0):
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                assets.prepare(self.root, [self.spec], False, 0)
        self.assertEqual(list(self.root.iterdir()), [])
    def test_plan_is_read_only_and_excludes_wiki_by_default(self):
        absent = self.root / "absent"
        result = assets.plan(absent, [self.spec], 0)
        self.assertFalse(absent.exists())
        self.assertFalse(result["training_started"])
        self.assertNotIn("wiki", {s["category"] for s in assets.selected_specs(assets.DEFAULT_ASSETS)})
    def test_git_blob_checksum(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(self.data)
        entry = dict(self.entry, hash="gitblob",
                     checksum=hashlib.sha1(b"blob " + str(len(self.data)).encode() + b"\0" + self.data).hexdigest())
        self.assertTrue(assets.verify(self.target, entry))
    def test_traversal_is_rejected(self):
        with self.assertRaises(ValueError):
            assets.safe_path(self.root, "../outside")
    def test_exclusive_lock_rejects_concurrent_run_and_is_released(self):
        with assets.exclusive_run(self.root):
            with self.assertRaisesRegex(ValueError, "Another downloader"):
                with assets.exclusive_run(self.root):
                    self.fail("Concurrent downloader")
        self.assertFalse((self.root / '.assets.lock').exists())
    def test_catalogue_contains_seven_primary_benchmarks_and_correct_judge(self):
        evaluation = next(s for s in assets.CATALOGUE if s["key"] == "arpo_evaluation")
        names = {e["name"].split("/")[0] for e in evaluation["files"]}
        self.assertTrue({"gsm8k", "math500", "aime24", "hotpotqa", "2wiki", "musique", "bamboogle"}.issubset(names))
        judge = next(s for s in assets.CATALOGUE if s["key"] == "judge")
        self.assertEqual(judge["repo"], "Qwen/Qwen3-32B")
        self.assertEqual(sum(e["name"].endswith(".safetensors") for e in judge["files"]), 17)
    def test_cli_no_run_neither_downloads_nor_creates_root(self):
        absent = self.root / "absent"
        result = subprocess.run([sys.executable, "-B", assets.__file__, "--root", str(absent),
                                 "--assets", "datasets"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("READ_ONLY", result.stdout)
        self.assertFalse(absent.exists())
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
