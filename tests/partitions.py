# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import json
from pathlib import Path
import tempfile
import unittest
from splits import audit
class SplitAuditTests(unittest.TestCase):
    def test_exact_duplicate_across_modes_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = {split: root / f"{split}.jsonl" for split in ("train", "validation", "test")}
            paths["train"].write_text(json.dumps({"prompt": [{"role": "user", "content": "Who is A?"}]}) + "\n")
            paths["validation"].write_text(json.dumps({"prompt": [{"role": "user", "content": "who  is A?"}]}) + "\n")
            paths["test"].write_text(json.dumps({"prompt": [{"role": "user", "content": "Who is B?"}]}) + "\n")
            result = audit(paths, "prompt")
            self.assertFalse(result["disjoint"])
            self.assertEqual(result["exact_overlap"]["train_validation"], 1)
    def test_unicode_identity_matches_dataset_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = {split: root/f'{split}.json' for split in ('train', 'validation', 'test')}
            questions = ['Hello world', '\uff28\uff45\uff4c\uff4c\uff4f world', 'Other']
            for path, question in zip(paths.values(), questions):
                path.write_text(json.dumps([{'question': question}]), encoding='utf-8-sig')
            self.assertEqual(audit(paths, 'question')['exact_overlap']['train_validation'], 1)
    def test_duplicate_rows_and_duplicate_real_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = {split: root/f'{split}.jsonl' for split in ('train', 'validation', 'test')}
            for split, path in paths.items():
                path.write_text(json.dumps({'question': split})+'\n', encoding='utf-8')
            paths['train'].write_text('{"question":"Q"}\n{"question":"q"}\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Duplicate normalized question'):
                audit(paths, 'question')
            paths['validation'] = root/'folder/../train.jsonl'
            (root/'folder').mkdir()
            with self.assertRaisesRegex(ValueError, 'paths must differ'):
                audit(paths, 'question')
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
