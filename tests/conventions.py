# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

# Keep maintained Python sources compact, ASCII-only and portable.
import ast
import io
from pathlib import Path
import re
import tokenize
import unittest
ROOT = Path(__file__).resolve().parents[1]
TITLE = 'EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents'
ROLES = {'metric': 'Context Metric', 'relations': 'Conditional Relations', 'dynamic': 'Capability Admission',
    'identity': 'Identity Alignment', 'residual': 'Identity Alignment', 'mixing': 'Directional Fusion',
    'bandit': 'Adoption Teacher', 'subtasks': 'Task Decomposition', 'stage': 'Co Evolution',
    'drift': 'Actor Drift', 'runtime': 'Rollout Integration'}
def maintained_sources():
    paths = [path for path in ROOT.rglob("*.py") if not any(part in {"assets", "runs", "__pycache__", ".git", ".test-tmp", "baselines", "archives"} or part.startswith("tmp") for part in path.relative_to(ROOT).parts[:-1])]
    return paths
def python_sources():
    return [path for path in ROOT.rglob('*.py') if not any(part in {'assets', 'runs', '__pycache__', '.git', '.test-tmp', 'archives'} or
        part.startswith(('tmp', '.evocond-backup', 'SEARL-upstream-snapshot')) for part in path.relative_to(ROOT).parts[:-1])]
class CodeStyleTests(unittest.TestCase):
    def test_filenames_use_one_lowercase_word(self):
        for path in maintained_sources():
            with self.subTest(file=path.relative_to(ROOT.parent)):
                self.assertRegex(path.stem, r"^[a-z]+$")
    def test_official_filenames_use_lowercase_words(self):
        for path in (ROOT/'baselines').rglob('*.py'):
            if any(part in {'__pycache__', '.git', '.test-tmp', 'archives', 'runs', 'assets', 'SEARL-upstream-snapshot'} or part.startswith('.evocond-backup') for part in path.relative_to(ROOT).parts):
                continue
            with self.subTest(file=path.relative_to(ROOT)):
                self.assertRegex(path.stem, r'^[a-z]+$')
    def test_every_source_starts_with_title_and_separator(self):
        for path in python_sources():
            with self.subTest(file=path.relative_to(ROOT)):
                lines = path.read_text(encoding='utf-8-sig').splitlines()
                self.assertGreaterEqual(len(lines), 5)
                self.assertEqual(lines[0], '# -*- coding: utf-8 -*-')
                self.assertEqual(lines[1], '# '+TITLE)
                self.assertEqual(lines[2], '')
                self.assertEqual(lines[-2:], ['# Author: Weiliang Zhu', '# Email: wlzchina05@gmail.com'])
        paper = ROOT.parent/'writing'/'EvoCond.md'
        if paper.is_file():
            self.assertEqual(paper.read_text(encoding='utf-8-sig').splitlines()[0], '# '+TITLE)
    def test_sources_use_ascii_and_single_blank_separators(self):
        for path in maintained_sources():
            with self.subTest(file=path.relative_to(ROOT.parent)):
                source = path.read_text(encoding="utf-8")
                self.assertTrue(source.isascii())
                lines = source.splitlines()
                self.assertFalse(any(not a.strip() and not b.strip() for a, b in zip(lines, lines[1:])))
                ast.parse(source)
    def test_comments_and_docstrings_are_single_line(self):
        for path in maintained_sources():
            with self.subTest(file=path.relative_to(ROOT.parent)):
                source = path.read_text(encoding="utf-8")
                tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
                for token in tokens:
                    if token.type == tokenize.COMMENT:
                        self.assertLessEqual(len(token.string), 160 if token.start[0] == 1 else 100)
                starts = {token.start: token for token in tokens if token.type == tokenize.STRING}
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(node, clean=False) is not None:
                        expression = node.body[0]
                        token = starts[(expression.lineno, expression.col_offset)]
                        self.assertEqual(token.start[0], token.end[0])
    def test_cli_defaults_do_not_embed_absolute_file_paths(self):
        for path in maintained_sources():
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                    continue
                for item in node.keywords:
                    if item.arg != "default":
                        continue
                    value = item.value
                    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "Path" and value.args:
                        value = value.args[0]
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        with self.subTest(file=path.name, line=node.lineno):
                            self.assertFalse(value.value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[/\\]", value.value))
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
