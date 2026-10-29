# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

# Keep maintained Python sources compact, ASCII-only and portable.
import ast
from collections import Counter, defaultdict
import io
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import tokenize
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import MagicMock, patch
import warnings
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

def native_tree(path):
    return ast.parse((ROOT/'baselines/SEARL-official'/path).read_text(encoding='utf-8'))

def native_method(path, class_name, method_name, scope):
    cls = next(node for node in native_tree(path).body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), path, 'exec'), scope)
    return scope[method_name]
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
                self.assertEqual(lines[-2:], ['# Author: Weiliang Zhu; Other main author: Yaming Lv', '# Email: wlzchina05@gmail.com'])
        paper = ROOT.parent/'writing'/'EvoCond.md'
        if paper.is_file():
            self.assertEqual(paper.read_text(encoding='utf-8-sig').splitlines()[0], '# '+TITLE)
    def test_sources_use_ascii_and_single_blank_separators(self):
        for path in python_sources():
            with self.subTest(file=path.relative_to(ROOT.parent)):
                source = path.read_text(encoding="utf-8")
                self.assertTrue(source.isascii())
                lines = source.splitlines()
                self.assertFalse(any(not a.strip() and not b.strip() for a, b in zip(lines, lines[1:])))
                ast.parse(source)
    def test_decoded_literals_do_not_hide_chinese_prose(self):
        for path in python_sources():
            with warnings.catch_warnings(), self.subTest(file=path.relative_to(ROOT)):
                warnings.simplefilter('error', SyntaxWarning)
                tree = ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        self.assertNotRegex(node.value, r'[\u3400-\u9fff]', msg=f'{path.name}:{node.lineno}')

    def test_legacy_configuration_categories_remain_readable(self):
        for template, environment in (('template', 'environment'),
                (bytes.fromhex('e6a8a1e69dbfe5ad97e6aeb5').decode(), bytes.fromhex('e78eafe5a283e5ad97e6aeb5').decode())):
            config = dict(params=[dict(name='pattern', category=template),
                dict(name='timeout', category=environment, default=10)])
            scope = dict(ConfigParser=SimpleNamespace(load_config=lambda _: config), re=re)
            build = native_method('envs/tool_manager/configuration.py', 'ConfigManager', '_build_tools', scope)
            manager = SimpleNamespace(verl_config=SimpleNamespace(config_path='data/tools.json'), _create_tools=lambda _: [])
            build(manager)
            self.assertIn('pattern', manager.template_fields)
            self.assertIn('timeout', manager.env_fields)
            self.assertEqual(manager.timeout, 10)

    def test_generated_mcp_source_keeps_executable_markers_and_arguments(self):
        tree = native_tree('envs/tools/legacy.py')
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'create_and_execute_mcp')
        assignment = next(node for node in ast.walk(function) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'execution_script' for target in node.targets))
        scope = dict(import_block='', code='def add(x, y):\n    return x+y', name='add', inputs_json=json.dumps(dict(x=1, y=2)))
        source = eval(compile(ast.Expression(body=assignment.value), 'mcp-source', 'eval'), scope)
        ast.parse(source, feature_version=(3, 10))
        self.assertIn("target_func = globals()['add']", source)
        self.assertIn('__MCP_CREATION_SUCCESS__', source)
        self.assertIn('__MCP_RESULT_START__{json.dumps(result)}__MCP_RESULT_END__', source)

    def test_graph_cache_defaults_are_relative_and_explicit_paths_are_preserved(self):
        scope = dict(SentenceTransformer=MagicMock(), TfidfVectorizer=MagicMock(), threading=threading,
            nx=SimpleNamespace(DiGraph=MagicMock()), defaultdict=defaultdict, Counter=Counter, Path=Path, Optional=Optional)
        initialize = native_method('envs/tool_manager/centralized/retrieval.py', 'GraphRAGRetrieval', '__init__', scope)
        for value, expected in ((None, 'cache/graphrag'), ('outputs/job/cache', 'outputs/job/cache')):
            instance = SimpleNamespace()
            with patch.object(Path, 'mkdir'), patch('builtins.print'):
                initialize(instance, 'assets/models/e5-base-v2', device='cpu', cache_dir=value)
            self.assertEqual(instance.cache_dir.as_posix(), expected)
            self.assertFalse(instance.cache_dir.is_absolute())

    def test_ipc_lock_and_endpoint_use_the_system_temporary_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            bind, lock = MagicMock(), MagicMock()
            socket = SimpleNamespace(bind=bind)
            scope = dict(os=os, tempfile=SimpleNamespace(gettempdir=lambda: folder), FileLock=lock,
                zmq=SimpleNamespace(Context=lambda: SimpleNamespace(socket=lambda _: socket), REP=1),
                threading=SimpleNamespace(Thread=MagicMock()))
            initialize = native_method('verl/workers/rollout/vllm_rollout/rollout.py', 'vLLMAsyncRollout', '_init_zeromq', scope)
            instance = SimpleNamespace(config=SimpleNamespace(tensor_model_parallel_size=1), _loop_forever=lambda: None)
            with patch.dict(os.environ, RAY_LOCAL_WORLD_SIZE='1'):
                address = initialize(instance)
            expected = 'ipc://'+os.path.join(folder, f'verl_vllm_zmq_{os.getpid()}.ipc')
            self.assertEqual(address, expected)
            bind.assert_called_once_with(expected)
            lock.assert_called_once_with(os.path.join(folder, 'verl_vllm_zmq.lock'))
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
        for path in python_sources():
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
                            self.assertFalse(value.value.startswith(("/", "\\", "~/")) or re.match(r"^[A-Za-z]:[/\\]", value.value))
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
