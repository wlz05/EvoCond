# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

# Keep maintained Python sources compact, ASCII-only and portable.
import ast
from collections import Counter, defaultdict
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import re
import sys
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

    def test_replacing_graph_tools_removes_stale_indices_and_history(self):
        scope = dict(Dict=dict, Any=object, Optional=Optional, nx=SimpleNamespace(DiGraph=object), Counter=Counter)
        load = native_method('envs/tool_manager/centralized/retrieval.py', 'GraphRAGRetrieval', 'load_tools', scope)
        instance = SimpleNamespace(lock=threading.RLock(), mcps={'old': {}}, tool_embeddings={'old': [1.]},
            tfidf_matrix=object(), tfidf_mcp_ids=['old'], communities={'stale': {'old'}},
            community_summaries={'stale': 'old'}, global_summary='old', enable_community_detection=True,
            execution_history={'old': {}, 'new': {'co_occurrence': {'old': 1, 'new': 2}}},
            _build_embeddings=MagicMock(), _build_tfidf_index=MagicMock(), _detect_communities=MagicMock(),
            _build_community_summaries=MagicMock(), _build_global_summary=MagicMock())
        graph = MagicMock()
        with patch('builtins.print'):
            load(instance, {'new': {}}, graph)
        self.assertEqual(instance.tool_embeddings, {})
        self.assertEqual(instance.tfidf_mcp_ids, [])
        self.assertIsNone(instance.tfidf_matrix)
        self.assertEqual(instance.communities, {})
        self.assertEqual(instance.community_summaries, {})
        self.assertEqual(instance.global_summary, '')
        self.assertEqual(instance.execution_history, {'new': {'co_occurrence': Counter(new=2)}})
        instance._build_embeddings.assert_called_once()
        with patch('builtins.print'):
            load(instance, {}, graph)
        self.assertEqual(instance.execution_history, {})

    def test_graph_cache_load_replaces_history_without_reusing_old_entries(self):
        scope = dict(Optional=Optional, Path=Path, Counter=Counter)
        load = native_method('envs/tool_manager/centralized/retrieval.py', 'GraphRAGRetrieval', 'load_cache', scope)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'cache.json'
            cache = dict(execution_history={'new': {'success_count': 3, 'co_occurrence': {'new': 1}}},
                communities={'c': ['new']}, community_summaries={'c': 'new'}, global_summary='new')
            path.write_text(json.dumps(cache), encoding='utf-8')
            instance = SimpleNamespace(lock=threading.RLock(), mcps={'new': {}},
                execution_history=defaultdict(dict, new={'failure_count': 99}, old={'success_count': 99}),
                communities={}, community_summaries={}, global_summary='')
            with patch('builtins.print'):
                load(instance, str(path))
            self.assertEqual(dict(instance.execution_history), cache['execution_history'])
            self.assertEqual(instance.communities, {'c': {'new'}})
            original = deepcopy(dict(instance.execution_history))
            for field in ('execution_history', 'communities', 'community_summaries'):
                broken = deepcopy(cache)
                if field == 'execution_history':
                    broken[field]['missing'] = {}
                elif field == 'communities':
                    broken[field]['c'] = ['missing']
                else:
                    broken[field]['stale'] = 'old summary'
                path.write_text(json.dumps(broken), encoding='utf-8')
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load(instance, str(path))
                self.assertEqual(dict(instance.execution_history), original)

    def test_empty_lexical_vocabulary_keeps_semantic_retrieval_available(self):
        method = native_method('envs/tool_manager/centralized/retrieval.py', 'GraphRAGRetrieval', '_build_tfidf_index', {})
        instance = SimpleNamespace(mcps={'tool': {'description': 'the'}}, tfidf_vectorizer=MagicMock(),
            tfidf_matrix=object(), tfidf_mcp_ids=['old'])
        instance.tfidf_vectorizer.fit_transform.side_effect = ValueError('empty vocabulary; stop words only')
        with patch('builtins.print'):
            method(instance)
        self.assertIsNone(instance.tfidf_matrix)
        self.assertEqual(instance.tfidf_mcp_ids, [])
        instance.tfidf_vectorizer.fit_transform.side_effect = ValueError('unrelated malformed index')
        with self.assertRaisesRegex(ValueError, 'unrelated'):
            method(instance)

    def test_native_restore_rejects_missing_iterator_before_loading_workers(self):
        from tests.experiments import ExperimentTests
        fixture = ExperimentTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        folder = fixture.checkpoint()
        (folder/'data.pt').unlink()
        torch = SimpleNamespace(load=MagicMock())
        method = native_method('verl/trainer/ppo/trainer.py', 'RayPPOTrainer', '_load_checkpoint',
            dict(os=os, sys=sys, torch=torch, find_latest_ckpt_path=lambda _: None))
        config = SimpleNamespace(resume_mode='resume_path', resume_from_path=str(folder),
            default_hdfs_dir=None, default_local_dir=str(folder.parent), del_local_ckpt_after_load=False)
        actor = MagicMock()
        instance = SimpleNamespace(config=SimpleNamespace(trainer=config), _method_actor=lambda: actor,
            actor_rollout_wg=MagicMock(), global_steps=0)
        with patch('builtins.print'), self.assertRaises(ValueError):
            method(instance)
        self.assertEqual(instance.global_steps, 0)
        torch.load.assert_not_called()
        instance.actor_rollout_wg.load_checkpoint.assert_not_called()
        actor.runtime_restore.remote.assert_not_called()

    def test_completed_native_checkpoint_does_not_collect_an_extra_batch(self):
        class Options(SimpleNamespace):
            def get(self, key, default=None):
                return getattr(self, key, default)
        tracking = SimpleNamespace(Tracking=MagicMock())
        omega = SimpleNamespace(OmegaConf=SimpleNamespace(to_container=lambda *args, **kwargs: {}))
        scope = dict(tqdm=MagicMock(side_effect=AssertionError('Completed schedule must not collect')))
        fit = native_method('verl/trainer/ppo/trainer.py', 'RayPPOTrainer', 'fit', scope)
        instance = SimpleNamespace(config=Options(trainer=Options(project_name='unit', experiment_name='resume', logger=[])),
            _method_actor=lambda: None, val_reward_fn=None, total_training_steps=27)
        instance._load_checkpoint = lambda: setattr(instance, 'global_steps', 27)
        with patch.dict(sys.modules, {'omegaconf': omega, 'verl.utils.tracking': tracking}), patch('builtins.print'):
            fit(instance)
        self.assertEqual(instance.global_steps, 27)
        scope['tqdm'].assert_not_called()

    def test_completed_native_checkpoint_still_runs_final_evaluation(self):
        class Options(SimpleNamespace):
            def get(self, key, default=None):
                return getattr(self, key, default)
        tracking = SimpleNamespace(Tracking=MagicMock())
        omega = SimpleNamespace(OmegaConf=SimpleNamespace(to_container=lambda *args, **kwargs: {}))
        scope = dict(pprint=MagicMock(), tqdm=MagicMock(side_effect=AssertionError('Evaluation cannot train')))
        fit = native_method('verl/trainer/ppo/trainer.py', 'RayPPOTrainer', 'fit', scope)
        instance = SimpleNamespace(config=Options(trainer=Options(project_name='unit', experiment_name='test',
            logger=[], val_only=True)), _method_actor=lambda: None, val_reward_fn=object(), total_training_steps=27,
            _validate=MagicMock(return_value={'acc': 1.}))
        instance._load_checkpoint = lambda: setattr(instance, 'global_steps', 27)
        with patch.dict(sys.modules, {'omegaconf': omega, 'verl.utils.tracking': tracking}), patch('builtins.print'):
            fit(instance)
        instance._validate.assert_called_once()
        scope['tqdm'].assert_not_called()

    def test_failed_paired_commit_does_not_advance_the_latest_pointer(self):
        class Options(SimpleNamespace):
            def get(self, key, default=None):
                return getattr(self, key, default)
        from tests.experiments import ExperimentTests
        fixture = ExperimentTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        folder = fixture.checkpoint()
        pointer = folder.parent/'latest_checkpointed_iteration.txt'
        pointer.write_text('4', encoding='utf-8')
        scope = dict(os=os, torch=SimpleNamespace(save=MagicMock()), ray=SimpleNamespace(get=lambda _: None))
        save = native_method('verl/trainer/ppo/trainer.py', 'RayPPOTrainer', '_save_checkpoint', scope)
        instance = SimpleNamespace(config=Options(trainer=Options(default_local_dir=str(folder.parent),
            default_hdfs_dir=None)), global_steps=6, actor_rollout_wg=MagicMock(), use_critic=False,
            train_dataloader=SimpleNamespace(state_dict=lambda: {}), _method_actor=lambda: MagicMock())
        with patch.dict(sys.modules, {'verl.utils.fs': SimpleNamespace(local_mkdir_safe=lambda _: None)}), \
                patch('control.runtime.seal_checkpoint', side_effect=OSError('incomplete rank write')), \
                patch('builtins.print'), self.assertRaises(OSError):
            save(instance)
        self.assertEqual(pointer.read_text(encoding='utf-8'), '4')

    def test_repeated_native_save_cannot_overwrite_an_existing_checkpoint(self):
        from tests.experiments import ExperimentTests
        fixture = ExperimentTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        folder = fixture.checkpoint()
        before = {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob('*') if path.is_file()}
        save = native_method('verl/trainer/ppo/trainer.py', 'RayPPOTrainer', '_save_checkpoint', dict(os=os))
        instance = SimpleNamespace(config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(folder.parent))),
            global_steps=5, _method_actor=lambda: MagicMock(), actor_rollout_wg=MagicMock())
        with patch.dict(sys.modules, {'verl.utils.fs': SimpleNamespace(local_mkdir_safe=lambda _: None)}), \
                self.assertRaises(FileExistsError):
            save(instance)
        instance.actor_rollout_wg.save_checkpoint.assert_not_called()
        after = {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob('*') if path.is_file()}
        self.assertEqual(after, before)

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
