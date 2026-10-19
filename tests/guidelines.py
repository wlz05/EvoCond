# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Synthetic controls, not a model performance experiment.'
from copy import deepcopy
from pathlib import Path
import ast
import tempfile
import unittest
from unittest.mock import patch
from control.dynamic import digest
from control.guidance import present
from control.runtime import execution_context_fingerprint
from guidance import variants
from learning import summarize
import experiment as experiment
from tests import runtimes as fixture_module
def view():
    rows = [{'name': name, 'version': name*64, 'semantic_score': sem, 'structural_score': structure}
            for name, sem, structure in (('a', .8, .1), ('b', .7, .9), ('c', .1, 1.))]
    return {'format': 'evocond_searl_retrieval_v2', 'candidates': rows,
            'teacher': {'structure_weight': 1.}, 'tools': [rows[2], rows[1]]}
class GuidanceTests(unittest.TestCase):
    def test_capabilities_identical_only_order_changes(self):
        from control.ranking import FORMAT
        source = view()
        source['format'] = FORMAT
        before = deepcopy(source)
        ranked, a = present(source, 'ranked', top_k=2)
        neutral, b = present(source, 'candidate_only', top_k=2)
        self.assertEqual([r['name'] for r in ranked], ['b', 'a'])
        self.assertEqual([r['name'] for r in neutral], ['a', 'b'])
        self.assertEqual(a['candidate_pool_hash'], b['candidate_pool_hash'])
        self.assertFalse(b['no_memory'])
        self.assertEqual(before, source)
        self.assertEqual([r['name'] for r in present(source, 'legacy', top_k=2)[0]], ['c', 'b'])
    def test_changed_guidance_invalidates_teacher_execution_context(self):
        self.assertNotEqual(execution_context_fingerprint({'guidance_mode': 'ranked'}),
                            execution_context_fingerprint({'guidance_mode': 'candidate_only'}))
        with self.assertRaises(ValueError):
            experiment.teacher_options({'guidance_mode': 'invented'})
    def test_control_configs_do_not_change_other_mechanisms(self):
        cfg = {'format': experiment.FORMAT, 'run_id': 'study',
               'teacher': {'fixed_alpha': .5, 'binding_credit': 'composite'}}
        controls = variants(cfg, {'ranked': 'a.json', 'candidate_only': 'b.json'})
        self.assertEqual(controls['ranked']['teacher']['fixed_alpha'], .5)
        self.assertEqual(controls['candidate_only']['teacher']['binding_credit'], 'composite')
        self.assertEqual(cfg['run_id'], 'study')
        with self.assertRaises(ValueError):
            variants(cfg, {'ranked': 'a.json', 'candidate_only': 'a.json'})
    def test_runtime_uses_control_and_persists_actual_presentations(self):
        from control.ranking import FORMAT
        fixture = fixture_module.RuntimeTests()
        fixture.setUp()
        try:
            runtime = fixture.teacher_runtime()
            for name in ('a', 'b', 'c'):
                runtime.dispatcher.create(fixture.tool(name), {'x': 1, 'y': 2}, task_id='fixture')
            episode = runtime.begin('compute', fixture_module.PLAN, split='validation')
            source = view()
            for row in source['candidates']:
                row['version'] = runtime.dispatcher.registry['aliases'][row['name']]
            source['format'] = FORMAT
            source['teacher'] = {'structure_weight': 1.}
            from control.budget import select
            source['structure_receipt'] = select({}, allowed_versions=[])[1]
            # Pure control is applied to the actual view, not fabricated tool prose.
            with patch('control.ranking.rank', return_value=source):
                runtime.config['guidance_mode'] = 'candidate_only'
                runtime.select(episode, top_k=2)
            chosen = runtime.context(episode)['selected_versions']
            self.assertEqual(set(chosen)-{'submit_revised_plan'}, {'a', 'b'})
            self.assertEqual(runtime.context(episode)['guidance_presentations'][0]['presented_order'], ['a', 'b'])
            runtime.finish(episode)
            import json
            row = json.loads((runtime.root/'binding_events.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(row['guidance_mode'], 'candidate_only')
            self.assertEqual(len(row['guidance_presentations']), 1)
            self.assertEqual(len(row['structure_presentations']), 1)
        finally:
            fixture.tearDown()
    def test_proxy_audit_distinguishes_correctness_from_cost_signal(self):
        def row(group, success, advantage, n):
            return {'group': group, 'success': success, 'advantage': advantage,
                    'turns': [{'prompt_ids': [1]*n, 'completion_ids': [2]}]}
        rows = [row('cost', True, 1., 8), row('cost', True, -1., 8),
                row('none', True, 0., 3), row('none', True, 0., 3)]
        report = summarize(rows, prompt_limit=4)
        self.assertEqual(report['groups_with_only_cost_signal'], 1)
        self.assertEqual(report['zero_advantage_groups'], 1)
        self.assertEqual(report['discarded_prefix_tokens_if_suffix_windowed'], 8)
        self.assertEqual(report['nonzero_advantage_turns'], 2)
    def test_legacy_optimizer_skips_zero_signal_before_any_update(self):
        import update
        tree = ast.parse(Path(update.__file__).read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
        loop = next(n for n in fn.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'key')
        skip = next(n for n in loop.body if isinstance(n, ast.If))
        self.assertTrue(any(isinstance(n, ast.Continue) for n in ast.walk(skip)))
        self.assertIn('zero_group_advantage', ast.unparse(skip))
        self.assertLess(loop.body.index(skip), next(i for i, n in enumerate(loop.body)
                        if isinstance(n, ast.Expr) and 'zero_grad' in ast.unparse(n)))
if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
