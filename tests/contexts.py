# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Visible contexts preserve soft uncertainty and explicit hard conflicts.'
import unittest
from control import context, subtasks
class Encoder:
    def encode(self, text):
        return [1.0, .5]
class ContextTests(unittest.TestCase):
    def test_entity_slots_preserve_operation_and_budget(self):
        result = context.normalize('Search Paris with budget 20', ['Paris'])
        self.assertEqual(result, 'Search <entity> with budget 20')
    def test_unknown_soft_fields_are_not_hard_exclusions(self):
        value = context.visible('search')
        self.assertTrue(context.compatible({'network': True}, value))
        self.assertFalse(context.compatible({'network': True}, context.visible('search', requirements={'network': False})))
    def test_features_are_frozen_semantics_and_explicit_availability(self):
        missing = context.features(context.visible('search'), Encoder())
        available = context.features(context.visible('search', available={'string': True}), Encoder())
        self.assertEqual(missing[:2], available[:2])
        self.assertEqual(missing[2], 0)
        self.assertEqual(available[2:4], [1, 1])
    def test_subtask_io_survives_dag_creation(self):
        plan = subtasks.make_plan('sum', [dict(id='a', goal='sum', depends_on=[], evidence_required='observed',
                                             inputs={'x': 'number'}, outputs={'sum': 'number'})])
        self.assertEqual(subtasks.ready(plan)[0]['inputs'], {'x': 'number'})
    def test_nonfinite_context_is_rejected(self):
        with self.assertRaises(ValueError):
            context.visible('sum', constraints={'budget': float('nan')})

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
