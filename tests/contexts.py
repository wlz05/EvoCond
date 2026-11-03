# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Visible contexts preserve soft uncertainty and explicit hard conflicts.'
import unittest
from copy import deepcopy
from control import context, subtasks
from model.client import Script, parse, validated
from state.data import decode
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

    def test_json_exponent_overflow_is_rejected_at_every_depth(self):
        for raw in ('1e999', '-1e999', '{"value": [1e999]}', '{"value": NaN}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                decode(raw)
        self.assertEqual(decode('{"value": 1e300}'), {'value': 1e300})

    def test_model_parser_preserves_wrappers_and_nested_objects(self):
        expected = {'nested': {'value': 3}, 'text': 'a {quoted} bracket'}
        raw = '{"nested": {"value": 3}, "text": "a {quoted} bracket"}'
        for text in (raw, '```json\n'+raw+'\n```', 'Result: '+raw):
            with self.subTest(text=text):
                self.assertEqual(parse(text), expected)

    def test_invalid_model_envelope_cannot_fall_through_to_a_nested_answer(self):
        for raw in ('{"value": 1e999, "nested": {"ok": true}}',
                '{"value": 1, "value": 2, "nested": {"ok": true}}',
                '{"broken":, "nested": {"ok": true}}', '[{"ok": true}]',
                '{"ok": true} {"different": false}', '{"ok": true} [1]'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse(raw)

    def test_model_and_validator_mutations_do_not_corrupt_repair_inputs(self):
        payload = {'values': [1]}
        original = deepcopy(payload)
        requests = []
        def generate(prompt, request):
            requests.append(deepcopy(request))
            request['values'].append(9)
            return {'values': [2 if len(requests) == 1 else 3]}
        def check(data):
            number = data['values'][0]
            data['values'].append(8)
            if number != 3:
                raise ValueError('Need three')
            return number
        self.assertEqual(validated(Script(generate), 'schema', payload, check), 3)
        self.assertEqual(payload, original)
        self.assertEqual(requests[1]['values'], [1])
        self.assertEqual(requests[1]['repair']['previous'], {'values': [2]})

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
