# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Whole-structure ablation learns contextual projections without changing main identity maps.'
from copy import deepcopy
import unittest
import numpy as np
from control import context, identity, mixing, relations, whole
from tests.identities import Encoder as Frozen, registry
class Encoder(Frozen):
    def encode(self, text):
        return [1.0, .5 if 'Add' in text else -.5, .1]
    encode_query = encode
class WholeTests(unittest.TestCase):
    def fixture(self):
        value, encoder = identity.attach(registry()), Encoder()
        alignment = identity.fit(value, encoder, identity.initialize(value, encoder, 'frozen'), steps=2)
        store = identity.catalogue(value)
        left, right = list(value['semantic'])
        memory = relations.new(bandwidth=.25)
        for index, goal in enumerate(('Add', 'Format')):
            view = context.visible(goal)
            event = dict(source=store['versions'][left], target=store['versions'][right], source_version=left,
                target_version=right, context=view, features=context.features(view, encoder), feedback=1-2*index,
                requirements=None, granularity='terminal')
            memory = relations.observe(memory, [event], base_id='train'+str(index), actor='actor')
        value['relations'] = memory
        return value, encoder, alignment
    def test_whole_projection_trains_residual_columns_on_training_contexts(self):
        value, encoder, alignment = self.fixture()
        initial = whole.initialize(alignment)
        before = deepcopy(alignment)
        fitted = whole.fit(value, encoder, alignment, initial, None, actor='actor', steps=10)
        self.assertTrue(fitted['trained'])
        self.assertEqual(fitted['fit_ids'], ['train0', 'train1'])
        self.assertEqual(fitted['samples'], 4)
        size = alignment['dimension']
        self.assertFalse(np.allclose(np.asarray(initial['whole_map'])[:, size:], np.asarray(fitted['whole_map'])[:, size:]))
        self.assertEqual(alignment, before)
        self.assertEqual(initial['loss'], [])
        self.assertLess(fitted['loss'][-1], fitted['loss'][0])
    def test_wrong_actor_does_not_supply_alignment_examples(self):
        value, encoder, alignment = self.fixture()
        fitted = whole.fit(value, encoder, alignment, whole.initialize(alignment), None, actor='other')
        self.assertFalse(fitted['trained'])
        self.assertEqual(fitted['samples'], 0)
    def test_whole_ablation_refuses_missing_projection(self):
        value, encoder, alignment = self.fixture()
        with self.assertRaises(ValueError):
            mixing.rank(value, encoder, alignment, context.visible('Add'), variant='whole')
    def test_whole_cold_start_keeps_semantic_score(self):
        value, encoder, alignment = self.fixture()
        fitted = whole.fit(value, encoder, alignment, whole.initialize(alignment), None, actor='actor', steps=2)
        result = mixing.rank(value, encoder, alignment, context.visible('Add'), variant='whole', whole_alignment=fitted)
        for row in result['candidates']:
            self.assertEqual(row['semantic'], row['structure'])
    def test_malformed_whole_projection_is_rejected(self):
        _, _, alignment = self.fixture()
        fitted = whole.initialize(alignment)
        fitted['whole_map'][0].pop()
        with self.assertRaises(ValueError):
            whole.validate(fitted, alignment['dimension'])
    def test_training_sample_budget_is_bounded(self):
        value, encoder, alignment = self.fixture()
        fitted = whole.fit(value, encoder, alignment, whole.initialize(alignment), None, actor='actor', samples_limit=2)
        self.assertEqual(fitted['samples'], 2)
        self.assertEqual(fitted['samples_limit'], 2)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
