# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Canonical versions and identity-only contrastive optimization checks.'
import unittest
from control import dynamic, identity, relations
from tests import runtimes as fixtures
Backend = fixtures.Backend
class Encoder:
    def encode(self, text):
        return [1.0, .5 if text.startswith('Add') or '"postconditions": "Add' in text else -.5, .1]
    encode_query = encode
def registry():
    value = dynamic.empty_registry()
    for name, description in (('add', 'Add two numbers'), ('format', 'Format a number')):
        spec = fixtures.RuntimeTests.tool(None, name)
        spec['description'] = description
        value, _ = dynamic.admit(value, spec, Backend(), probe_inputs={'x': 1, 'y': 2}, task_id='train')
    return value
class IdentityTests(unittest.TestCase):
    def test_equal_schema_different_capability_keeps_separate_identity(self):
        value = identity.catalogue(registry())
        self.assertEqual(len(value['nodes']), 2)
        artifact = identity.initialize(registry(), Encoder(), 'encoder')
        vectors = [identity.node_vector(registry(), artifact, Encoder(), key) for key in value['nodes']]
        self.assertNotEqual(vectors[0], vectors[1])
    def test_unknown_versions_are_not_compatible(self):
        value = registry()
        left, right = value['semantic']
        self.assertFalse(identity.eligible(value, left, right))
    def test_behavior_receipt_preserves_versions_under_one_capability(self):
        value = registry()
        left, right = value['semantic']
        receipt = dict(decision='merge_verified', behavioral_probes=2, behavioral_probe_sha256=['a', 'b'])
        updated = identity.link(value, left, right, receipt)
        self.assertEqual(len(identity.catalogue(updated)['nodes']), 1)
        self.assertEqual(len(updated['semantic']), 2)
        self.assertTrue(identity.eligible(updated, left, right))
    def test_capability_merge_retains_opposite_conditional_relations(self):
        value = registry()
        value, _ = dynamic.admit(value, fixtures.RuntimeTests.tool(None, 'plus'), Backend(),
            probe_inputs={'x': 2, 'y': 3}, task_id='train')
        value = identity.attach(value)
        versions = {spec['name']: version for version, spec in value['semantic'].items()}
        left, right, target = (versions[name] for name in ('add', 'plus', 'format'))
        store = identity.catalogue(value)
        events = [dict(source=store['versions'][source], target=store['versions'][target],
            source_version=source, target_version=target, features=[float(i)],
            feedback=1.0 if i == 0 else -1.0, granularity='subtask')
            for i, source in enumerate((left, right))]
        value['relations'] = relations.observe(relations.new(bandwidth=.1), events,
            base_id='same_question', actor='fixed_actor')
        original = identity.attach(value)
        receipt = dict(decision='merge_verified', behavioral_probes=2, behavioral_probe_sha256=['a', 'b'])
        updated = identity.link(value, left, right, receipt)
        self.assertEqual(value, original)
        self.assertEqual(len(identity.catalogue(updated)['nodes']), 2)
        self.assertEqual(len(updated['semantic']), 3)
        memory = updated['relations']
        self.assertEqual(len(memory['edges']), 1)
        entries = next(iter(memory['edges'].values()))['events']
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['feedback'], 0)
        self.assertEqual({r['source_version'] for r in entries[0]['samples']}, {left, right})
        self.assertEqual([(r['features'], r['feedback']) for r in entries[0]['samples']],
                         [([0.0], 1.0), ([1.0], -1.0)])
        canonical, successor = store['versions'][left], store['versions'][target]
        positive = relations.estimate(memory, canonical, successor, [0.0], {})
        negative = relations.estimate(memory, canonical, successor, [1.0], {})
        pooled = relations.estimate(memory, canonical, successor, [0.0], {}, unconditional=True)
        self.assertGreater(positive['weight'], 0)
        self.assertLess(negative['weight'], 0)
        self.assertEqual(pooled['weight'], 0)
        self.assertTrue(identity.eligible(updated, right, left))
    def test_similarity_alone_is_not_merge_evidence(self):
        value = registry()
        with self.assertRaises(ValueError):
            identity.link(value, *value['semantic'], dict(decision='merge_verified', similarity=1))
    def test_alignment_loss_has_no_reward_or_context_features(self):
        value = registry()
        initial = identity.initialize(value, Encoder(), 'encoder')
        fitted = identity.fit(value, Encoder(), initial, steps=10)
        self.assertEqual(fitted['objective'], 'canonical_identity_only_no_context_or_rewards')
        self.assertLess(fitted['loss'][-1], fitted['loss'][0])
        self.assertEqual(initial['loss'], [])
        self.assertTrue(fitted['frozen'])
        self.assertNotIn('identities', fitted)
    def test_admission_reuses_compatible_identity_only_after_distinct_probes(self):
        value = registry()
        spec = fixtures.RuntimeTests.tool(None, 'plus')
        value, receipt = identity.admit(value, spec, Backend(), probe_inputs={'x': 1, 'y': 2},
            equivalence_inputs=[{'x': 3, 'y': 4}], task_id='new', encoder=Encoder())
        self.assertEqual(receipt['capability_decision'], 'compatible_version')
        self.assertEqual(len(identity.catalogue(value)['nodes']), 2)
    def test_no_multinput_probes_is_conservative_new_identity(self):
        value = registry()
        value, receipt = identity.admit(value, fixtures.RuntimeTests.tool(None, 'plus'), Backend(), probe_inputs={'x': 1, 'y': 2},
            task_id='new', encoder=Encoder())
        self.assertEqual(receipt['capability_decision'], 'new_capability')
        self.assertEqual(len(identity.catalogue(value)['nodes']), 3)
    def test_new_capability_cannot_rotate_existing_identity_coordinates(self):
        value = registry()
        initial = identity.fit(value, Encoder(), identity.initialize(value, Encoder(), 'e'), steps=2)
        canonical = next(iter(identity.catalogue(value)['nodes']))
        before = identity.node_vector(value, initial, Encoder(), canonical)
        enlarged, _ = dynamic.admit(value, fixtures.RuntimeTests.tool(None, 'fresh'), Backend(), probe_inputs={'x': 3, 'y': 4}, task_id='new')
        after = identity.fit(enlarged, Encoder(), initial, steps=2)
        self.assertEqual(after, initial)
        self.assertEqual(identity.node_vector(enlarged, after, Encoder(), canonical), before)
    def test_cold_start_structure_cannot_change_own_score(self):
        from control import context, mixing
        value = registry()
        initial = identity.initialize(value, Encoder(), 'e')
        result = mixing.rank(value, Encoder(), initial, context.visible('Add'), recall=4, top_k=1)
        for row in result['candidates']:
            self.assertEqual(row['semantic'], row['structure'])
            self.assertEqual(row['verified_uses'], 0)
    def test_compatible_versions_do_not_consume_multiple_candidate_slots(self):
        from control import context, mixing
        value = registry()
        value = identity.link(value, *value['semantic'], dict(decision='merge_verified', behavioral_probes=2, behavioral_probe_sha256=['a', 'b']))
        initial = identity.initialize(value, Encoder(), 'e')
        result = mixing.rank(value, Encoder(), initial, context.visible('Add'), recall=4, top_k=1)
        self.assertEqual(len(result['candidates']), 1)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
