# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Check observable access without inventing target recall or task attribution.'
from copy import deepcopy
from hashlib import sha256
import json
import unittest
from control import exposure
class ExposureTests(unittest.TestCase):
    def receipt(self, target=('b', 'a'), schemas=('a', 'b', 'c')):
        access = dict(active_versions=4, active_identities=3, recalled_versions=['a', 'b', 'c'],
            recall_truncated=False, retention_horizon=128, history_query_limit=64)
        rows = []
        for versions, alphas in ((('a', 'b'), [0.0]), (target, [.5, 1.0])):
            payload = json.dumps(dict(tools=[v for v in versions if v in schemas]), separators=(',', ':'))
            rows.append(dict(alphas=alphas, ranking=[dict(version=v) for v in versions],
                payload=payload, payload_hash=sha256(payload.encode('utf-8')).hexdigest()))
        return access, rows, [dict(tool_version=v) for v in schemas]
    def test_order_changes_do_not_claim_new_candidate_access(self):
        access, classes, schemas = self.receipt()
        original = deepcopy((access, classes, schemas))
        receipt = exposure.build(access, classes, 1, schemas)
        result = exposure.summarize([dict(exposure=receipt)])
        self.assertEqual(result['payload_changes'], 1)
        self.assertEqual(result['order_only_changes'], 1)
        self.assertEqual(result['candidate_set_changes'], 0)
        self.assertEqual(result['schema_drops'], 0)
        self.assertEqual((access, classes, schemas), original)
        self.assertEqual(receipt['scope'], 'observed_exposure_not_target_recall_or_adoption_utility')
    def test_different_sets_are_not_reported_as_order_only(self):
        access, classes, schemas = self.receipt(('a', 'c'))
        receipt = exposure.build(access, classes, 1, schemas)
        result = exposure.summarize([dict(exposure=receipt)])
        self.assertEqual(result['candidate_set_changes'], 1)
        self.assertEqual(result['order_only_changes'], 0)
        self.assertEqual(result['selected_payload_bytes'], len(classes[1]['payload'].encode('utf-8')))
    def test_missing_schema_is_not_a_missing_target_label(self):
        access, classes, schemas = self.receipt(schemas=('a', 'b'))
        receipt = exposure.build(access, classes, 0, schemas)
        result = exposure.summarize([dict(exposure=receipt)])
        self.assertEqual(result['schema_drops'], 1)
        self.assertEqual(receipt['schema_missing_versions'], ['c'])
        self.assertEqual(result['payload_changes'], 0)
        self.assertNotIn('target_recall', receipt)
    def test_old_or_partial_logs_remain_unknown_not_zero(self):
        access, classes, schemas = self.receipt()
        receipt = exposure.build(access, classes, 0, schemas)
        for selections in ([], [{}], [dict(exposure=receipt), {}]):
            self.assertEqual(exposure.summarize(selections), {key: None for key in exposure.METRICS})
    def test_wrong_fingerprint_and_out_of_pool_payload_are_rejected(self):
        access, classes, schemas = self.receipt()
        classes[1]['payload'] += ' '
        with self.assertRaises(ValueError):
            exposure.build(access, classes, 1, schemas)
        access, classes, schemas = self.receipt(('a', 'outside'), ('a', 'b', 'c', 'outside'))
        with self.assertRaises(ValueError):
            exposure.build(access, classes, 1, schemas)
    def test_invalid_selection_and_duplicate_semantic_class_are_rejected(self):
        access, classes, schemas = self.receipt()
        for selected in (-1, 2, True, None):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                exposure.build(access, classes, selected, schemas)
        classes[1]['alphas'].append(0.0)
        with self.assertRaises(ValueError):
            exposure.build(access, classes, 1, schemas)
    def test_tampered_summary_cannot_invent_counts_or_order_changes(self):
        access, classes, schemas = self.receipt()
        receipt = exposure.build(access, classes, 0, schemas)
        for key, value in (('active_identities', 5), ('history_query_limit', 129), ('recall_truncated', 1), ('format', 'unknown')):
            row = deepcopy(receipt)
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                exposure.summarize([dict(exposure=row)])
        for key, value in (('bytes', -1), ('versions', ['b', 'a']), ('sha256', 'bad')):
            row = deepcopy(receipt)
            row['selected'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                exposure.summarize([dict(exposure=row)])
if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
