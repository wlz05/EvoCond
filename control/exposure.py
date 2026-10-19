# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Observe candidate access and actual payload changes without judging usefulness.'
from __future__ import annotations
from hashlib import sha256
FORMAT = 'evocond_retrieval_exposure_v1'
METRICS = ('payload_changes', 'candidate_set_changes', 'order_only_changes', 'schema_drops',
    'selected_payload_bytes', 'semantic_payload_bytes')
def unique(values):
    return isinstance(values, list) and all(isinstance(v, str) and v for v in values) and len(set(values)) == len(values)
def validate(row):
    if not isinstance(row, dict) or row.get('format') != FORMAT:
        raise ValueError('Unknown retrieval exposure receipt')
    for key in ('active_versions', 'active_identities', 'retention_horizon', 'history_query_limit', 'payload_classes'):
        if type(row.get(key)) is not int or row[key] < (1 if key in ('retention_horizon', 'history_query_limit', 'payload_classes') else 0):
            raise ValueError('Exposure counts must be declared nonnegative integers')
    if row['active_identities'] > row['active_versions'] or row['history_query_limit'] > row['retention_horizon']:
        raise ValueError('Exposure counts exceed the declared access budgets')
    recalled, missing = row.get('recalled_versions'), row.get('schema_missing_versions')
    if not unique(recalled) or not unique(missing) or len(recalled) > row['active_identities']:
        raise ValueError('Recalled and missing versions must be unique access records')
    if not set(missing) <= set(recalled) or type(row.get('recall_truncated')) is not bool:
        raise ValueError('Schema drops and recall truncation must be explicit')
    for key in ('semantic', 'selected'):
        packet = row.get(key)
        if not isinstance(packet, dict) or not unique(packet.get('versions')):
            raise ValueError('Exposure requires actual ordered version lists')
        if not set(packet['versions']) <= set(recalled)-set(missing):
            raise ValueError('Presented versions must be available recalled candidates')
        fingerprint = packet.get('sha256')
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
            raise ValueError('Exposure requires a valid payload fingerprint')
        if type(packet.get('bytes')) is not int or packet['bytes'] < 0:
            raise ValueError('Payload bytes must be measured nonnegative integers')
    if row['semantic']['sha256'] == row['selected']['sha256'] and row['semantic'] != row['selected']:
        raise ValueError('Identical payloads cannot have different byte counts or version orders')
    return row
def build(access, classes, selected, schemas):
    recalled = access['recalled_versions']
    if len(set(recalled)) != len(recalled):
        raise ValueError('Recalled versions must be unique')
    if type(selected) is not int or not 0 <= selected < len(classes):
        raise ValueError('Identify the actual payload class')
    semantic = [row for row in classes if 0.0 in row['alphas']]
    if len(semantic) != 1:
        raise ValueError('Exactly one class must contain the semantic action')
    available = {row['tool_version'] for row in schemas if row.get('tool_version') is not None}
    def packet(row):
        if not isinstance(row['payload'], str) or sha256(row['payload'].encode('utf-8')).hexdigest() != row['payload_hash']:
            raise ValueError('Payload fingerprint must match actual UTF-8 bytes')
        versions = [item['version'] for item in row['ranking'] if item['version'] in available]
        if len(set(versions)) != len(versions) or not set(versions) <= set(recalled):
            raise ValueError('Presented versions must be unique recalled candidates')
        return dict(versions=versions, sha256=row['payload_hash'], bytes=len(row['payload'].encode('utf-8')))
    return validate(dict(format=FORMAT, **access, schema_missing_versions=[v for v in recalled if v not in available],
        semantic=packet(semantic[0]), selected=packet(classes[selected]), payload_classes=len(classes),
        scope='observed_exposure_not_target_recall_or_adoption_utility'))
def summarize(selections):
    rows = [row.get('exposure') for row in selections]
    if not rows or any(row is None for row in rows):
        return {key: None for key in METRICS}
    for row in rows:
        validate(row)
    changed = [row['selected']['sha256'] != row['semantic']['sha256'] for row in rows]
    sets = [set(row['selected']['versions']) != set(row['semantic']['versions']) for row in rows]
    reordered = [not different and row['selected']['versions'] != row['semantic']['versions']
        for row, different in zip(rows, sets)]
    return dict(payload_changes=sum(changed), candidate_set_changes=sum(sets), order_only_changes=sum(reordered),
        schema_drops=sum(len(row['schema_missing_versions']) for row in rows),
        selected_payload_bytes=sum(row['selected']['bytes'] for row in rows),
        semantic_payload_bytes=sum(row['semantic']['bytes'] for row in rows))

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
