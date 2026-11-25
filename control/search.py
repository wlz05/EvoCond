# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Record document search and score fixed-prefix search probes.'
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import string
import unicodedata
from urllib.parse import urlsplit
from control.contracts import strict_json
from control.isolation import canonical_hash

FORMAT = 'evocond_search_probe_v1'
CELLS = ('old_semantic', 'old_structural', 'new_semantic', 'new_structural', 'old_repeat')
HEADING = 'Following are the search results given by the wiki search engine:'
EMPTY = 'No relevant documents found'
MAX_BYTES = 1048576

class SearchResponseError(ValueError):
    pass

def digest(text):
    return sha256(text.encode('utf-8')).hexdigest()

def is_hash(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None

def revision(path):
    hasher = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def render(text):
    title, _, body = text.partition('\n')
    return '\nsearch results: "'+title.strip('"')+'"\n'+body

def endpoint(value):
    if not isinstance(value, str):
        raise ValueError('Declare the HTTP(S) /retrieve endpoint')
    parsed = urlsplit(value)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.path != '/retrieve'
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError('Declare the HTTP(S) /retrieve endpoint')
    return value

def session():
    import requests
    from requests.adapters import HTTPAdapter
    client = requests.Session()
    client.trust_env = False
    for scheme in ('http://', 'https://'):
        client.mount(scheme, HTTPAdapter(pool_connections=32, pool_maxsize=32))
    return client

def fetch(client, url, query, topk):
    validate(dict(query=query, topk=topk, hits=[]))
    with client.post(endpoint(url), json=dict(queries=[query], topk=topk, return_scores=True),
                     timeout=(5, 10), stream=True, allow_redirects=False) as reply:
        if reply.status_code != 200:
            raise SearchResponseError(f'retrieval HTTP status {reply.status_code}')
        data = bytearray()
        for chunk in reply.iter_content(16384):
            data.extend(chunk)
            if len(data) > MAX_BYTES:
                raise SearchResponseError('response exceeded size limit')
    return response(query, topk, json.loads(data))

def response(query, topk, data):
    batches = data.get('result') if isinstance(data, dict) else None
    if batches == []:
        batches = [[]]
    if not isinstance(batches, list) or len(batches) != 1 or not isinstance(batches[0], list):
        raise ValueError('Search needs exactly one ranked result list')
    hits = []
    for rank, item in enumerate(batches[0], 1):
        text = item['document']['contents']
        if not isinstance(text, str):
            raise ValueError('Document contents must be text')
        hits.append(dict(passage_id=digest(text), rank=rank, reference=render(text)))
    visible = hits[:topk]
    output = HEADING+''.join(hit['reference'] for hit in visible) if visible else EMPTY
    return output, dict(query=query, topk=topk, corpus_revision=data.get('corpus_revision'),
        index_revision=data.get('index_revision'), hits=hits)

def validate(record, *, identified=False):
    if (not isinstance(record, dict) or not isinstance(record.get('query'), str) or not record['query'].strip()
            or len(record['query']) > 16000 or type(record.get('topk')) is not int or not 1 <= record['topk'] <= 10
            or not isinstance(record.get('hits'), list)):
        raise ValueError('Invalid document search receipt')
    for key in ('corpus_revision', 'index_revision'):
        if (identified or record.get(key) is not None) and not is_hash(record.get(key)):
            raise ValueError('Search evidence requires pinned corpus and index hashes')
    for rank, hit in enumerate(record['hits'], 1):
        if (not isinstance(hit, dict) or not is_hash(hit.get('passage_id')) or type(hit.get('rank')) is not int
                or hit['rank'] != rank or not isinstance(hit.get('reference'), str) or not hit['reference']):
            raise ValueError('Search evidence requires ordered passage references')
    return record

def observe(context, record, *, prefix=False):
    validate(record, identified='search_probe' in context)
    row = deepcopy(record)
    for key in ('corpus_revision', 'index_revision'):
        if 'search_probe' in context and row[key] != context['search_probe'][key]:
            raise ValueError('Search service snapshot differs from the shared prefix')
    row.update(turn=context['assistant_turn'], prefix=prefix, visible=[], first_seen=[])
    context.setdefault('document_searches', []).append(row)

def delivered(context, rendered):
    records = context.get('document_searches', [])
    seen = {value for row in records for value in row['visible']}
    cursor = 0
    for row in records:
        if row['turn'] != context['assistant_turn'] or row.get('delivery_checked'):
            continue
        for hit in row['hits'][:row['topk']]:
            index = rendered.find(hit['reference'], cursor)
            if index < 0:
                continue
            cursor = index+len(hit['reference'])
            key = hit['passage_id']
            row['visible'].append(key)
            if key not in seen:
                row['first_seen'].append(key)
                seen.add(key)
        row['delivery_checked'] = True

def evidence(context):
    return [{**{key: value for key, value in row.items() if key != 'hits'},
        'returned': [hit['passage_id'] for hit in row['hits']]} for row in context.get('document_searches', [])]

def repetitions(events):
    queries = [' '.join(row['query'].casefold().split()) for row in events]
    return len(queries)-len(set(queries))

def manifest(value):
    fields = {'format', 'budget', 'endpoint', 'corpus_revision', 'index_revision', 'prefixes', 'checkpoints'}
    if (not isinstance(value, dict) or value.keys() != fields or value.get('format') != FORMAT
            or type(value.get('budget')) is not int or not 1 <= value['budget'] <= 32
            or not isinstance(value.get('prefixes'), dict) or not value['prefixes']):
        raise ValueError('Declare a fixed-prefix search study and a positive call budget')
    for key in ('corpus_revision', 'index_revision'):
        if not is_hash(value.get(key)):
            raise ValueError('Declare a SHA256 for each retrieval artifact')
    endpoint(value.get('endpoint'))
    paths = value.get('checkpoints')
    if not isinstance(paths, dict) or paths.keys() != {'old', 'new', 'bank'} or any(
            not isinstance(path, str) or not path for path in paths.values()):
        raise ValueError('Declare old, new and bank checkpoint paths')
    for task, prefix in value['prefixes'].items():
        if (not is_hash(task) or not isinstance(prefix, dict) or set(prefix) != {'plan', 'search'}
                or not isinstance(prefix['plan'], str) or not prefix['plan'].strip()):
            raise ValueError('Each question hash needs a fixed plan and first search')
        search = validate(prefix['search'], identified=True)
        if any(search[key] != value[key] for key in ('corpus_revision', 'index_revision')):
            raise ValueError('Prefix retrieval revision mismatch')
    return value

def prepare(spec):
    if not isinstance(spec, dict) or not isinstance(spec.get('prefixes'), dict):
        raise ValueError('Declare format, budget, endpoint, revisions, prefixes and checkpoint paths')
    rows = {}
    for task, row in spec['prefixes'].items():
        if not isinstance(row, dict) or set(row) - {'plan', 'query', 'topk'} or not is_hash(task):
            raise ValueError('Each question hash needs a fixed plan and first-hop query')
        rows[task] = dict(plan=row.get('plan'), search=dict(query=row.get('query'), topk=row.get('topk', 3),
            hits=[], corpus_revision=spec.get('corpus_revision'), index_revision=spec.get('index_revision')))
    manifest(dict(spec, prefixes=rows))
    with session() as client:
        for row in rows.values():
            original = row['search']
            _, receipt = fetch(client, spec['endpoint'], original['query'], original['topk'])
            if any(receipt[k] != spec[k] for k in ('corpus_revision', 'index_revision')):
                raise ValueError('The search sidecar lacks the declared corpus and index revisions')
            row['search'] = receipt
    return manifest(dict(spec, prefixes=rows))

def prefix_text(record):
    hits = record['hits'][:record['topk']]
    return HEADING+''.join(hit['reference'] for hit in hits) if hits else EMPTY

def coverage(path, expected, facts):
    remaining = {key for rows in facts.values() for group in rows for key in group}
    hasher = sha256()
    with Path(path).open('rb') as stream:
        for line in stream:
            hasher.update(line)
            if remaining and line.strip():
                row = json.loads(line)
                remaining.discard(digest(row['contents']))
    if hasher.hexdigest() != expected:
        raise ValueError('Coverage corpus differs from the frozen retrieval corpus')
    return {task: [sorted(set(group)-remaining) for group in rows] for task, rows in facts.items()}

def hotpot(spec, dataset, corpus):
    from control.runtime import task_identity
    spec = manifest(spec)
    if not isinstance(dataset, list):
        raise ValueError('HotpotQA annotations must be an array of original examples')
    def flatten(value):
        return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())
    indexed, annotations = {}, {}
    for row in dataset:
        task = task_identity(row['question'])
        if task not in spec['prefixes']:
            continue
        if task in annotations:
            raise ValueError('HotpotQA contains repeated questions in the selected panel')
        context = dict(row['context'])
        groups = []
        for title, index in row['supporting_facts']:
            sentences = context.get(title)
            if not isinstance(sentences, list) or type(index) is not int or not 0 <= index < len(sentences):
                raise ValueError('HotpotQA supporting fact is outside the original context')
            sentence = flatten(sentences[index])
            if not sentence:
                raise ValueError('HotpotQA supporting fact has no text')
            aliases = []
            groups.append(aliases)
            indexed.setdefault(flatten(title), []).append((sentence, aliases))
        if not groups or not isinstance(row['answer'], str) or not normalize(row['answer']):
            raise ValueError('HotpotQA example has no supporting facts or answer')
        annotations[task] = dict(answers=[row['answer']], facts=groups)
    if annotations.keys() != spec['prefixes'].keys():
        raise ValueError('HotpotQA annotations do not cover all requested questions')
    hasher = sha256()
    with Path(corpus).open('rb') as stream:
        for line in stream:
            hasher.update(line)
            if not line.strip():
                continue
            text = json.loads(line)['contents']
            title, _, body = text.partition('\n')
            if matches := indexed.get(flatten(title.strip('"'))):
                normalized = ' '+flatten(body)+' '
                key = digest(text)
                for sentence, aliases in matches:
                    if ' '+sentence+' ' in normalized and key not in aliases:
                        aliases.append(key)
    if hasher.hexdigest() != spec['corpus_revision']:
        raise ValueError('Annotation corpus differs from the frozen retrieval corpus')
    return annotations

def normalize(answer):
    text = answer.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())

def answer_scores(output, answers):
    matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', output, flags=re.S)
    raw = matches[-1].strip() if matches else ''
    boxed = re.fullmatch(r'\\boxed\{(.*)\}', raw, flags=re.S)
    answer = normalize(boxed[1] if boxed else raw)
    exact, f1 = 0, 0.
    for target in map(normalize, answers):
        exact = max(exact, int(answer == target))
        if answer != target and {answer, target} & {'yes', 'no', 'noanswer'}:
            continue
        overlap = sum((Counter(answer.split()) & Counter(target.split())).values())
        if overlap:
            f1 = max(f1, 2*overlap/(len(answer.split())+len(target.split())))
    return exact, f1

def interval(values, seed):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values)))/len(values) for _ in range(1000))
    return dict(mean=sum(values)/len(values), interval=[means[25], means[974]])

def load_exports(spec, exports):
    by_cell, deployments, banks = {}, set(), set()
    actors = {}
    protocol = canonical_hash(spec)
    for cell, path in exports.items():
        rows = {}
        with Path(path).open(encoding='utf-8') as stream:
            for line in stream:
                row = strict_json(line, max_text=16*1024*1024)
                task = row['question_sha256']
                if task in rows or task not in spec['prefixes']:
                    raise ValueError('Duplicate or unexpected probe question')
                probe = row['search_probe']
                if (probe['cell'] != cell or not is_hash(probe['checkpoint']) or not is_hash(probe['bank'])
                        or probe['split'] != 'validation' or probe['task'] != task
                        or probe['protocol_sha256'] != protocol
                        or not all(is_hash(probe.get(k)) for k in ('source_sha256', 'environment_sha256'))
                        or probe['budget'] != spec['budget'] or any(probe[k] != spec[k]
                            for k in ('corpus_revision', 'index_revision'))):
                    raise ValueError('Search study receipt differs from its cell protocol')
                deployments.add((probe['source_sha256'], probe['environment_sha256']))
                banks.add(probe['bank'])
                actors.setdefault(cell.split('_')[0], set()).add(probe['checkpoint'])
                rows[task] = row
        if rows.keys() != spec['prefixes'].keys():
            raise ValueError('Every probe cell must account for every scheduled question')
        by_cell[cell] = rows
    if len(deployments) != 1 or len(banks) != 1 or any(len(items) != 1 for items in actors.values()):
        raise ValueError('Probe cells must share an evaluation source, environment and bank and use one checkpoint per age')
    if len(actors) == 2 and actors['old'] == actors['new']:
        raise ValueError('Require two distinct actor checkpoints')
    return by_cell

def check_repeat(by_cell):
    for task, original in by_cell['old_semantic'].items():
        repeat = by_cell['old_repeat'][task]
        if (original['search_probe']['prefix_sha256'] != repeat['search_probe']['prefix_sha256']
                or any(original[key] != repeat[key] for key in ('document_searches', 'tool_list_selections', 'output'))):
            raise ValueError('Same-prefix same-list repeat changed the search trajectory or answer')
    return dict(replay_passed=True, questions=len(by_cell['old_semantic']))

def panel(spec, annotations, corpus, exports):
    spec = manifest(spec)
    if set(annotations) != set(spec['prefixes']) or set(exports) != set(CELLS):
        raise ValueError('Require all annotated questions and four cells plus the repeat control')
    by_cell = load_exports(spec, exports)
    check_repeat(by_cell)
    facts = {}
    for task, row in annotations.items():
        groups, answers = row.get('facts'), row.get('answers')
        if (not isinstance(groups, list) or not groups or any(not isinstance(g, list) or any(not is_hash(v) for v in g) for g in groups)
                or not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not normalize(a) for a in answers)):
            raise ValueError('Annotations require passage aliases per supporting fact and nonempty answers')
        facts[task] = groups
    covered = coverage(corpus, spec['corpus_revision'], facts)
    paired = []
    for task in facts:
        scores = {}
        controls = []
        for cell in CELLS:
            row = by_cell[cell][task]
            probe, events = row['search_probe'], row['document_searches']
            selections = row['tool_list_selections']
            if not selections or not is_hash(selections[0].get('visible_prompt_sha256')):
                raise ValueError('Initial tool presentation was never used by the actor')
            selection = selections[0]
            controls.append((probe['prefix_sha256'], selection['semantic_payload_hash'], selection['candidate_hash'],
                sorted(selection['versions'])))
            if (not events or not events[0]['prefix'] or not events[0].get('delivery_checked') or len(events) > spec['budget']
                    or any(e['prefix'] or any(e[k] != spec[k] for k in ('corpus_revision', 'index_revision')) for e in events[1:])):
                raise ValueError('Probe search history exceeds its frozen budget or revision')
            first = events[0]
            source = spec['prefixes'][task]['search']
            if (first['query'] != source['query'] or first['topk'] != source['topk']
                    or first['returned'] != [hit['passage_id'] for hit in source['hits']]
                    or first['visible'] != first['returned'][:source['topk']]):
                raise ValueError('Executed first-hop evidence differs from the study manifest')
            if cell in ('old_semantic', 'new_semantic', 'old_repeat') and selection['payload_hash'] != selection['semantic_payload_hash']:
                raise ValueError('Semantic control received a different payload')
            if any(item['payload_hash'] != item['semantic_payload_hash'] for item in selections[1:]):
                raise ValueError('The study only intervenes at the first presentation')
            seen, curve = set(), []
            for index in range(spec['budget']):
                if index < len(events):
                    event = events[index]
                    if (not set(event['visible']) <= set(event['returned'][:event['topk']])
                            or event['first_seen'] != list(dict.fromkeys(key for key in event['visible'] if key not in seen))):
                        raise ValueError('Visible evidence must come from the forwarded ranked results')
                    seen.update(event['visible'])
                curve.append(int(all(seen.intersection(group) for group in covered[task])))
            em, f1 = answer_scores(row['output'], annotations[task]['answers'])
            scores[cell] = dict(curve=curve, em=em, f1=f1, calls=len(events), repeated=repetitions(events),
                cost=row.get('accounted_cost'), payload_hash=selection['payload_hash'],
                input_tokens=row['actor_input_tokens'], output_tokens=row['actor_output_tokens'],
                initial_versions=selection['versions'])
        if any(control != controls[0] for control in controls[1:]):
            raise ValueError('A cell changed the prefix, candidate pool or pinned tool set')
        paired.append(dict(task=task, covered=all(covered[task]), scores=scores))
    eligible = [row['scores'] for row in paired if row['covered']]
    curves = {}
    for index in range(spec['budget']):
        cells = {cell: [row[cell]['curve'][index] for row in eligible] for cell in CELLS}
        old = [b-a for a,b in zip(cells['old_semantic'], cells['old_structural'])]
        new = [b-a for a,b in zip(cells['new_semantic'], cells['new_structural'])]
        curves[index+1] = None if not eligible else dict(
            complete={cell: interval(values, index) for cell, values in cells.items()},
            old_gain=interval(old, index), new_gain=interval(new, index),
            gain_change=interval([b-a for a,b in zip(old,new)], index),
            replaced=sum(a == 1 and b == 0 and s == 1 for a,b,s in zip(old,new,cells['new_semantic'])),
            replaced_with_extra_searches=sum(a == 1 and b == 0 and s == 1 and row['new_structural']['calls'] > row['new_semantic']['calls']
                for a,b,s,row in zip(old,new,cells['new_semantic'],eligible)),
            harmed=sum(a == -1 for a in new),
            harm_after_old_help=sum(a == 1 and b == -1 for a,b in zip(old,new)))
    summary = {}
    for cell in CELLS:
        rows = [row['scores'][cell] for row in paired]
        summary[cell] = {key: sum(row[key] for row in rows)/len(rows)
            for key in ('em', 'f1', 'calls', 'input_tokens', 'output_tokens')}
        summary[cell]['repeat_search_rate'] = sum(row['repeated'] for row in rows)/sum(row['calls'] for row in rows)
        summary[cell]['accounted_cost'] = None if any(row['cost'] is None for row in rows) else sum(row['cost'] for row in rows)/len(rows)
    changed = {age: sum(row['scores'][age+'_semantic']['initial_versions'] != row['scores'][age+'_structural']['initial_versions']
        for row in paired) for age in ('old', 'new')}
    return dict(questions=len(paired), corpus_covered=len(eligible), initial_order_changed=changed,
        curves=curves, summary=summary, paired=paired)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
