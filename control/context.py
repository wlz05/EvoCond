# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Visible, entity-normalized context features for conditional relations.'
from __future__ import annotations
from copy import deepcopy
import json
import math
import re
import unicodedata
FORMAT = 'evocond_visible_context_v1'
def normalize(text, slots=()):
    text = unicodedata.normalize('NFKC', str(text))
    for slot in sorted(set(slots), key=lambda value: (-len(value), value)):
        if not isinstance(slot, str) or not slot:
            raise ValueError('Entity slots must be nonempty strings')
        text = re.sub(re.escape(slot), '<entity>', text, flags=re.I)
    text = re.sub(r'https?://\S+', '<url>', text)
    text = re.sub(r'"[^"\n]+"|\x27[^\x27\n]+\x27', '<entity>', text)
    return ' '.join(text.split())
def unit(values, dimension=None, *, zero=False):
    vector = [float(value) for value in values]
    if not vector or (dimension is not None and len(vector) != dimension) or not all(math.isfinite(x) for x in vector):
        raise ValueError('Invalid finite context vector')
    norm = math.hypot(*vector)
    if norm <= 1e-12:
        if zero:
            return [0.0]*len(vector)
        raise ValueError('Zero context embedding')
    return [value/norm for value in vector]
def embedding(encoder, text, *, query=True):
    function = getattr(encoder, 'encode_query', encoder.encode) if query else encoder.encode
    return unit(function(text))
def visible(goal, *, constraints=None, available=None, progress=None, requirements=None, slots=()):
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError('Visible context needs a current goal')
    fields = {'constraints': constraints, 'available': available, 'progress': progress, 'requirements': requirements}
    if any(value is not None and not isinstance(value, dict) for value in fields.values()):
        raise ValueError('Explicit visible fields must be objects or missing')
    json.dumps(fields, allow_nan=False)
    def clean(value):
        if isinstance(value, str):
            return normalize(value, slots)
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    return {'format': FORMAT, 'goal': normalize(goal, slots), **clean(deepcopy(fields))}
def features(context, encoder):
    if context.get('format') != FORMAT:
        raise ValueError('Unknown visible context schema')
    semantic = embedding(encoder, json.dumps({key: context[key] for key in ('goal', 'constraints', 'progress')}, sort_keys=True))
    available = context['available']
    types = ('string', 'number', 'boolean', 'object', 'array')
    explicit = [float(available is not None)]
    explicit.extend(float(bool(available and available.get(kind, False))) for kind in types)
    return semantic + explicit
def compatible(requirements, context):
    current = context.get('requirements') or {}
    return not any(key in current and current[key] is not None and current[key] != value
                   for key, value in (requirements or {}).items() if value is not None)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
