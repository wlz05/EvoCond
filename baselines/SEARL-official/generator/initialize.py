# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from .base import BaseGenerator, _GENERATORS
from .generate import APIGenerator

def get_generator(name: str) -> BaseGenerator:
    '\n    Return constructor for specified generator\n    '
    name = "".join(name.lower().split("_"))
    if name in _GENERATORS:
        return _GENERATORS[name]
    else:
        raise Exception("Error: Trying to access a generator that has not been registered")

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
