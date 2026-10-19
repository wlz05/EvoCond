# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Create deterministic normalized text vectors."""
from collections import Counter
from hashlib import blake2b
from math import isfinite, log, sqrt
import re
import unicodedata
def fold(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())
class Embed:
    """Map text into a compact test vector."""
    def __init__(self, dim: int = 128):
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 16:
            raise ValueError("dim must be at least 16")
        self.dim = dim
    def terms(self, text: str) -> list[str]:
        chunks = re.findall(r"[^\W_]+", fold(text), flags=re.UNICODE)
        output: list[str] = []
        for chunk in chunks:
            output.append(chunk)
            span = 3 if chunk.isascii() else 2
            if len(chunk) > span:
                output.extend(chunk[pos:pos + span] for pos in range(len(chunk) - span + 1))
        return output
    def vector(self, text: str) -> list[float]:
        values = [0.0] * self.dim
        counts = Counter(self.terms(text))
        # Hash terms into a fixed vector.
        for term, count in counts.items():
            digest = blake2b(term.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            values[index] += sign * sqrt(count) * (1.0 + log(1.0 + len(term)))
        size = sqrt(sum(value * value for value in values))
        return [value / size for value in values] if size else values
def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions differ")
    if any(not isfinite(value) for value in left + right):
        raise ValueError("vectors must be finite")
    leftsize = sqrt(sum(value * value for value in left))
    rightsize = sqrt(sum(value * value for value in right))
    if leftsize == 0.0 or rightsize == 0.0:
        return 0.0
    return sum(one * two for one, two in zip(left, right)) / (leftsize * rightsize)
def word(name: str) -> str:
    clean = fold(name)
    value = "".join(re.findall(r"[a-z]+", clean))
    if not value:
        digest = blake2b(clean.encode("utf-8"), digest_size=5).digest()
        value = "tool" + "".join(chr(97 + item % 26) for item in digest)
    return value[:32]
def suffix(index: int) -> str:
    output = ""
    value = index
    while value:
        value, part = divmod(value - 1, 26)
        output = chr(97 + part) + output
    return output or "a"
def unique(base: str, used: set[str]) -> str:
    if base not in used:
        return base
    index = 2
    while True:
        tail = suffix(index)
        value = base[:32 - len(tail)] + tail
        if value not in used:
            return value
        index += 1

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
