"""A small BM25 index for the tool data of the language bench.

The ranking function is the usual Okapi BM25. Tokens are lower-cased words with accents removed and a trailing plural ``s`` dropped, and the
stop words of both languages are ignored, so ``zapatillas`` finds ``zapatilla`` but ``sneakers`` finds nothing in a Spanish catalogue: a search
in another language than the data comes back empty, which is the failure this bench counts. (The design follows the shopping search of
BabelArena, Apache-2.0: BM25 over the localised text of the products, an empty list when nothing scores. See ``provenance/babelarena``.)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from .canon import words as _words

STOP = frozenset("""a an the of and or to in on for with without by at from is are was be it its this that these those de del la las el los un una unos unas y o
en con sin por para al lo se su sus que es son""".split())


def stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token


def tokens(text: str) -> list[str]:
    return [stem(t) for t in _words(text) if t not in STOP]


class Bm25:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokens(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for c in self.docs:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokens(query)
        out = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in q:
                f = doc.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * (length / self.avg if self.avg else 1)))
            out.append(s)
        return out

    def rank(self, query: str) -> list[int]:
        """Indices of the documents that match at least one query token, best first (ties keep the catalogue order)."""
        scored = [(s, i) for i, s in enumerate(self.scores(query)) if s > 0]
        return [i for _, i in sorted(scored, key=lambda x: (-x[0], x[1]))]
