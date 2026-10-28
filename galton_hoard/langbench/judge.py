"""The local language judge: which language a reply is written in, looking only at its prose, and which language a tool argument is in.

Nothing here calls a model. A reply is reduced to its prose by removing what the bench keeps in its original form in both languages: ids, tool
names, numbers, prices and dates, quoted entity names from the canonical map, brands, places and people. The words that remain are voted by
function words that only one of the two languages has (``el``, ``de``, ``cuesta``... against ``the``, ``of``, ``costs``...) and by letters that
only one of them uses (``ñ``, accents, ``¿``). The consistency of a reply is the share of the voting words that are in the expected language;
a reply with fewer than ``MIN_WORDS`` voting words has no prose to judge and gets ``None``.

It is a lexical judge for two languages: it cannot tell a third language apart and an LLM judge would read tone and register that this one
ignores. Its figures are comparable between models and runs of this bench, not with another benchmark's judge.
"""

from __future__ import annotations

import re
from typing import Optional

from .canon import CANON, LANGS, every_label, fold, words

MIN_WORDS = 3
INCONSISTENT_BELOW = 0.8

ES_ONLY = frozenset(fold(w) for w in """el la los las un una unos unas del al que y en es son está están estoy estamos hay por para con sin se su sus lo le les más pero como muy
tiene tienen cuesta cuestan precio precios producto productos carrito añadido añadida añadidos disponible disponibles unidades unidad quedan cuál cuáles cuánto cuántos cuántas también
entre sobre desde hasta ya sí ser fue ha han hemos tu tus mi mis nuestro nuestra nuestros gastamos gastos gasto mes meses hogar cita citas envío envíos despensa caduca caducan
esta este estos estas ese esa quién dónde cuándo porque según donde cuando aquí allí ahí además solo sólo todo todos toda todas otro otra otros otras segundos cada puedes puede
puedo tengo tenemos he has fue fueron será serán hay había pagó pagaron compró total está agotado agotada llegará llega antes después ahora luego""".split())
EN_ONLY = frozenset("""the of and to is are was were in on for with without it its this that these those has have had price prices product products cart added available
units unit which what how much many also between from until your our you we i my be been at by or not but more most only there their they them he she his her who where when
because according here about all each other can could will would should costs cost spent paid bought arrives arrive before after now then out stock left remaining""".split())
_SHARED = {"total", "no", "a", "me", "stock", "e", "he", "has", "son", "in"}            # words both languages use (or that mean different things), never counted
ES_ONLY = frozenset(ES_ONLY | {"de"}) - frozenset(fold(w) for w in _SHARED)
EN_ONLY = EN_ONLY - frozenset(_SHARED)
ES_ONLY, EN_ONLY = ES_ONLY - EN_ONLY, EN_ONLY - ES_ONLY

_ES_LETTERS = re.compile(r"[áéíóúñü¿¡]")
_ES_ENDINGS = ("cion", "ciones", "mente", "ando", "iendo", "dad", "dades")
_EN_ENDINGS = ("ing", "tion", "ness", "ment", "ly")

_ID = re.compile(r"\b[A-Z]{1,3}-\d{2,}\b")
_TOOL = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")
_NUMBERISH = re.compile(r"[\d][\d.,:/\-]*|€|eur\b|euros?\b|%")
_FENCE = re.compile(r"```.*?```|`[^`]*`", re.S)


_NAMES: Optional[re.Pattern[str]] = None


def _names() -> re.Pattern[str]:
    global _NAMES
    if _NAMES is None:
        _NAMES = re.compile(r"(?<![^\W_])(?:" + "|".join(re.escape(n) for n in every_label()) + r")(?![^\W_])", re.I)
    return _NAMES


def prose(text: str) -> str:
    """The reply without ids, tool names, entity names, numbers and code: only the words the model composed."""
    t = _FENCE.sub(" ", str(text))
    t = _ID.sub(" ", t)
    t = _TOOL.sub(" ", t)
    t = _names().sub(" ", t)
    t = _NUMBERISH.sub(" ", t)
    return t


_SHARED_FOLDED = frozenset(fold(w) for w in _SHARED)


def vote(token: str) -> Optional[str]:
    """``"es"``, ``"en"`` or ``None`` for one folded word."""
    if token in _SHARED_FOLDED:
        return None
    if token in ES_ONLY:
        return "es"
    if token in EN_ONLY:
        return "en"
    if len(token) > 5:
        if token.endswith(_ES_ENDINGS):
            return "es"
        if token.endswith(_EN_ENDINGS):
            return "en"
    return None



def tally(text: str) -> dict[str, int]:
    counts = {"es": 0, "en": 0}
    for raw in re.findall(r"[^\W\d_]+", prose(text), flags=re.U):
        if _ES_LETTERS.search(raw.lower()) and fold(raw) not in EN_ONLY:
            counts["es"] += 1
            continue
        v = vote(fold(raw))
        if v:
            counts[v] += 1
    return counts


def consistency(text: str, expected: str) -> dict[str, object]:
    """``score`` is the share of the language-marked words of the prose that are in ``expected``; ``None`` when there are too few of them."""
    counts = tally(text)
    other = "en" if expected == "es" else "es"
    n = counts[expected] + counts[other]
    score = round(counts[expected] / n, 4) if n >= MIN_WORDS else None
    return {"score": score, "expected": expected, "words": counts, "judged": n >= MIN_WORDS}


# ------------------------------------------------------------------------------------------------- arguments of the tools
def _vocab(lang: str) -> frozenset[str]:
    out: set[str] = set()
    for table in CANON.values():
        for entry in table.values():
            out.update(words(str(entry[lang])))
            for alias in entry["alias"][lang]:                      # type: ignore[index]
                out.update(words(alias))
    return frozenset(out)


_VOCAB = {lang: _vocab(lang) for lang in LANGS}
_ONLY = {lang: frozenset(w for w in _VOCAB[lang] if w not in _VOCAB["en" if lang == "es" else "es"] and len(w) > 2) for lang in LANGS}


def arg_language(text: str) -> dict[str, int]:
    """Votes for each language in a tool argument: words that exist only in the catalogue of one language, plus its function words."""
    votes = {"es": 0, "en": 0}
    for tok in words(text):
        stem = tok[:-1] if len(tok) > 3 and tok.endswith("s") else tok
        marked = False
        for lang in LANGS:
            if tok in _ONLY[lang] or stem in _ONLY[lang]:
                votes[lang] += 1
                marked = True
        if marked:
            continue
        v = vote(tok)
        if v:
            votes[v] += 1
    return votes


def arg_matches_data(text: str, data_lang: str) -> Optional[bool]:
    """Is a free-text tool argument in the language of the data? ``None`` when it carries no signal (an id, a number, a brand)."""
    v = arg_language(text)
    other = "en" if data_lang == "es" else "es"
    if v[data_lang] == 0 and v[other] == 0:
        return None
    return v[data_lang] >= v[other]
