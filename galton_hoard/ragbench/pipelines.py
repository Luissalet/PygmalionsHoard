"""The pipelines a case set can be run against. Each one answers a question with a trace:

``{traced, retrieved, ranked, context, answer}``: the passages the retriever returned in its order, the order after the reranker (None when
there is none), the passages the generator received, and its answer. ``traced`` is False when a pipeline reports no passages at all.

* ``builtin``: a plain pipeline over the corpus of the case set: documents cut in fragments, a BM25 retriever, an optional reranker and a
  generator model. Every piece can be swapped for a broken one (a retriever that returns passages at random, a reranker that reverses the
  order, a generator that never sees the passages) to check that a failure is attributed to the piece that caused it.
* ``faustus``: the document RAG of a running Faustus, through its ``/api/rag/answer`` route (what its chat retrieves and what its model
  answers with it) or its notes search.
* ``http``: any other pipeline behind one HTTP request, described by where its reply keeps the passages and the answer.
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable, Optional, Sequence

import httpx

from ..checkers.jsoncheck import get_path
from ..errors import GaltonError
from ..hoard_link.docs.chunking import chunk_text
from ..langbench.bm25 import Bm25
from . import metrics as M

KINDS = ("builtin", "faustus", "http")
RETRIEVERS = ("bm25", "random", "none")
RERANKERS = ("none", "overlap", "model", "reverse", "random")
GENERATORS = ("model", "extractive", "no_context")
SOURCES = ("documents", "brain")
NO_ANSWER = "No lo sé: los pasajes no lo dicen."
MAX_PASSAGE_CHARS = 4000
MAX_DEPTH = 50

GEN_SYSTEM = ("Respondes preguntas usando SOLO los pasajes que se te dan. Si los pasajes no contienen la respuesta, responde exactamente: "
              f"«{NO_ANSWER}» No añadas datos que no estén en los pasajes. Responde en el idioma de la pregunta, en una o dos frases, sin preámbulos.")
FREE_SYSTEM = "Responde a la pregunta en una o dos frases, en el idioma de la pregunta, sin preámbulos."

Ask = Callable[[list[dict[str, str]]], str]
Rate = Callable[[str, str], float]


def passage(pid: Any, text: Any, doc: Any = "", score: Any = None) -> dict[str, Any]:
    out = {"id": str(pid if pid is not None else "")[:200], "doc": str(doc or "")[:300], "text": str(text or "")[:MAX_PASSAGE_CHARS]}
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        out["score"] = round(float(score), 4)
    return out


def _int(raw: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    try:
        value = int(raw.get(key, default) if raw.get(key) is not None else default)
    except (TypeError, ValueError):
        raise GaltonError("invalid", "rb_pipeline_field", field=key) from None
    if not low <= value <= high:
        raise GaltonError("invalid", "rb_pipeline_field", field=key)
    return value


def _choice(raw: dict[str, Any], key: str, default: str, options: Sequence[str]) -> str:
    value = str(raw.get(key) or default).strip().lower()
    if value not in options:
        raise GaltonError("invalid", "rb_pipeline_option", field=key, value=value, options=", ".join(options))
    return value


def normalise(raw: Any, *, faustus_url: str = "") -> dict[str, Any]:
    """Validate a pipeline description and fill its defaults. The generator ``model`` stays as the reference the caller gave (resolved later)."""
    if not isinstance(raw, dict):
        raise GaltonError("invalid", "rb_pipeline_shape")
    kind = _choice(raw, "kind", "builtin", KINDS)
    out: dict[str, Any] = {"kind": kind, "name": str(raw.get("name") or "").strip()[:120], "model": raw.get("model") or "",
                           "depth": _int(raw, "depth", 10, 1, MAX_DEPTH), "top_k": _int(raw, "top_k", 3, 1, MAX_DEPTH),
                           "timeout_s": _int(raw, "timeout_s", 180, 5, 1800), "max_tokens": _int(raw, "max_tokens", 400, 16, 4000)}
    if kind == "builtin":
        out.update(retriever=_choice(raw, "retriever", "bm25", RETRIEVERS), reranker=_choice(raw, "reranker", "none", RERANKERS),
                   generator=_choice(raw, "generator", "model" if raw.get("model") else "extractive", GENERATORS),
                   chunk_chars=_int(raw, "chunk_chars", 600, 80, 8000), chunk_overlap=_int(raw, "chunk_overlap", 80, 0, 4000), seed=_int(raw, "seed", 0, 0, 10 ** 9))
        if out["generator"] != "extractive" and not out["model"]:
            raise GaltonError("invalid", "rb_pipeline_needs_model", what=out["generator"])
        if out["reranker"] == "model" and not (raw.get("rerank_model") or out["model"]):
            raise GaltonError("invalid", "rb_pipeline_needs_model", what="reranker")
        out["rerank_model"] = raw.get("rerank_model") or ""
    elif kind == "faustus":
        url = str(raw.get("url") or faustus_url or "").strip().rstrip("/")
        if not url.lower().startswith(("http://", "https://")):
            raise GaltonError("invalid", "rb_pipeline_url", url=url)
        out.update(url=url, token=str(raw.get("token") or ""), source=_choice(raw, "source", "documents", SOURCES), answer=_choice(raw, "answer", "faustus", ("faustus", "model")))
        if out["answer"] == "model" and not out["model"]:
            raise GaltonError("invalid", "rb_pipeline_needs_model", what="generator")
    else:
        url = str(raw.get("url") or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            raise GaltonError("invalid", "rb_pipeline_url", url=url)
        headers = raw.get("headers") or {}
        if not isinstance(headers, dict):
            raise GaltonError("invalid", "rb_pipeline_field", field="headers")
        out.update(url=url, method=_choice(raw, "method", "POST", ("post", "get")).upper(), headers={str(k): str(v) for k, v in headers.items()},
                   body=raw.get("body") if raw.get("body") is not None else {"question": "{question}", "k": "{k}"},
                   passages_path=str(raw.get("passages_path") or ""), ranked_path=str(raw.get("ranked_path") or ""), context_path=str(raw.get("context_path") or ""),
                   answer_path=str(raw.get("answer_path") or ""), text_field=str(raw.get("text_field") or ""), id_field=str(raw.get("id_field") or ""),
                   doc_field=str(raw.get("doc_field") or ""), score_field=str(raw.get("score_field") or ""))
    if not out["name"]:
        out["name"] = describe(out)
    return out


def describe(p: dict[str, Any]) -> str:
    if p["kind"] == "builtin":
        parts = [p["retriever"], f"{p['chunk_chars']}/{p['chunk_overlap']}"]
        if p["reranker"] != "none":
            parts.append(f"rerank {p['reranker']}")
        parts += [f"top {p['top_k']}", p["generator"]]
        return " · ".join(parts)
    if p["kind"] == "faustus":
        return f"Faustus {p['source']} · k {p['depth']}"
    return f"HTTP {p['url']}"[:120]


def public(p: dict[str, Any]) -> dict[str, Any]:
    """The pipeline as it is stored and shown: no token, no header values."""
    out = {k: v for k, v in p.items() if k not in ("token", "headers")}
    if p.get("token"):
        out["token_set"] = True
    if p.get("headers"):
        out["headers"] = sorted(p["headers"])
    if isinstance(out.get("model"), dict):
        out["model"] = out["model"].get("model") or out["model"].get("name") or ""
    if isinstance(out.get("rerank_model"), dict):
        out["rerank_model"] = out["rerank_model"].get("model") or ""
    return out


# ------------------------------------------------------------------------------------------------ the pieces of the built-in pipeline
class Index:
    """The corpus cut in fragments with a BM25 index over them."""

    def __init__(self, corpus: Sequence[dict[str, Any]], chunk_chars: int = 600, chunk_overlap: int = 80):
        self.passages: list[dict[str, Any]] = []
        for doc in corpus:
            for chunk in chunk_text(doc.get("text") or "", size=chunk_chars, overlap=chunk_overlap, min_tail=min(200, chunk_chars // 3)):
                if chunk.text.strip():
                    self.passages.append(passage(f"{doc.get('id') or doc.get('name')}#{chunk.ordinal}", chunk.text.strip(), doc.get("name") or doc.get("id")))
        self.bm25 = Bm25([p["text"] for p in self.passages])

    def retrieve(self, question: str, how: str = "bm25", depth: int = 10, seed: int = 0) -> list[dict[str, Any]]:
        if how == "none" or not self.passages:
            return []
        if how == "random":                                                   # the broken retriever: the same wrong passages for the same question
            rng = random.Random(f"{seed}:{M.norm(question)}")
            return [dict(self.passages[i]) for i in rng.sample(range(len(self.passages)), min(depth, len(self.passages)))]
        scores = self.bm25.scores(question)
        order = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], i))[:depth]
        return [{**self.passages[i], "score": round(scores[i], 4)} for i in order]


def rerank(question: str, passages: Sequence[dict[str, Any]], how: str, *, rate: Optional[Rate] = None, seed: int = 0) -> Optional[list[dict[str, Any]]]:
    """The passages in the reranker's order, or None when the pipeline has no reranker."""
    rows = [dict(p) for p in passages]
    if how == "none":
        return None
    if how == "reverse":
        return rows[::-1]
    if how == "random":
        random.Random(f"{seed}:rerank:{M.norm(question)}").shuffle(rows)
        return rows
    if how == "model":
        if rate is None:
            raise GaltonError("invalid", "rb_pipeline_needs_model", what="reranker")
        scored = [(rate(question, p["text"]), i) for i, p in enumerate(rows)]
    else:
        scored = [(M.coverage(question, p["text"]), i) for i, p in enumerate(rows)]
    order = sorted(scored, key=lambda x: (-x[0], x[1]))
    return [{**rows[i], "rerank_score": round(float(s), 4)} for s, i in order]


def extractive(question: str, context: Sequence[dict[str, Any]]) -> str:
    """A generator with no model: the sentence of the passages that shares most content words with the question, or the refusal."""
    best, best_score = "", 0.0
    for p in context:
        for sentence in M.sentences(p.get("text")):
            if sentence.lstrip().startswith("#"):                       # a Markdown heading names the subject, it does not answer
                continue
            score = M.coverage(question, sentence)
            if score > best_score:
                best, best_score = sentence, score
    return best if best_score >= 0.34 else NO_ANSWER


def generator_messages(question: str, context: Sequence[dict[str, Any]], *, with_passages: bool = True) -> list[dict[str, str]]:
    if not with_passages:
        return [{"role": "system", "content": FREE_SYSTEM}, {"role": "user", "content": question.strip()}]
    shown = "\n\n".join(f"[{i}] {p['text']}" for i, p in enumerate(context, 1)) or "(ningún pasaje)"
    return [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": f"PASAJES:\n{shown}\n\nPREGUNTA: {question.strip()}"}]


def builtin_trace(index: Index, p: dict[str, Any], question: str, *, ask: Optional[Ask] = None, rate: Optional[Rate] = None) -> dict[str, Any]:
    t0 = time.monotonic()
    retrieved = index.retrieve(question, p["retriever"], p["depth"], p["seed"])
    t1 = time.monotonic()
    ranked = rerank(question, retrieved, p["reranker"], rate=rate, seed=p["seed"])
    t2 = time.monotonic()
    context = (ranked if ranked is not None else retrieved)[:p["top_k"]]
    if p["generator"] == "extractive":
        answer = extractive(question, context)
    else:
        if ask is None:
            raise GaltonError("invalid", "rb_pipeline_needs_model", what=p["generator"])
        answer = ask(generator_messages(question, context, with_passages=p["generator"] == "model"))
    t3 = time.monotonic()
    return {"traced": True, "retrieved": retrieved, "ranked": ranked, "context": context, "answer": answer,
            "latency_ms": {"retrieve": _ms(t0, t1), "rerank": _ms(t1, t2), "generate": _ms(t2, t3)}}


def _ms(a: float, b: float) -> int:
    return int((b - a) * 1000)


# ------------------------------------------------------------------------------------------------ pipelines behind HTTP
TEXT_FIELDS = ("text", "content", "document", "page_content", "passage", "chunk", "snippet", "body")
ID_FIELDS = ("id", "chunk_id", "doc_id", "path")
DOC_FIELDS = ("doc", "filename", "source", "title", "file", "path", "name")
SCORE_FIELDS = ("score", "similarity", "relevance", "rerank_score")
ANSWER_PATHS = ("answer", "response", "output", "result.answer", "data.answer", "text")


def _pick(row: dict[str, Any], named: str, candidates: Sequence[str]) -> Any:
    if named:
        found, value = get_path(row, named)
        return value if found else None
    for key in candidates:
        if row.get(key) not in (None, ""):
            return row[key]
    meta = row.get("metadata")
    if isinstance(meta, dict):
        for key in candidates:
            if meta.get(key) not in (None, ""):
                return meta[key]
    return None


def rows_to_passages(rows: Any, p: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for n, row in enumerate(rows if isinstance(rows, list) else [], 1):
        if isinstance(row, str):
            out.append(passage(n, row))
        elif isinstance(row, dict):
            text = _pick(row, p.get("text_field", ""), TEXT_FIELDS)
            if text in (None, ""):
                continue
            out.append(passage(_pick(row, p.get("id_field", ""), ID_FIELDS) or n, text, _pick(row, p.get("doc_field", ""), DOC_FIELDS), _pick(row, p.get("score_field", ""), SCORE_FIELDS)))
    return out[:MAX_DEPTH]


def fill(template: Any, values: dict[str, Any]) -> Any:
    """Put the question and ``k`` into a request body: a string that is exactly ``{k}`` becomes the number, placeholders inside a text are replaced."""
    if isinstance(template, str):
        for key, value in values.items():
            if template == "{" + key + "}":
                return value
        for key, value in values.items():
            template = template.replace("{" + key + "}", str(value))
        return template
    if isinstance(template, dict):
        return {k: fill(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [fill(v, values) for v in template]
    return template


def _request(client: httpx.Client, p: dict[str, Any], method: str, url: str, *, body: Any = None, headers: Optional[dict[str, str]] = None) -> Any:
    try:
        if method == "GET":
            response = client.get(url, params={k: v for k, v in (body or {}).items() if not isinstance(v, (dict, list))}, headers=headers, timeout=p["timeout_s"])
        else:
            response = client.post(url, json=body, headers=headers, timeout=p["timeout_s"])
    except httpx.HTTPError as exc:
        raise GaltonError("unavailable", "rb_pipeline_unreachable", name=p["name"], error=type(exc).__name__) from exc
    if response.status_code >= 400:
        detail = response.text[:300]
        try:
            body_json = response.json()
            if isinstance(body_json, dict):
                detail = str(body_json.get("detail") or body_json.get("error") or detail)[:300]
        except ValueError:
            pass
        raise GaltonError("unavailable", "rb_pipeline_error", name=p["name"], http=response.status_code, detail=detail)
    try:
        return response.json()
    except ValueError as exc:
        raise GaltonError("unavailable", "rb_pipeline_error", name=p["name"], http=response.status_code, detail="the reply is not JSON") from exc


def _local_answer(p: dict[str, Any], question: str, context: Sequence[dict[str, Any]], ask: Optional[Ask]) -> str:
    if ask is None:
        raise GaltonError("invalid", "rb_pipeline_needs_model", what="generator")
    return ask(generator_messages(question, context))


def faustus_trace(client: httpx.Client, p: dict[str, Any], question: str, *, ask: Optional[Ask] = None) -> dict[str, Any]:
    """Ask Faustus what it retrieves for the question and, unless a local generator was chosen, what it answers with it."""
    headers = {"Authorization": f"Bearer {p['token']}"} if p.get("token") else None
    local = p["answer"] == "model"
    t0 = time.monotonic()
    data = _request(client, p, "POST", p["url"] + ("/api/rag/search" if local else "/api/rag/answer"),
                    body={"query": question, "k": p["depth"], "source": p["source"]}, headers=headers)
    t1 = time.monotonic()
    if not isinstance(data, dict):
        raise GaltonError("unavailable", "rb_pipeline_error", name=p["name"], http=200, detail="unexpected reply")
    rows = data.get("results") if isinstance(data.get("results"), list) else []
    retrieved = rows_to_passages(rows, {})
    used = [bool(r.get("used")) if isinstance(r, dict) else False for r in rows][:len(retrieved)]
    context = [row for row, flag in zip(retrieved, used) if flag]
    if local:
        answer = _local_answer(p, question, context, ask)
    else:
        answer = str(data.get("answer") or "")
    t2 = time.monotonic()
    return {"traced": True, "retrieved": retrieved, "ranked": None, "context": context, "answer": answer, "served_by": str(data.get("model") or ""),
            "threshold": data.get("threshold"), "latency_ms": {"retrieve": _ms(t0, t1), "rerank": 0, "generate": _ms(t1, t2)}}


def http_trace(client: httpx.Client, p: dict[str, Any], question: str, *, ask: Optional[Ask] = None) -> dict[str, Any]:
    t0 = time.monotonic()
    data = _request(client, p, p["method"], p["url"], body=fill(p["body"], {"question": question, "k": p["depth"], "top_k": p["top_k"]}), headers=p.get("headers") or None)
    t1 = time.monotonic()

    def at(path: str) -> tuple[bool, Any]:
        return get_path(data, path) if path else (False, None)

    found_rows, rows = at(p["passages_path"])
    if not p["passages_path"] and isinstance(data, dict):
        for key in ("results", "passages", "documents", "contexts", "sources", "chunks"):
            if isinstance(data.get(key), list):
                found_rows, rows = True, data[key]
                break
    traced = bool(found_rows and isinstance(rows, list))
    retrieved = rows_to_passages(rows, p) if traced else []
    found_ranked, ranked_rows = at(p["ranked_path"])
    ranked = rows_to_passages(ranked_rows, p) if found_ranked and isinstance(ranked_rows, list) else None
    found_context, context_rows = at(p["context_path"])
    if found_context and isinstance(context_rows, list):
        context = rows_to_passages(context_rows, p)
    else:
        context = (ranked if ranked is not None else retrieved)[:p["top_k"]]
    answer: Any = None
    for path in ([p["answer_path"]] if p["answer_path"] else ANSWER_PATHS):
        found, value = get_path(data, path)
        if found and isinstance(value, str):
            answer = value
            break
    if answer is None:
        if not p.get("model"):
            raise GaltonError("unavailable", "rb_pipeline_no_answer", name=p["name"])
        answer = _local_answer(p, question, context, ask)
    t2 = time.monotonic()
    return {"traced": traced, "retrieved": retrieved, "ranked": ranked, "context": context, "answer": str(answer),
            "latency_ms": {"retrieve": _ms(t0, t1), "rerank": 0, "generate": _ms(t1, t2)}}
