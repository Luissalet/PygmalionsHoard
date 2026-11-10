"""The RAG pipeline bench: case sets with their documents, runs of a pipeline over a set, and the reading of a run by piece.

A case is a question, the passages that answer it and the expected answer (or no passage at all, for a question the documents cannot answer).
A run asks a pipeline every accepted case, keeps what each piece did (retrieved passages, order after the reranker, passages given to the
generator, answer) and scores the pieces apart: retrieval (recall@k, MRR, nDCG), reranking (the same before and after), generation
(faithfulness, coverage of the expected answer, abstention). Every failed case is attributed to the piece that lost it (``attribution``),
and two runs over the same set are compared case by case.

Models are servers that are already running (a registered server model or ``{kind: server, url, model}``): the bench loads nothing itself.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from ..backends import Cancelled, ChatRequest
from ..checkers.textutil import split_reasoning
from ..errors import GaltonError
from ..logitjudge import rubric as R, scoring as S
from ..logitjudge.scorers import ServerScorer
from ..messages import item as msg_item, text
from ..suiteval.stats import sign_test, wilson
from . import attribution as A, generate as G, metrics as M, pipelines as P
from .judge import MODES, NO_LOGITS, GenerationJudge
from .store import ACTIVE, CASE_STATES, TERMINAL, RagStore

log = logging.getLogger("galton.ragbench")

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".jsonl", ".html", ".htm", ".yaml", ".yml", ".log", ".tex", ".xml", ".ini", ".toml"}
MAX_DOCS = 300
MAX_DOC_CHARS = 400_000
MAX_CORPUS_CHARS = 8_000_000
MAX_CASES = 500
DEFAULT_KS = (1, 3, 5, 10)
STORED_PASSAGE_CHARS = 1500
ABORT_AFTER = 5                      # failures of the pipeline before any case worked: the pipeline is down, not the cases
RELEVANCE = {"id": "relevance", "type": "score", "min": 0, "max": 9, "instructions": "How useful is the PASSAGE to answer the QUESTION?",
             "anchors": {"0": "unrelated", "5": "on the subject but without the answer", "9": "it contains the answer"}}
DELTA_NOISE = 0.02


def normalise_settings(raw: Optional[dict[str, Any]]) -> dict[str, Any]:
    raw = dict(raw or {})
    ks_in = raw.get("ks") or list(DEFAULT_KS)
    try:
        ks = sorted({int(k) for k in ks_in})
    except (TypeError, ValueError):
        raise GaltonError("invalid", "rb_setting", setting="ks") from None
    if not ks or len(ks) > 6 or ks[0] < 1 or ks[-1] > P.MAX_DEPTH:
        raise GaltonError("invalid", "rb_setting", setting="ks")
    out: dict[str, Any] = {"ks": ks}
    for key, default, low in (("match_threshold", M.MATCH_THRESHOLD, 0.3), ("faithful_min", 0.6, 0.0), ("coverage_min", 0.6, 0.0)):
        try:
            value = float(raw.get(key, default) if raw.get(key) is not None else default)
        except (TypeError, ValueError):
            raise GaltonError("invalid", "rb_setting", setting=key) from None
        if not low <= value <= 1.0:
            raise GaltonError("invalid", "rb_setting", setting=key)
        out[key] = value
    mode = str(raw.get("judge_mode") or "auto").strip().lower()
    if mode not in MODES:
        raise GaltonError("invalid", "rb_judge_mode", mode=mode)
    out["judge_mode"] = mode
    try:
        out["parallel"] = max(1, min(8, int(raw.get("parallel") or 4)))
    except (TypeError, ValueError):
        raise GaltonError("invalid", "rb_setting", setting="parallel") from None
    return out


class _Models:
    """The model connections one run (or one call) opens, closed together."""

    def __init__(self, bench: "RagBench", cancel: Callable[[], bool] = lambda: False):
        self.bench, self.cancel = bench, cancel
        self.backends: list[Any] = []
        self.clients: list[Any] = []

    def client(self) -> httpx.Client:
        client = self.bench.client_factory() if self.bench.client_factory else httpx.Client(trust_env=False, timeout=180.0)
        self.clients.append(client)
        return client

    def ask(self, c: dict[str, Any], *, max_tokens: int = 400, timeout_s: float = 180.0) -> P.Ask:
        backend = self.bench.backend_factory(c["api"] or "openai", c["url"], c["model"] or c["name"])
        self.backends.append(backend)

        def ask(messages: list[dict[str, str]]) -> str:
            budget = max_tokens
            for attempt in (1, 2):
                done = backend.chat(ChatRequest(messages=messages, temperature=0.0, max_tokens=budget, effort="off", timeout_s=timeout_s), self.cancel)
                if done.error:
                    raise GaltonError("unavailable", "rb_model_failed", name=c["name"], error=str(done.error)[:200])
                if done.truncated and attempt == 1:                           # the model spent the budget thinking: once more, with room
                    budget = budget * 3 + 600
                    continue
                return split_reasoning(done.text)[0].strip()
            return ""
        return ask

    def scorer(self, c: dict[str, Any], *, probe: bool = True) -> tuple[Optional[ServerScorer], Any]:
        settings = self.bench.settings
        scorer = ServerScorer(c, client_factory=self.bench.client_factory, top_logprobs=int(settings.get("logit.top_logprobs")),
                              timeout_s=float(settings.get("logit.timeout_s")), no_thinking=bool(settings.get("logit.no_thinking")))
        client = scorer.client_factory()
        self.clients.append(client)
        if probe and not scorer.probe(client).get("supported"):
            return None, None
        return scorer, client

    def close(self) -> None:
        for thing in [*self.backends, *self.clients]:
            try:
                thing.close()
            except Exception:  # noqa: BLE001 - closing a connection never fails a run
                pass


class RagBench:
    def __init__(self, store: RagStore, runner: Any, settings: Any, *, backend_factory: Callable[..., Any], client_factory: Optional[Callable[[], httpx.Client]] = None,
                 clock: Callable[[], float] = time.time, emit: Callable[..., None] = lambda *a, **k: None, faustus_token: Callable[[], str] = lambda: ""):
        self.store, self.runner, self.settings, self.clock, self.emit = store, runner, settings, clock, emit
        self.backend_factory, self.client_factory, self.faustus_token = backend_factory, client_factory, faustus_token
        self.threads: dict[str, tuple[threading.Thread, threading.Event]] = {}
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ models
    def _server(self, ref: Any) -> dict[str, Any]:
        c = self.runner.resolve_contestants([ref])[0]
        if c["kind"] != "server" or not c.get("url"):
            raise GaltonError("invalid", "rb_model_server", name=c["name"])
        if c.get("remote") and not c.get("remote_ok"):
            raise GaltonError("remote_not_allowed", "remote_not_allowed", name=c["name"])
        self.runner.ensure_served(c)
        return c

    def _rate(self, c: dict[str, Any], models: _Models) -> P.Rate:
        scorer, client = models.scorer(c)
        question = R.normalise_rubric([RELEVANCE])[0]
        ask = models.ask(c, max_tokens=16, timeout_s=120.0)

        def rate(query: str, passage: str) -> float:
            messages = R.build_messages(f"QUESTION:\n{query}\n\nPASSAGE:\n{passage[:1800]}", question)
            if scorer is not None:
                try:
                    return S.answer_for(question, scorer.score(client, messages, question["labels"]))["expected"] / 9.0
                except GaltonError as exc:
                    if exc.key not in NO_LOGITS:
                        raise
            digit = re.search(r"\d", ask(messages))
            return int(digit.group()) / 9.0 if digit else 0.0
        return rate

    def _judge(self, ref: Any, mode: str, models: _Models) -> GenerationJudge:
        ref = ref or str(self.settings.get("judge.contestant") or "")
        if mode == "lexical" or (mode == "auto" and not ref):
            return GenerationJudge("lexical")
        if not ref:
            raise GaltonError("invalid", "rb_judge_needs_model", mode=mode)
        c = self._server(ref)
        scorer = client = None
        if mode in ("auto", "logit"):
            scorer, client = models.scorer(c, probe=mode == "auto")
        return GenerationJudge(mode, scorer=scorer, client=client, ask=models.ask(c, max_tokens=500, timeout_s=180.0), name=c["name"])

    @staticmethod
    def _generates(p: dict[str, Any]) -> bool:
        """Does this bench's own generator model write the answers of the pipeline?"""
        if p["kind"] == "builtin":
            return p["generator"] != "extractive"
        if p["kind"] == "faustus":
            return p["answer"] == "model"
        return bool(p.get("model"))

    def _models_needed(self, p: dict[str, Any]) -> list[Any]:
        refs = [p["model"]] if self._generates(p) and p.get("model") else []
        if p["kind"] == "builtin" and p["reranker"] == "model":
            refs.append(p.get("rerank_model") or p["model"])
        return refs

    def _prepare(self, pipeline: dict[str, Any], set_row: Optional[dict[str, Any]], models: _Models) -> tuple[Callable[[str], dict[str, Any]], dict[str, Any]]:
        """The function that answers one question with a trace, and the pipeline with its models resolved."""
        p = dict(pipeline)
        ask: Optional[P.Ask] = None
        if self._generates(p) and p.get("model"):
            c = self._server(p["model"])
            p["model_name"] = c["name"]
            ask = models.ask(c, max_tokens=p["max_tokens"], timeout_s=p["timeout_s"])
        if p["kind"] == "builtin":
            corpus = (set_row or {}).get("corpus") or []
            if not corpus:
                raise GaltonError("invalid", "rb_no_corpus", name=(set_row or {}).get("name") or "?")
            index = P.Index(corpus, p["chunk_chars"], p["chunk_overlap"])
            p["fragments"] = len(index.passages)
            rate = self._rate(self._server(p["rerank_model"] or p["model"]), models) if p["reranker"] == "model" else None
            return (lambda q: {"cut": p["top_k"], **P.builtin_trace(index, p, q, ask=ask, rate=rate)}), p
        client = models.client()
        if p["kind"] == "faustus":
            if not p.get("token"):
                p["token"] = self.faustus_token()
            return (lambda q: P.faustus_trace(client, p, q, ask=ask)), p
        return (lambda q: P.http_trace(client, p, q, ask=ask)), p

    # ------------------------------------------------------------------ sets
    def _set(self, ref: str, *, corpus: bool = True) -> dict[str, Any]:
        found = self.store.find_set(str(ref or "").strip())
        if found is None:
            raise GaltonError("not_found", "rb_no_set", ref=ref)
        if not corpus:
            found.pop("corpus", None)
        return found

    def _set_card(self, s: dict[str, Any]) -> dict[str, Any]:
        runs = self.store.runs(set_id=s["id"], limit=200)
        return {"id": s["id"], "name": s["name"], "description": s["description"], "documents": s["documents"], "chars": sum(d["chars"] for d in s["documents"]),
                "cases": self.store.case_counts(s["id"]), "runs": len(runs), "created_ts": s["created_ts"], "updated_ts": s["updated_ts"]}

    def sets(self) -> dict[str, Any]:
        rows = [self._set_card(s) for s in self.store.sets()]
        return {"sets": rows, "count": len(rows)}

    @staticmethod
    def _read_paths(paths: list[str]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for raw in paths:
            path = Path(str(raw).strip())
            if not path.is_absolute():
                raise GaltonError("invalid", "path_not_absolute", path=str(raw))
            if path.is_dir():
                files = sorted(f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in TEXT_SUFFIXES)
            elif path.is_file():
                files = [path]
            else:
                raise GaltonError("not_found", "file_missing", path=str(raw))
            for f in files[:MAX_DOCS]:
                try:
                    out.append({"name": f.name, "text": f.read_text(encoding="utf-8", errors="replace")})
                except OSError as exc:
                    raise GaltonError("invalid", "rb_doc_unreadable", path=str(f), error=type(exc).__name__) from exc
        return out

    def set_save(self, *, id: str = "", name: str = "", description: str = "", documents: Optional[list[Any]] = None, paths: Optional[list[str]] = None,
                 cases: Optional[list[dict[str, Any]]] = None, replace_documents: bool = False) -> dict[str, Any]:
        """Create a case set, or add documents and cases to an existing one (``id``)."""
        existing = self._set(id) if id else None
        docs_in: list[dict[str, str]] = []
        for n, raw in enumerate(documents or [], 1):
            if isinstance(raw, str):
                docs_in.append({"name": f"doc-{n}", "text": raw})
            elif isinstance(raw, dict) and str(raw.get("text") or "").strip():
                docs_in.append({"name": str(raw.get("name") or f"doc-{n}")[:200], "text": str(raw["text"])})
            else:
                raise GaltonError("invalid", "rb_doc_shape", n=n)
        docs_in += self._read_paths(list(paths or []))
        corpus = [] if (existing is None or replace_documents) else list(existing["corpus"])
        names = {d["name"] for d in corpus}
        for doc in docs_in:
            text_ = doc["text"].replace("\r\n", "\n").strip()
            if not text_:
                continue
            name_, n = doc["name"], 2
            while name_ in names:
                name_, n = f"{doc['name']} ({n})", n + 1
            names.add(name_)
            corpus.append({"id": f"d{len(corpus) + 1}", "name": name_, "text": text_[:MAX_DOC_CHARS]})
        if len(corpus) > MAX_DOCS or sum(len(d["text"]) for d in corpus) > MAX_CORPUS_CHARS:
            raise GaltonError("too_large", "rb_corpus_too_large", docs=MAX_DOCS, chars=MAX_CORPUS_CHARS)
        if existing is None:
            if not name.strip():
                raise GaltonError("invalid", "rb_set_name")
            row = self.store.create_set(name.strip()[:120], description.strip()[:1000], corpus)
        else:
            fields: dict[str, Any] = {"corpus": corpus}
            if name.strip():
                fields["name"] = name.strip()[:120]
            if description.strip():
                fields["description"] = description.strip()[:1000]
            row = self.store.update_set(existing["id"], **fields)
        for raw in cases or []:
            self.case_save(set=row["id"], **{k: v for k, v in raw.items() if k in ("question", "expected", "passages", "answerable", "require", "status", "notes")})
        return self.set_get(row["id"])

    def set_get(self, ref: str, *, status: str = "", with_text: bool = False) -> dict[str, Any]:
        s = self._set(ref)
        if status and status not in CASE_STATES:
            raise GaltonError("invalid", "rb_case_state", state=status)
        card = self._set_card(s)
        if with_text:
            card["corpus"] = s["corpus"]
        runs = [self.card(r) for r in self.store.runs(set_id=s["id"], limit=30)]
        return {"set": card, "cases": self.store.cases(s["id"], status), "runs": runs}

    def set_delete(self, ref: str) -> dict[str, Any]:
        s = self._set(ref, corpus=False)
        for run in self.store.runs(set_id=s["id"], states=ACTIVE, limit=50):
            self.cancel(run["id"])
        self.store.delete_set(s["id"])
        return {"deleted": s["id"], "name": s["name"]}

    # ------------------------------------------------------------------ cases
    @staticmethod
    def _passages(raw: Any) -> list[dict[str, str]]:
        out = []
        for n, p in enumerate(raw or [], 1):
            if isinstance(p, str):
                p = {"text": p}
            if not isinstance(p, dict):
                raise GaltonError("invalid", "rb_passage_shape", n=n)
            row = {k: str(p.get(k) or "").strip()[:P.MAX_PASSAGE_CHARS] for k in ("id", "doc", "text")}
            if not (row["id"] or row["doc"] or row["text"]):
                raise GaltonError("invalid", "rb_passage_shape", n=n)
            out.append({k: v for k, v in row.items() if v})
        return out[:20]

    def case_save(self, *, set: str = "", id: str = "", question: Optional[str] = None, expected: Optional[str] = None, passages: Any = None,
                  answerable: Optional[bool] = None, require: Optional[str] = None, status: Optional[str] = None, notes: Optional[str] = None) -> dict[str, Any]:
        """Add a case to a set, or change one (``id``): its text, its passages, or its review state."""
        old = self.store.case(id) if id else None
        if id and old is None:
            raise GaltonError("not_found", "rb_no_case", id=id)
        if status is not None and status not in CASE_STATES:
            raise GaltonError("invalid", "rb_case_state", state=status)
        if require is not None and require not in ("all", "any"):
            raise GaltonError("invalid", "rb_setting", setting="require")
        fields: dict[str, Any] = {}
        if question is not None:
            fields["question"] = " ".join(question.split())[:2000]
        if expected is not None:
            fields["expected"] = expected.strip()[:4000]
        if passages is not None:
            fields["passages"] = self._passages(passages)
        for key, value in (("answerable", answerable), ("require", require), ("status", status), ("notes", None if notes is None else notes[:1000])):
            if value is not None:
                fields[key] = value
        merged = {**(old or {"question": "", "expected": "", "passages": [], "answerable": True}), **fields}
        if not merged["question"]:
            raise GaltonError("invalid", "rb_case_question")
        if merged["answerable"] and not merged["passages"] and not merged["expected"]:
            raise GaltonError("invalid", "rb_case_empty")
        if not merged["answerable"]:
            fields["passages"], fields["expected"] = [], ""
        if old is not None:
            return self.store.update_case(old["id"], **fields)
        s = self._set(set, corpus=False)
        if sum(self.store.case_counts(s["id"]).values()) >= MAX_CASES:
            raise GaltonError("too_large", "rb_too_many_cases", max=MAX_CASES)
        return self.store.add_case(s["id"], question=fields["question"], expected=fields.get("expected", ""), passages=fields.get("passages", []),
                                   answerable=merged["answerable"], require=fields.get("require", "all"), status=fields.get("status", "accepted"),
                                   origin="manual", notes=fields.get("notes", ""))

    def cases_review(self, *, set: str, status: str, ids: Optional[list[str]] = None, all_proposed: bool = False) -> dict[str, Any]:
        s = self._set(set, corpus=False)
        if status not in CASE_STATES:
            raise GaltonError("invalid", "rb_case_state", state=status)
        mine = {c["id"]: c for c in self.store.cases(s["id"])}
        chosen = [c["id"] for c in mine.values() if c["status"] == "proposed"] if all_proposed else [str(i) for i in ids or []]
        unknown = [i for i in chosen if i not in mine]
        if unknown:
            raise GaltonError("not_found", "rb_no_case", id=unknown[0])
        for cid in chosen:
            self.store.update_case(cid, status=status)
        return {"set": s["id"], "changed": len(chosen), "status": status, "cases": self.store.case_counts(s["id"])}

    def case_remove(self, cid: str) -> dict[str, Any]:
        old = self.store.case(cid)
        if old is None:
            raise GaltonError("not_found", "rb_no_case", id=cid)
        self.store.delete_case(cid)
        return {"deleted": cid, "set": old["set_id"]}

    def cases_generate(self, *, set: str, model: Any, n: int = 10, per_fragment: int = 1, unanswerable: int = 0, chunk_chars: int = 900, seed: int = 0) -> dict[str, Any]:
        s = self._set(set)
        if not s["corpus"]:
            raise GaltonError("invalid", "rb_no_corpus", name=s["name"])
        c = self._server(model)
        models = _Models(self)
        try:
            out = G.propose(s["corpus"], models.ask(c, max_tokens=700, timeout_s=180.0), n=n, per_fragment=per_fragment, unanswerable=unanswerable,
                            chunk_chars=chunk_chars, seed=seed, known=[k["question"] for k in self.store.cases(s["id"])])
        finally:
            models.close()
        room = MAX_CASES - sum(self.store.case_counts(s["id"]).values())
        saved = [self.store.add_case(s["id"], question=k["question"], expected=k["expected"], passages=k["passages"], answerable=k["answerable"], status="proposed",
                                     origin="generated", notes=k["notes"]) for k in out["cases"][:max(0, room)]]
        return {"set": s["id"], "model": c["name"], "proposed": len(saved), "asked": out["asked"], "fragments": out["fragments"], "discarded": out["discarded"],
                "cases": saved, "counts": self.store.case_counts(s["id"])}

    # ------------------------------------------------------------------ one question, no run
    def probe(self, *, pipeline: dict[str, Any], question: str, set: str = "") -> dict[str, Any]:
        """Ask a pipeline one question and show its trace, to check an adapter before a run. Nothing is stored."""
        s = self._set(set) if set else None
        models = _Models(self)
        try:
            fn, p = self._prepare(P.normalise(pipeline, faustus_url=str(self.settings.get("faustus.url") or "")), s, models)
            trace = fn(question)
        finally:
            models.close()
        return {"pipeline": P.public(p), "question": question, **{k: trace.get(k) for k in ("traced", "retrieved", "ranked", "context", "answer", "latency_ms")}}

    # ------------------------------------------------------------------ runs
    def card(self, run: dict[str, Any]) -> dict[str, Any]:
        summary = run.get("summary") or {}
        head = {k: summary.get(k) for k in ("cases", "passed", "pass_rate")} if summary else {}
        if summary:
            head["main_piece"] = (summary.get("attribution") or {}).get("main")
        return {"id": run["id"], "label": run["label"], "state": run["state"], "set": run["set_id"], "set_name": run["set_name"], "pipeline": run["pipeline"],
                "settings": {k: v for k, v in (run["settings"] or {}).items() if k != "case_ids"}, "total": run["total"], "done": run["done"], "error": run["error"], "created_ts": run["created_ts"],
                "started_ts": run["started_ts"], "finished_ts": run["finished_ts"], "head": head, "caller": run["caller"]}

    def _run(self, rid: str) -> dict[str, Any]:
        run = self.store.run(str(rid or "").strip())
        if run is None:
            raise GaltonError("not_found", "rb_no_run", id=rid)
        return run

    def run_start(self, *, set: str, pipeline: dict[str, Any], judge: Any = "", settings: Optional[dict[str, Any]] = None, label: str = "", include_proposed: bool = False,
                  limit: int = 0, source: str = "ui", caller: str = "", wait_s: float = 0.0, start_thread: bool = True) -> dict[str, Any]:
        s = self._set(set, corpus=False)
        rs = normalise_settings(settings)
        cases = [c for c in self.store.cases(s["id"]) if c["status"] == "accepted" or (include_proposed and c["status"] == "proposed")]
        if limit > 0:
            cases = cases[:limit]
        if not cases:
            raise GaltonError("invalid", "rb_no_cases", name=s["name"])
        p = P.normalise(pipeline, faustus_url=str(self.settings.get("faustus.url") or ""))
        judge_name = ""
        for ref in self._models_needed(p):                               # resolved before a run exists, so a wrong model name fails the call and not the run
            self._server(ref)
        ref = judge or str(self.settings.get("judge.contestant") or "")
        if rs["judge_mode"] != "lexical" and ref:
            judge_name = self._server(ref)["name"]
        elif rs["judge_mode"] in ("logit", "model"):
            raise GaltonError("invalid", "rb_judge_needs_model", mode=rs["judge_mode"])
        run = self.store.create_run(set_id=s["id"], set_name=s["name"], label=(label.strip() or f"{p['name']} · {s['name']}")[:160], pipeline=P.public(p),
                                    settings={**rs, "judge": judge_name, "case_ids": [c["id"] for c in cases]}, total=len(cases), source=source, caller=caller)
        if start_thread:
            cancel = threading.Event()
            thread = threading.Thread(target=self.execute, args=(run["id"], cancel, p, judge), name=f"ragbench-{run['id']}", daemon=True)
            with self.lock:
                self.threads[run["id"]] = (thread, cancel)
            thread.start()
            if wait_s > 0:
                thread.join(wait_s)
        else:
            self.execute(run["id"], threading.Event(), p, judge)
        return self.status(run["id"])

    def _one(self, case: dict[str, Any], fn: Callable[[str], dict[str, Any]], judge: GenerationJudge, rs: dict[str, Any], state: dict[str, Any]) -> Optional[dict[str, Any]]:
        if state["cancel"].is_set() or state["abort"]:
            return None
        answerable = bool(case["answerable"])
        refs = case["passages"] if answerable else []
        trace: dict[str, Any] = {"traced": False, "retrieved": [], "ranked": None, "context": [], "answer": ""}
        gen: dict[str, Any] = {}
        error = ""
        try:
            trace = fn(case["question"])
            gen = judge.judge(case["question"], trace["context"], case["expected"] if answerable else "", trace["answer"])
        except Cancelled:
            return None
        except GaltonError as exc:
            error = exc.message
        except Exception as exc:  # noqa: BLE001 - one case that breaks is a failed case, not a failed run
            log.exception("ragbench case %s failed", case["id"])
            error = f"{type(exc).__name__}: {exc}"[:300]
        with state["lock"]:
            if error and not gen:
                state["errors"] += 1
                if state["errors"] >= ABORT_AFTER and not state["worked"]:
                    state["abort"] = error
            elif not error:
                state["worked"] += 1
        threshold, ks = rs["match_threshold"], rs["ks"]
        retrieved, ranked, context = trace.get("retrieved") or [], trace.get("ranked"), trace.get("context") or []
        after = ranked if ranked is not None else retrieved
        found_r, found_k, found_c = M.hits(refs, retrieved, threshold), M.hits(refs, after, threshold), M.hits(refs, context, threshold)
        cut = int(trace.get("cut") or len(context))
        traced = bool(trace.get("traced"))
        verdict = A.judge_case(answerable=answerable, n_refs=len(refs), traced=traced, found_retrieved=found_r, found_ranked=found_k, found_context=found_c, cut=cut,
                               gen=gen, require=case.get("require") or "all", faithful_min=rs["faithful_min"], coverage_min=rs["coverage_min"], error=error if not gen else "", reranked=ranked is not None)
        metrics: dict[str, Any] = {"generation": gen, "cut": cut, "context_size": len(context)}
        if answerable and traced and refs and not (error and not gen):
            before = {"mrr": round(M.mrr(found_r), 4), "ndcg": M._round(M.ndcg_at(found_r, len(refs), cut)), "recall": M._round(M.recall_at(found_r, len(refs), cut)),
                      "first_rank": M.first_rank(found_r)}
            now = {"mrr": round(M.mrr(found_k), 4), "ndcg": M._round(M.ndcg_at(found_k, len(refs), cut)), "recall": M._round(M.recall_at(found_k, len(refs), cut)),
                   "first_rank": M.first_rank(found_k)}
            metrics["retrieval"] = {**M.retrieval_metrics(found_r, len(refs), ks), "retrieved": len(retrieved)}
            metrics["rerank"] = {"applied": ranked is not None, "before": before, "after": now}
            metrics["context_recall"] = M._round(M.recall_at(found_c, len(refs), len(context)))

        def keep(rows: Any, found: list[list[int]]) -> list[dict[str, Any]]:
            return [{**{k: v for k, v in row.items() if k != "text"}, "text": row.get("text", "")[:STORED_PASSAGE_CHARS], "hit": bool(h)} for row, h in zip(rows or [], found)][:30]

        stored = {"traced": traced, "retrieved": keep(retrieved, found_r), "ranked": keep(ranked, found_k) if ranked is not None else None, "context": keep(context, found_c),
                  "answer": str(trace.get("answer") or "")[:6000], "latency_ms": trace.get("latency_ms") or {}, "expected": case["expected"], "references": refs,
                  "answerable": answerable}
        return {"case": case, "trace": stored, "metrics": metrics, "verdict": verdict, "error": error}

    def execute(self, rid: str, cancel: threading.Event, pipeline: dict[str, Any], judge_ref: Any = "") -> dict[str, Any]:
        run = self.store.run(rid)
        if run is None or run["state"] != "queued":
            return run or {}
        self.store.update_run(rid, state="running", started_ts=self.clock())
        rs = run["settings"]
        models = _Models(self, cancel.is_set)
        try:
            s = self.store.set(run["set_id"])
            fn, p = self._prepare(pipeline, s, models)
            judge = self._judge(judge_ref, rs["judge_mode"], models)
            by_id = {c["id"]: c for c in self.store.cases(run["set_id"])}
            cases = [by_id[i] for i in rs.get("case_ids", []) if i in by_id]
            state = {"cancel": cancel, "abort": "", "errors": 0, "worked": 0, "lock": threading.Lock()}
            done = 0
            with ThreadPoolExecutor(max_workers=rs["parallel"], thread_name_prefix=f"ragbench-{rid[-6:]}") as pool:
                futures = [pool.submit(self._one, c, fn, judge, rs, state) for c in cases]
                for future in as_completed(futures):
                    row = future.result()
                    if row is None:
                        continue
                    case = row["case"]
                    self.store.put_result(rid, case["id"], position=case["position"], question=case["question"], trace=row["trace"], metrics=row["metrics"],
                                          verdict=row["verdict"], ok=row["verdict"]["ok"], piece=row["verdict"]["piece"], error=row["error"])
                    done += 1
                    self.store.update_run(rid, done=done)
            results = self.store.results(rid)
            summary = self.summarise(results, rs, P.public(p), judge)
            if state["abort"]:
                self.store.update_run(rid, state="failed", finished_ts=self.clock(), error=state["abort"], summary=summary, pipeline=P.public(p))
            else:
                self.store.update_run(rid, state="cancelled" if cancel.is_set() else "done", finished_ts=self.clock(), summary=summary, pipeline=P.public(p))
        except GaltonError as exc:
            self.store.update_run(rid, state="failed", finished_ts=self.clock(), error=exc.message)
        except Exception as exc:  # noqa: BLE001 - a stored run must never stay 'running'
            log.exception("ragbench run %s failed", rid)
            self.store.update_run(rid, state="failed", finished_ts=self.clock(), error=f"{type(exc).__name__}: {exc}"[:300])
        finally:
            models.close()
        final = self.store.run(rid)
        self.emit("ragbench.finished", {"run": rid, "state": final["state"]})
        return final

    # ------------------------------------------------------------------ reading a run
    @staticmethod
    def summarise(results: list[dict[str, Any]], rs: dict[str, Any], pipeline: dict[str, Any], judge: Optional[GenerationJudge] = None) -> dict[str, Any]:
        """A run by piece: retrieval, reranking, generation, the pass rate and what the failures are attributed to."""
        ks = [str(k) for k in rs["ks"]]
        scored = [r for r in results if not (r["error"] and not r["metrics"].get("generation"))]
        verdicts = [r["verdict"] for r in results]
        passed = sum(1 for v in verdicts if v.get("ok"))
        with_retrieval = [r["metrics"]["retrieval"] for r in scored if r["metrics"].get("retrieval")]
        reranks = [r["metrics"]["rerank"] for r in scored if r["metrics"].get("rerank")]
        gens = [r["metrics"]["generation"] for r in scored if r["metrics"].get("generation")]

        def side(name: str, key: str) -> Optional[float]:
            return M.mean([x[name][key] for x in reranks])

        def delta(key: str) -> dict[str, Any]:
            before, after = side("before", key), side("after", key)
            return {"before": before, "after": after, "delta": None if before is None or after is None else round(after - before, 4)}

        def share(rows: list[dict[str, Any]], test: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
            n = sum(1 for r in rows if test(r))
            return {"cases": len(rows), "abstained": n, "rate": round(n / len(rows), 4) if rows else None}

        answerable = [r for r in scored if r["trace"].get("answerable")]
        with_passage = [r for r in answerable if r["verdict"].get("context_ok")]
        without_passage = [r for r in answerable if r["verdict"].get("context_ok") is False]
        unanswerable = [r for r in scored if not r["trace"].get("answerable")]
        abstained = lambda r: bool(r["metrics"]["generation"].get("abstained"))  # noqa: E731
        modes: dict[str, int] = {}
        for g in gens:
            modes[g.get("mode") or "?"] = modes.get(g.get("mode") or "?", 0) + 1
        warnings = []
        if judge is not None and judge.notes:
            warnings.append(msg_item(text("rb_warn_judge_fallback", mode=judge.mode)))
        if modes.get("lexical"):
            warnings.append(msg_item(text("rb_warn_lexical", n=modes["lexical"])))
        if judge is not None and judge.name and judge.name == pipeline.get("model_name"):
            warnings.append(msg_item(text("rb_warn_self_judged", name=judge.name)))
        untraced = sum(1 for r in scored if not r["trace"].get("traced"))
        if untraced:
            warnings.append(msg_item(text("rb_warn_untraced", n=untraced)))
        if len(results) < 10:
            warnings.append(msg_item(text("rb_warn_few_cases", n=len(results))))
        latency = {k: M.mean([float((r["trace"].get("latency_ms") or {}).get(k) or 0) for r in scored]) for k in ("retrieve", "rerank", "generate")}
        return {
            "cases": len(results), "passed": passed, "pass_rate": round(passed / len(results), 4) if results else None, "pass_ci95": wilson(passed, len(results)),
            "errors": len(results) - len(scored),
            "retrieval": {"cases": len(with_retrieval), "recall": {k: M.mean([x["recall"][k] for x in with_retrieval]) for k in ks},
                          "ndcg": {k: M.mean([x["ndcg"][k] for x in with_retrieval]) for k in ks}, "mrr": M.mean([x["mrr"] for x in with_retrieval]),
                          "hit_rate": M.mean([1.0 if x["first_rank"] else 0.0 for x in with_retrieval]), "mean_retrieved": M.mean([float(x["retrieved"]) for x in with_retrieval])},
            "rerank": {"applied": any(x["applied"] for x in reranks), "cases": len(reranks), "mrr": delta("mrr"), "ndcg": delta("ndcg"), "recall": delta("recall"),
                       "rescued": sum(1 for v in verdicts if v.get("rescued")), "demoted": sum(1 for v in verdicts if v.get("demoted"))},
            "generation": {"judge": (judge.name if judge else "") or rs.get("judge") or "", "judge_modes": modes,
                           "faithfulness": M.mean([g.get("faithfulness") for g in gens]), "coverage": M.mean([g.get("coverage") for g in gens]),
                           "with_passage": {"cases": len(with_passage), "right": sum(1 for r in with_passage if r["verdict"].get("answer_ok")),
                                            "rate": round(sum(1 for r in with_passage if r["verdict"].get("answer_ok")) / len(with_passage), 4) if with_passage else None},
                           "abstention": {"with_passage": share(with_passage, abstained), "without_passage": share(without_passage, abstained),
                                          "unanswerable": share(unanswerable, abstained)}},
            "attribution": A.summarise(verdicts), "latency_ms": latency, "warnings": warnings,
        }

    def status(self, rid: str) -> dict[str, Any]:
        return {"run": self.card(self._run(rid))}

    @staticmethod
    def _row(r: dict[str, Any]) -> dict[str, Any]:
        gen = r["metrics"].get("generation") or {}
        ret = r["metrics"].get("retrieval") or {}
        return {"case": r["case_id"], "position": r["position"], "question": r["question"], "ok": r["ok"], "piece": r["piece"], "cause": r["verdict"].get("cause", ""),
                "generator_note": r["verdict"].get("generator_note", ""), "retrieved_ok": r["verdict"].get("retrieved_ok"), "context_ok": r["verdict"].get("context_ok"),
                "answer_ok": r["verdict"].get("answer_ok"), "first_rank": ret.get("first_rank"), "mrr": ret.get("mrr"), "context_recall": r["metrics"].get("context_recall"),
                "faithfulness": gen.get("faithfulness"), "coverage": gen.get("coverage"), "abstained": gen.get("abstained"), "judge_mode": gen.get("mode"),
                "answerable": r["trace"].get("answerable"), "answer": str(r["trace"].get("answer") or "")[:240], "error": r["error"]}

    def result(self, rid: str, *, case: str = "", only_failed: bool = False) -> dict[str, Any]:
        run = self._run(rid)
        if case:
            row = self.store.result(run["id"], case)
            if row is None:
                raise GaltonError("not_found", "rb_no_result", id=case, run=run["id"])
            return {"run": self.card(run), "case": {**self._row(row), "expected": row["trace"].get("expected"), "references": row["trace"].get("references"),
                                                    "answer": row["trace"].get("answer"), "retrieved": row["trace"].get("retrieved"), "ranked": row["trace"].get("ranked"),
                                                    "context": row["trace"].get("context"), "metrics": row["metrics"], "verdict": row["verdict"],
                                                    "latency_ms": row["trace"].get("latency_ms")}}
        rows = [self._row(r) for r in self.store.results(run["id"])]
        if only_failed:
            rows = [r for r in rows if not r["ok"]]
        return {"run": self.card(run), "summary": run["summary"], "cases": rows}

    def list(self, *, set: str = "", limit: int = 30) -> dict[str, Any]:
        sid = self._set(set, corpus=False)["id"] if set else ""
        rows = [self.card(r) for r in self.store.runs(set_id=sid, limit=limit)]
        return {"runs": rows, "count": len(rows)}

    def cancel(self, rid: str) -> dict[str, Any]:
        run = self._run(rid)
        if run["state"] in TERMINAL:
            return {"run": run["id"], "state": run["state"], "cancelled": False}
        with self.lock:
            entry = self.threads.get(run["id"])
        if entry is not None and entry[0].is_alive():
            entry[1].set()
            return {"run": run["id"], "state": run["state"], "cancelled": True}
        self.store.update_run(run["id"], state="cancelled", finished_ts=self.clock())
        return {"run": run["id"], "state": "cancelled", "cancelled": True}

    def delete(self, rid: str) -> dict[str, Any]:
        run = self._run(rid)
        with self.lock:
            entry = self.threads.get(run["id"])
        if run["state"] in ACTIVE and entry is not None and entry[0].is_alive():
            raise GaltonError("conflict", "rb_run_active", id=run["id"])
        self.store.delete_run(run["id"])
        return {"deleted": run["id"], "label": run["label"]}

    # ------------------------------------------------------------------ two runs, case by case
    def compare(self, a: str, b: str) -> dict[str, Any]:
        """Run ``b`` against run ``a`` (the baseline), case by case: what got fixed, what broke, and in which piece."""
        ra, rb = self._run(a), self._run(b)
        for run in (ra, rb):
            if run["state"] not in ("done", "cancelled", "failed") or not run["summary"]:
                raise GaltonError("conflict", "rb_run_not_finished", id=run["id"])
        rows_a = {r["case_id"]: r for r in self.store.results(ra["id"])}
        rows_b = {r["case_id"]: r for r in self.store.results(rb["id"])}
        if ra["set_id"] != rb["set_id"]:                                  # two sets: the same question is the same case
            by_question = {M.norm(r["question"]): r for r in rows_b.values()}
            rows_b = {cid: by_question[M.norm(r["question"])] for cid, r in rows_a.items() if M.norm(r["question"]) in by_question}
        shared = [cid for cid in rows_a if cid in rows_b]
        cases, improved, regressed = [], 0, 0
        moves: dict[str, int] = {}

        def value(row: dict[str, Any], *path: str) -> Optional[float]:
            cur: Any = row["metrics"]
            for part in path:
                cur = cur.get(part) if isinstance(cur, dict) else None
            return float(cur) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else None

        measures = {"mrr": ("retrieval", "mrr"), "context_recall": ("context_recall",), "faithfulness": ("generation", "faithfulness"), "coverage": ("generation", "coverage")}
        sums: dict[str, list[float]] = {k: [] for k in measures}
        for cid in sorted(shared, key=lambda c: rows_a[c]["position"]):
            x, y = rows_a[cid], rows_b[cid]
            change = "same_ok" if x["ok"] and y["ok"] else "same_failed" if not x["ok"] and not y["ok"] else "improved" if y["ok"] else "regressed"
            improved += change == "improved"
            regressed += change == "regressed"
            if change != "same_ok":
                key = f"{x['piece'] or 'ok'}>{y['piece'] or 'ok'}"
                moves[key] = moves.get(key, 0) + 1
            deltas = {}
            for name, path in measures.items():
                va, vb = value(x, *path), value(y, *path)
                if va is not None and vb is not None:
                    deltas[name] = round(vb - va, 4)
                    sums[name].append(vb - va)
            cases.append({"case": cid, "question": x["question"], "change": change, "a": {"ok": x["ok"], "piece": x["piece"], "cause": x["verdict"].get("cause", "")},
                          "b": {"ok": y["ok"], "piece": y["piece"], "cause": y["verdict"].get("cause", "")}, "delta": deltas, "case_b": y["case_id"]})
        p_value = sign_test(improved, regressed)
        if not shared:
            verdict = "no_shared_cases"
        elif improved == regressed:
            verdict = "no_clear_difference"
        elif p_value is not None and p_value < 0.05:
            verdict = "b_better" if improved > regressed else "a_better"
        else:
            verdict = "b_better_weak" if improved > regressed else "a_better_weak"
        pa, pb = ra["pipeline"], rb["pipeline"]
        differs = {k: {"a": pa.get(k), "b": pb.get(k)} for k in sorted(set(pa) | set(pb)) if pa.get(k) != pb.get(k) and k not in ("name", "fragments")}
        sa, sb = ra["summary"], rb["summary"]

        def pair(*path: str) -> dict[str, Any]:
            def dig(summary: dict[str, Any]) -> Any:
                cur: Any = summary
                for part in path:
                    cur = cur.get(part) if isinstance(cur, dict) else None
                return cur
            va, vb = dig(sa), dig(sb)
            both = isinstance(va, (int, float)) and isinstance(vb, (int, float))
            return {"a": va, "b": vb, "delta": round(vb - va, 4) if both else None}

        top = str(max((k for k in sa.get("retrieval", {}).get("recall", {})), key=int, default="")) if sa.get("retrieval", {}).get("recall") else ""
        pieces = {"pass_rate": pair("pass_rate"), "retrieval_mrr": pair("retrieval", "mrr"), "retrieval_hit_rate": pair("retrieval", "hit_rate"),
                  "rerank_recall_at_cut": pair("rerank", "recall", "after"), "generation_faithfulness": pair("generation", "faithfulness"),
                  "generation_coverage": pair("generation", "coverage"), "generation_right_with_passage": pair("generation", "with_passage", "rate")}
        if top:
            pieces[f"retrieval_recall_at_{top}"] = pair("retrieval", "recall", top)
        moved = max(((k, v["delta"]) for k, v in pieces.items() if v["delta"] is not None and k != "pass_rate" and abs(v["delta"]) > DELTA_NOISE), key=lambda kv: abs(kv[1]), default=None)
        return {"a": self.card(ra), "b": self.card(rb), "shared": len(shared), "only_a": len(rows_a) - len(shared), "only_b": len(self.store.results(rb["id"])) - len(shared),
                "improved": improved, "regressed": regressed, "same": len(shared) - improved - regressed, "sign_test_p": p_value, "verdict": verdict, "differs": differs,
                "pieces": pieces, "largest_change": {"measure": moved[0], "delta": moved[1]} if moved else None,
                "attribution": {"a": sa.get("attribution", {}).get("by_piece"), "b": sb.get("attribution", {}).get("by_piece")}, "moves": moves,
                "mean_delta": {k: (round(sum(v) / len(v), 4) if v else None) for k, v in sums.items()}, "cases": cases}

    # ------------------------------------------------------------------ lifecycle
    def abandon_stale(self) -> None:
        """A run that was working when the process stopped cannot finish: say so instead of leaving it as running."""
        for run in self.store.runs(states=ACTIVE, limit=200):
            self.store.update_run(run["id"], state="failed", finished_ts=self.clock(), error=text("run_stopped"))

    def stop_all(self, timeout: float = 30.0) -> list[threading.Thread]:
        with self.lock:
            entries = list(self.threads.values())
        for _, cancel in entries:
            cancel.set()
        deadline = time.monotonic() + timeout
        for thread, _ in entries:
            thread.join(max(0.0, deadline - time.monotonic()))
        return [t for t, _ in entries if t.is_alive()]
