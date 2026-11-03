"""The logit judge: a rubric of one-token questions is asked about shared contexts and answered from next-token probabilities.

A run is stored as an ordinary run (``source`` = ``logit_judge``) so it appears in the run history; its ``summary.logit`` holds the rubric,
one result set per target and the deltas of every other set against the first (base against adapter, or one server against another).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from .. import adhoc, messages as msgs
from ..errors import GaltonError
from ..util import new_id
from . import rubric as R, scoring as S
from .scorers import HFRunner, ServerScorer

log = logging.getLogger("galton.logitjudge")
SOURCE = "logit_judge"
MAX_REQUESTS = 4000                     # questions x items x result sets cap


class LogitJudge:
    def __init__(self, store: Any, settings: Any, *, data_dir: Path, clock: Callable[[], float] = time.time, emit: Callable[..., None] = lambda *a, **k: None,
                 client_factory: Optional[Callable[[], httpx.Client]] = None, hf_runner: Optional[Callable[..., Any]] = None, meta_reader: Any = None, digest_fn: Any = None):
        self.store, self.settings, self.data_dir, self.clock, self.emit = store, settings, Path(data_dir), clock, emit
        self.client_factory, self.hf_runner = client_factory, hf_runner
        self.lock = threading.Lock()
        self.active: dict[str, Any] = {}
        self.stopping = False
        self.adhoc_kwargs = {k: v for k, v in (("meta_reader", meta_reader), ("digest_fn", digest_fn)) if v is not None}

    # ------------------------------------------------------------------ targets
    def _python(self) -> str:
        configured = str(self.settings.get("logit.python") or "").strip()
        if configured:
            return configured
        try:
            from pygmalion_hoard.settings import WIN_PYTHON
            if Path(WIN_PYTHON).is_file():
                return WIN_PYTHON
        except Exception:  # noqa: BLE001 — the evaluation module can run without the training package
            pass
        import os
        return os.environ.get("PYGMALION_ENV_PYTHON", "").strip()

    def resolve_target(self, raw: Any) -> dict[str, Any]:
        """``{kind: 'hf', base_path, adapter_path?, name?}`` for a local Hugging Face model, otherwise a server contestant (id, name or {kind: 'server', url, model})."""
        if isinstance(raw, dict) and raw.get("kind") == "hf":
            base = str(raw.get("base_path") or "").strip()
            if not base or not Path(base).is_absolute():
                raise GaltonError("invalid", "lj_hf_path", field="base_path")
            if not (Path(base) / "config.json").is_file():
                raise GaltonError("not_found", "lj_hf_missing", path=base)
            adapter = str(raw.get("adapter_path") or "").strip()
            if adapter and not (Path(adapter) / "adapter_config.json").is_file():
                raise GaltonError("not_found", "lj_adapter_missing", path=adapter)
            name = str(raw.get("name") or Path(base).name)[:80]
            return {"kind": "hf", "name": name, "base_path": base, "adapter_path": adapter, "device": str(raw.get("device") or "auto")}
        contestant = adhoc.resolve_one(self.store, self.settings, raw, **self.adhoc_kwargs)
        if contestant["kind"] != "server" or not contestant.get("url"):
            raise GaltonError("invalid", "lj_server_only", name=contestant["name"])
        if contestant.get("remote") and not contestant.get("remote_ok"):
            raise GaltonError("remote_not_allowed", "remote_not_allowed", name=contestant["name"])
        return {"kind": "server", "name": contestant["name"], "id": contestant["id"], "contestant": contestant}

    def _sets_of(self, targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """One result set per variant: a server is one set; an HF model is one set (base) or two (base, then with its adapter)."""
        sets = []
        for t in targets:
            if t["kind"] == "hf" and t["adapter_path"]:
                sets.append({**t, "variant": "base", "label": f"{t['name']} (base)"})
                sets.append({**t, "variant": "adapter", "label": f"{t['name']} (+ {Path(t['adapter_path']).name})"})
            else:
                sets.append({**t, "variant": "base", "label": t["name"]})
        return sets

    # ------------------------------------------------------------------ planning and running
    def plan(self, rubric: Any, items: Any, targets: Any) -> dict[str, Any]:
        questions, contexts = R.normalise_rubric(rubric), R.normalise_items(items)
        if not isinstance(targets, list) or not targets:
            raise GaltonError("invalid", "lj_targets_empty")
        resolved = [self.resolve_target(t) for t in targets]
        sets = self._sets_of(resolved)
        n_requests = len(questions) * len(contexts)
        if n_requests * len(sets) > MAX_REQUESTS:
            raise GaltonError("invalid", "lj_too_many_requests", n=n_requests * len(sets), max=MAX_REQUESTS)
        notes = []
        if self.hf_runner is None and any(t["kind"] == "hf" for t in resolved):
            python = self._python()
            if not python or not Path(python).is_file():
                raise GaltonError("unavailable", "lj_no_python")
        return {"questions": len(questions), "items": len(contexts), "requests_per_set": n_requests, "sets": [s["label"] for s in sets], "notes": notes,
                "questions_detail": questions, "targets": [{k: v for k, v in t.items() if k != "contestant"} for t in resolved]}

    def _requests(self, questions: list[dict[str, Any]], contexts: list[dict[str, str]]) -> list[dict[str, Any]]:
        return [{"key": f"{c['id']}::{q['id']}", "item": c["id"], "question": q["id"], "labels": q["labels"], "messages": R.build_messages(c["context"], q)}
                for c in contexts for q in questions]

    def _score_server(self, target: dict[str, Any], requests: list[dict[str, Any]], cancel: threading.Event) -> tuple[dict[str, dict[str, Any]], list[str]]:
        scorer = ServerScorer(target["contestant"], client_factory=self.client_factory, top_logprobs=int(self.settings.get("logit.top_logprobs")),
                              timeout_s=float(self.settings.get("logit.timeout_s")), no_thinking=bool(self.settings.get("logit.no_thinking")))
        out: dict[str, dict[str, Any]] = {}
        client = scorer.client_factory()
        try:
            for r in requests:
                if cancel.is_set() or self.stopping:
                    raise GaltonError("conflict", "lj_cancelled")
                out[r["key"]] = scorer.score(client, r["messages"], r["labels"])
        finally:
            client.close()
        return out, []

    def _score_hf(self, target: dict[str, Any], requests: list[dict[str, Any]], variants: list[str], run_id: str) -> dict[str, dict[str, dict[str, Any]]]:
        args = {"base_path": target["base_path"], "adapter_path": target["adapter_path"], "variants": variants, "device": target.get("device", "auto"),
                "requests": [{"key": r["key"], "messages": r["messages"], "labels": r["labels"]} for r in requests]}
        if self.hf_runner is not None:                      # tests and other front ends bring their own way of running the worker
            return self.hf_runner(args)
        gpus = [int(g) for g in (self.settings.get("gpus.allowed") or [])]
        runner = HFRunner(self._python(), work_dir=self.data_dir / "logit", gpus=gpus, timeout_s=float(self.settings.get("logit.hf_timeout_s")))
        self.active[f"{run_id}:hf"] = runner
        try:
            return runner.run(args)
        finally:
            self.active.pop(f"{run_id}:hf", None)

    def run(self, rubric: Any, items: Any, targets: Any, *, label: str = "", source: str = "ui", caller: str = "") -> dict[str, Any]:
        plan = self.plan(rubric, items, targets)
        questions, contexts = plan["questions_detail"], R.normalise_items(items)
        resolved = [self.resolve_target(t) for t in targets]
        sets = self._sets_of(resolved)
        requests = self._requests(questions, contexts)
        rid = new_id("lj")
        name = label or f"logit judge: {', '.join(s['label'] for s in sets)[:100]}"
        self.store.create_run(id=rid, label=name, state="running", suites=[], contestants=[s["label"] for s in sets], source=SOURCE, caller=caller or source,
                              settings={"judge": "logit", "questions": len(questions), "items": len(contexts)}, started_ts=self.clock())
        cancel = threading.Event()
        self.active[rid] = cancel
        results: list[dict[str, Any]] = []
        try:
            hf_cache: dict[tuple[str, str], dict[str, Any]] = {}
            for s in sets:
                raw: dict[str, Any]
                if s["kind"] == "server":
                    raw, _ = self._score_server(s, requests, cancel)
                else:
                    ident = (s["base_path"], s["adapter_path"])
                    if ident not in hf_cache:
                        variants = ["base", "adapter"] if s["adapter_path"] else ["base"]
                        hf_cache[ident] = self._score_hf(s, requests, variants, rid)
                    raw = hf_cache[ident][s["variant"]]
                results.append(self._assemble(s, questions, contexts, requests, raw))
            payload = self._payload(questions, contexts, results)
            self.store.update_run(rid, state="done", finished_ts=self.clock(), summary={"logit": payload})
        except GaltonError as exc:
            self.store.update_run(rid, state="cancelled" if exc.key == "lj_cancelled" else "failed", finished_ts=self.clock(), error=exc.message)
            raise
        except Exception as exc:  # noqa: BLE001 — a stored run must never stay 'running'
            self.store.update_run(rid, state="failed", finished_ts=self.clock(), error=f"{type(exc).__name__}: {exc}"[:300])
            raise
        finally:
            self.active.pop(rid, None)
        self.emit("logit_judge.done", {"run": rid})
        return {"run": rid, "label": name, **payload}

    def _assemble(self, s: dict[str, Any], questions: list[dict[str, Any]], contexts: list[dict[str, str]], requests: list[dict[str, Any]],
                  raw: dict[str, dict[str, Any]]) -> dict[str, Any]:
        qmap = {q["id"]: q for q in questions}
        by_item: dict[str, dict[str, Any]] = {c["id"]: {"id": c["id"], "answers": {}} for c in contexts}
        for r in requests:
            by_item[r["item"]]["answers"][r["question"]] = S.answer_for(qmap[r["question"]], raw[r["key"]])
        items = list(by_item.values())
        return {"label": s["label"], "kind": s["kind"], "variant": s["variant"], "items": items, "aggregate": S.aggregate(questions, items)}

    def _payload(self, questions: list[dict[str, Any]], contexts: list[dict[str, str]], results: list[dict[str, Any]]) -> dict[str, Any]:
        compare = [{"label": other["label"], "against": results[0]["label"], "deltas": S.deltas(questions, results[0], other)} for other in results[1:]]
        warnings = []
        for r in results:
            low = sum(1 for it in r["items"] for a in it["answers"].values() if a.get("low_mass"))
            if low:
                warnings.append(msgs.item(msgs.text("lj_low_mass", set=r["label"], n=low)))
        return {"rubric": questions, "contexts": [{"id": c["id"], "preview": c["context"][:240]} for c in contexts], "sets": results, "compare": compare, "warnings": warnings}

    # ------------------------------------------------------------------ reading and probing
    def result(self, ref: str = "") -> dict[str, Any]:
        runs = self.store.runs(source=SOURCE, limit=1 if not ref else 500)
        run = next((r for r in runs if r["id"] == ref), None) if ref else (runs[0] if runs else None)
        if run is None:
            raise GaltonError("not_found", "lj_no_run", id=ref or "(latest)")
        logit = (run.get("summary") or {}).get("logit")
        return {"run": self.card(run), **(logit or {})}

    def card(self, run: dict[str, Any]) -> dict[str, Any]:
        return {"id": run["id"], "label": run["label"], "state": run["state"], "error": run["error"], "created_ts": run["created_ts"], "finished_ts": run["finished_ts"],
                "sets": run["contestants"], "caller": run["caller"]}

    def list(self, limit: int = 30) -> dict[str, Any]:
        rows = self.store.runs(source=SOURCE, limit=limit)
        out = []
        for r in rows:
            logit = (r.get("summary") or {}).get("logit") or {}
            out.append({**self.card(r), "questions": (r["settings"] or {}).get("questions"), "items": (r["settings"] or {}).get("items"),
                        "compare": [{"label": c["label"], "against": c["against"]} for c in logit.get("compare", [])]})
        return {"runs": out, "count": len(out)}

    def probe(self, target: Any) -> dict[str, Any]:
        t = self.resolve_target(target)
        if t["kind"] != "server":
            return {"supported": True, "kind": "hf", "name": t["name"], "note": "A Hugging Face model gives the logits directly; no server is involved."}
        scorer = ServerScorer(t["contestant"], client_factory=self.client_factory, top_logprobs=int(self.settings.get("logit.top_logprobs")),
                              timeout_s=float(self.settings.get("logit.timeout_s")), no_thinking=bool(self.settings.get("logit.no_thinking")))
        client = scorer.client_factory()
        try:
            return {"kind": "server", **scorer.probe(client)}
        finally:
            client.close()

    def cancel(self, ref: str = "") -> dict[str, Any]:
        entries = [(k, v) for k, v in list(self.active.items()) if not ref or k == ref or k.startswith(ref + ":")]
        for _, v in entries:
            (v.kill if isinstance(v, HFRunner) else v.set)()
        return {"cancelled": [k for k, _ in entries]}

    def stop(self) -> None:
        self.stopping = True
        self.cancel()
