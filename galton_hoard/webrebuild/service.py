"""The website-rebuild bench inside the app: models explore a reference site in a browser and rebuild it; the rebuilt site is scored case by case.

A run is a background thread (it does not use the GPU lane: the models are served somewhere else). First every reference case is run on the
reference at each width (1280 and 390 px): the checkpoint pictures are kept and a case whose own checks fail on the reference is left out.
Then, for each model, one session rebuilds every site in turn (``agent.RebuildAgent``) and the rebuilt site is put through the same cases.
Scores come from ``hoard_link.webfidelity``: a case at a width passes when its checks pass and every checkpoint reaches the threshold of
normalized SSIM; a site's score is the share of its cases that pass. The zones that differ most are drawn on the candidate's picture.

Files go to ``<data>/webrebuild/<run>/<task>/reference/`` (reference pictures) and ``<run>/<task>/<slot>/`` (``site/`` the files the model
wrote, ``shots/`` its pictures and the diff pictures).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Optional

from ..backends import Cancelled, ChatRequest, OllamaBackend
from ..briefbench import shot as shots
from ..briefbench.service import _price, cost_of, parse_prices
from ..errors import GaltonError
from ..hoard_link import proc
from ..hoard_link import webfidelity as wf
from ..jobevents import RunJobEvents
from ..messages import text
from ..runner import SECONDS_PER_REASONING_TOKEN, Runner, normalise_settings
from . import tasks as tasks_lib
from .agent import RebuildAgent
from .browser import Browser
from .serve import Site
from .store import ACTIVE, TERMINAL, RebuildStore

log = logging.getLogger("galton.webrebuild")

MAX_MODELS = 8
MAX_CONSECUTIVE_INFRA = 2
DEFAULTS = {"max_tokens": 12000, "timeout_s": 600, "temperature": 0.2, "max_steps": 30, "threshold": 0.99}
BOUNDS = {"max_steps": (4, 200)}
SHOT = re.compile(r"^[A-Za-z0-9_.-]{1,200}\.png$")
SLOT = re.compile(r"^(reference|m\d{1,2})$")
SITE_FILE = re.compile(r"^[A-Za-z0-9_./-]{1,200}$")


class RebuildJobEvents(RunJobEvents):
    def url(self, run_id: str) -> str:
        base = (self._base_url() or "").rstrip("/")
        return f"{base}/#/webs/{run_id}" if base else ""


def normalise_rebuild_settings(raw: Optional[dict[str, Any]]) -> dict[str, Any]:
    raw = dict(raw or {})
    mine: dict[str, Any] = {}
    for key, (low, high) in BOUNDS.items():
        value = raw.pop(key, None)
        if value is None:
            mine[key] = DEFAULTS[key]
            continue
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise GaltonError("invalid", "setting_number", setting=key) from exc
        if not low <= number <= high:
            raise GaltonError("invalid", "setting_range", setting=key, low=str(low), high=str(high))
        mine[key] = number
    threshold = raw.pop("threshold", None)
    try:
        mine["threshold"] = DEFAULTS["threshold"] if threshold is None else float(threshold)
    except (TypeError, ValueError) as exc:
        raise GaltonError("invalid", "setting_number", setting="threshold") from exc
    if not 0.5 <= mine["threshold"] <= 1.0:
        raise GaltonError("invalid", "setting_range", setting="threshold", low="0.5", high="1")
    widths = raw.pop("widths", None) or list(tasks_lib.WIDTHS)
    try:
        widths = sorted({int(w) for w in widths}, reverse=True)
    except (TypeError, ValueError) as exc:
        raise GaltonError("invalid", "setting_number", setting="widths") from exc
    if not widths or any(not 320 <= w <= 2560 for w in widths) or len(widths) > 3:
        raise GaltonError("invalid", "setting_range", setting="widths", low="320", high="2560")
    mine["widths"] = widths
    for key in ("max_tokens", "timeout_s", "temperature"):
        if raw.get(key) is None:
            raw[key] = DEFAULTS[key]
    return {**normalise_settings(raw), **mine}


def _mean(values: list[Any], digits: int = 4) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), digits) if values else None


class WebRebuild:
    def __init__(self, store: RebuildStore, runner: Runner, settings: Any, *, data_dir: Path, emit: Optional[Callable[[str, dict], None]] = None,
                 clock: Callable[[], float] = time.time, base_url: Callable[[], str] = lambda: "",
                 browser_factory: Optional[Callable[[str], Any]] = None):
        self.store, self.runner, self.settings, self.data_dir, self.clock = store, runner, settings, Path(data_dir), clock
        self.jobs = RebuildJobEvents(emit or (lambda *_: None), clock=clock, base_url=base_url)
        self.browser_factory = browser_factory or Browser
        self.threads: dict[str, tuple[threading.Thread, threading.Event]] = {}
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ settings
    @property
    def root(self) -> Path:
        configured = str(self.settings.get("webrebuild.folder") or "").strip()
        return Path(configured) if configured else self.data_dir / "webrebuild"

    def python(self) -> str:
        return shots.browser_python(str(self.settings.get("webrebuild.browser_python") or ""),
                                    str(self.settings.get("briefbench.browser_python") or self.settings.get("videobench.browser_python") or ""))

    @property
    def currency(self) -> str:
        return str(self.settings.get("briefbench.currency") or "USD").strip()[:8] or "USD"

    # ------------------------------------------------------------------ what can be run
    def catalog(self) -> dict[str, Any]:
        py = self.python()
        return {"tasks": [tasks_lib.public(t) for t in tasks_lib.builtin()], "widths": list(tasks_lib.WIDTHS), "threshold": DEFAULTS["threshold"],
                "browser": {"python": py, "found": bool(py)}, "currency": self.currency, "prices": parse_prices(self.settings.get("briefbench.prices")),
                "step_kinds": list(wf.STEP_KINDS), "defaults": {k: DEFAULTS[k] for k in ("max_steps", "max_tokens", "temperature")}}

    def task(self, tid: str) -> dict[str, Any]:
        known = tasks_lib.by_id()
        if tid not in known:
            raise GaltonError("not_found", "wr_no_task", id=tid)
        return tasks_lib.public(known[tid], full=True)

    def resolve_models(self, models: list[Any]) -> list[dict[str, Any]]:
        if not models:
            raise GaltonError("invalid", "bk_no_models")
        if len(models) > MAX_MODELS:
            raise GaltonError("invalid", "bk_too_many_models", n=len(models), max=MAX_MODELS)
        out = []
        for n, ref in enumerate(models, 1):
            c = self.runner.resolve_contestants([ref])[0]
            out.append({"slot": f"m{n}", "id": c["id"], "name": c["name"], "contestant": c})
        return out

    def _prices(self, models: list[dict[str, Any]], given: Optional[dict[str, Any]]) -> dict[str, dict[str, float]]:
        table = {str(k).strip().lower(): (k, v) for source in (parse_prices(self.settings.get("briefbench.prices")), given or {}) for k, v in source.items()}
        out: dict[str, dict[str, float]] = {}
        for m in models:
            hit = table.get(m["id"].lower()) or table.get(m["name"].lower())
            if hit:
                out[m["slot"]] = _price(hit[1], m["name"])
        return out

    def plan(self, *, models: list[Any], task_ids: Optional[list[str]] = None, urls: Optional[list[dict[str, Any]]] = None,
             prices: Optional[dict[str, Any]] = None, settings: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        rs = normalise_rebuild_settings(settings)
        rows = self.resolve_models(models)
        chosen = tasks_lib.resolve(task_ids, urls)
        priced = self._prices(rows, prices)
        py = self.python()
        cases = sum(len(t["cases"]) for t in chosen) * len(rs["widths"])
        notes = [text("wr_plan_note", models=len(rows), tasks=len(chosen), cases=cases, steps=rs["max_steps"])]
        notes += [text("bk_price_missing", model=m["name"]) for m in rows if m["slot"] not in priced]
        if not py:
            notes.append(text("wr_no_browser_note"))
        return {"models": [{"slot": m["slot"], "name": m["name"], "priced": m["slot"] in priced} for m in rows], "tasks": [tasks_lib.public(t) for t in chosen],
                "cases_per_model": cases, "browser": {"python": py, "found": bool(py)}, "settings": rs, "currency": self.currency, "notes": notes}

    # ------------------------------------------------------------------ runs
    def start(self, *, models: list[Any], task_ids: Optional[list[str]] = None, urls: Optional[list[dict[str, Any]]] = None,
              prices: Optional[dict[str, Any]] = None, settings: Optional[dict[str, Any]] = None, label: str = "", source: str = "ui", caller: str = "",
              start_thread: bool = True) -> dict[str, Any]:
        rs = normalise_rebuild_settings(settings)
        rows = self.resolve_models(models)
        for m in rows:
            c = m["contestant"]
            if not c["enabled"]:
                raise GaltonError("invalid", "model_disabled", name=c["name"])
            if c["missing"]:
                raise GaltonError("not_found", "model_gone", name=c["name"])
            self.runner.ensure_served(c)
        chosen = tasks_lib.resolve(task_ids, urls)
        if not self.python():
            raise GaltonError("unavailable", "bk_no_browser")
        priced = self._prices(rows, prices)
        run = self.store.create_run(label=label or text("label_webrebuild", models=len(rows), tasks=len(chosen)), settings=rs, prices=priced,
                                    models=[{"slot": m["slot"], "id": m["id"], "name": m["name"]} for m in rows],
                                    tasks=[{k: v for k, v in t.items() if k in ("id", "title", "summary", "level", "kind", "url", "cases")} for t in chosen],
                                    references={}, progress={"phase": "queued", "built": 0, "warnings": []}, total=len(rows) * len(chosen), done=0,
                                    source=source, caller=caller)
        for t in chosen:
            for m in rows:
                self.store.upsert_item(run["id"], t["id"], m["slot"], model_id=m["id"], model_name=m["name"], state="queued")
        self.jobs.queued(run)
        if start_thread:
            self._spawn(run["id"])
        return self.card(self.store.run(run["id"]))

    def _spawn(self, rid: str) -> None:
        with self.lock:
            if rid in self.threads and self.threads[rid][0].is_alive():
                return
            cancel = threading.Event()
            t = threading.Thread(target=self.execute, args=(rid, cancel), name=f"webrebuild-{rid}", daemon=True)
            self.threads[rid] = (t, cancel)
            t.start()

    def cancel(self, rid: str) -> dict[str, Any]:
        run = self.store.run(rid)
        if run["state"] in TERMINAL:
            return {"run": rid, "state": run["state"], "cancelled": False, "note": text("run_finished")}
        with self.lock:
            entry = self.threads.get(rid)
        if entry is not None:
            entry[1].set()
        self.store.update_run(rid, cancel=True)
        if run["state"] == "queued" and (entry is None or not entry[0].is_alive()):
            self.store.update_run(rid, state="cancelled", finished_ts=self.clock())
            return {"run": rid, "state": "cancelled", "cancelled": True}
        return {"run": rid, "state": run["state"], "cancelled": True}

    def abandon_stale(self) -> None:
        for run in self.store.runs(states=ACTIVE, limit=200):
            self.store.update_run(run["id"], state="failed", finished_ts=self.clock(), error=text("run_stopped"))

    def stop_all(self, timeout: float = 30.0) -> list[threading.Thread]:
        with self.lock:
            entries = list(self.threads.values())
        for _, cancel in entries:
            cancel.set()
        deadline = time.monotonic() + timeout
        for t, _ in entries:
            t.join(max(0.0, deadline - time.monotonic()))
        return [t for t, _ in entries if t.is_alive()]

    def delete(self, rid: str) -> dict[str, Any]:
        run = self.store.run(rid)
        if run["state"] in ACTIVE:
            raise GaltonError("conflict", "wr_run_active", id=rid)
        self.store.delete(rid)
        return {"deleted": rid}

    # ------------------------------------------------------------------ bookkeeping
    def _progress(self, rid: str, **fields: Any) -> None:
        with self.store.lock:
            run = self.store.run(rid)
            prog = {**run["progress"], **fields}
            self.store.update_run(rid, progress=prog, done=int(prog.get("built", 0)))

    def _warn(self, rid: str, message: Any) -> None:
        run = self.store.run(rid)
        self._progress(rid, warnings=[*run["progress"].get("warnings", []), str(message)][-10:])

    def _task_dir(self, rid: str, tid: str) -> Path:
        return self.root / rid / tid

    @staticmethod
    def _reference_site(task: dict[str, Any]):
        if task["kind"] == "url":
            return nullcontext(task["url"])
        return Site(Path(tasks_lib.SITES / task["id"]), name=task["id"])

    # ------------------------------------------------------------------ executing
    def execute(self, rid: str, cancel_event: Optional[threading.Event] = None) -> dict[str, Any]:
        run = self.store.run(rid)
        if run["state"] != "queued":
            return run
        cancel_event = cancel_event or threading.Event()
        cancel = lambda: cancel_event.is_set() or bool(self.store.run(rid)["cancel"])  # noqa: E731
        self.store.update_run(rid, state="running", started_ts=self.clock())
        self.jobs.started(run)
        error, state = "", "done"
        browser = None
        try:
            browser = self.browser_factory(self.python())
            self._phase_reference(run, browser, cancel)
            run = self.store.run(rid)
            for model in run["models"]:
                if cancel():
                    break
                self._build_model(run, model, browser, cancel)
            state = "cancelled" if cancel() else "done"
        except GaltonError as exc:
            state, error = "failed", str(exc.message)
        except (Cancelled, proc.Cancelled):
            state = "cancelled"
        except Exception as exc:  # noqa: BLE001
            log.exception("rebuild run %s failed", rid)
            state, error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            if browser is not None:
                browser.close()
        if state == "done":
            self._progress(rid, phase="done")
        final = self.store.update_run(rid, state=state, finished_ts=self.clock(), error=error, summary=self.summary(rid))
        self.jobs.finished(final, state, error)
        with self.lock:
            self.threads.pop(rid, None)
        return final

    def _run_case(self, browser: Any, base: str, case: dict[str, Any], width: int, out: Path) -> dict[str, Any]:
        reply = browser.call({"cmd": "case", "base": base, "width": width, "height": tasks_lib.height_for(width), "case": case, "out": str(out)}, timeout_s=120)
        if "checks" not in reply:
            return {"id": case["id"], "checks": [], "checkpoints": [], "error": str(reply.get("error") or "the browser gave no answer")[:300]}
        return reply

    def _phase_reference(self, run: dict[str, Any], browser: Any, cancel: Callable[[], bool]) -> None:
        rid = run["id"]
        self._progress(rid, phase="reference")
        refs: dict[str, Any] = {}
        for task in run["tasks"]:
            if cancel():
                return
            out = self._task_dir(rid, task["id"]) / "reference"
            rows = []
            with self._reference_site(task) as base:
                for case in task["cases"]:
                    for width in run["settings"]["widths"]:
                        r = self._run_case(browser, base, case, width, out)
                        valid = not r.get("error") and all(c.get("ok") for c in r.get("checks", []))
                        rows.append({"id": case["id"], "width": width, "valid": valid, "error": r.get("error", ""),
                                     "checks": r.get("checks", []), "checkpoints": [{**cp, "file": Path(cp["file"]).name if cp.get("file") else ""} for cp in r.get("checkpoints", [])]})
            invalid = [f"{x['id']}@{x['width']}" for x in rows if not x["valid"]]
            if invalid:
                self._warn(rid, text("wr_reference_case_failed", task=task["id"], cases=", ".join(invalid[:6])))
            refs[task["id"]] = {"cases": rows, "valid": sum(1 for x in rows if x["valid"])}
        self.store.update_run(rid, references=refs)

    def _chat_fn(self, session: Any, rs: dict[str, Any], cancel: Callable[[], bool]):
        def chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
            extra = self.runner.reasoning_allowance(rs, session)
            timeout = float(rs["timeout_s"] or self.settings.get("runner.timeout_s")) + extra * SECONDS_PER_REASONING_TOKEN
            req = ChatRequest(messages=messages, tools=tools, temperature=float(rs["temperature"]), max_tokens=int(rs["max_tokens"]) + extra, seed=rs["seed"],
                              effort=rs["effort"], timeout_s=timeout, reasoning_tokens=extra, tools_fallback=True)
            comp = session.backend.chat(req, cancel)
            if session.guard is not None:
                session.guard.mark_own()
            return comp
        return chat

    def _build_model(self, run: dict[str, Any], model: dict[str, Any], browser: Any, cancel: Callable[[], bool]) -> None:
        rid, slot = run["id"], model["slot"]
        rs = run["settings"]
        self._progress(rid, phase="building", current=model["name"])
        try:
            c = self.runner.store.contestant(model["id"])
            with self.runner.session(c, rs, cancel, lambda s: None) as session:
                api = "ollama" if isinstance(session.backend, OllamaBackend) else "openai"
                chat = self._chat_fn(session, rs, cancel)
                infra = 0
                for task in run["tasks"]:
                    if cancel():
                        return
                    item = self._build_one(run, task, model, browser, chat, api, cancel)
                    infra = infra + 1 if item.get("outcome") == "none" else 0
                    run_now = self.store.run(rid)
                    self._progress(rid, built=sum(1 for i in run_now["items"].values() if i.get("state") == "done"))
                    if infra >= MAX_CONSECUTIVE_INFRA:
                        self._mark_rest_none(rid, slot, run["tasks"], "server_error", text("bk_infra_abort", n=infra))
                        return
        except (Cancelled, proc.Cancelled):
            raise
        except GaltonError as exc:
            if not cancel():
                self._mark_rest_none(rid, slot, run["tasks"], "session", text("bk_session_failed", error=str(exc.message)[:240]))

    def _mark_rest_none(self, rid: str, slot: str, tasks: list[dict[str, Any]], reason: str, detail: str) -> None:
        for t in tasks:
            if self.store.item(rid, t["id"], slot)["state"] != "done":
                self.store.upsert_item(rid, t["id"], slot, state="done", outcome="none", reason=reason, detail=detail[:300])

    def _build_one(self, run: dict[str, Any], task: dict[str, Any], model: dict[str, Any], browser: Any, chat: Any, api: str,
                   cancel: Callable[[], bool]) -> dict[str, Any]:
        rid, slot, rs = run["id"], model["slot"], run["settings"]
        folder = self._task_dir(rid, task["id"]) / slot
        site_dir = folder / "site"
        site_dir.mkdir(parents=True, exist_ok=True)
        self.store.upsert_item(rid, task["id"], slot, state="building")
        started = self.clock()
        with self._reference_site(task) as ref_base, Site(site_dir, name=f"{task['id']}-{slot}") as cand_base:
            def explore(which: str, width: int, action: dict[str, Any]) -> dict[str, Any]:
                base = ref_base if which == "reference" else cand_base
                return browser.call({"cmd": "explore", "base": base, "width": width, "height": tasks_lib.height_for(width), "action": action}, timeout_s=60)
            agent = RebuildAgent(chat, explore, site_dir, api=api, max_steps=int(rs["max_steps"]), cancel=cancel)
            res = agent.run(tasks_lib.intro(task))
            browser.call({"cmd": "reset"}, timeout_s=30)
            wall = round(self.clock() - started, 1)
            price = run["prices"].get(slot)
            metrics = {"steps": res.steps, "tokens_in": res.tokens_in, "tokens_out": res.tokens_out, "latency_ms": round(res.latency_ms, 1), "build_s": wall,
                       "cost": cost_of(price, res.tokens_in, res.tokens_out), "stop": res.stop, "note": res.note, "files": res.files, "trace": res.trace[-60:]}
            if res.stop == "cancelled":
                return self.store.upsert_item(rid, task["id"], slot, state="done", outcome="none", reason="cancelled", **metrics)
            if res.stop == "server_error" and not res.files:
                return self.store.upsert_item(rid, task["id"], slot, state="done", outcome="none", reason="server_error", detail=res.error, **metrics)
            self.store.upsert_item(rid, task["id"], slot, state="scoring", **metrics)
            scored = self._score(run, task, slot, cand_base, browser)      # a site without index.html is scored too: its cases fail
        has_index = (site_dir / "index.html").is_file()
        return self.store.upsert_item(rid, task["id"], slot, state="done", outcome="done", reason="" if has_index else "no_site",
                                      detail="" if has_index else text("wr_no_index"), score_s=round(self.clock() - started - wall, 1), **scored)

    def _score(self, run: dict[str, Any], task: dict[str, Any], slot: str, cand_base: str, browser: Any) -> dict[str, Any]:
        rid = run["id"]
        threshold = float(run["settings"]["threshold"])
        refs = (self.store.run(rid).get("references") or {}).get(task["id"], {}).get("cases", [])
        ref_dir = self._task_dir(rid, task["id"]) / "reference"
        shots_dir = self._task_dir(rid, task["id"]) / slot / "shots"
        by_case = {c["id"]: c for c in task["cases"]}
        cases_out = []
        for ref in refs:
            if not ref["valid"]:
                continue
            case = by_case[ref["id"]]
            got = self._run_case(browser, cand_base, case, int(ref["width"]), shots_dir)
            got_cps = [{**cp, "file": Path(cp["file"]).name if cp.get("file") else ""} for cp in got.get("checkpoints", [])]
            cps = []
            for rcp, ccp in wf.pair_checkpoints(ref["checkpoints"], got_cps):
                row = {"name": rcp["name"], "route": rcp["route"], "width": rcp["width"], "reference": rcp["file"], "candidate": ccp["file"] if ccp else "",
                       "score": None, "ssim": None, "regions": [], "diff": ""}
                if ccp and ccp.get("file") and rcp.get("file"):
                    m = wf.normalized_ssim(ref_dir / rcp["file"], shots_dir / ccp["file"])
                    row.update(score=m["score"], ssim=m["ssim"], baseline=m["baseline"])
                    if m["score"] < threshold:
                        regions = wf.diff_regions(ref_dir / rcp["file"], shots_dir / ccp["file"], limit=6)
                        if regions:
                            name = "diff--" + ccp["file"]
                            wf.overlay(shots_dir / ccp["file"], regions).save(shots_dir / name, "PNG", optimize=True)
                            row.update(regions=regions, diff=name)
                cps.append(row)
            scored = wf.score_case(f"{case['id']}@{ref['width']}", got.get("checks", []), cps, threshold)
            cases_out.append({**scored, "case": case["id"], "width": ref["width"], "error": got.get("error", ""),
                              "failed_checks": [c for c in got.get("checks", []) if not c.get("ok")][:8], "checkpoint_rows": cps,
                              "page_errors": got.get("page_errors", [])})
        site = wf.score_site(cases_out)
        return {"score": site["score"], "passed": site["passed"], "cases": site["cases"], "visual_mean": site["visual_mean"],
                "functional_rate": site["functional_rate"], "case_results": cases_out}

    # ------------------------------------------------------------------ reading
    def card(self, run: dict[str, Any]) -> dict[str, Any]:
        return {"id": run["id"], "label": run.get("label", ""), "state": run["state"], "created_ts": run["created_ts"], "started_ts": run.get("started_ts"),
                "finished_ts": run.get("finished_ts"), "models": [m["name"] for m in run["models"]], "tasks": [t["id"] for t in run["tasks"]],
                "progress": run.get("progress", {}), "done": run.get("done", 0), "total": run.get("total", 0), "error": run.get("error", ""),
                "threshold": run["settings"]["threshold"], "widths": run["settings"]["widths"]}

    def summary(self, rid: str) -> dict[str, Any]:
        run = self.store.run(rid)
        items = list(run["items"].values())
        per_model = []
        for m in run["models"]:
            mine = [i for i in items if i["slot"] == m["slot"] and i.get("outcome") == "done"]
            costs = [i.get("cost") for i in mine]
            per_model.append({"slot": m["slot"], "model": m["name"], "sites": len(mine), "score": _mean([i.get("score") for i in mine]),
                              "cases_passed": sum(int(i.get("passed") or 0) for i in mine), "cases": sum(int(i.get("cases") or 0) for i in mine),
                              "visual_mean": _mean([i.get("visual_mean") for i in mine]), "functional_rate": _mean([i.get("functional_rate") for i in mine]),
                              "tokens_in": sum(int(i.get("tokens_in") or 0) for i in mine), "tokens_out": sum(int(i.get("tokens_out") or 0) for i in mine),
                              "cost": round(sum(c for c in costs if c is not None), 6) if mine and all(c is not None for c in costs) else None,
                              "build_s": round(sum(float(i.get("build_s") or 0) for i in mine), 1), "steps": _mean([i.get("steps") for i in mine], 1),
                              "none": sum(1 for i in items if i["slot"] == m["slot"] and i.get("outcome") == "none")})

        def best(key: str, low: bool) -> Optional[str]:
            rows = [r for r in per_model if r.get(key) is not None and r["sites"]]
            if not rows:
                return None
            val = (min if low else max)(r[key] for r in rows)
            hits = [r["model"] for r in rows if r[key] == val]
            return hits[0] if len(hits) == 1 else None
        return {"models": per_model, "best": best("score", False), "fastest": best("build_s", True), "cheapest": best("cost", True),
                "currency": self.currency, "threshold": run["settings"]["threshold"]}

    def _row(self, i: dict[str, Any]) -> dict[str, Any]:
        keys = ("task", "slot", "model_name", "state", "outcome", "reason", "detail", "score", "passed", "cases", "visual_mean", "functional_rate", "steps",
                "tokens_in", "tokens_out", "latency_ms", "build_s", "cost", "stop", "note")
        return {k: i.get(k) for k in keys}

    def result(self, rid: str) -> dict[str, Any]:
        run = self.store.run(rid)
        refs = run.get("references") or {}
        tasks_out = []
        for t in run["tasks"]:
            rows = [self._row(i) for i in run["items"].values() if i["task"] == t["id"]]
            tasks_out.append({"id": t["id"], "title": t.get("title", t["id"]), "kind": t["kind"], "url": t.get("url", ""),
                              "reference_cases": len(refs.get(t["id"], {}).get("cases", [])), "reference_valid": refs.get(t["id"], {}).get("valid"),
                              "rows": sorted(rows, key=lambda r: r["slot"])})
        return {"run": self.card(run), "summary": run.get("summary") or self.summary(rid), "tasks": tasks_out}

    def item_detail(self, rid: str, tid: str, slot: str) -> dict[str, Any]:
        item = self.store.item(rid, tid, slot)
        base = f"/api/webrebuild/{rid}/{tid}"
        cases = []
        for c in item.get("case_results") or []:
            rows = []
            for cp in c.get("checkpoint_rows", []):
                rows.append({**cp, "reference_url": f"{base}/reference/{cp['reference']}" if cp.get("reference") else "",
                             "candidate_url": f"{base}/{slot}/{cp['candidate']}" if cp.get("candidate") else "",
                             "diff_url": f"{base}/{slot}/{cp['diff']}" if cp.get("diff") else ""})
            cases.append({**{k: v for k, v in c.items() if k != "checkpoint_rows"}, "checkpoints": rows})
        return {**{k: v for k, v in item.items() if k != "case_results"}, "case_results": cases}

    def file_path(self, rid: str, tid: str, slot: str, name: str) -> Path:
        if not SLOT.match(slot or "") or not SHOT.match(name or "") or not re.match(r"^[a-z0-9-]{1,40}$", tid or ""):
            raise GaltonError("not_found", "wr_no_file", name=str(name)[:80])
        self.store.run(rid)
        folder = self._task_dir(rid, tid) / ("reference" if slot == "reference" else f"{slot}/shots")
        path = folder / name
        if not path.is_file():
            raise GaltonError("not_found", "wr_no_file", name=name)
        return path

    def code(self, rid: str, tid: str, slot: str, path: str = "index.html", max_chars: int = 60_000) -> dict[str, Any]:
        """A file the model wrote, with the list of all of them."""
        if not re.match(r"^m\d{1,2}$", slot or "") or not SITE_FILE.match(path or "") or ".." in path.split("/"):
            raise GaltonError("not_found", "wr_no_file", name=str(path)[:80])
        self.store.item(rid, tid, slot)
        site = self._task_dir(rid, tid) / slot / "site"
        files = sorted(p.relative_to(site).as_posix() for p in site.rglob("*") if p.is_file()) if site.is_dir() else []
        target = site / path
        if not target.is_file():
            raise GaltonError("not_found", "wr_no_file", name=path)
        content = target.read_text(encoding="utf-8", errors="replace")
        return {"path": path, "files": files, "chars": len(content), "content": content[:max_chars], "truncated": len(content) > max_chars}

    # ------------------------------------------------------------------ a direct check of two sites
    def score_sites(self, *, reference: str, candidate: str, cases: Optional[list[dict[str, Any]]] = None, routes: Optional[list[str]] = None,
                    widths: Optional[list[int]] = None, threshold: Optional[float] = None) -> dict[str, Any]:
        """Run the same cases on two sites that are already running and score the second against the first, now (no model involved)."""
        rs = normalise_rebuild_settings({"widths": widths, "threshold": threshold})
        ref_task = tasks_lib.url_task({"url": reference, "cases": cases, "routes": routes}, 1)
        cand = tasks_lib.url_task({"url": candidate}, 2)["url"]
        if not self.python():
            raise GaltonError("unavailable", "bk_no_browser")
        cid = "check-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime(self.clock()))
        out = self.root / "checks" / cid
        browser = self.browser_factory(self.python())
        try:
            rows = []
            for case in ref_task["cases"]:
                for width in rs["widths"]:
                    ref = self._run_case(browser, ref_task["url"], case, width, out / "reference")
                    got = self._run_case(browser, cand, case, width, out / "candidate")
                    cps = []
                    for rcp, ccp in wf.pair_checkpoints(ref.get("checkpoints", []), got.get("checkpoints", [])):
                        row = {"name": rcp["name"], "route": rcp["route"], "width": width, "score": None, "regions": []}
                        if ccp and ccp.get("file") and rcp.get("file"):
                            m = wf.normalized_ssim(rcp["file"], ccp["file"])
                            row.update(score=m["score"], ssim=m["ssim"])
                            if m["score"] < rs["threshold"]:
                                row["regions"] = wf.diff_regions(rcp["file"], ccp["file"], limit=4)
                        cps.append(row)
                    valid = not ref.get("error") and all(c.get("ok") for c in ref.get("checks", []))
                    rows.append({**wf.score_case(f"{case['id']}@{width}", got.get("checks", []), cps, rs["threshold"]), "reference_valid": valid,
                                 "checkpoint_rows": cps, "failed_checks": [c for c in got.get("checks", []) if not c.get("ok")][:6]})
        finally:
            browser.close()
        valid_rows = [r for r in rows if r["reference_valid"]]
        return {"site": wf.score_site(valid_rows), "cases": rows, "skipped": len(rows) - len(valid_rows), "threshold": rs["threshold"], "folder": str(out)}
