"""The brief bench inside the app: the same brief goes to several models and the answers are put side by side with what each one took and cost.

A run is a background thread (it does not use the GPU lane: the models are served somewhere else). For each model, one session answers every brief; a page is rendered
right after it is written, so the grid fills as the run goes. Then, if a judge was named, a fresh session that did not write any of the answers scores them without knowing
which model wrote each. Files go to ``data/briefbench/<run>/<brief>/<slot>/`` (the page, its pictures) and ``<run>/<brief>/grid.png`` (the comparison sheet).

What is measured per answer: time (the whole call), time to the first token and decode speed when the server reports them, tokens in and out, and the cost those tokens would have
had at the price the person gave for that model (per million tokens). A model without a price has an unknown cost, never a zero one; a command-line agent has unknown tokens too.
An answer that never reached a phase is ``none``: it is left out of the means, it is not a score of zero.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ..backends import Cancelled, ChatRequest
from ..errors import GaltonError
from ..hoard_link import proc
from ..jobevents import RunJobEvents
from ..messages import text
from ..runner import SECONDS_PER_REASONING_TOKEN, Runner, normalise_settings
from . import cli as cli_lib
from . import grid as grid_lib
from . import shot as shots
from .briefs import BRIEFS, BY_ID, CATEGORIES, KINDS, custom, public
from .contract import build_messages, extract_answer, inline_libraries, library_file, static_check
from .judge import DIMENSIONS, RUBRIC, judge_messages, parse_judge
from .store import ACTIVE, TERMINAL, BriefStore

log = logging.getLogger("galton.briefbench")

MAX_BRIEFS = 20
MAX_MODELS = 12
MAX_CONSECUTIVE_INFRA = 3
DEFAULTS = {"max_tokens": 16000, "timeout_s": 900, "temperature": 0.7, "settle_ms": 1500, "motion_ms": 1200, "render_timeout_s": 120, "judge_images": "auto", "judge_max_tokens": 4000}
BOUNDS = {"settle_ms": (0, 30000), "motion_ms": (0, 10000), "render_timeout_s": (20, 900), "judge_max_tokens": (200, 32000)}
FILES = re.compile(r"^(index\.html|shot\.png|shot2\.png|thumb\.png|answer\.txt)$")
SLOT = re.compile(r"^m\d{1,2}$")


class BriefJobEvents(RunJobEvents):
    def url(self, run_id: str) -> str:
        base = (self._base_url() or "").rstrip("/")
        return f"{base}/#/encargos/{run_id}" if base else ""


def normalise_brief_settings(raw: Optional[dict[str, Any]]) -> dict[str, Any]:
    """The settings of a run: the shared run settings (token cap, temperature, timeout, seed, effort, device...) plus the ones of this bench."""
    raw = dict(raw or {})
    mine: dict[str, Any] = {}
    for key, (low, high) in BOUNDS.items():
        value = raw.pop(key, None)
        if value is None:
            mine[key] = DEFAULTS[key]
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise GaltonError("invalid", "setting_number", setting=key) from exc
        if not low <= number <= high:
            raise GaltonError("invalid", "setting_range", setting=key, low=f"{low:g}", high=f"{high:g}")
        mine[key] = int(number)
    images = raw.pop("judge_images", None)
    if images is None or str(images).strip().lower() == "auto":
        mine["judge_images"] = "auto"
    elif str(images).strip().lower() in ("1", "true", "yes", "on", "0", "false", "no", "off"):
        mine["judge_images"] = str(images).strip().lower() in ("1", "true", "yes", "on")
    else:
        raise GaltonError("invalid", "setting_bool", setting="judge_images")
    for key in ("max_tokens", "timeout_s", "temperature"):
        if raw.get(key) is None:
            raw[key] = DEFAULTS[key]
    return {**normalise_settings(raw), **mine}


def _mean(values: list[Any], digits: int = 2) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), digits) if values else None


def _price(value: Any, model: str) -> dict[str, float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = {"input": value, "output": value}
    if not isinstance(value, dict):
        raise GaltonError("invalid", "bk_bad_price", model=model)
    try:
        out = {"input": float(value.get("input", 0) or 0), "output": float(value.get("output", 0) or 0)}
    except (TypeError, ValueError) as exc:
        raise GaltonError("invalid", "bk_bad_price", model=model) from exc
    if out["input"] < 0 or out["output"] < 0:
        raise GaltonError("invalid", "bk_bad_price", model=model)
    return out


def parse_prices(lines: Any) -> dict[str, Any]:
    """The prices written in the setting, one per line: ``model name = input, output`` (per million tokens), or ``model name = price`` for the same price both ways."""
    out: dict[str, Any] = {}
    for raw in lines or []:
        name, sep, value = str(raw).partition("=")
        parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
        if not sep or not name.strip() or name.strip().startswith("#") or not parts or len(parts) > 2:
            continue
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            continue
        out[name.strip()] = {"input": nums[0], "output": nums[-1]}
    return out


def cost_of(price: Optional[dict[str, float]], tokens_in: Optional[int], tokens_out: Optional[int]) -> Optional[float]:
    """What the tokens would have cost at ``price`` (per million tokens). None when there is no price or the server did not report the tokens."""
    if not price or tokens_out is None:
        return None
    return round(((tokens_in or 0) * price["input"] + tokens_out * price["output"]) / 1_000_000, 6)


class BriefBench:
    def __init__(self, store: BriefStore, runner: Runner, settings: Any, *, data_dir: Path, emit: Optional[Callable[[str, dict], None]] = None,
                 clock: Callable[[], float] = time.time, base_url: Callable[[], str] = lambda: "", render_fn: Optional[Callable[..., dict[str, Any]]] = None):
        self.store, self.runner, self.settings, self.data_dir, self.clock = store, runner, settings, Path(data_dir), clock
        self.jobs = BriefJobEvents(emit or (lambda *_: None), clock=clock, base_url=base_url)
        self.render_fn = render_fn or shots.render
        self.threads: dict[str, tuple[threading.Thread, threading.Event]] = {}
        self.lock = threading.Lock()
        self.active: Optional[str] = None

    # ------------------------------------------------------------------ what can be run
    @property
    def root(self) -> Path:
        configured = str(self.settings.get("briefbench.folder") or "").strip()
        return Path(configured) if configured else self.data_dir / "briefbench"

    @property
    def libraries_dir(self) -> str:
        return str(self.settings.get("briefbench.libraries") or "").strip()

    @property
    def currency(self) -> str:
        return str(self.settings.get("briefbench.currency") or "USD").strip()[:8] or "USD"

    def python(self) -> str:
        return shots.browser_python(str(self.settings.get("briefbench.browser_python") or ""), str(self.settings.get("videobench.browser_python") or ""))

    def cli_agents(self) -> list[str]:
        return sorted(cli_lib.parse(self.settings.get("briefbench.cli") or []))

    def _libraries_for(self, brief: dict[str, Any]) -> dict[str, str]:
        found = {n: library_file(self.libraries_dir, n) for n in brief.get("libraries") or []}
        return {n: f for n, f in found.items() if f}

    def catalog(self, *, category: str = "", kind: str = "", query: str = "") -> dict[str, Any]:
        q = query.strip().lower()
        rows = [b for b in BRIEFS if (not category or b["category"] == category) and (not kind or b["kind"] == kind) and (not q or q in b["prompt"].lower() or q in b["id"] or q in b["title"].lower())]
        py = self.python()
        names = sorted({n for b in BRIEFS for n in b["libraries"]})
        return {"briefs": [public(b) for b in rows], "total": len(BRIEFS), "matching": len(rows), "categories": CATEGORIES, "kinds": list(KINDS),
                "libraries": {n: library_file(self.libraries_dir, n) for n in names}, "libraries_folder": self.libraries_dir, "cli_agents": self.cli_agents(),
                "browser": {"python": py, "found": bool(py)}, "currency": self.currency, "prices": parse_prices(self.settings.get("briefbench.prices")),
                "rubric": {d: RUBRIC[d] for d in DIMENSIONS}}

    def brief(self, bid: str) -> dict[str, Any]:
        if bid not in BY_ID:
            raise GaltonError("not_found", "bk_no_brief", id=bid)
        return {**public(BY_ID[bid], full=True), "libraries_available": self._libraries_for(BY_ID[bid])}

    def resolve_briefs(self, ids: Optional[list[str]], brief: str = "", briefs: Optional[list[dict[str, Any]]] = None, *, kind: str = "page", title: str = "",
                       libraries: Optional[list[str]] = None, criteria: str = "") -> list[dict[str, Any]]:
        chosen: list[dict[str, Any]] = []
        for i in [str(x).strip() for x in (ids or []) if str(x).strip()]:
            if i not in BY_ID:
                raise GaltonError("not_found", "bk_no_brief", id=i)
            chosen.append(dict(BY_ID[i]))
        own = [{"prompt": brief, "title": title, "kind": kind, "libraries": libraries or [], "criteria": criteria}] if brief.strip() else []
        own += [dict(b) for b in briefs or []]
        for n, b in enumerate(own, 1):
            prompt = str(b.get("prompt") or "").strip()
            k = str(b.get("kind") or kind or "page").strip().lower()
            if len(prompt) < 10:
                raise GaltonError("invalid", "bk_brief_text")
            if k not in KINDS:
                raise GaltonError("invalid", "bk_bad_kind", kind=k)
            chosen.append(custom(prompt, title=str(b.get("title") or ""), kind=k, libraries=[str(x) for x in (b.get("libraries") or [])][:6],
                                 criteria=str(b.get("criteria") or ""), index=n))
        if not chosen:
            raise GaltonError("invalid", "bk_no_briefs")
        if len(chosen) > MAX_BRIEFS:
            raise GaltonError("invalid", "bk_too_many_briefs", n=len(chosen), max=MAX_BRIEFS)
        return chosen

    def resolve_models(self, models: list[Any]) -> list[dict[str, Any]]:
        """The contestants of a run in order: ``{slot, kind: model|cli, id, name, contestant | argv}``."""
        if not models:
            raise GaltonError("invalid", "bk_no_models")
        if len(models) > MAX_MODELS:
            raise GaltonError("invalid", "bk_too_many_models", n=len(models), max=MAX_MODELS)
        out = []
        for n, ref in enumerate(models, 1):
            slot = f"m{n}"
            if cli_lib.is_cli(ref):
                name, argv = cli_lib.lookup(self.settings.get("briefbench.cli") or [], ref)
                out.append({"slot": slot, "kind": "cli", "id": f"{cli_lib.ID_PREFIX}{name}", "name": name, "argv": argv})
                continue
            c = self.runner.resolve_contestants([ref])[0]
            out.append({"slot": slot, "kind": "model", "id": c["id"], "name": c["name"], "contestant": c})
        return out

    def _prices(self, models: list[dict[str, Any]], given: Optional[dict[str, Any]]) -> dict[str, dict[str, float]]:
        """The price per million tokens of each slot: what the call gave, else the setting, matched by id or name (case-insensitive)."""
        table = {str(k).strip().lower(): (k, v) for source in (parse_prices(self.settings.get("briefbench.prices")), given or {}) for k, v in source.items()}
        out: dict[str, dict[str, float]] = {}
        for m in models:
            hit = table.get(m["id"].lower()) or table.get(m["name"].lower())
            if hit:
                out[m["slot"]] = _price(hit[1], m["name"])
        return out

    def _judge_mode(self, c: dict[str, Any], rs: dict[str, Any]) -> str:
        want = rs["judge_images"]
        if want == "auto":
            want = bool(c.get("vision"))
        return "images" if want else "text"

    def plan(self, *, models: list[Any], judge: Any = None, brief_ids: Optional[list[str]] = None, brief: str = "", briefs: Optional[list[dict[str, Any]]] = None, kind: str = "page",
             title: str = "", libraries: Optional[list[str]] = None, criteria: str = "", prices: Optional[dict[str, Any]] = None, settings: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        rs = normalise_brief_settings(settings)
        rows = self.resolve_models(models)
        chosen = self.resolve_briefs(brief_ids, brief, briefs, kind=kind, title=title, libraries=libraries, criteria=criteria)
        jury = None
        if judge is not None and judge != "":
            if cli_lib.is_cli(judge):
                raise GaltonError("invalid", "bk_cli_judge")
            jury = self.runner.resolve_contestants([judge])[0]
        priced = self._prices(rows, prices)
        py = self.python()
        pages = [b for b in chosen if b["kind"] == "page"]
        notes = [text("bk_plan_note", models=len(rows), briefs=len(chosen))]
        for b in chosen:
            for name in (b.get("libraries") or []):
                if not library_file(self.libraries_dir, name):
                    notes.append(text("bk_libraries_missing", name=name))
        notes += [text("bk_price_missing", model=m["name"]) for m in rows if m["slot"] not in priced and m["kind"] == "model"]
        return {"models": [{"slot": m["slot"], "name": m["name"], "kind": m["kind"], "priced": m["slot"] in priced} for m in rows],
                "briefs": [public(b) for b in chosen], "judge": jury["name"] if jury else None, "judge_mode": self._judge_mode(jury, rs) if jury else None,
                "same_model_judges": bool(jury) and any(m["id"] == jury["id"] for m in rows), "browser": {"python": py, "found": bool(py), "needed": bool(pages)},
                "calls": {"answers": len(rows) * len(chosen), "judge": len(rows) * len(chosen) if jury else 0}, "settings": rs, "currency": self.currency, "notes": notes}

    # ------------------------------------------------------------------ runs
    def start(self, *, models: list[Any], judge: Any = None, brief_ids: Optional[list[str]] = None, brief: str = "", briefs: Optional[list[dict[str, Any]]] = None, kind: str = "page",
              title: str = "", libraries: Optional[list[str]] = None, criteria: str = "", prices: Optional[dict[str, Any]] = None, settings: Optional[dict[str, Any]] = None,
              label: str = "", source: str = "ui", caller: str = "", start_thread: bool = True) -> dict[str, Any]:
        rs = normalise_brief_settings(settings)
        rows = self.resolve_models(models)
        for m in rows:
            if m["kind"] != "model":
                continue
            c = m["contestant"]
            if not c["enabled"]:
                raise GaltonError("invalid", "model_disabled", name=c["name"])
            if c["missing"]:
                raise GaltonError("not_found", "model_gone", name=c["name"])
            self.runner.ensure_served(c)
        jury_id = ""
        if judge is not None and judge != "":
            if cli_lib.is_cli(judge):
                raise GaltonError("invalid", "bk_cli_judge")
            j = self.runner.resolve_contestants([judge])[0]
            if not any(m["id"] == j["id"] for m in rows):
                self.runner.ensure_served(j)
            jury_id = j["id"]
        chosen = self.resolve_briefs(brief_ids, brief, briefs, kind=kind, title=title, libraries=libraries, criteria=criteria)
        if any(b["kind"] == "page" for b in chosen) and not self.python():
            raise GaltonError("unavailable", "bk_no_browser")
        priced = self._prices(rows, prices)
        n = len(rows) * len(chosen)
        run = self.store.create_run(label=label or text("label_briefbench", models=len(rows), briefs=len(chosen)), state="queued", jury=jury_id, settings=rs, prices=priced,
                                    models=[{"slot": m["slot"], "id": m["id"], "name": m["name"], "kind": m["kind"]} for m in rows],
                                    briefs=[{k: v for k, v in b.items() if k in ("id", "title", "kind", "category", "libraries", "prompt", "criteria")} for b in chosen],
                                    progress={"phase": "queued", "answered": 0, "judged": 0, "warnings": []}, total=n + (n if jury_id else 0), done=0, source=source, caller=caller)
        for b in chosen:
            for m in rows:
                self.store.upsert_item(run["id"], b["id"], m["slot"], contestant_id=m["id"], model_name=m["name"], state="queued")
        self.jobs.queued(run)
        if start_thread:
            self._spawn(run["id"])
        return self.card(self.store.run(run["id"]))

    def _spawn(self, rid: str) -> None:
        with self.lock:
            if rid in self.threads and self.threads[rid][0].is_alive():
                return
            cancel = threading.Event()
            t = threading.Thread(target=self.execute, args=(rid, cancel), name=f"briefbench-{rid}", daemon=True)
            self.threads[rid] = (t, cancel)
            t.start()

    def cancel(self, rid: str) -> dict[str, Any]:
        run = self.store.run(rid)
        if run["state"] in TERMINAL:
            return {"run": run["id"], "state": run["state"], "cancelled": False, "note": text("run_finished")}
        with self.lock:
            entry = self.threads.get(run["id"])
        if entry is not None:
            entry[1].set()
        self.store.update_run(run["id"], cancel=True)
        if run["state"] == "queued" and (entry is None or not entry[0].is_alive()):
            self.store.update_run(run["id"], state="cancelled", finished_ts=self.clock())
            return {"run": run["id"], "state": "cancelled", "cancelled": True}
        return {"run": run["id"], "state": run["state"], "cancelled": True, "note": text("run_stopping_own")}

    def abandon_stale(self) -> None:
        """A run that was running when the process died cannot finish: say so."""
        for run in self.store.runs(states=ACTIVE, limit=100):
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

    # ------------------------------------------------------------------ bookkeeping
    def _progress(self, rid: str, **fields: Any) -> None:
        with self.lock:
            run = self.store.run(rid)
            prog = {**run["progress"], **fields}
            self.store.update_run(rid, progress=prog, done=int(prog.get("answered", 0)) + int(prog.get("judged", 0)))

    def _warn(self, rid: str, message: Any) -> None:
        run = self.store.run(rid)
        warnings = [*run["progress"].get("warnings", []), str(message)]
        self._progress(rid, warnings=warnings[-10:])

    def _folder(self, rid: str, bid: str, slot: str) -> Path:
        return self.root / rid / bid / slot

    # ------------------------------------------------------------------ executing
    def execute(self, rid: str, cancel_event: Optional[threading.Event] = None) -> dict[str, Any]:
        run = self.store.run(rid)
        if run["state"] != "queued":
            return run
        cancel_event = cancel_event or threading.Event()
        cancel = lambda: cancel_event.is_set() or bool(self.store.run(rid)["cancel"])  # noqa: E731
        rs = run["settings"]
        self.store.update_run(rid, state="running", started_ts=self.clock())
        self.jobs.started(run)
        self.active = rid
        error = ""
        try:
            self._phase_answer(run, rs, cancel)
            if not cancel() and run["jury"]:
                self._phase_judge(run, rs, cancel)
            state = "cancelled" if cancel() else "done"
        except GaltonError as exc:
            state, error = "failed", str(exc.message)
            log.info("brief run %s failed: %s", rid, exc.message)
        except (Cancelled, proc.Cancelled):
            state = "cancelled"
        except Exception as exc:  # noqa: BLE001
            log.exception("brief run %s failed", rid)
            state, error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            self.active = None
        if state == "done":
            self._progress(rid, phase="done", answered=len(run["models"]) * len(run["briefs"]), judged=len(run["models"]) * len(run["briefs"]) if run["jury"] else 0)
        summary = self.summary(rid)
        final = self.store.update_run(rid, state=state, finished_ts=self.clock(), error=error, summary=summary)
        self.jobs.finished(final, state, error)
        with self.lock:
            self.threads.pop(rid, None)
        return final

    def _chat(self, session: Any, rs: dict[str, Any], cancel: Callable[[], bool], messages: list[dict[str, Any]], *, max_tokens: int, temperature: float,
              images: Optional[list[bytes]] = None, seed: Optional[int] = None):
        extra = self.runner.reasoning_allowance(rs, session)
        timeout = float(rs["timeout_s"] or self.settings.get("runner.timeout_s")) + extra * SECONDS_PER_REASONING_TOKEN
        req = ChatRequest(messages=messages, images=images or [], temperature=temperature, max_tokens=max_tokens + extra, seed=seed, effort=rs["effort"], timeout_s=timeout,
                          reasoning_tokens=extra, tools_fallback=False)
        comp = session.backend.chat(req, cancel)
        comp.max_tokens_sent = req.max_tokens
        if session.guard is not None:
            session.guard.mark_own()
        if session.identity is not None and not comp.error:
            session.identity.answer(comp.served_model)
        return comp

    def _verify(self, rid: str, session: Any, who: str) -> None:
        """On a shared server, say so when it could not be confirmed to have served the same model all along."""
        if session.identity is None:
            return
        state = "unknown"
        for _ in range(3):
            state = session.identity.check(force=True)
            if state == "fresh":
                return
            self.runner.sleep(1.0)
        self._warn(rid, text("bk_identity_unverified", who=who, state=state))

    # -- phase 1: every model answers every brief; a page is rendered as soon as it is written -------------------------
    def _phase_answer(self, run: dict[str, Any], rs: dict[str, Any], cancel: Callable[[], bool]) -> None:
        rid = run["id"]
        briefs = run["briefs"]
        self._progress(rid, phase="answering")
        for model in run["models"]:
            if cancel():
                return
            self._progress(rid, current=model["name"])
            try:
                if model["kind"] == "cli":
                    self._answer_cli(run, rs, model, briefs, cancel)
                else:
                    self._answer_model(run, rs, model, briefs, cancel)
            except (Cancelled, proc.Cancelled):
                raise
            except GaltonError as exc:
                if cancel():
                    return
                self._mark_rest_none(rid, model, briefs, "session", text("bk_session_failed", error=str(exc.message)[:240]))      # one model failing does not stop the others
            self._count_answered(rid)

    def _count_answered(self, rid: str) -> None:
        items = self.store.items(rid)
        self._progress(rid, answered=sum(1 for i in items if i["state"] not in ("queued", "writing")))

    def _mark_rest_none(self, rid: str, model: dict[str, Any], briefs: list[dict[str, Any]], reason: str, detail: Any) -> None:
        for b in briefs:
            if self.store.item(rid, b["id"], model["slot"])["state"] in ("queued", "writing"):
                self.store.upsert_item(rid, b["id"], model["slot"], state="done", outcome="none", reason=reason, detail=str(detail)[:300])

    def _answer_model(self, run: dict[str, Any], rs: dict[str, Any], model: dict[str, Any], briefs: list[dict[str, Any]], cancel: Callable[[], bool]) -> None:
        rid = run["id"]
        c = self.runner.store.contestant(model["id"])
        price = run["prices"].get(model["slot"])
        infra = 0
        with self.runner.session(c, rs, cancel, lambda s: None) as session:
            self._progress(rid, **{f"runs_on_{model['slot']}": str(session.runs_on)})
            for n, b in enumerate(briefs):
                if cancel():
                    break
                self.store.upsert_item(rid, b["id"], model["slot"], state="writing")
                seed = (int(rs["seed"]) + n) if rs["seed"] is not None else None
                comp = self._chat(session, rs, cancel, build_messages(b, self._libraries_for(b)), max_tokens=int(rs["max_tokens"]), temperature=float(rs["temperature"]), seed=seed)
                metrics = {"tokens_in": comp.prompt_tokens, "tokens_out": comp.completion_tokens, "latency_ms": comp.latency_ms, "ttft_ms": comp.ttft_ms, "load_ms": comp.load_ms,
                           "decode_tps": comp.decode_tps, "finish_reason": comp.finish_reason or "", "cost": cost_of(price, comp.prompt_tokens, comp.completion_tokens)}
                if comp.error:
                    infra += 1
                    self.store.upsert_item(rid, b["id"], model["slot"], state="done", outcome="none", reason="server_error", detail=str(comp.error)[:300], **metrics)
                    if infra >= MAX_CONSECUTIVE_INFRA:
                        self._mark_rest_none(rid, model, briefs, "server_error", text("bk_infra_abort", n=infra))
                        break
                else:
                    infra = 0
                    self._receive(run, rs, b, model, comp.text, metrics, cancel)
                self._count_answered(rid)
            self._verify(rid, session, model["name"])

    def _answer_cli(self, run: dict[str, Any], rs: dict[str, Any], model: dict[str, Any], briefs: list[dict[str, Any]], cancel: Callable[[], bool]) -> None:
        rid = run["id"]
        timeout = float(self.settings.get("briefbench.cli_timeout_s"))
        argv = cli_lib.lookup(self.settings.get("briefbench.cli") or [], model["id"])[1]          # read now: the setting may have been edited since the run was queued
        for b in briefs:
            if cancel():
                break
            self.store.upsert_item(rid, b["id"], model["slot"], state="writing")
            workdir = self._folder(rid, b["id"], model["slot"]) / "work"
            done = cli_lib.run(argv, b["prompt"], workdir, timeout_s=timeout, cancel=cancel)
            if done["error"] and not done["text"].strip() and not (workdir / "index.html").is_file():
                self.store.upsert_item(rid, b["id"], model["slot"], state="done", outcome="none", reason="cli_error", detail=text("bk_cli_failed", error=done["error"]), latency_ms=done["wall_ms"])
            else:
                reply = cli_lib.collected(workdir, done["text"], b["kind"])
                self._receive(run, rs, b, model, reply, {"latency_ms": done["wall_ms"], "finish_reason": "cli"}, cancel)
            self._count_answered(rid)
        import shutil
        for b in briefs:
            shutil.rmtree(self._folder(rid, b["id"], model["slot"]) / "work", ignore_errors=True)

    def _receive(self, run: dict[str, Any], rs: dict[str, Any], b: dict[str, Any], model: dict[str, Any], reply: str, metrics: dict[str, Any], cancel: Callable[[], bool]) -> None:
        """What an answer is worth once it is in: save it, render it if it is a page, and leave it ready for the judge (or done when nobody judges)."""
        rid, slot = run["id"], model["slot"]
        folder = self._folder(rid, b["id"], slot)
        folder.mkdir(parents=True, exist_ok=True)
        answer = extract_answer(reply, b["kind"])
        finish = metrics.get("finish_reason") or "-"
        if not answer:
            (folder / "answer.txt").write_text((reply or "")[:20000], encoding="utf-8")
            if b["kind"] == "page":
                self.store.upsert_item(rid, b["id"], slot, state="done", outcome="unrendered", reason="no_html", detail=text("bk_no_html", finish=finish), folder=str(folder), **metrics)
            else:
                self.store.upsert_item(rid, b["id"], slot, state="done", outcome="none", reason="empty", detail=text("bk_no_text", finish=finish), folder=str(folder), **metrics)
            return
        if b["kind"] == "text":
            (folder / "answer.txt").write_text(answer, encoding="utf-8")
            self.store.upsert_item(rid, b["id"], slot, state="ready" if run["jury"] else "done", outcome="done", chars=len(answer), folder=str(folder), **metrics)
            return
        (folder / "index.html").write_text(answer, encoding="utf-8")
        self.store.upsert_item(rid, b["id"], slot, state="rendering", chars=len(answer), static={**static_check(answer), "truncated": finish == "length"}, folder=str(folder), **metrics)
        self._render(run, rs, b, slot, folder, answer)

    def _render(self, run: dict[str, Any], rs: dict[str, Any], b: dict[str, Any], slot: str, folder: Path, html: str) -> None:
        rid = run["id"]
        scratch = folder / "_render.html"
        scratch.write_text(inline_libraries(html, self.libraries_dir), encoding="utf-8")
        try:
            report = self.render_fn(scratch, folder, python=self.python(), settle_ms=int(rs["settle_ms"]), motion_ms=int(rs["motion_ms"]), timeout_s=30.0,
                                    total_timeout_s=float(rs["render_timeout_s"]))
        finally:
            scratch.unlink(missing_ok=True)
        if not report.get("ok"):
            self.store.upsert_item(rid, b["id"], slot, state="done", outcome="unrendered", reason="render_failed", detail=str(report.get("error") or "")[:300], render=report)
            return
        report["motion_ms"] = int(rs["motion_ms"])
        try:
            auto = shots.measure(folder, report["files"])
            shots.thumbnail(folder)
        except Exception as exc:  # noqa: BLE001
            self.store.upsert_item(rid, b["id"], slot, state="done", outcome="unrendered", reason="pictures_unreadable", detail=f"{type(exc).__name__}: {exc}"[:300], render=report)
            return
        auto["usable"] = bool(auto["not_blank"] and not report.get("page_errors"))
        self.store.upsert_item(rid, b["id"], slot, state="ready" if run["jury"] else "done", outcome="done", render=report, auto=auto)

    # -- phase 2: the judge --------------------------------------------------------------------------------------------
    def _phase_judge(self, run: dict[str, Any], rs: dict[str, Any], cancel: Callable[[], bool]) -> None:
        rid = run["id"]
        pending = [i for i in self.store.items(rid) if i["state"] == "ready"]
        self._progress(rid, phase="judging", judged=0)
        briefs = {b["id"]: b for b in run["briefs"]}
        skipped = len(run["models"]) * len(run["briefs"]) - len(pending)
        c = self.runner.store.contestant(run["jury"])
        mode = self._judge_mode(c, rs)
        self._progress(rid, judge_mode=mode)
        if not pending:
            return
        infra = 0
        try:
            with self.runner.session(c, rs, cancel, lambda s: None) as session:
                self._progress(rid, judge_runs_on=str(session.runs_on))
                for n, item in enumerate(pending):
                    if cancel():
                        break
                    self._progress(rid, judged=skipped + n)
                    b = briefs[item["brief_id"]]
                    folder = Path(item["folder"])
                    body = ""
                    if b["kind"] == "text":
                        body = (folder / "answer.txt").read_text(encoding="utf-8", errors="replace")
                    elif mode == "text" and (folder / "index.html").is_file():
                        body = (folder / "index.html").read_text(encoding="utf-8", errors="replace")
                    images = []
                    if b["kind"] == "page" and mode == "images":
                        images = [shots.png_bytes(folder / f, width=960) for f in (item["render"].get("files") or []) if (folder / f).is_file()]
                    messages = judge_messages(b, body, item["auto"], item["render"], item["static"], images=bool(images))
                    verdict, last_error, comp = None, "", None
                    for attempt in range(2):
                        retry = [] if attempt == 0 else [{"role": "assistant", "content": comp.text[:2000]}, {"role": "user", "content": "Reply with the JSON object only, as specified."}]
                        comp = self._chat(session, rs, cancel, [*messages, *retry], max_tokens=int(rs["judge_max_tokens"]), temperature=0.0, images=images)
                        if comp.error:
                            last_error = str(comp.error)
                            break
                        verdict = parse_judge(comp.text)
                        if verdict is not None:
                            break
                    if verdict is None:
                        infra += 1 if last_error else 0
                        self.store.upsert_item(rid, item["brief_id"], item["slot"], state="done", reason="judge_error" if last_error else "judge_unreadable",
                                               detail=last_error[:300] or text("bk_judge_unreadable"))
                        if infra >= MAX_CONSECUTIVE_INFRA:
                            for rest in [i for i in pending if self.store.item(rid, i["brief_id"], i["slot"])["state"] == "ready"]:
                                self.store.upsert_item(rid, rest["brief_id"], rest["slot"], state="done", reason="judge_error", detail=text("bk_infra_abort", n=infra))
                            break
                    else:
                        infra = 0
                        self.store.upsert_item(rid, item["brief_id"], item["slot"], state="done", judge=verdict, judge_mode=mode, mean=verdict["mean"])
                self._progress(rid, judged=skipped + len(pending))
                self._verify(rid, session, "judge")
        except (Cancelled, proc.Cancelled):
            raise
        except GaltonError as exc:
            if cancel():
                return
            for i in self.store.items(rid):
                if i["state"] == "ready":
                    self.store.upsert_item(rid, i["brief_id"], i["slot"], state="done", reason="judge_error", detail=text("bk_session_failed", error=str(exc.message)[:240]))
            raise

    # ------------------------------------------------------------------ reading
    def card(self, run: dict[str, Any]) -> dict[str, Any]:
        jury = None
        if run["jury"]:
            found = self.runner.store.find_contestant(run["jury"])
            jury = {"id": run["jury"], "name": found["name"] if found else run["jury"]}
        return {"id": run["id"], "label": run["label"], "state": run["state"], "models": [{k: m[k] for k in ("slot", "id", "name", "kind")} for m in run["models"]], "judge": jury,
                "same_model_judges": bool(run["jury"]) and any(m["id"] == run["jury"] for m in run["models"]),
                "briefs": [{k: b[k] for k in ("id", "title", "kind")} for b in run["briefs"]], "n": len(run["models"]) * len(run["briefs"]), "settings": run["settings"],
                "currency": self.currency, "progress": run["progress"], "total": run["total"], "done": run["done"], "source": run["source"], "created_ts": run["created_ts"],
                "started_ts": run["started_ts"], "finished_ts": run["finished_ts"], "error": run["error"], "active": run["id"] == self.active, "cancel": run["cancel"]}

    @staticmethod
    def _speed(i: dict[str, Any]) -> Optional[float]:
        if i["decode_tps"]:
            return round(float(i["decode_tps"]), 1)
        if i["tokens_out"] and i["latency_ms"]:
            return round(i["tokens_out"] / (i["latency_ms"] / 1000), 1)
        return None

    def summary(self, rid: str) -> dict[str, Any]:
        run = self.store.run(rid)
        items = self.store.items(rid)
        models = []
        for m in run["models"]:
            mine = [i for i in items if i["slot"] == m["slot"]]
            answered = [i for i in mine if i["state"] not in ("queued", "writing") and i["outcome"] != "none"]
            done = [i for i in mine if i["outcome"] == "done"]
            judged = [i for i in done if i["mean"] is not None]
            lat = [i["latency_ms"] / 1000 for i in answered if i["latency_ms"]]
            costs = [i["cost"] for i in answered if i["cost"] is not None]
            tin = [i["tokens_in"] for i in answered if i["tokens_in"] is not None]
            tout = [i["tokens_out"] for i in answered if i["tokens_out"] is not None]
            tout_time = sum(i["latency_ms"] / 1000 for i in answered if i["tokens_out"] and i["latency_ms"])
            models.append({"slot": m["slot"], "id": m["id"], "name": m["name"], "kind": m["kind"], "briefs": len(mine), "answered": len(answered), "usable": len(done),
                           "none": sum(1 for i in mine if i["outcome"] == "none"), "failed": sum(1 for i in mine if i["outcome"] == "unrendered"),
                           "latency_s_mean": _mean(lat, 1), "latency_s_total": round(sum(lat), 1) if lat else None, "tokens_in_total": sum(tin) if tin else None,
                           "tokens_out_total": sum(tout) if tout else None, "tokens_per_s": round(sum(tout) / tout_time, 1) if tout and tout_time else None,
                           "price": run["prices"].get(m["slot"]), "cost_total": round(sum(costs), 6) if costs else None, "cost_known": len(costs), "cost_unknown": len(answered) - len(costs),
                           "judged": len(judged), "mean_score": _mean([i["mean"] for i in judged]),
                           "by_dimension": {d: _mean([i["judge"]["scores"].get(d) for i in judged if d in (i["judge"].get("scores") or {})]) for d in DIMENSIONS}})

        def best(key: str, low: bool) -> Optional[str]:
            pool = [m for m in models if m[key] is not None]
            if not pool:
                return None
            pick = (min if low else max)(pool, key=lambda m: m[key])
            return pick["name"] if sum(1 for m in pool if m[key] == pick[key]) == 1 or len(pool) == 1 else None
        return {"models": models, "briefs": len(run["briefs"]), "items": len(items), "judged": sum(1 for i in items if i["mean"] is not None),
                "none": sum(1 for i in items if i["outcome"] == "none"), "failed": sum(1 for i in items if i["outcome"] == "unrendered"),
                "best_score": best("mean_score", False), "fastest": best("latency_s_mean", True), "cheapest": best("cost_total", True), "currency": self.currency,
                "judge_mode": run["progress"].get("judge_mode", ""), "same_model_judges": bool(run["jury"]) and any(m["id"] == run["jury"] for m in run["models"])}

    def _row(self, rid: str, i: dict[str, Any]) -> dict[str, Any]:
        folder = Path(i["folder"]) if i["folder"] else None
        scores = (i["judge"] or {}).get("scores", {})
        return {"slot": i["slot"], "model": i["model_name"], "state": i["state"], "outcome": i["outcome"], "reason": i["reason"], "detail": i["detail"],
                "latency_s": round(i["latency_ms"] / 1000, 2) if i["latency_ms"] is not None else None, "ttft_s": round(i["ttft_ms"] / 1000, 2) if i["ttft_ms"] is not None else None,
                "tokens_in": i["tokens_in"], "tokens_out": i["tokens_out"], "tokens_per_s": self._speed(i), "cost": i["cost"], "chars": i["chars"], "mean": i["mean"], "scores": scores,
                "judge_summary": (i["judge"] or {}).get("summary", ""), "moves": (i["auto"] or {}).get("moves"), "usable": (i["auto"] or {}).get("usable"),
                "page_errors": len((i["render"] or {}).get("page_errors") or []), "finish_reason": i["finish_reason"],
                "has_shot": bool(folder) and (folder / "shot.png").is_file(), "has_page": bool(folder) and (folder / "index.html").is_file(),
                "has_text": bool(folder) and (folder / "answer.txt").is_file()}

    def result(self, rid: str) -> dict[str, Any]:
        run = self.store.run(rid)
        items = self.store.items(rid)
        blocks = []
        for b in run["briefs"]:
            rows = [self._row(rid, i) for i in items if i["brief_id"] == b["id"]]
            blocks.append({"id": b["id"], "title": b["title"], "kind": b["kind"], "grid_url": f"/api/briefbench/{rid}/{b['id']}/grid.png", "items": rows})
        return {"run": self.card(run), "summary": run["summary"] or self.summary(rid), "briefs": blocks, "markdown": self.markdown(blocks, self.currency)}

    @staticmethod
    def markdown(blocks: list[dict[str, Any]], currency: str) -> str:
        lines = []
        for b in blocks:
            lines += [f"### {b['title']} ({b['kind']})", "", f"| model | outcome | time s | tokens in | tokens out | tok/s | cost {currency} | score |", "|---|---|---|---|---|---|---|---|"]
            for r in b["items"]:
                cells = [r["model"], (r["outcome"] or r["state"]) + (f" ({r['reason']})" if r["reason"] else ""), r["latency_s"], r["tokens_in"], r["tokens_out"], r["tokens_per_s"],
                         r["cost"], r["mean"]]
                lines.append("| " + " | ".join("-" if c is None else str(c) for c in cells) + " |")
            lines.append("")
        return "\n".join(lines).strip()

    def item_detail(self, rid: str, bid: str, slot: str, *, answer: bool = False) -> dict[str, Any]:
        run = self.store.run(rid)
        item = self.store.item(rid, bid, slot)
        brief = next((b for b in run["briefs"] if b["id"] == bid), {"id": bid})
        out = {**item, "brief": brief, "currency": self.currency, "tokens_per_s": self._speed(item), "price": run["prices"].get(slot)}
        folder = Path(item["folder"]) if item["folder"] else None
        out["files"] = {n: (folder / n).is_file() for n in ("index.html", "shot.png", "shot2.png", "answer.txt")} if folder else {}
        if answer and folder:
            for n in ("index.html", "answer.txt"):
                if (folder / n).is_file():
                    out["answer"] = (folder / n).read_text(encoding="utf-8", errors="replace")
                    break
        return out

    def file_path(self, rid: str, bid: str, slot: str, name: str) -> Path:
        """A file of an item the screen may show. Only the names this bench writes are served."""
        if not FILES.match(name) or not SLOT.match(slot):
            raise GaltonError("not_found", "bk_no_file", name=name)
        item = self.store.item(rid, bid, slot)
        path = (Path(item["folder"]) / name) if item["folder"] else None
        if path is None or not path.is_file():
            raise GaltonError("not_found", "bk_no_file", name=name)
        return path

    def page_html(self, rid: str, bid: str, slot: str) -> str:
        """The page of an item as it is played: with the libraries it names put in."""
        return inline_libraries(self.file_path(rid, bid, slot, "index.html").read_text(encoding="utf-8", errors="replace"), self.libraries_dir)

    def grid_path(self, rid: str, bid: str) -> Path:
        """The comparison sheet of a brief, made now from the items it has (a model that has not answered yet shows as waiting)."""
        run = self.store.run(rid)
        brief = next((b for b in run["briefs"] if b["id"] == bid), None)
        if brief is None:
            raise GaltonError("not_found", "bk_no_file", name="grid.png")
        items = [i for i in self.store.items(rid) if i["brief_id"] == bid]
        return grid_lib.compose(items, f"{brief['title']} · {len(items)} model(s)", self.root / rid / bid / "grid.png", currency=self.currency)

    def delete(self, rid: str) -> dict[str, Any]:
        import shutil
        run = self.store.run(rid)
        if run["state"] in ACTIVE:
            raise GaltonError("invalid", "bk_run_active", id=rid)
        shutil.rmtree(self.root / run["id"], ignore_errors=True)
        self.store.delete_run(run["id"])
        return {"deleted": run["id"]}
