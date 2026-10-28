"""The video bank inside the app: a run is a background thread (it does not use the GPU lane: the model is served somewhere else), its items are in the database
and its files (page, frames, contact sheet) under ``data/videobench/<run>/<brief>/``.

Three phases, so that the writer and the jury are two different sessions: the model writes a page for each brief; each page is rendered in a browser (no model);
the jury, a fresh session that did not write the page, looks at the frames and scores them. A call that could not be answered leaves an item as ``none``: it is
not a score of zero and it is left out of the means.
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
from . import render as rendering
from .contract import DEFAULT_DURATION, MAX_DURATION, build_messages, extract_html, static_check
from .jury import DIMENSIONS, SCORED, RUBRIC, jury_messages, parse_jury
from .prompts import CATEGORIES, FILTER, PROMPTS, PROMPTS_BY_ID, SOURCE, TAGS, public, select
from .store import ACTIVE, TERMINAL, VideoStore

log = logging.getLogger("galton.videobench")

MAX_PROMPTS = 200
MAX_CONSECUTIVE_INFRA = 3
DEFAULTS = {"max_tokens": 16000, "timeout_s": 900, "temperature": 0.7, "frames": 12, "duration_cap": MAX_DURATION, "default_duration": DEFAULT_DURATION, "render_timeout_s": 240,
            "jury_images": "auto", "jury_max_tokens": 4000}
BOUNDS = {"frames": (4, 24), "duration_cap": (2, 60), "default_duration": (2, 60), "render_timeout_s": (20, 1800), "jury_max_tokens": (200, 32000)}
FILES = re.compile(r"^(index\.html|sheet\.png|key[12]\.png|frames/f\d{2}\.png)$")


class VideoJobEvents(RunJobEvents):
    def url(self, run_id: str) -> str:
        base = (self._base_url() or "").rstrip("/")
        return f"{base}/#/videos/{run_id}" if base else ""


def normalise_video_settings(raw: Optional[dict[str, Any]]) -> dict[str, Any]:
    """The settings of a video run: the shared run settings (token cap, temperature, timeout, seed, effort, device...) plus the ones of this bench."""
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
    images = raw.pop("jury_images", None)
    if images is None or str(images).strip().lower() == "auto":
        mine["jury_images"] = "auto"
    elif str(images).strip().lower() in ("1", "true", "yes", "on", "0", "false", "no", "off"):
        mine["jury_images"] = str(images).strip().lower() in ("1", "true", "yes", "on")
    else:
        raise GaltonError("invalid", "setting_bool", setting="jury_images")
    for key in ("max_tokens", "timeout_s", "temperature"):
        if raw.get(key) is None:
            raw[key] = DEFAULTS[key]
    return {**normalise_settings(raw), **mine}


def _mean(values: list[float], digits: int = 2) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), digits) if values else None


class VideoBench:
    def __init__(self, store: VideoStore, runner: Runner, settings: Any, *, data_dir: Path, emit: Optional[Callable[[str, dict], None]] = None,
                 clock: Callable[[], float] = time.time, base_url: Callable[[], str] = lambda: "", render_fn: Optional[Callable[..., dict[str, Any]]] = None):
        self.store, self.runner, self.settings, self.data_dir, self.clock = store, runner, settings, Path(data_dir), clock
        self.jobs = VideoJobEvents(emit or (lambda *_: None), clock=clock, base_url=base_url)
        self.render_fn = render_fn or rendering.render
        self.threads: dict[str, tuple[threading.Thread, threading.Event]] = {}
        self.lock = threading.Lock()
        self.active: Optional[str] = None

    # ------------------------------------------------------------------ what can be run
    @property
    def root(self) -> Path:
        configured = str(self.settings.get("videobench.folder") or "").strip()
        return Path(configured) if configured else self.data_dir / "videobench"

    def python(self) -> str:
        return rendering.browser_python(str(self.settings.get("videobench.browser_python") or ""))

    def catalog(self, *, category: str = "", tag: str = "", standalone: Optional[bool] = None, query: str = "", limit: int = 60, offset: int = 0) -> dict[str, Any]:
        rows = select((), category=category, tag=tag, standalone=standalone, query=query)
        py = self.python()
        return {"source": SOURCE, "filter": FILTER, "total": len(PROMPTS), "standalone": sum(1 for p in PROMPTS if p["standalone"]), "matching": len(rows),
                "categories": {c: sum(1 for p in PROMPTS if p["category"] == c) for c in CATEGORIES}, "tags": TAGS,
                "prompts": [public(p) for p in rows[max(0, offset):max(0, offset) + max(1, min(limit, 200))]],
                "rubric": {d: RUBRIC[d] for d in DIMENSIONS}, "scored_dimensions": list(SCORED), "browser": {"python": py, "found": bool(py)},
                "license": "The list of prompts is MIT; every video belongs to its author (author and post are kept with each brief). The rubric is MIT too (see provenance/videobench)."}

    def prompt(self, pid: str) -> dict[str, Any]:
        p = PROMPTS_BY_ID.get(pid)
        if p is None:
            raise GaltonError("not_found", "vb_no_prompt", id=pid)
        return public(p, full=True)

    def resolve(self, prompts: Optional[list[str]], count: int, seed: int, category: str, tag: str, standalone: Optional[bool]) -> list[dict[str, Any]]:
        ids = [str(i).strip() for i in (prompts or []) if str(i).strip()]
        unknown = [i for i in ids if i not in PROMPTS_BY_ID]
        if unknown:
            raise GaltonError("not_found", "vb_no_prompt", id=unknown[0])
        rows = select(ids, category=category, tag=tag, standalone=standalone, count=int(count or 0), seed=int(seed or 0))
        if not rows:
            raise GaltonError("invalid", "vb_no_prompts")
        if len(rows) > MAX_PROMPTS:
            raise GaltonError("invalid", "vb_too_many", n=len(rows), max=MAX_PROMPTS)
        return rows

    def plan(self, *, model: Any, jury: Any = None, prompts: Optional[list[str]] = None, count: int = 10, seed: int = 0, category: str = "", tag: str = "",
             standalone: Optional[bool] = True, settings: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        rs = normalise_video_settings(settings)
        rows = self.runner.resolve_contestants([model])
        jury_rows = self.runner.resolve_contestants([jury]) if jury else rows
        chosen = self.resolve(prompts, count, seed, category, tag, standalone)
        py = self.python()
        mode = self._jury_mode(jury_rows[0], rs)
        return {"model": rows[0]["name"], "jury": jury_rows[0]["name"], "same_model_judges": jury_rows[0]["id"] == rows[0]["id"], "prompts": [public(p) for p in chosen], "n": len(chosen),
                "jury_mode": mode, "browser": {"python": py, "found": bool(py)}, "settings": rs,
                "calls": {"writing": len(chosen), "jury": len(chosen)},
                "note": text("vb_plan_note", n=len(chosen))}

    @staticmethod
    def _jury_mode(c: dict[str, Any], rs: dict[str, Any]) -> str:
        want = rs["jury_images"]
        if want == "auto":
            want = bool(c.get("vision"))
        return "images" if want else "text"

    # ------------------------------------------------------------------ runs
    def start(self, *, model: Any, jury: Any = None, prompts: Optional[list[str]] = None, count: int = 10, seed: int = 0, category: str = "", tag: str = "",
              standalone: Optional[bool] = True, settings: Optional[dict[str, Any]] = None, label: str = "", source: str = "ui", caller: str = "",
              start_thread: bool = True) -> dict[str, Any]:
        rs = normalise_video_settings(settings)
        rows = self.runner.resolve_contestants([model])
        c = rows[0]
        if not c["enabled"]:
            raise GaltonError("invalid", "model_disabled", name=c["name"])
        if c["missing"]:
            raise GaltonError("not_found", "model_gone", name=c["name"])
        self.runner.ensure_served(c)
        j = self.runner.resolve_contestants([jury])[0] if jury else c
        if j["id"] != c["id"]:
            self.runner.ensure_served(j)
        chosen = self.resolve(prompts, count, seed, category, tag, standalone)
        if not self.python():
            raise GaltonError("unavailable", "vb_no_browser")
        run = self.store.create_run(label=label or text("label_videobench", model=c["name"], n=len(chosen)), state="queued", model=c["id"], jury=j["id"],
                                    prompts=[p["id"] for p in chosen], settings=rs, progress={"phase": "queued", "written": 0, "rendered": 0, "judged": 0, "warnings": []},
                                    total=len(chosen) * 3, done=0, source=source, caller=caller)
        for p in chosen:
            self.store.upsert_item(run["id"], p["id"], state="queued")
        self.jobs.queued(run)
        if start_thread:
            self._spawn(run["id"])
        return self.card(self.store.run(run["id"]))

    def _spawn(self, rid: str) -> None:
        with self.lock:
            if rid in self.threads and self.threads[rid][0].is_alive():
                return
            cancel = threading.Event()
            t = threading.Thread(target=self.execute, args=(rid, cancel), name=f"videobench-{rid}", daemon=True)
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
            done = int(prog.get("written", 0)) + int(prog.get("rendered", 0)) + int(prog.get("judged", 0))
            self.store.update_run(rid, progress=prog, done=done)

    def _warn(self, rid: str, message: Any) -> None:
        run = self.store.run(rid)
        warnings = [*run["progress"].get("warnings", []), str(message)]
        self._progress(rid, warnings=warnings[-10:])

    def _folder(self, rid: str, pid: str) -> Path:
        return self.root / rid / pid

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
            prompts = [PROMPTS_BY_ID[i] for i in run["prompts"] if i in PROMPTS_BY_ID]
            self._phase_write(run, prompts, rs, cancel)
            if not cancel():
                self._phase_render(run, rs, cancel)
            if not cancel():
                self._phase_judge(run, rs, cancel)
            state = "cancelled" if cancel() else "done"
        except GaltonError as exc:
            state, error = "failed", str(exc.message)
            log.info("video run %s failed: %s", rid, exc.message)
        except (Cancelled, proc.Cancelled):
            state = "cancelled"
        except Exception as exc:  # noqa: BLE001
            log.exception("video run %s failed", rid)
            state, error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            self.active = None
        if state == "done":
            n = len(run["prompts"])
            self._progress(rid, phase="done", written=n, rendered=n, judged=n)
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
        self._warn(rid, text("vb_identity_unverified", who=who, state=state))

    # -- phase 1: the model writes the pages -----------------------------------------------------------------------------
    def _phase_write(self, run: dict[str, Any], prompts: list[dict[str, Any]], rs: dict[str, Any], cancel: Callable[[], bool]) -> None:
        rid = run["id"]
        c = self.runner.store.contestant(run["model"])
        self._progress(rid, phase="writing")
        infra = 0
        try:
            with self.runner.session(c, rs, cancel, lambda s: None) as session:
                self._progress(rid, writer_runs_on=str(session.runs_on))
                for n, p in enumerate(prompts):
                    if cancel():
                        break
                    self._progress(rid, written=n)
                    item = self.store.item(rid, p["id"])
                    if item["state"] not in ("queued", "writing"):
                        continue
                    self.store.upsert_item(rid, p["id"], state="writing")
                    seed = (int(rs["seed"]) + n) if rs["seed"] is not None else None
                    comp = self._chat(session, rs, cancel, build_messages(p["prompt"]), max_tokens=int(rs["max_tokens"]), temperature=float(rs["temperature"]), seed=seed)
                    if comp.error:
                        infra += 1
                        self.store.upsert_item(rid, p["id"], state="done", outcome="none", reason="server_error", detail=str(comp.error)[:300],
                                               tokens_in=comp.prompt_tokens, tokens_out=comp.completion_tokens, latency_ms=comp.latency_ms)
                        if infra >= MAX_CONSECUTIVE_INFRA:
                            self._mark_rest_none(rid, prompts[n + 1:], "server_error", text("vb_infra_abort", n=infra))
                            break
                        continue
                    infra = 0
                    html = extract_html(comp.text)
                    truncated = comp.finish_reason == "length"
                    if not html:
                        self.store.upsert_item(rid, p["id"], state="done", outcome="unrendered", reason="no_html",
                                               detail=text("vb_no_html", finish=comp.finish_reason or "-"), tokens_in=comp.prompt_tokens,
                                               tokens_out=comp.completion_tokens, latency_ms=comp.latency_ms, static={"truncated": truncated})
                    else:
                        folder = self._folder(rid, p["id"])
                        folder.mkdir(parents=True, exist_ok=True)
                        (folder / "index.html").write_text(html, encoding="utf-8")
                        self.store.upsert_item(rid, p["id"], state="written", html_chars=len(html), tokens_in=comp.prompt_tokens, tokens_out=comp.completion_tokens,
                                               latency_ms=comp.latency_ms, static={**static_check(html), "truncated": truncated}, folder=str(folder))
                self._progress(rid, written=len(prompts))
                self._verify(rid, session, "writer")
        except (Cancelled, proc.Cancelled):
            raise
        except GaltonError as exc:
            if cancel():
                return
            self._mark_rest_none(rid, [q for q in prompts if self.store.item(rid, q["id"])["state"] in ("queued", "writing")], "session", text("vb_session_failed", error=str(exc.message)[:240]))
            self._progress(rid, written=len(prompts))
            raise

    def _mark_rest_none(self, rid: str, rest: list[dict[str, Any]], reason: str, detail: Any) -> None:
        for q in rest:
            if self.store.item(rid, q["id"])["state"] in ("queued", "writing"):
                self.store.upsert_item(rid, q["id"], state="done", outcome="none", reason=reason, detail=str(detail)[:300])

    # -- phase 2: render ------------------------------------------------------------------------------------------------
    def _phase_render(self, run: dict[str, Any], rs: dict[str, Any], cancel: Callable[[], bool]) -> None:
        rid = run["id"]
        self._progress(rid, phase="rendering")
        python = self.python()
        items = self.store.items(rid)
        for n, item in enumerate(items):
            if cancel():
                return
            self._progress(rid, rendered=n)
            if item["state"] != "written":
                continue
            pid = item["prompt_id"]
            folder = Path(item["folder"])
            self.store.upsert_item(rid, pid, state="rendering")
            report = self.render_fn(folder / "index.html", folder / "frames", python=python, frames=int(rs["frames"]), duration_cap=float(rs["duration_cap"]),
                                    default_duration=float(rs["default_duration"]), timeout_s=20.0, total_timeout_s=float(rs["render_timeout_s"]))
            if not report.get("ok"):
                self.store.upsert_item(rid, pid, state="done", outcome="unrendered", reason="render_failed", detail=str(report.get("error") or "")[:300], render=report)
                continue
            try:
                auto = rendering.measure(folder / "frames", report["files"], report["times"])
                rendering.contact_sheet(folder / "frames", report["files"], report["times"], folder / "sheet.png")
                self._key_frames(folder, report)
            except Exception as exc:  # noqa: BLE001
                self.store.upsert_item(rid, pid, state="done", outcome="unrendered", reason="frames_unreadable", detail=f"{type(exc).__name__}: {exc}"[:300], render=report)
                continue
            usable = bool(auto["moves"] and auto["not_blank"] and report.get("deterministic") and not report.get("page_errors"))
            auto["usable"] = usable
            self.store.upsert_item(rid, pid, state="rendered", render=report, auto=auto)
        self._progress(rid, rendered=len(items))

    @staticmethod
    def _key_frames(folder: Path, report: dict[str, Any]) -> None:
        """Two larger frames for the jury: about a third and about four fifths of the way through."""
        import shutil
        files = report["files"]
        for name, frac in (("key1.png", 0.34), ("key2.png", 0.8)):
            shutil.copyfile(folder / "frames" / files[min(len(files) - 1, round(frac * (len(files) - 1)))], folder / name)

    # -- phase 3: the jury ---------------------------------------------------------------------------------------------
    def _phase_judge(self, run: dict[str, Any], rs: dict[str, Any], cancel: Callable[[], bool]) -> None:
        rid = run["id"]
        pending = [i for i in self.store.items(rid) if i["state"] == "rendered"]
        self._progress(rid, phase="judging")
        if not pending:
            return
        c = self.runner.store.contestant(run["jury"])
        mode = self._jury_mode(c, rs)
        self._progress(rid, jury_mode=mode)
        infra = 0
        try:
            with self.runner.session(c, rs, cancel, lambda s: None) as session:
                self._progress(rid, jury_runs_on=str(session.runs_on))
                for n, item in enumerate(pending):
                    if cancel():
                        break
                    self._progress(rid, judged=n)
                    pid = item["prompt_id"]
                    folder = Path(item["folder"])
                    brief = PROMPTS_BY_ID[pid]["prompt"]
                    html = (folder / "index.html").read_text(encoding="utf-8") if mode == "text" else ""
                    messages = jury_messages(brief, item["auto"], item["render"], images=(mode == "images"), html=html)
                    images = [rendering.png_bytes(folder / "sheet.png"), rendering.png_bytes(folder / "key1.png", width=960), rendering.png_bytes(folder / "key2.png", width=960)] if mode == "images" else []
                    verdict, last_error = None, ""
                    for attempt in range(2):
                        comp = self._chat(session, rs, cancel, messages if attempt == 0 else [*messages, {"role": "assistant", "content": comp.text[:2000]},
                                                                                              {"role": "user", "content": "Reply with the JSON object only, as specified."}],
                                          max_tokens=int(rs["jury_max_tokens"]), temperature=0.0, images=images)
                        if comp.error:
                            last_error = str(comp.error)
                            break
                        verdict = parse_jury(comp.text)
                        if verdict is not None:
                            break
                    if verdict is None:
                        infra += 1 if last_error else 0
                        self.store.upsert_item(rid, pid, state="done", outcome="none", reason="jury_error" if last_error else "jury_unreadable",
                                               detail=last_error[:300] or text("vb_jury_unreadable"))
                        if infra >= MAX_CONSECUTIVE_INFRA:
                            for rest in [i for i in pending if self.store.item(rid, i["prompt_id"])["state"] == "rendered"]:
                                self.store.upsert_item(rid, rest["prompt_id"], state="done", outcome="none", reason="server_error", detail=text("vb_infra_abort", n=infra))
                            break
                    else:
                        infra = 0
                        self.store.upsert_item(rid, pid, state="done", outcome="done", jury=verdict, jury_mode=mode, mean=verdict["mean"])
                self._progress(rid, judged=len(pending))
                self._verify(rid, session, "jury")
        except (Cancelled, proc.Cancelled):
            raise
        except GaltonError as exc:
            if cancel():
                return
            for i in self.store.items(rid):
                if i["state"] == "rendered":
                    self.store.upsert_item(rid, i["prompt_id"], state="done", outcome="none", reason="session", detail=text("vb_session_failed", error=str(exc.message)[:240]))
            raise

    # ------------------------------------------------------------------ reading
    def names(self, ids: list[str]) -> dict[str, str]:
        out = {}
        for cid in ids:
            c = self.runner.store.find_contestant(cid)
            out[cid] = c["name"] if c else cid
        return out

    def card(self, run: dict[str, Any]) -> dict[str, Any]:
        names = self.names([run["model"], run["jury"]])
        return {"id": run["id"], "label": run["label"], "state": run["state"], "model": {"id": run["model"], "name": names[run["model"]]},
                "jury": {"id": run["jury"], "name": names[run["jury"]]}, "same_model_judges": run["model"] == run["jury"], "n": len(run["prompts"]), "settings": run["settings"],
                "progress": run["progress"], "total": run["total"], "done": run["done"], "source": run["source"], "created_ts": run["created_ts"], "started_ts": run["started_ts"],
                "finished_ts": run["finished_ts"], "error": run["error"], "active": run["id"] == self.active, "cancel": run["cancel"]}

    def summary(self, rid: str) -> dict[str, Any]:
        run = self.store.run(rid)
        items = self.store.items(rid)
        asked = [i for i in items if (i["outcome"] != "none" or i["auto"] or i["render"]) and i["state"] not in ("queued", "writing")]      # the model answered (or tried to)
        rendered = [i for i in items if i["auto"]]
        judged = [i for i in items if i["outcome"] == "done"]
        usable = [i for i in rendered if i["auto"].get("usable")]
        by_dim = {d: _mean([i["jury"]["scores"].get(d) for i in judged if d in (i["jury"].get("scores") or {})]) for d in SCORED}
        by_cat: dict[str, Any] = {}
        for i in judged:
            cat = PROMPTS_BY_ID.get(i["prompt_id"], {}).get("category", "")
            by_cat.setdefault(cat, []).append(i["mean"])
        n_asked = len(asked)
        return {"prompts": len(items), "answered": n_asked, "rendered": len(rendered), "usable": len(usable), "judged": len(judged), "none": sum(1 for i in items if i["outcome"] == "none"),
                "render_rate": round(len(rendered) / n_asked, 3) if n_asked else None, "usable_rate": round(len(usable) / n_asked, 3) if n_asked else None,
                "deterministic_rate": _mean([1.0 if i["render"].get("deterministic") else 0.0 for i in rendered], 3), "mean_score": _mean([i["mean"] for i in judged]),
                "by_dimension": by_dim, "by_category": {k: {"n": len(v), "mean": _mean(v)} for k, v in by_cat.items()},
                "tokens_out_mean": _mean([float(i["tokens_out"]) for i in asked if i["tokens_out"]], 0), "latency_s_mean": _mean([i["latency_ms"] / 1000 for i in asked if i["latency_ms"]], 1),
                "jury_mode": run["progress"].get("jury_mode", ""), "same_model_judges": run["model"] == run["jury"]}

    def result(self, rid: str) -> dict[str, Any]:
        run = self.store.run(rid)
        items = self.store.items(rid)
        rows = []
        for i in items:
            meta = PROMPTS_BY_ID.get(i["prompt_id"], {})
            rows.append({"prompt": i["prompt_id"], "category": meta.get("category", ""), "author": meta.get("author", ""), "post_url": meta.get("post_url", ""), "state": i["state"],
                         "outcome": i["outcome"], "reason": i["reason"], "detail": i["detail"], "mean": i["mean"], "scores": (i["jury"] or {}).get("scores", {}),
                         "usable": (i["auto"] or {}).get("usable"), "deterministic": (i["render"] or {}).get("deterministic"), "moves": (i["auto"] or {}).get("moves"),
                         "duration": (i["render"] or {}).get("duration"), "html_chars": i["html_chars"], "tokens_out": i["tokens_out"],
                         "summary": (i["jury"] or {}).get("summary", ""), "has_sheet": bool(i["folder"]) and (Path(i["folder"]) / "sheet.png").is_file()})
        return {"run": self.card(run), "summary": run["summary"] or self.summary(rid), "items": rows, "markdown": self.markdown(rows)}

    @staticmethod
    def markdown(rows: list[dict[str, Any]]) -> str:
        head = "| brief | category | outcome | moves | deterministic | mean | " + " | ".join(SCORED) + " |\n|" + "---|" * (6 + len(SCORED))
        lines = [head]
        for r in rows:
            sc = r["scores"] or {}
            lines.append(f"| {r['prompt']} | {r['category']} | {r['outcome'] or r['state']}{' (' + r['reason'] + ')' if r['reason'] else ''} | {r['moves'] if r['moves'] is not None else '-'} | "
                         f"{r['deterministic'] if r['deterministic'] is not None else '-'} | {r['mean'] if r['mean'] is not None else '-'} | " + " | ".join(str(sc.get(d, "-")) for d in SCORED) + " |")
        return "\n".join(lines)

    def item_detail(self, rid: str, pid: str, *, html: bool = False) -> dict[str, Any]:
        item = self.store.item(rid, pid)
        out = {**item, "brief": public(PROMPTS_BY_ID[pid], full=True) if pid in PROMPTS_BY_ID else {"id": pid}}
        out["files"] = {n: (Path(item["folder"]) / n).is_file() for n in ("index.html", "sheet.png", "key1.png", "key2.png")} if item["folder"] else {}
        if html and item["folder"] and (Path(item["folder"]) / "index.html").is_file():
            out["html"] = (Path(item["folder"]) / "index.html").read_text(encoding="utf-8")
        return out

    def file_path(self, rid: str, pid: str, name: str) -> Path:
        """A file of an item the screen may show. Only the names this bench writes are served."""
        if not FILES.match(name):
            raise GaltonError("not_found", "vb_no_file", name=name)
        item = self.store.item(rid, pid)
        path = (Path(item["folder"]) / name) if item["folder"] else None
        if path is None or not path.is_file():
            raise GaltonError("not_found", "vb_no_file", name=name)
        return path

    def delete(self, rid: str) -> dict[str, Any]:
        import shutil
        run = self.store.run(rid)
        if run["state"] in ACTIVE:
            raise GaltonError("invalid", "vb_run_active", id=rid)
        shutil.rmtree(self.root / run["id"], ignore_errors=True)
        self.store.delete_run(run["id"])
        return {"deleted": run["id"]}
