"""Jobs: two lanes (``gpu`` and ``cpu``) with one job at a time each, pipelines that chain a job's successors, progress that survives a
restart, and cancel, resume and delete.

A job moves through ``queued`` -> (``waiting_gpu``) -> ``running`` -> ``done`` | ``failed`` | ``cancelled``; what was running when
the app stopped becomes ``interrupted`` and can be resumed (a training run continues from its last checkpoint).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from . import procs
from .errors import PygmalionError
from .gpus import GpuManager, Grant
from .logclean import clean_lines
from .jobevents import JobEvents
from .lineage import Lineage, is_quantized
from .messages import ERRORS, error_text, text
from .settings import Settings
from .store import ACTIVE_STATES, Store
from .util import tail_lines
from .workdir import Work

log = logging.getLogger("pygmalion.jobs")

LANE_OF = {"download": "cpu", "dataset_build": "cpu", "train": "gpu", "merge_lora": "cpu", "merge_models": "cpu", "convert": "cpu",
           "imatrix": "gpu", "quantize": "cpu", "perplexity": "gpu", "ctx_extend": "cpu", "publish": "cpu", "evaluate": "gpu"}
KINDS = tuple(LANE_OF)
LANES = ("gpu", "cpu")
RESUMABLE = ("interrupted", "failed", "cancelled")
TERMINAL = ("done", "failed", "cancelled", "interrupted")
PERSIST_EVERY_S = 0.5
MEASURED_AFTER_STEPS = 1        # a step is enough for the measured speed to replace the estimate (the first step includes warm-up, so the ETA settles)
REF_RE = re.compile(r"^\$[a-z_]+(?::[A-Za-z0-9_]+)?$")


class JobCancelled(Exception):
    """Raised inside a runner when the job's cancel flag is set."""


@dataclass
class Outcome:
    """What a worker reported on its output."""

    result: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    hint: str = ""
    key: str = ""                  # the catalogue key of the error, when the worker named one (messages.ERRORS)
    params: dict[str, Any] = field(default_factory=dict)
    returncode: Optional[int] = None


class JobContext:
    """What a runner may use: the job, its folder and log, progress, the GPU lease and ways to run the trainer's worker or a tool."""

    def __init__(self, manager: "JobManager", job: dict[str, Any], cancel: threading.Event):
        self.manager = manager
        self.job = job
        self.cancel = cancel
        self.deps = manager.deps
        self.dir = Path(job["job_dir"])
        self.log_path = self.dir / "job.log"
        self.progress_path = self.dir / "progress.jsonl"
        self._last_persist = 0.0
        self._progress: dict[str, Any] = dict(job.get("progress") or {})
        self._output: Optional[str] = None

    # -- bookkeeping ---------------------------------------------------------------------------------------------
    @property
    def id(self) -> str:
        return self.job["id"]

    @property
    def params(self) -> dict[str, Any]:
        return self.job["params"]

    @property
    def resuming(self) -> bool:
        return bool(self.job.get("attempts", 0) > 1 or self.params.get("resume"))

    def check_cancel(self) -> None:
        if self.cancel.is_set():
            raise JobCancelled()

    def log(self, message: str) -> None:
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(message + "\n")
        except OSError:
            pass

    def progress(self, force: bool = False, **fields: Any) -> None:
        """Merge fields into the job's progress; persisted at most every half second (always for ``force``)."""
        clean = {k: v for k, v in fields.items() if v is not None}
        self._progress.update(clean)
        step, total = self._progress.get("step"), self._progress.get("total")
        if isinstance(step, (int, float)) and isinstance(total, (int, float)) and total:
            self._progress["pct"] = round(min(100.0, 100.0 * step / total), 1)
        self._progress["updated_ts"] = self.manager.clock()
        if clean.get("loss") is not None or clean.get("eval_loss") is not None:
            try:
                with open(self.progress_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"t": round(self._progress["updated_ts"], 1), **{k: clean[k] for k in ("step", "loss", "lr", "eval_loss", "tokens_per_s", "gpu_mem_mb") if k in clean}}) + "\n")
            except OSError:
                pass
        now = time.monotonic()
        if force or now - self._last_persist >= PERSIST_EVERY_S:
            self._last_persist = now
            self.manager.store.update_job(self.id, progress=self._progress)
        self.manager.events.progress(self.job, self._progress)

    def message(self, text: str, **fields: Any) -> None:
        self.progress(force=True, message=text, **fields)

    def set_output(self, artifact_id: str) -> None:
        self._output = artifact_id

    @property
    def output(self) -> Optional[str]:
        return self._output

    def reserve_output_id(self) -> str:
        """An artifact id chosen on the first attempt and kept for every retry, so a resumed run writes in the same folder."""
        existing = self.params.get("_out_id")
        if existing:
            return existing
        from .util import new_id
        value = new_id("a", self.manager.clock())
        params = dict(self.params)
        params["_out_id"] = value
        self.manager.store.update_job(self.id, params=params)
        self.job["params"] = params
        return value

    def set_vram(self, estimate: dict[str, Any]) -> None:
        self.manager.store.update_job(self.id, vram=estimate)

    # -- the GPU --------------------------------------------------------------------------------------------------
    def acquire_gpu(self, total_mb: int, purpose: str) -> Grant:
        store = self.manager.store
        store.update_job(self.id, state="waiting_gpu")

        def waiting(position: Optional[int]) -> None:
            store.update_job(self.id, queue_position=position)

        grant = self.deps.gpus.acquire(int(total_mb), purpose, cancel=self.cancel, on_wait=waiting)
        store.update_job(self.id, state="running", queue_position=None, gpus=grant.gpus)
        self.job["gpus"] = list(grant.gpus)                 # the progress events carry the GPU the job now holds
        self.log(f"GPU {grant.gpus} granted ({grant.via}).")
        return grant

    # -- programs --------------------------------------------------------------------------------------------------
    def secrets(self) -> list[str]:
        token = self.deps.config.secret("HF_TOKEN")
        return [token] if token else []

    def run_tool(self, argv: list[str], *, on_line: Optional[Callable[[str], None]] = None, gpus: Optional[list[int]] = None,
                 env: Optional[dict[str, str]] = None, cwd: Optional[str | Path] = None, timeout_s: Optional[float] = None) -> procs.ProcResult:
        """Run an external program under this job: logged, cancellable, killed as a tree. A cancel raises ``JobCancelled``."""
        result = procs.run_streaming(argv, cwd=cwd or self.dir, env=procs.build_env(env, gpus if gpus is not None else []), log_path=self.log_path,
                                     on_line=on_line, cancel=self.cancel, timeout_s=timeout_s, grace_s=self.deps.settings.int("jobs.stop_grace_s"),
                                     secrets=self.secrets())
        if result.cancelled:
            raise JobCancelled()
        return result

    def run_worker(self, name: str, args: dict[str, Any], *, gpus: Optional[list[int]] = None, env: Optional[dict[str, str]] = None,
                   label: str = "args") -> Outcome:
        """Run ``workers/<name>.py`` in the trainer environment and follow its JSON-lines protocol."""
        work = self.deps.work
        python, script = work.python(), work.worker(name)
        if not python or not Path(python).is_file():
            raise PygmalionError("env_missing", "env_not_set")
        if not script.is_file():
            raise PygmalionError("env_missing", "worker_missing", name=name, script=script)
        args_path = self.dir / f"{label}.json"
        args_path.write_text(json.dumps(args, indent=2, ensure_ascii=False), encoding="utf-8")
        outcome = Outcome()

        def on_line(line: str) -> None:
            line = line.strip()
            if not line.startswith("{"):
                return
            try:
                event = json.loads(line)
            except ValueError:
                return
            if not isinstance(event, dict):
                return
            kind = event.get("event")
            if kind == "progress":
                self.progress(**{k: event.get(k) for k in ("step", "total", "loss", "lr", "tokens_per_s", "avg_tokens_per_s", "eta_s", "gpu_mem_mb", "epoch", "mb_per_s")})
            elif kind == "eval":
                self.progress(force=True, eval_loss=event.get("eval_loss"), eval_step=event.get("step"))
                self._append_eval(event)
            elif kind == "log":
                self.log(f"[{name}] {event.get('msg', '')}")
            elif kind == "artifact":
                outcome.artifacts.append({"path": event.get("path"), "kind": event.get("kind")})
            elif kind == "result":
                outcome.result = event.get("data") or {}
            elif kind == "error":
                outcome.error, outcome.hint = str(event.get("msg") or ""), str(event.get("hint") or "")
                outcome.key = str(event.get("key") or "")
                outcome.params = event.get("params") if isinstance(event.get("params"), dict) else {}

        extra = {"PYTHONUTF8": "1", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1", **(env or {})}
        if name != "download":                     # everything but the download works on local files only
            extra.setdefault("HF_HUB_OFFLINE", "1")
            extra.setdefault("TRANSFORMERS_OFFLINE", "1")
        result = procs.run_streaming([python, str(script), "--args", str(args_path)], cwd=self.dir, env=procs.build_env(extra, gpus if gpus is not None else []),
                                     log_path=self.log_path, on_line=on_line, cancel=self.cancel, grace_s=self.deps.settings.int("jobs.stop_grace_s"),
                                     secrets=self.secrets(), stop_file=self.dir / "STOP")
        outcome.returncode = result.returncode
        if result.cancelled:
            raise JobCancelled()
        if not result.ok:
            if outcome.key in ERRORS:             # the worker named an entry of the catalogue: the app words it (and the UI translates it)
                named = PygmalionError("failed", outcome.key, **outcome.params)
                if outcome.hint:                  # a more specific hint than the catalogue's (a library's error text told which)
                    named.hint = outcome.hint
                raise named
            if outcome.error:
                raise PygmalionError("failed", outcome.error, outcome.hint)
            tail = " | ".join(x for x in result.tail[-3:] if x.strip())[:300]
            raise PygmalionError("failed", "worker_exit", name=name, code=result.returncode, tail=tail)
        return outcome

    def _append_eval(self, event: dict[str, Any]) -> None:
        try:
            with open(self.progress_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": round(self.manager.clock(), 1), "step": event.get("step"), "eval_loss": event.get("eval_loss")}) + "\n")
        except OSError:
            pass


@dataclass
class Deps:
    """Everything the runners need from the rest of the app."""

    store: Store
    work: Work
    settings: Settings
    gpus: GpuManager
    lineage: Lineage
    datasets: Any
    publisher: Any
    evaluator: Any
    hub: Any
    config: Any


Runner = Callable[[JobContext], dict[str, Any]]


def resolve_refs(value: Any, outputs: dict[str, str]) -> Any:
    """Replace ``"$name"`` strings in a step's parameters with the artifact ids the pipeline has produced so far."""
    if isinstance(value, str) and REF_RE.match(value):
        name = value[1:]
        if name not in outputs:
            if outputs:
                raise PygmalionError("invalid", "pipeline_ref_missing", name=name, options=["$" + k for k in sorted(outputs)])
            raise PygmalionError("invalid", "pipeline_ref_none", name=name)
        return outputs[name]
    if isinstance(value, dict):
        return {k: resolve_refs(v, outputs) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_refs(v, outputs) for v in value]
    return value


class JobManager:
    def __init__(self, deps: Deps, runners: dict[str, Runner], *, clock: Callable[[], float] = time.time, enabled: bool = True,
                 paused: Callable[[], bool] = lambda: False, emit: Callable[[str, dict[str, Any]], None] = lambda t, d: None,
                 base_url: Callable[[], str] = lambda: ""):
        self.deps = deps
        self.store = deps.store
        self.work = deps.work
        self.runners = runners
        self.clock = clock
        self.enabled = enabled
        self.paused = paused
        self.emit = emit
        self.events = JobEvents(lambda t, d: self.emit(t, d), clock=clock, base_url=base_url)   # looks self.emit up at send time
        self._stop = threading.Event()
        self._threads: dict[str, threading.Thread] = {}
        self._wake = {lane: threading.Event() for lane in LANES}
        self._cancels: dict[str, threading.Event] = {}
        self._running: dict[str, Optional[str]] = {lane: None for lane in LANES}
        self._lock = threading.RLock()
        self.finished = 0
        self._done = threading.Condition()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> int:
        """Mark what was running as interrupted, then start the lane threads (unless the scheduler is disabled)."""
        stale = [j for j in self.store.jobs(states=["running", "waiting_gpu"], limit=2000)]
        interrupted = self.store.mark_interrupted()
        for old in stale:
            self.events.finished(old, "interrupted", "The app stopped while this job was running.")
        if interrupted:
            log.info("%d job(s) were running when the app stopped: marked interrupted", interrupted)
        if self.enabled and not self.alive():
            self._stop.clear()
            for lane in LANES:
                thread = threading.Thread(target=self._lane_loop, args=(lane,), name=f"pygmalion-lane-{lane}", daemon=True)
                self._threads[lane] = thread
                thread.start()
        return interrupted

    def alive(self) -> bool:
        return any(t.is_alive() for t in self._threads.values())

    def stop(self, timeout: float = 20.0) -> list[str]:
        """Cancel what runs, wake the lanes and join them within one shared ``timeout``. Returns the names of the lane threads that were still
        running afterwards (a job that would not stop). Safe to call twice; ``start`` can follow."""
        self._stop.set()
        with self._lock:
            for event in self._cancels.values():
                event.set()
        for event in self._wake.values():
            event.set()
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in list(self._threads.values()):
            if thread is not threading.current_thread():
                thread.join(max(0.0, deadline - time.monotonic()))
        late = [t.name for t in self._threads.values() if t.is_alive()]
        if late:
            log.warning("lane thread(s) still running after %.1f s: %s", timeout, ", ".join(late))
        else:
            self._threads.clear()
        return late

    # ------------------------------------------------------------------ submitting
    def lane_for(self, kind: str, params: Optional[dict[str, Any]] = None) -> str:
        if kind not in LANE_OF:
            raise PygmalionError("invalid", "job_kind_unknown", kind=kind, options=list(KINDS))
        if kind == "merge_lora" and (params or {}).get("device") == "cuda":
            return "gpu"
        return LANE_OF[kind]

    def submit(self, kind: str, params: dict[str, Any], *, title: str = "", then: Optional[list[dict[str, Any]]] = None,
               pipeline_id: Optional[str] = None, step: int = 0) -> dict[str, Any]:
        lane = self.lane_for(kind, params)
        for nxt in then or []:
            if nxt.get("kind") not in LANE_OF:
                raise PygmalionError("invalid", "job_kind_unknown_step", kind=nxt.get("kind"), options=list(KINDS))
        job = self.store.create_job(kind, lane, params, title=title or kind, then=then, pipeline_id=pipeline_id, step=step)
        folder = self.work.job_dir(job["id"])
        folder.mkdir(parents=True, exist_ok=True)
        job = self.store.update_job(job["id"], job_dir=str(folder))
        self._wake[lane].set()
        self.events.queued(job)
        return job

    def pipeline(self, steps: list[dict[str, Any]], title: str = "") -> dict[str, Any]:
        """Queue the first step; each later step is created when the one before it finishes."""
        if not steps:
            raise PygmalionError("invalid", "pipeline_empty")
        first = steps[0]
        return self.submit(first["kind"], first.get("params") or {}, title=first.get("title") or title or first["kind"], then=steps[1:])

    # ------------------------------------------------------------------ control
    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.store.job(job_id)
        if job["state"] == "queued":
            cancelled = self.store.update_job(job_id, state="cancelled", finished_ts=self.clock(), error=str(text("job_cancelled_early")))
            self.events.finished(cancelled, "cancelled")
            return cancelled
        if job["state"] in ("running", "waiting_gpu"):
            self.store.update_job(job_id, cancel_requested=True)
            with self._lock:
                event = self._cancels.get(job_id)
            if event is not None:
                event.set()
            else:   # state says running but nothing runs it here (stale row): close it
                closed = self.store.update_job(job_id, state="cancelled", finished_ts=self.clock(), error=str(error_text("cancelled")))
                self.events.finished(closed, "cancelled")
                return closed
            return self.store.job(job_id)
        raise PygmalionError("conflict", "job_not_cancellable", id=job_id, state=job["state"])

    def resume(self, job_id: str) -> dict[str, Any]:
        job = self.store.job(job_id)
        if job["state"] not in RESUMABLE:
            raise PygmalionError("conflict", "job_not_resumable", id=job_id, state=job["state"])
        job = self.store.update_job(job_id, state="queued", error="", hint="", finished_ts=None, cancel_requested=False, queue_position=None)
        self._wake[job["lane"]].set()
        return job

    def delete(self, job_id: str) -> dict[str, Any]:
        job = self.store.job(job_id)
        if job["state"] in ACTIVE_STATES:
            raise PygmalionError("conflict", "job_still_active", id=job_id, state=job["state"])
        self.store.delete_job(job_id)
        if job.get("job_dir"):
            shutil.rmtree(job["job_dir"], ignore_errors=True)
        return {"deleted": job_id, "artifacts_kept": True}

    # ------------------------------------------------------------------ views
    def status(self) -> dict[str, Any]:
        self.reap_stale()
        counts = {lane: len(self.store.jobs(states=["queued"], lane=lane, limit=2000)) for lane in LANES}
        return {"enabled": self.enabled, "running": self.alive(), "paused": bool(self.paused()),
                "lanes": {lane: {"queued": counts[lane], "current": self._current(lane)} for lane in LANES}, "finished": self.finished}

    def _current(self, lane: str) -> Optional[str]:
        """The job a lane is running, as the database sees it: a lane whose job already ended (or was removed) is idle."""
        with self._lock:
            job_id = self._running[lane]
        if not job_id:
            return None
        try:
            return job_id if self.store.job(job_id)["state"] in ACTIVE_STATES else None
        except PygmalionError:
            return None

    def curve(self, job_id: str, max_points: int = 400) -> dict[str, Any]:
        job = self.store.job(job_id)
        train: list[list[float]] = []
        evals: list[list[float]] = []
        path = Path(job["job_dir"]) / "progress.jsonl" if job.get("job_dir") else None
        if path and path.is_file():
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get("loss") is not None and row.get("step") is not None:
                        train.append([row["step"], row["loss"]])
                    if row.get("eval_loss") is not None and row.get("step") is not None:
                        evals.append([row["step"], row["eval_loss"]])
        total = len(train)
        if total > max_points:
            stride = total / max_points
            train = [train[int(i * stride)] for i in range(max_points)] + [train[-1]]
        last = train[-1][1] if train else None
        return {"train": train, "eval": evals, "points": total, "last_loss": last, "min_loss": min((p[1] for p in train), default=None)}

    def log_tail(self, job_id: str, lines: int = 60) -> list[str]:
        job = self.store.job(job_id)
        if not job.get("job_dir"):
            return []
        # logs written by an older version still hold escape sequences and one line per spinner frame: clean them on the way out
        return clean_lines(tail_lines(Path(job["job_dir"]) / "job.log", lines * 4, max_bytes=256_000))[-lines:]

    def timing(self, job: dict[str, Any]) -> Optional[dict[str, Any]]:
        """How long a job takes: the planner's estimate until the job has measured its own speed, then the measured tokens per second and the
        worker's real ETA. ``source`` says which one the numbers are (``estimate``, ``measured`` or ``done``)."""
        estimate = job["params"].get("_estimate") or {}
        progress = job["progress"] or {}
        measured = progress.get("avg_tokens_per_s") or progress.get("tokens_per_s")
        if job["state"] == "running" and measured and progress.get("eta_s") is not None and (progress.get("step") or 0) >= MEASURED_AFTER_STEPS:
            return {"source": "measured", "tokens_per_s": int(measured), "eta_s": int(progress["eta_s"]), "estimated_s": estimate.get("seconds"),
                    "estimated_tokens_per_s": estimate.get("tokens_per_s")}
        if job["state"] in ACTIVE_STATES and estimate:
            return {"source": "estimate", "tokens_per_s": estimate.get("tokens_per_s"), "eta_s": estimate.get("seconds"),
                    "estimated_s": estimate.get("seconds"), "estimated_tokens_per_s": estimate.get("tokens_per_s")}
        if job["state"] == "done" and estimate and job.get("started_ts") and job.get("finished_ts"):
            return {"source": "done", "elapsed_s": int(job["finished_ts"] - job["started_ts"]), "estimated_s": estimate.get("seconds"),
                    "estimated_tokens_per_s": estimate.get("tokens_per_s"), "tokens_per_s": progress.get("avg_tokens_per_s")}
        return None

    def view(self, job: dict[str, Any], detail: bool = False) -> dict[str, Any]:
        out = {k: job[k] for k in ("id", "kind", "title", "state", "lane", "step", "pipeline_id", "created_ts", "started_ts", "finished_ts", "error", "hint",
                                   "queue_position", "attempts", "out_artifact")}
        out["progress"] = job["progress"]
        out["gpus"] = job["gpus"]
        out["next"] = [s.get("kind") for s in job["then"]]
        out["resumable"] = job["state"] in RESUMABLE
        out["pct"] = (job["progress"] or {}).get("pct")
        out["timing"] = self.timing(job)
        if detail:
            out.update(params={k: v for k, v in job["params"].items() if not k.startswith("_")}, result=job["result"], vram=job["vram"],
                       curve=self.curve(job["id"], 200), log_tail=self.log_tail(job["id"], 40),
                       pipeline=self.pipeline_steps(job))
        return out

    def pipeline_steps(self, job: dict[str, Any]) -> list[dict[str, Any]]:
        """Every step of the job's pipeline in order: the jobs that exist with their states, then the steps the last one still has to start
        (``planned``, no id yet). A step that is created later changes from planned to queued under the same position."""
        made = sorted(self.store.jobs(pipeline_id=job["pipeline_id"], limit=200), key=lambda x: x["step"])
        steps = [{"id": j["id"], "kind": j["kind"], "title": j["title"], "state": j["state"], "step": j["step"], "error": bool(j["error"])} for j in made]
        if made:
            last = made[-1]
            for offset, nxt in enumerate(last["then"], start=1):
                steps.append({"id": None, "kind": nxt.get("kind"), "title": nxt.get("title") or nxt.get("kind"), "state": "planned", "step": last["step"] + offset,
                              "error": False})
        return steps

    # ------------------------------------------------------------------ running
    def wait(self, job_id: str, timeout: float = 30.0) -> dict[str, Any]:
        """Block until the job is no longer active (or the timeout passes) and return it. Without lane threads it runs the queue inline."""
        deadline = time.monotonic() + timeout
        if not self.alive():
            self.run_until_idle()
            return self.store.job(job_id)
        with self._done:
            while time.monotonic() < deadline:
                job = self.store.job(job_id)
                if job["state"] not in ACTIVE_STATES:
                    return job
                self._done.wait(0.25)
        return self.store.job(job_id)

    def run_until_idle(self, limit: int = 200) -> int:
        """Run queued jobs one after another in the calling thread (tests and MCP-only mode); returns how many ran."""
        ran = 0
        while ran < limit and not self._stop.is_set():
            job = None
            for lane in LANES:
                job = self.store.next_queued(lane)
                if job:
                    break
            if job is None:
                break
            self._execute(job)
            ran += 1
        return ran

    def _lane_loop(self, lane: str) -> None:
        while not self._stop.is_set():
            try:
                self.reap_stale()
                if self.paused():
                    self._wake[lane].wait(1.0)
                    self._wake[lane].clear()
                    continue
                job = self.store.next_queued(lane)
                if job is None:
                    self._wake[lane].wait(1.0)
                    self._wake[lane].clear()
                    continue
                self._execute(job)
            except Exception:  # noqa: BLE001 — one bad iteration (a locked database, a full disk) must not end the lane for good
                log.exception("lane %s: iteration failed", lane)
                self._stop.wait(1.0)

    def _execute(self, job: dict[str, Any]) -> None:
        cancel = threading.Event()
        # the lane read the job a moment ago; a cancel may have landed since, and must not be overwritten by "running". Claiming and
        # registering are one step under the lock, so `reap_stale` never sees a running row that nobody has registered yet
        with self._lock:
            if self._stop.is_set():       # `stop` sets the flag before it cancels what is registered: a job claimed after it would never be cancelled
                return
            if not self.store.claim_job(job["id"]):
                return
            self._cancels[job["id"]] = cancel
            self._running[job["lane"]] = job["id"]
        try:
            self._run_claimed(job, cancel)
        except Exception as exc:  # noqa: BLE001 — the row must not stay "running" because bookkeeping failed
            log.exception("job %s (%s): bookkeeping failed", job["id"], job["kind"])
            self._force_close(job["id"], f"{type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._cancels.pop(job["id"], None)
                if self._running.get(job["lane"]) == job["id"]:
                    self._running[job["lane"]] = None
            with self._done:
                self._done.notify_all()

    def _force_close(self, job_id: str, detail: str) -> None:
        try:
            current = self.store.job(job_id)
            if current["state"] in ACTIVE_STATES:
                crash = PygmalionError("failed", "job_crashed", detail=detail[:500])
                self.store.update_job(job_id, state="failed", finished_ts=self.clock(), error=crash.message, hint=crash.hint, queue_position=None)
        except Exception:  # noqa: BLE001
            log.exception("job %s could not be closed", job_id)

    def _run_claimed(self, job: dict[str, Any], cancel: threading.Event) -> None:
        runner = self.runners.get(job["kind"])
        started = self.clock()
        job = self.store.update_job(job["id"], state="running", started_ts=started, finished_ts=None, attempts=job["attempts"] + 1,
                                    error="", hint="", cancel_requested=False)
        if not job["job_dir"]:
            job = self.store.update_job(job["id"], job_dir=str(self.work.job_dir(job["id"])))
        Path(job["job_dir"]).mkdir(parents=True, exist_ok=True)
        self.events.started(job)
        ctx = JobContext(self, job, cancel)
        state, error, hint, result = "done", "", "", {}
        try:
            if runner is None:
                raise PygmalionError("invalid", "job_no_runner", kind=job["kind"])
            ctx.log(f"== {job['kind']} started (attempt {job['attempts']})")
            result = runner(ctx) or {}
            if cancel.is_set():
                raise JobCancelled()
        except JobCancelled:
            state, error = "cancelled", str(error_text("cancelled"))
        except PygmalionError as exc:
            state, error, hint = "failed", exc.message, exc.hint
            if cancel.is_set():
                state, error, hint = "cancelled", str(error_text("cancelled")), ""
        except Exception as exc:  # noqa: BLE001 — a bug in one runner must not stop the lane
            log.exception("job %s (%s) crashed", job["id"], job["kind"])
            crash = PygmalionError("failed", "job_crashed", detail=f"{type(exc).__name__}: {exc}"[:500])
            state, error, hint = "failed", crash.message, crash.hint
        ctx.log(f"== {state}" + (f": {error}" if error else ""))
        progress = dict(ctx._progress)
        if state == "done":
            progress["pct"] = 100.0
        job = self.store.update_job(job["id"], state=state, finished_ts=self.clock(), error=error, hint=hint, result=result, progress=progress,
                                    out_artifact=ctx.output or job.get("out_artifact"), queue_position=None)
        self.store.add_run(job["kind"], job["id"], state == "done", int((self.clock() - started) * 1000), error or "ok")
        self.finished += 1
        self.events.finished(job, state, error, result)
        if state == "done" and job["then"]:
            try:
                self._spawn_next(job)
            except PygmalionError as exc:
                self.store.update_job(job["id"], error=str(text("next_step_failed", reason=exc.message)), hint=exc.hint)
                log.warning("pipeline %s stopped: %s", job["pipeline_id"], exc.message)

    def reap_stale(self) -> int:
        """Close the rows that say a job runs while nothing in this process runs it (a lane thread that died, a write that failed): they stay
        "running" for ever otherwise, and the sidebar and the badge say so. They become ``interrupted`` and can be resumed."""
        if not self.alive():
            return 0
        reaped = 0
        with self._lock:
            for job in self.store.jobs(states=["running", "waiting_gpu"], limit=50):
                if job["id"] in self._cancels:
                    continue
                self.store.update_job(job["id"], state="interrupted", finished_ts=self.clock(), error=str(text("job_lost")), queue_position=None)
                reaped += 1
        return reaped

    # ------------------------------------------------------------------ pipelines
    def pipeline_outputs(self, pipeline_id: str) -> dict[str, str]:
        """Names a later step may refer to (``$adapter``, ``$gguf``, ``$quant:Q4_K_M``, ``$prev``...) and the artifacts they stand for."""
        outputs: dict[str, str] = {}
        previous = None
        for job in sorted(self.store.jobs(pipeline_id=pipeline_id, limit=200), key=lambda j: j["step"]):
            if job["state"] != "done" or not job.get("out_artifact"):
                continue
            try:
                art = self.store.artifact(job["out_artifact"])
            except PygmalionError:
                continue
            previous = art["id"]
            if art["kind"] != "gguf":                  # a GGUF is named below: $gguf is the unquantized one, $quant the quantized one
                outputs[art["kind"]] = art["id"]
            if job["params"].get("_as"):
                outputs[str(job["params"]["_as"])] = art["id"]
            if art["kind"] == "gguf":
                if ((art.get("recipe") or {}).get("params") or {}).get("adapter"):
                    outputs["adapter_gguf"] = art["id"]
                elif is_quantized(art):
                    qtype = ((art.get("recipe") or {}).get("params") or {}).get("qtype") or (art.get("metrics") or {}).get("quant")
                    outputs[f"quant:{qtype}"] = art["id"]
                    outputs["quant"] = art["id"]
                else:
                    outputs["gguf"] = art["id"]
        if previous:
            outputs["prev"] = previous
        return outputs

    def _spawn_next(self, job: dict[str, Any]) -> dict[str, Any]:
        step = job["then"][0]
        outputs = self.pipeline_outputs(job["pipeline_id"])
        params = resolve_refs(step.get("params") or {}, outputs)
        return self.submit(step["kind"], params, title=step.get("title") or step["kind"], then=job["then"][1:], pipeline_id=job["pipeline_id"],
                           step=job["step"] + 1)
