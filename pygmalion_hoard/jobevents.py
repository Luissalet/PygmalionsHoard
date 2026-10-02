"""Canonical job events for the family hub: ``pygmalion.job.queued|started|progress|done|failed|cancelled``.

Data of every event: ``{job_id, title, kind, progress (0..1), gpu, eta_s, url, error}``. ``kind`` is ``train``, ``merge`` (both
LoRA merges and model merges), ``convert``, ``quantize`` or ``publish``; the other kinds of this app keep their own name (``download``,
``dataset_build``, ``imatrix``, ``perplexity``, ``ctx_extend``, ``evaluate``). ``job_kind`` carries the app's own kind and ``pipeline``
the pipeline id when the job belongs to one. A ``publish`` job that finished also carries ``model``: the name the model was published
under (the Ollama tag; the backend id when only a llama.cpp backend was written), so a rule of the hub can ask Galton to measure it.

The old ``pygmalion.job_queued`` / ``pygmalion.job_done`` are no longer emitted; the hub maps them onto these. Progress events are
throttled (``min_interval_s``). Events are hints: a failing ``emit`` never reaches the job.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

CANONICAL_KIND = {"merge_lora": "merge", "merge_models": "merge"}


def canonical_kind(kind: str) -> str:
    return CANONICAL_KIND.get(kind, kind)


def published_model(result: dict[str, Any]) -> str:
    """The name a finished publish job put the model under: the Ollama tag, else the llama.cpp backend id."""
    if not isinstance(result, dict):
        return ""
    ollama, llama = result.get("ollama") or {}, result.get("llama") or {}
    return str(ollama.get("tag") or llama.get("id") or "")


class JobEvents:
    def __init__(self, emit: Callable[[str, dict[str, Any]], None], *, clock: Callable[[], float] = time.time,
                 base_url: Callable[[], str] = lambda: "", min_interval_s: float = 5.0):
        self._emit, self._clock, self._base_url, self.min_interval_s = emit, clock, base_url, float(min_interval_s)
        self._lock = threading.Lock()
        self._last_progress: dict[str, float] = {}

    def url(self, job_id: str) -> str:
        base = (self._base_url() or "").rstrip("/")
        return f"{base}/#/trabajos/{job_id}" if base else ""

    def _data(self, job: dict[str, Any], **extra: Any) -> dict[str, Any]:
        data: dict[str, Any] = {"job_id": job["id"], "title": str(job.get("title") or job["kind"])[:120], "kind": canonical_kind(job["kind"]),
                                "job_kind": job["kind"], "url": self.url(job["id"])}
        if job.get("pipeline_id"):
            data["pipeline"] = job["pipeline_id"]
        gpus = job.get("gpus") or []
        if gpus:
            data["gpu"] = int(gpus[0])
        data.update({k: v for k, v in extra.items() if v is not None})
        return data

    def _send(self, name: str, data: dict[str, Any]) -> None:
        try:
            self._emit(f"pygmalion.job.{name}", data)
        except Exception:  # noqa: BLE001 — events are hints; the database is the truth
            pass

    def queued(self, job: dict[str, Any]) -> None:
        self._send("queued", self._data(job, progress=0.0))

    def started(self, job: dict[str, Any]) -> None:
        with self._lock:
            self._last_progress[job["id"]] = self._clock()
        self._send("started", self._data(job, progress=0.0))

    def progress(self, job: dict[str, Any], progress: dict[str, Any]) -> None:
        """Called on every progress update of a running job; sends at most one event per ``min_interval_s``."""
        pct = progress.get("pct")
        if not isinstance(pct, (int, float)):
            return
        now = self._clock()
        with self._lock:
            if now - self._last_progress.get(job["id"], 0.0) < self.min_interval_s:
                return
            self._last_progress[job["id"]] = now
        eta = progress.get("eta_s")
        self._send("progress", self._data(job, progress=round(max(0.0, min(1.0, float(pct) / 100.0)), 3),
                                          eta_s=int(eta) if isinstance(eta, (int, float)) else None))

    def finished(self, job: dict[str, Any], state: str, error: str = "", result: Optional[dict[str, Any]] = None) -> None:
        """``state`` is the job's final state; ``interrupted`` is reported as ``failed`` (the app stopped under it)."""
        name = {"done": "done", "failed": "failed", "cancelled": "cancelled", "interrupted": "failed"}.get(state)
        if name is None:
            return
        extra: dict[str, Any] = {}
        if name == "done":
            extra["progress"] = 1.0
            if job["kind"] == "publish":
                model = published_model(result or {})
                if model:
                    extra["model"] = model
        if name == "failed":
            extra["error"] = (error or "interrupted")[:300]
        with self._lock:
            self._last_progress.pop(job["id"], None)
        self._send(name, self._data(job, **extra))
