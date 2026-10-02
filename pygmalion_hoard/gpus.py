"""GPU inventory and leases. The apps use only the GPUs in the ``gpus.allowed`` setting (default 2 and 3; 0 and 1 belong to the
owner of the computer) and always through a lease from the family hub, so two apps never load onto the same free gigabytes."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from . import procs
from . import vram as V
from .errors import PygmalionError
from .hoard_link import _hubclient, lease as _lease_factory
from .messages import text
from .procs import run_capture
from .settings import Settings

log = logging.getLogger("pygmalion.gpus")
OWNER = "pygmalion"


def _nvidia_smi() -> str:
    found = procs.which("nvidia-smi")
    if found:
        return found
    if sys.platform == "win32":
        legacy = Path(__import__("os").environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
        if legacy.is_file():
            return str(legacy)
    return "nvidia-smi"


def query_gpus(timeout_s: float = 4.0) -> list[dict[str, Any]]:
    """Per GPU: index, name, total_mb, used_mb, free_mb. Empty when there is no NVIDIA driver."""
    code, out, _err = run_capture([_nvidia_smi(), "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                                  timeout_s=timeout_s)
    if code != 0:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            total, used = int(parts[-2]), int(parts[-1])
            gpus.append({"index": int(parts[0]), "name": ",".join(parts[1:-2]).strip(), "total_mb": total, "used_mb": used,
                         "free_mb": total - used})
        except ValueError:
            continue
    return gpus


@dataclass
class Grant:
    """GPUs held for a job: the indices to expose with CUDA_VISIBLE_DEVICES and the leases to release."""

    gpus: list[int]
    leases: list[Any] = field(default_factory=list)
    via: str = "none"
    warning: str = ""

    def release(self) -> None:
        for held in self.leases:
            try:
                held.release()
            except Exception:  # noqa: BLE001 — releasing must never mask the job's own outcome
                log.warning("could not release a GPU lease", exc_info=True)
        self.leases = []


class GpuManager:
    def __init__(self, settings: Settings, *, probe: Callable[[], list[dict[str, Any]]] = query_gpus,
                 lease_factory: Callable[..., Any] = _lease_factory, hub_status: Optional[Callable[[], Optional[dict[str, Any]]]] = None,
                 offline: bool = False):
        self.settings = settings
        self.probe = probe
        self.lease_factory = lease_factory
        self._hub_status = hub_status or self._default_hub_status
        self.offline = offline
        self._helpers: dict[threading.Thread, threading.Event] = {}     # helper threads waiting for a lease, with their "abandoned" flag
        self._helpers_lock = threading.Lock()

    def allowed(self) -> list[int]:
        return self.settings.intlist("gpus.allowed")

    def reserved(self) -> list[int]:
        return self.settings.intlist("gpus.reserved")

    def inventory(self) -> list[dict[str, Any]]:
        allowed, reserved = set(self.allowed()), set(self.reserved())
        out = []
        for g in self.probe():
            out.append({**g, "allowed": g["index"] in allowed, "reserved": g["index"] in reserved})
        return out

    def allowed_inventory(self) -> list[dict[str, Any]]:
        return [g for g in self.inventory() if g["allowed"]]

    def _default_hub_status(self) -> Optional[dict[str, Any]]:
        if self.offline:
            return None
        url = _hubclient.hub_url()
        status, body = _hubclient.fetch(url + "/api/lease", timeout=1.5)
        return body if status == 200 and isinstance(body, dict) else None

    def status(self) -> dict[str, Any]:
        """What the UI strip shows: every GPU with its memory, whether it is allowed, and who holds leases."""
        inventory = self.inventory()
        hub = self._hub_status()
        leases = (hub or {}).get("leases") or []
        queue = (hub or {}).get("queue") or []
        for g in inventory:
            g["leases"] = [{"owner": x.get("owner"), "purpose": x.get("purpose"), "vram_mb": x.get("vram_mb")}
                           for x in leases if x.get("gpu") == g["index"]]
        return {"gpus": inventory, "allowed": self.allowed(), "reserved": self.reserved(), "hub": bool(hub), "queue": queue,
                "inventory": bool(inventory), "note": "" if inventory else text("gpu_status_none")}

    # ------------------------------------------------------------------ planning
    def plan(self, total_mb: int) -> dict[str, Any]:
        inventory = self.allowed_inventory()
        fits = V.fit_on_gpus(total_mb, inventory)
        pick = V.pick_gpus(total_mb, inventory)
        warn = ""
        if inventory and not any(f["fits_free"] for f in fits):
            warn = text("gpu_plan_wait" if pick.get("fits") else "gpu_plan_no_fit", total_mb=total_mb)
        elif not inventory:
            warn = text("gpu_plan_no_inventory")
        return {"fits": fits, "pick": pick, "warning": warn, "allowed": self.allowed()}

    # ------------------------------------------------------------------ leases
    def acquire(self, total_mb: int, purpose: str, *, cancel: Optional[threading.Event] = None,
                on_wait: Optional[Callable[[Optional[int]], None]] = None, priority: int = 0) -> Grant:
        """Hold ``total_mb`` on allowed GPUs. Blocks while the hub queues the request; returns early (raising) when ``cancel`` is set."""
        allowed = self.allowed()
        inventory = self.allowed_inventory()
        pick = V.pick_gpus(total_mb, inventory)
        if inventory and pick.get("fits") is False:
            raise PygmalionError("gpu_unavailable", "gpu_no_fit", reason=pick["reason"])
        gpus = pick["gpus"] or allowed[:1]
        if not gpus:
            raise PygmalionError("gpu_unavailable", "no_gpu_allowed")
        share = max(1, int(total_mb / len(gpus)))
        # try the GPUs of a single-GPU plan in order of free memory, so a busy one does not block a free one
        grant = Grant(gpus=[], via="hub")
        try:
            for index in gpus:
                if index not in allowed:
                    raise PygmalionError("forbidden", "gpu_index_not_allowed", index=index)
                held = self.lease_factory(vram_mb=share, purpose=purpose, owner=OWNER, gpu=index, priority=priority)
                self._acquire_cancellable(held, cancel, on_wait)
                if held.gpu is not None and held.gpu not in allowed:
                    held.release()
                    raise PygmalionError("forbidden", "gpu_hub_placed", gpu=held.gpu)
                grant.leases.append(held)
                grant.gpus.append(held.gpu if held.gpu is not None else index)
                grant.via = held.via or grant.via
                if getattr(held, "warning", None):
                    grant.warning = held.warning
        except BaseException:
            grant.release()
            raise
        return grant

    def close(self, timeout: float = 5.0) -> list[str]:
        """Give up every lease request that is still waiting (a lease that is granted afterwards is released at once) and join the helper
        threads within ``timeout``. Returns the names of those that did not end."""
        with self._helpers_lock:
            helpers = dict(self._helpers)
        for abandoned in helpers.values():
            abandoned.set()
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in helpers:
            thread.join(max(0.0, deadline - time.monotonic()))
        return [t.name for t in helpers if t.is_alive()]

    def _acquire_cancellable(self, held: Any, cancel: Optional[threading.Event], on_wait: Optional[Callable[[Optional[int]], None]]) -> None:
        """``held.acquire()`` in a helper thread so the caller can report the queue position and leave on cancel."""
        box: dict[str, Any] = {}
        done = threading.Event()
        abandoned = threading.Event()
        gate = threading.Lock()

        def release_once() -> None:
            with gate:
                if box.get("released"):
                    return
                box["released"] = True
            try:
                held.release()
            except Exception:  # noqa: BLE001 — nothing useful to do if the hub already dropped it
                pass

        def run() -> None:
            try:
                held.acquire()
            except BaseException as exc:  # noqa: BLE001
                box["error"] = exc
            finally:
                done.set()
                if abandoned.is_set() and "error" not in box:
                    release_once()
                with self._helpers_lock:
                    self._helpers.pop(threading.current_thread(), None)

        thread = threading.Thread(target=run, name="pygmalion-lease", daemon=True)
        with self._helpers_lock:
            self._helpers[thread] = abandoned
        thread.start()
        while not done.wait(0.5):
            if on_wait:
                position = (getattr(held, "info", None) or {}).get("position")
                on_wait(position)
            if (cancel is not None and cancel.is_set()) or abandoned.is_set():
                abandoned.set()
                if done.is_set() and "error" not in box:     # granted in the meantime: give it back
                    release_once()
                raise PygmalionError("failed", "cancelled_waiting_gpu")
        if "error" in box:
            error = box["error"]
            if isinstance(error, PygmalionError):
                raise error
            raise PygmalionError("gpu_unavailable", "gpu_lease_failed", detail=str(error)) from error
