"""Wiring: database, settings, work folder, GPU manager, datasets, lineage, publisher, evaluator and the job manager behind one object
that the API routers and the agent tools share. Also the views (dashboard, status, settings) built from them."""

from __future__ import annotations

import logging
import os
import secrets as _secrets
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from . import SERVICE, __version__
from .config import PREFIX, Config, load_dotenv
from .datasets.service import DatasetService
from .db import Database
from .envcheck import EnvChecker
from .errors import PygmalionError
from .events import EventPump
from .evaluate import Evaluator, Galton
from .reference import Reference
from .gpus import GpuManager, query_gpus
from .messages import text
from .hf import HubClient, local_model_folders, repo_of_folder, scan_local
from .jobs import Deps, JobManager
from .lineage import Lineage
from .planner import Planner
from .publish import Publisher
from .runners import Runners
from .settings import SECRET_KEYS, SPECS, Settings
from .store import Store
from .workdir import Work

log = logging.getLogger("pygmalion")

SECRET_ENV = {"hf.token": "HF_TOKEN"}
DASHBOARD_CACHE_S = 30.0


def write_token(config: Config) -> str:
    """The MCP token is persistent: created once, reused on every later start."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    try:
        existing = config.token_path.read_text(encoding="utf-8").strip()
    except OSError:
        existing = ""
    if len(existing) >= 32:
        return existing
    token = _secrets.token_hex(32)
    config.token_path.write_text(token, encoding="utf-8")
    try:
        config.token_path.chmod(0o600)
    except OSError:
        pass
    return token


def write_url(config: Config) -> None:
    try:
        config.url_path.write_text(f"http://127.0.0.1:{config.port}", encoding="utf-8")
    except OSError:
        pass


def _family_call(app: str, tool: str, arguments: Optional[dict[str, Any]] = None, timeout: float = 60.0) -> dict[str, Any]:
    from .hoard_link import family
    return family.call(app, tool, arguments, timeout=timeout)


class Services:
    def __init__(self, config: Config, *, clock_fn: Callable[[], float] = time.time, hub_transport: Optional[httpx.BaseTransport] = None,
                 galton_transport: Optional[httpx.BaseTransport] = None, family_call: Optional[Callable[..., dict]] = None,
                 teacher: Optional[Callable[[list[dict[str, str]], int], str]] = None, probe: Optional[Callable[[], list[dict[str, Any]]]] = None,
                 lease_factory: Optional[Callable[..., Any]] = None, hub_status: Optional[Callable[[], Optional[dict]]] = None,
                 launcher: Any = None, capture: Optional[Callable[..., Any]] = None, port_is_free: Optional[Callable[[int], bool]] = None):
        self.config = config
        self.clock = clock_fn
        self.started_at = time.time()
        for d in (config.data_dir, config.logs_dir, config.datasets_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.token = write_token(config)
        write_url(config)
        self.db = Database(config.db_path)
        self.store = Store(self.db, clock_fn)
        self.settings = Settings(self.db, config.data_dir)
        self.work = Work(self.settings, config.effective_workers_dir())
        self.lineage = Lineage(self.store)
        gpu_kwargs: dict[str, Any] = {"offline": config.offline}
        if lease_factory is not None:
            gpu_kwargs["lease_factory"] = lease_factory
        if hub_status is not None:
            gpu_kwargs["hub_status"] = hub_status
        self.gpus = GpuManager(self.settings, probe=probe or query_gpus, **gpu_kwargs)
        self.env = EnvChecker(self.work, **({"capture": capture} if capture else {}), clock=clock_fn)
        self.hub = HubClient(token=lambda: self.config.secret("HF_TOKEN"), transport=hub_transport, offline=config.offline, timeout_s=config.http_timeout_s)
        self._family_call = family_call if family_call is not None else (None if config.offline else _family_call)
        self._teacher_fn = teacher
        self._link: Any = None
        self.datasets = DatasetService(self.store, self.settings, config.datasets_dir, data_dir=config.data_dir, family_call=self._family_call,
                                       teacher=self._teacher, clock=clock_fn)
        self.galton = Galton(self.settings, family_call=self._family_call, transport=galton_transport, offline=config.offline, clock=clock_fn)
        self.references = Reference(self.store, self.lineage)
        self.evaluator = Evaluator(self.store, self.lineage, self.galton, self.settings, datasets=self.datasets, references=self.references)
        self.publisher = Publisher(self.store, self.lineage, self.settings, self.work, **({"launcher": launcher} if launcher else {}),
                                   **({"is_free": port_is_free} if port_is_free else {}))
        self.planner = Planner(self.store, self.lineage, self.settings, self.gpus, references=self.references)
        deps = Deps(store=self.store, work=self.work, settings=self.settings, gpus=self.gpus, lineage=self.lineage, datasets=self.datasets,
                    publisher=self.publisher, evaluator=self.evaluator, hub=self.hub, config=config)
        self.events = EventPump(self._send_event, wanted=self._events_wanted)
        self._stopped = False
        self.jobs = JobManager(deps, Runners(deps).table(), clock=clock_fn, enabled=config.scheduler,
                               paused=lambda: self.settings.bool("scheduler.paused"), emit=self.emit)
        self._bases_lock = threading.Lock()
        self._dash_cache: Optional[tuple[float, dict[str, Any], Optional[dict[str, Any]]]] = None  # (when, slow blocks, env check they were built on)
        from .ops import Operations
        self.ops = Operations(self)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.work.ensure()
        self.jobs.start()

    def stop(self, timeout: float = 20.0) -> list[str]:
        """Stop everything this object started, in order, each with a timeout: the job lanes (they cancel what runs), the GPU lease helpers, the
        event worker, the language-model link and the database. Safe to call twice. Returns the names of the threads that did not end in time
        (empty on a clean stop), so a caller or a test can say so."""
        if self._stopped:
            return []
        self._stopped = True
        late: list[str] = []
        late += self.jobs.stop(timeout)
        late += self.gpus.close(min(timeout, 5.0))
        if not self.events.close(min(timeout, 5.0)):
            late.append("pygmalion-events")
        if self._link is not None:
            try:
                self._link.sync.close()
            except Exception:  # noqa: BLE001
                pass
            self._link = None
        if late:
            log.warning("stopped with threads still running: %s", ", ".join(late))
        self.db.close()
        return late

    def _events_wanted(self) -> bool:
        from .hoard_link import family
        state = family.status()
        return bool(state.get("enabled") and state.get("app"))

    @staticmethod
    def _send_event(type_: str, data: dict[str, Any]) -> None:
        from .hoard_link import family
        family.emit(type_, data, block=True, timeout=2.0)

    def emit(self, type_: str, data: dict[str, Any]) -> None:
        """An event for the family bus: queued for the one worker thread, dropped when nobody listens or the app is stopping."""
        try:
            self.events.emit(type_, data)
        except Exception:  # noqa: BLE001 — events are hints; the database is the truth
            pass

    # ------------------------------------------------------------------ the teacher model for synthetic data
    def _teacher(self) -> Optional[Callable[[list[dict[str, str]], int], str]]:
        if self._teacher_fn is not None:
            return self._teacher_fn
        if self.config.offline:
            return None
        if self._link is None:
            from .hoard_link import Link, LinkConfig
            env = dict(os.environ)
            if self.settings.get("teacher.model"):
                env["HOARD_LLM_MODEL"] = self.settings.get("teacher.model")
            self._link = Link(LinkConfig.load(None, env=env, app="pygmalion"))
        link = self._link
        effort = self.settings.get("teacher.effort")

        def chat(messages: list[dict[str, str]], max_tokens: int) -> str:
            result = link.sync.chat(messages, max_tokens=max_tokens, temperature=0.7, effort=effort)
            return str(getattr(result, "text", "") or "")

        return chat

    def teacher_status(self) -> dict[str, Any]:
        if self._teacher_fn is not None:
            return {"ok": True, "detail": "injected"}
        if self.config.offline:
            return {"ok": False, "detail": text("detail_offline")}
        try:
            self._teacher()
            res = self._link.sync.resolve("llm")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "detail": type(exc).__name__}
        return {"ok": bool(getattr(res, "resolved", False)), "detail": str(getattr(res, "reason", ""))[:160],
                "model": getattr(res, "model", None)}

    # ------------------------------------------------------------------ settings and secrets
    def settings_view(self) -> dict[str, Any]:
        values = self.settings.all()
        return {"values": values, "groups": self.settings.groups(), "secrets": self.secrets_status(), "defaults": self.settings.train_defaults(),
                "specs": {k: {"kind": s.kind, "choices": list(s.choices), "doc": s.doc, "group": s.group, "low": s.low, "high": s.high}
                          for k, s in SPECS.items()}}

    def set_settings(self, values: dict[str, Any], confirm_reserved: bool = False) -> dict[str, Any]:
        self.settings.set(values, confirm_reserved=confirm_reserved)
        self._dash_cache = None
        return self.settings_view()

    def set_secret(self, name: str, value: str) -> dict[str, Any]:
        """Write-only: the value is saved to ``data/secrets.env`` (owner-only) and never returned or logged."""
        key = name.strip().lower()
        if key not in SECRET_ENV:
            raise PygmalionError("invalid", "unknown_secret", name=name, options=list(SECRET_KEYS))
        env_name = f"{PREFIX}{SECRET_ENV[key]}"
        path = self.config.secrets_path
        lines = load_dotenv(path)
        value = (value or "").strip()
        if value:
            lines[env_name] = value
        else:
            lines.pop(env_name, None)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(f"{k}={v}\n" for k, v in lines.items()), encoding="utf-8")
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
        os.replace(tmp, path)
        self.config.secrets.pop(env_name, None)
        return self.secrets_status()[key]

    def secrets_status(self) -> dict[str, dict[str, Any]]:
        out = {}
        for key, env_name in SECRET_ENV.items():
            value = self.config.secret(env_name)
            out[key] = {"configured": bool(value), "source": "env" if os.environ.get(f"{PREFIX}{env_name}") else ("file" if value else "")}
        return out

    # ------------------------------------------------------------------ views
    def sync_bases(self) -> list[Path]:
        """Register every model folder of the work folder as a ``base`` artifact (what ``bases`` does with the full cards) and return them."""
        folders = local_model_folders(self.work.hf_dir)
        with self._bases_lock:                      # two requests must not both register the same new folder
            for folder in folders:
                try:
                    self.lineage.ensure_base(folder, repo_of_folder(folder.name) or folder.name, {"source": "local", "folder": folder.name})
                except PygmalionError:
                    continue
        return folders

    def counts(self) -> dict[str, int]:
        """Store counts, with ``bases`` read from the same place as ``bases_list``: the model folders on disk, registered on sight."""
        folders = self.sync_bases()
        counts = self.store.counts()
        counts["bases"] = len(folders)
        return counts

    def bases(self, deep: bool = False) -> list[dict[str, Any]]:
        """Models in the work folder (registered as ``base`` artifacts on first sight) with their support badges."""
        script = self.work.src_script("convert_hf_to_gguf.py")
        if deep and self.work.python_ok():
            self.env.check()
        cards = scan_local(self.work.hf_dir, script, self.env.trainable_architectures())
        with self._bases_lock:
            for card in cards:
                art = self.lineage.ensure_base(card["path"], card["repo_id"] or card["name"], {"source": "local", "folder": card["name"]})
                card["artifact"] = art["id"]
        return cards

    def environment(self, fresh: bool = False) -> dict[str, Any]:
        if fresh or self.env.last() is None:
            return self.env.check()
        return self.env.last()

    def status(self) -> dict[str, Any]:
        return {"service": SERVICE, "version": __version__, "data_dir": str(self.config.data_dir), "uptime_s": int(time.time() - self.started_at),
                "counts": self.counts(), "scheduler": self.jobs.status(), "offline": self.config.offline,
                "work_dir": str(self.work.root), "allowed_gpus": self.gpus.allowed(), "recent_runs": self.store.runs(limit=12)}

    def dashboard(self) -> dict[str, Any]:
        now = self.clock()
        env = self.env.last()
        # The slow blocks are cached for a few seconds, but never across an environment check: the panel must show its result at once.
        if self._dash_cache and now - self._dash_cache[0] < DASHBOARD_CACHE_S and self._dash_cache[2] is env:
            slow = self._dash_cache[1]
        else:
            slow = {"env": self._env_summary(env), "galton": self.galton.available(), "gpus": self.gpus.status()}
            self._dash_cache = (now, slow, env)
        active = [self.jobs.view(j) for j in self.store.jobs(states=["running", "waiting_gpu", "queued"], limit=12, oldest_first=True)]
        for view in active:
            view["curve"] = self.jobs.curve(view["id"], 60)["train"][-60:] if view["kind"] == "train" else []
        latest = [self.lineage.card(a) for a in self.store.artifacts(limit=8)]
        failed = [self.jobs.view(j) for j in self.store.jobs(states=["failed", "interrupted"], limit=5)]
        return {"now": now, **slow, "gpus": self.gpus.status(), "jobs": active, "failed": failed, "artifacts": latest, "counts": self.counts(),
                "scheduler": self.jobs.status(), "storage": self.lineage.storage(), "offline": self.config.offline}

    @staticmethod
    def _env_summary(env: Optional[dict[str, Any]]) -> dict[str, Any]:
        if env is None:
            return {"checked": False}
        return {"checked": True, "ok": env["ok"], "capabilities": env["capabilities"], "problems": len(env["problems"]),
                "torch": (env["trainer"].get("torch") or {}).get("version"), "cuda": (env["trainer"].get("torch") or {}).get("cuda_available")}
