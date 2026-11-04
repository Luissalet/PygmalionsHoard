"""SpeechBench: run speech engines over a corpus with references and keep WER, CER, latency and real-time factor.

Runs live in ``<data>/speechbench/runs/<id>.json`` (one file each, rewritten atomically as files finish), the command
engines and the chosen Whisper Python in ``<data>/speechbench/engines.json``. One run at a time per engine kind is not
enforced: runs are CPU work in their own thread and process, at below-normal priority."""

from __future__ import annotations

import statistics
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from ..hoard_link import atomic
from . import corpus as corpus_mod
from . import engines as eng
from .metrics import corpus_rates, score

STATES_ACTIVE = ("queued", "running")


class SpeechBench:
    def __init__(self, data_dir: Path, *, hub_apps: Optional[Callable[[], list[dict[str, Any]]]] = None,
                 clock: Callable[[], float] = time.time, emit: Optional[Callable[..., Any]] = None,
                 runner: Optional[dict[str, Callable[..., Any]]] = None):
        self.root = Path(data_dir) / "speechbench"
        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        self.hub_apps, self.clock, self.emit = hub_apps, clock, emit
        self.runner = runner or {"whisper": eng.run_whisper, "command": eng.run_command}
        self.lock = threading.RLock()
        self.threads: dict[str, tuple[threading.Thread, threading.Event]] = {}

    # ------------------------------------------------------------------ engines
    def _config(self) -> dict[str, Any]:
        data = atomic.read_json(self.root / "engines.json", None) or {}
        return {"commands": dict(data.get("commands") or {}), "whisper_python": str(data.get("whisper_python") or "")}

    def _save_config(self, config: dict[str, Any]) -> None:
        atomic.write_json_atomic(self.root / "engines.json", config)

    def whisper_python(self) -> Optional[str]:
        return eng.find_whisper_python(self._config()["whisper_python"], self.hub_apps)

    def engines(self) -> dict[str, Any]:
        config = self._config()
        python = self.whisper_python()
        whisper = [f"faster-whisper:{s}" for s in eng.WHISPER_SIZES] if python else []
        commands = [{**c, "id": f"cmd:{n}"} for n, c in sorted(config["commands"].items())]
        return {"available": whisper + [c["id"] for c in commands], "commands": commands,
                "whisper": {"python": python, "configured": config["whisper_python"] or None, "sizes": list(eng.WHISPER_SIZES),
                            "models_dir": self.models_dir(),
                            "note": None if python else "No Python with faster-whisper found: set speechbench.whisper_python or GALTON_WHISPER_PYTHON."}}

    @staticmethod
    def models_dir() -> str:
        try:
            from ..hoard_link.media.stt import default_models_dir

            return str(default_models_dir())
        except Exception:  # noqa: BLE001
            return ""

    def engine_save(self, name: str, command: Optional[list[str]] = None, *, timeout_s: float = 600, cwd: str = "",
                    note: str = "", delete: bool = False) -> dict[str, Any]:
        with self.lock:
            config = self._config()
            if delete:
                if config["commands"].pop(name, None) is None:
                    raise LookupError(f"No command engine named {name!r}.")
                self._save_config(config)
                return {"deleted": name}
            saved = eng.save_command(config["commands"], name, command or [], timeout_s, cwd, note)
            self._save_config(config)
            return {"saved": {**saved, "id": f"cmd:{name}"}}

    def set_whisper_python(self, path: str) -> dict[str, Any]:
        path = path.strip().strip('"')
        if path and not (Path(path).is_file() and eng.has_faster_whisper(path)):
            raise ValueError(f"{path} is not a Python with faster-whisper installed.")
        with self.lock:
            config = self._config()
            config["whisper_python"] = path
            self._save_config(config)
        return self.engines()["whisper"]

    # ------------------------------------------------------------------ corpus and runs
    def corpus_check(self, folder: str, limit: int = corpus_mod.MAX_ITEMS) -> dict[str, Any]:
        found = corpus_mod.load(folder, limit=limit)
        return {**{k: v for k, v in found.items() if k != "items"},
                "preview": [{k: i[k] for k in ("id", "duration_s")} | {"reference": i["reference"][:160]} for i in found["items"][:20]]}

    def _path(self, run_id: str) -> Path:
        if not run_id or any(c not in "0123456789abcdef" for c in run_id):
            raise LookupError(f"No speech run {run_id!r}.")
        return self.root / "runs" / f"{run_id}.json"

    def _load(self, run_id: str) -> dict[str, Any]:
        run = atomic.read_json(self._path(run_id), None)
        if not run:
            raise LookupError(f"No speech run {run_id!r}.")
        return run

    def _write(self, run: dict[str, Any]) -> None:
        with self.lock:
            atomic.write_json_atomic(self._path(run["id"]), run)

    def start(self, folder: str, engine_specs: list[str], *, language: Optional[str] = None, accents: bool = True,
              device: str = "cpu", compute: str = "int8", limit: int = corpus_mod.MAX_ITEMS, dry_run: bool = False,
              name: str = "") -> dict[str, Any]:
        config = self._config()
        engines = [eng.parse_engine(s, config["commands"]) for s in dict.fromkeys(engine_specs)]
        if not engines:
            raise ValueError("Name at least one engine (speechbench_engines lists them).")
        found = corpus_mod.load(folder, limit=limit)
        if not found["items"]:
            raise ValueError(f"No audio with a reference in {found['folder']} ({len(found['problems'])} problems).")
        python = self.whisper_python() if any(e["kind"] == "faster-whisper" for e in engines) else None
        if any(e["kind"] == "faster-whisper" for e in engines) and not python:
            raise ValueError("No Python with faster-whisper found: set it with speechbench_engines (whisper_python).")
        plan = {"folder": found["folder"], "files": found["files"], "audio_s": found["audio_s"], "problems": found["problems"],
                "engines": [e["id"] for e in engines], "language": language, "accents": accents, "device": device,
                "compute": compute, "whisper_python": python}
        if dry_run:
            return {"dry_run": True, **plan}
        run = {"id": uuid.uuid4().hex[:16], "name": name or Path(found["folder"]).name, "state": "queued",
               "created": self.clock(), "finished": None, **plan, "corpus": found["items"],
               "results": {e["id"]: {"items": {}, "load_s": None, "error": None} for e in engines}, "summary": {}}
        self._write(run)
        cancel = threading.Event()
        thread = threading.Thread(target=self._work, args=(run["id"], engines, cancel), name=f"speechbench-{run['id']}", daemon=True)
        with self.lock:
            self.threads[run["id"]] = (thread, cancel)
        thread.start()
        return self.card(run)

    def _record(self, run: dict[str, Any], engine_id: str, item: dict[str, Any], result: dict[str, Any]) -> None:
        entry = {k: result.get(k) for k in ("text", "first_text_s", "decode_s", "error", "language")}
        if result.get("text") is not None and not result.get("error"):
            entry["score"] = score(item["reference"], result["text"], accents=run["accents"])
        if item.get("duration_s") and result.get("decode_s") is not None:
            entry["rtf"] = round(result["decode_s"] / item["duration_s"], 3)
        run["results"][engine_id]["items"][item["id"]] = entry
        run["summary"][engine_id] = self.summarise(run, engine_id)
        self._write(run)

    def _work(self, run_id: str, engines: list[dict[str, Any]], cancel: threading.Event) -> None:
        run = self._load(run_id)
        run["state"] = "running"
        self._write(run)
        by_id = {i["id"]: i for i in run["corpus"]}
        try:
            for engine in engines:
                if cancel.is_set():
                    break
                if engine["kind"] == "faster-whisper":
                    def on_event(event: dict[str, Any], engine_id: str = engine["id"]) -> None:
                        if event.get("event") == "loaded":
                            run["results"][engine_id]["load_s"] = event.get("load_s")
                            self._write(run)
                        elif event.get("event") in ("file", "error") and event.get("id") in by_id:
                            self._record(run, engine_id, by_id[event["id"]], event)
                        elif event.get("event") == "failed":
                            run["results"][engine_id]["error"] = event.get("error") or "the Whisper worker failed"
                            self._write(run)

                    self.runner["whisper"](run["whisper_python"], engine, run["corpus"], language=run["language"], device=run["device"],
                                           compute=run["compute"], models_dir=self.models_dir(), on_event=on_event, cancel=cancel)
                else:
                    for item in run["corpus"]:
                        if cancel.is_set():
                            break
                        self._record(run, engine["id"], item, self.runner["command"](engine, item, language=run["language"], cancel=cancel))
            run["state"] = "cancelled" if cancel.is_set() else "done"
        except Exception as error:  # noqa: BLE001 - the run says why it stopped
            run["state"], run["error"] = "failed", f"{type(error).__name__}: {error}"
        run["finished"] = self.clock()
        run["summary"] = {e["id"]: self.summarise(run, e["id"]) for e in engines}
        self._write(run)
        with self.lock:
            self.threads.pop(run_id, None)
        if self.emit:
            try:
                self.emit("speechbench.finished", {"run": run_id, "state": run["state"]})
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def summarise(run: dict[str, Any], engine_id: str) -> dict[str, Any]:
        res = run["results"][engine_id]
        items = list(res["items"].values())
        durations = {i["id"]: i.get("duration_s") for i in run["corpus"]}
        done = [i for i in items if not i.get("error")]
        timed = [(i["decode_s"], durations.get(k)) for k, i in res["items"].items() if i.get("decode_s") is not None and durations.get(k)]
        firsts = [i["first_text_s"] for i in done if i.get("first_text_s") is not None]
        return {**corpus_rates(items), "files_done": len(items), "failures": len(items) - len(done),
                "load_s": res.get("load_s"), "decode_s": round(sum(d for d, _ in timed), 3) if timed else None,
                "rtf": round(sum(d for d, _ in timed) / sum(a for _, a in timed), 3) if timed else None,
                "first_text_s_median": round(statistics.median(firsts), 3) if firsts else None, "error": res.get("error")}

    def card(self, run: dict[str, Any]) -> dict[str, Any]:
        return {k: run.get(k) for k in ("id", "name", "state", "created", "finished", "folder", "files", "audio_s", "engines",
                                         "language", "device", "compute", "error")} | {"summary": run.get("summary") or {}}

    def get(self, run_id: str, *, items: bool = False, engine: Optional[str] = None) -> dict[str, Any]:
        run = self._load(run_id)
        out = {"run": self.card(run), "problems": run.get("problems") or []}
        if items:
            refs = {i["id"]: i for i in run["corpus"]}
            out["items"] = [{"id": k, "reference": refs.get(k, {}).get("reference"), "duration_s": refs.get(k, {}).get("duration_s"),
                             "engine": eid, **{f: v for f, v in r.items() if f != "score"},
                             "wer": (r.get("score") or {}).get("wer"), "cer": (r.get("score") or {}).get("cer")}
                            for eid, res in run["results"].items() if engine in (None, eid) for k, r in res["items"].items()]
        return out

    def list(self, limit: int = 20) -> dict[str, Any]:
        files = sorted((self.root / "runs").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        runs = [self.card(r) for r in (atomic.read_json(p, None) for p in files) if r]
        return {"runs": runs, "count": len(runs)}

    def cancel(self, run_id: str) -> dict[str, Any]:
        self._load(run_id)
        with self.lock:
            entry = self.threads.get(run_id)
        if entry:
            entry[1].set()
        return {"run": run_id, "cancelling": bool(entry)}

    def stop_all(self, timeout: float = 5.0) -> None:
        with self.lock:
            entries = list(self.threads.values())
        for thread, cancel in entries:
            cancel.set()
        for thread, _ in entries:
            thread.join(timeout)
