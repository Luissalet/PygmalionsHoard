"""Publishing: an Ollama tag (Modelfile + ``ollama create``) or a llama.cpp command backend in ``~/.hoard/backends.json`` that the
hub can start. Nothing is started automatically."""

from __future__ import annotations

import re
import socket
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from .errors import PygmalionError
from .gguf_meta import read_metadata, summarize
from .hoard_link.launch import Launcher
from .lineage import Lineage
from .messages import text
from .procs import run_capture, run_streaming
from .settings import Settings
from .store import Store
from .util import slug
from .workdir import Work

OLLAMA_NAME_RE = re.compile(r"^pyg-[a-z0-9][a-z0-9._-]{0,60}:[a-z0-9][a-z0-9._-]{0,40}$")
BACKEND_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,40}$")


def ollama_name(name: str, tag: str = "latest") -> str:
    """``pyg-<slug>:<tag>``: lowercase, and always prefixed so the app never touches a tag it did not create."""
    base = slug(name.removeprefix("pyg-"), 56)
    t = slug(tag or "latest", 40).replace("_", "-") or "latest"
    full = f"pyg-{base}:{t}"
    if not OLLAMA_NAME_RE.match(full):
        raise PygmalionError("invalid", "ollama_tag_invalid", tag=full)
    return full


def _fs_path(path: str | Path) -> str:
    """A path as a Modelfile wants it: forward slashes work on Windows too and need no escaping; quoted when it has spaces."""
    text = str(path).replace("\\", "/")
    return f'"{text}"' if re.search(r"\s|\"", text) else text


def modelfile(gguf: str | Path, *, adapter: Optional[str | Path] = None, num_ctx: Optional[int] = None, template: str = "", system: str = "",
              parameters: Optional[dict[str, Any]] = None) -> str:
    lines = [f"FROM {_fs_path(gguf)}"]
    if adapter:
        lines.append(f"ADAPTER {_fs_path(adapter)}")
    if num_ctx:
        lines.append(f"PARAMETER num_ctx {int(num_ctx)}")
    for key, value in (parameters or {}).items():
        if not re.match(r"^[a-z_]{2,30}$", str(key)):
            raise PygmalionError("invalid", "modelfile_parameter", key=key)
        for item in (value if isinstance(value, list) else [value]):
            lines.append(f'PARAMETER {key} {str(item).replace(chr(10), " ")}')
    if template:
        lines.append('TEMPLATE """' + template.replace('"""', '\\"\\"\\"') + '"""')
    if system:
        lines.append('SYSTEM """' + system.replace('"""', '\\"\\"\\"') + '"""')
    return "\n".join(lines) + "\n"


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) != 0


class Publisher:
    def __init__(self, store: Store, lineage: Lineage, settings: Settings, work: Work, *, launcher: Optional[Launcher] = None,
                 is_free: Callable[[int], bool] = port_free):
        self.store = store
        self.lineage = lineage
        self.settings = settings
        self.work = work
        self.launcher = launcher or Launcher(app="pygmalion")
        self.is_free = is_free

    # ------------------------------------------------------------------ Ollama
    def _gguf(self, ref: str) -> dict[str, Any]:
        art = self.lineage.resolve(ref, ("gguf",))
        if not Path(art["path"]).is_file():
            raise PygmalionError("not_found", "gguf_file_missing", name=art["name"], path=art["path"])
        return art

    def default_ctx(self, art: dict[str, Any], asked: Optional[int], trained: Optional[int]) -> tuple[Optional[int], Any]:
        """The ``num_ctx`` of a Modelfile. Ollama reserves the whole KV cache for it, so writing the trained context (often 128k to 256k tokens)
        asks for far more memory than any chat needs: the default is the ``publish.num_ctx`` setting, never above the trained context. A context
        variant is published at the length it was built for. A number the caller asks for is kept, with a note when it is above the trained one."""
        if asked:
            return int(asked), (text("publish_note_ctx_above", ctx=int(asked), trained=trained) if trained and asked > trained else None)
        if trained and any(a["kind"] == "ctx_variant" for a in self.lineage.ancestors(art["id"])):
            return int(trained), None
        wanted = self.settings.int("publish.num_ctx")
        if trained and wanted > trained:
            return int(trained), None
        return wanted or (int(trained) if trained else None), None

    def ollama_plan(self, ref: str, name: str = "", tag: str = "latest", num_ctx: Optional[int] = None, adapter: str = "",
                    template: str = "", system: str = "") -> dict[str, Any]:
        art = self._gguf(ref)
        adapter_art = self._gguf(adapter) if adapter else None
        tag_name = ollama_name(name or art["name"], tag)
        try:
            summary = summarize(read_metadata(art["path"]))
        except PygmalionError:
            summary = {}
        trained = summary.get("context_length") or None
        notes = []
        ctx, ctx_note = self.default_ctx(art, num_ctx, trained)
        if ctx_note:
            notes.append(ctx_note)
        if summary.get("rope_scaling_type") and summary.get("rope_scaling_type") != "none":
            notes.append(text("publish_note_rope", type=summary["rope_scaling_type"], factor=summary.get("rope_scaling_factor")))
        if not summary.get("has_chat_template") and not template:
            notes.append(text("publish_note_no_template"))
        content = modelfile(art["path"], adapter=adapter_art["path"] if adapter_art else None, num_ctx=ctx, template=template, system=system)
        return {"artifact": art, "adapter": adapter_art, "tag": tag_name, "modelfile": content, "num_ctx": ctx, "notes": notes, "summary": summary}

    def publish_ollama(self, ref: str, name: str = "", tag: str = "latest", num_ctx: Optional[int] = None, adapter: str = "", template: str = "",
                       system: str = "", cancel: Optional[threading.Event] = None, log_path: Optional[Path] = None,
                       job_id: Optional[str] = None) -> dict[str, Any]:
        exe = self.work.ollama_exe()
        if not exe:
            raise PygmalionError("tool_missing", "ollama_missing")
        plan = self.ollama_plan(ref, name, tag, num_ctx, adapter, template, system)
        art = plan["artifact"]
        folder = self.work.outputs_dir / art["id"]
        folder.mkdir(parents=True, exist_ok=True)
        mf = folder / f"Modelfile.{plan['tag'].split(':', 1)[1]}"
        mf.write_text(plan["modelfile"], encoding="utf-8")
        result = run_streaming([exe, "create", plan["tag"], "-f", str(mf)], cwd=folder, log_path=log_path, cancel=cancel, timeout_s=3600)
        if result.cancelled:
            raise PygmalionError("failed", "cancelled")
        if not result.ok:
            raise PygmalionError("failed", "ollama_create_failed", detail=" | ".join(result.tail[-3:])[:300])
        existing = self.store.find_artifact_by_name("ollama", plan["tag"])
        recipe = {"job_kind": "publish", "params": {"gguf": art["name"], "tag": plan["tag"], "num_ctx": plan["num_ctx"], "adapter": plan["adapter"]["name"] if plan["adapter"] else None}}
        parents = [art["id"]] + ([plan["adapter"]["id"]] if plan["adapter"] else [])
        if existing:
            metrics = {k: v for k, v in existing["metrics"].items() if k != "unpublished_ts"}
            metrics["modelfile"] = plan["modelfile"]
            existing = self.store.update_artifact(existing["id"], metrics=metrics, parents=parents)
        pub = existing or self.store.create_artifact("ollama", plan["tag"], path=plan["tag"], size=art["size"], parents=parents, job_id=job_id,
                                                      recipe=recipe, metrics={"modelfile": plan["modelfile"]}, dataset_version=art.get("dataset_version"))
        self.lineage.mark_published(art["id"], {"type": "ollama", "name": plan["tag"], "artifact": pub["id"], "ts": self.store.clock()})
        return {"artifact": pub["id"], "tag": plan["tag"], "modelfile": plan["modelfile"], "notes": plan["notes"], "from": art["id"]}

    # ------------------------------------------------------------------ llama.cpp
    def _used_ports(self) -> set[int]:
        ports = set()
        for cmd in self.launcher.config().get("commands") or []:
            m = re.search(r":(\d+)(?:/|$)", str(cmd.get("health", "")))
            if m:
                ports.add(int(m.group(1)))
        return ports

    def pick_port(self) -> int:
        start = self.settings.int("publish.llama_port_start")
        used = self._used_ports()
        for port in range(start, start + 100):
            if port not in used and self.is_free(port):
                return port
        raise PygmalionError("failed", "no_free_port")

    def backend_entry(self, art: dict[str, Any], *, backend_id: str, port: int, ctx: int, ngl: int, gpu: Optional[int], extra: list[str]) -> dict[str, Any]:
        exe = self.work.llama_bin("llama-server")
        if not exe:
            raise PygmalionError("tool_missing", "llama_server_missing")
        argv = [exe, "-m", art["path"], "-c", str(ctx), "-ngl", str(ngl), "--host", "127.0.0.1", "--port", str(port), "--jinja", "-a", backend_id, "-fa", "on", *extra]
        entry: dict[str, Any] = {"id": backend_id, "label": f"{art['name']} (Pygmalion)", "argv": argv, "cwd": str(Path(exe).parent),
                                 "health": f"http://127.0.0.1:{port}/health", "capabilities": ["llm"]}
        if gpu is not None:
            # PCI order: the index means the same GPU here as in nvidia-smi and the allowed list
            entry["env"] = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu)}
        return entry

    def publish_llama(self, ref: str, name: str = "", ctx: Optional[int] = None, ngl: int = 99, gpu: Optional[int] = None,
                      extra_args: Optional[list[str]] = None, port: Optional[int] = None) -> dict[str, Any]:
        art = self._gguf(ref)
        allowed = self.settings.intlist("gpus.allowed")
        if gpu is not None and gpu not in allowed:
            raise PygmalionError("forbidden", "gpu_not_in_allowed", gpu=gpu, allowed=allowed)
        if gpu is None and allowed:
            gpu = allowed[0]
        try:
            summary = summarize(read_metadata(art["path"]))
        except PygmalionError:
            summary = {}
        backend_id = slug(f"pyg-{(name or art['name']).removeprefix('pyg-')}", 40)
        if not BACKEND_ID_RE.match(backend_id):
            raise PygmalionError("invalid", "backend_id_invalid", id=backend_id)
        chosen = port or self.pick_port()
        entry = self.backend_entry(art, backend_id=backend_id, port=chosen, ctx=ctx or min(summary.get("context_length") or 8192, 32768), ngl=ngl,
                                   gpu=gpu, extra=[str(x) for x in (extra_args or [])])
        commands = [c for c in self.launcher.config().get("commands") or [] if c.get("id") != backend_id] + [entry]
        try:
            self.launcher.set_config({"commands": commands})
        except ValueError as exc:
            raise PygmalionError("invalid", "backends_refused", detail=str(exc)) from exc
        self.lineage.mark_published(art["id"], {"type": "llama", "name": backend_id, "port": chosen, "ts": self.store.clock()})
        return {"id": backend_id, "service": f"cmd:{backend_id}", "port": chosen, "entry": entry, "from": art["id"],
                "note": text("publish_note_llama")}

    # ------------------------------------------------------------------ unpublish
    def unpublish(self, ref: str, target: str = "all", name: str = "", log_path: Optional[Path] = None) -> dict[str, Any]:
        art = self.lineage.resolve(ref)
        removed: list[dict[str, Any]] = []
        entries = list(art["published"])
        if art["kind"] == "ollama":
            entries = [{"type": "ollama", "name": art["path"], "artifact": art["id"], "_self": True}]
        if not entries:
            raise PygmalionError("not_found", "not_published", name=art["name"])
        for entry in entries:
            kind = entry.get("type")
            if target not in ("all", kind) or (name and entry.get("name") != name):
                continue
            if kind == "ollama":
                exe = self.work.ollama_exe()
                if not exe:
                    raise PygmalionError("tool_missing", "ollama_missing")
                if not str(entry["name"]).startswith("pyg-"):
                    raise PygmalionError("forbidden", "only_pyg_tags")
                code, _out, err = run_capture([exe, "rm", str(entry["name"])], timeout_s=120)
                if code != 0 and "not found" not in err.lower():
                    raise PygmalionError("failed", "ollama_rm_failed", detail=err.strip()[:200])
            elif kind == "llama":
                commands = [c for c in self.launcher.config().get("commands") or [] if c.get("id") != entry.get("name")]
                self.launcher.set_config({"commands": commands})
            removed.append({"type": kind, "name": entry.get("name")})
            parent_id = entry.get("artifact") if entry.get("_self") else art["id"]
            if entry.get("_self"):
                self._mark_gone(art)
                for parent in art["parents"]:
                    try:
                        self.lineage.unmark_published(parent, "ollama", entry["name"])
                    except PygmalionError:
                        pass
            else:
                self.lineage.unmark_published(parent_id, kind, entry["name"])
                if kind == "ollama" and entry.get("artifact"):
                    try:
                        self._mark_gone(self.store.artifact(entry["artifact"]))
                    except PygmalionError:
                        pass
        if not removed:
            raise PygmalionError("not_found", "unpublish_nothing", entries=[f"{e.get('type')}: {e.get('name')}" for e in entries])
        return {"removed": removed, "artifact": art["id"]}

    def _mark_gone(self, pub: dict[str, Any]) -> None:
        metrics = dict(pub["metrics"])
        metrics["unpublished_ts"] = self.store.clock()
        self.store.update_artifact(pub["id"], metrics=metrics, notes=(pub["notes"] + "\n" + str(text("note_unpublished"))).strip())
