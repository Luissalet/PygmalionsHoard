"""The model's side of a rebuild: a tool loop where the model explores the reference in a browser and writes the files of its own site.

The model sees the reference only through the browser tools (a text picture of the page: every visible element with its position, size,
text, font, colours and box), never its source. It writes static files (HTML, CSS, JS, SVG) into its folder and can open its own pages the
same way to compare. The loop ends when the model calls ``finish``, when it stops calling tools, or when the steps run out.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..backends import Completion

SUFFIXES = (".html", ".htm", ".css", ".js", ".mjs", ".json", ".svg", ".txt")
MAX_FILE = 400_000
MAX_FILES = 40
SNAPSHOT_CHARS = 14_000
KEEP_FULL = 6                  # tool results kept whole; older ones are cut to their first line

SYSTEM = """Eres un desarrollador web. Tienes que reconstruir un sitio web de referencia como sitio estático propio (HTML, CSS y JavaScript) en tu carpeta.

Cómo lo ves: solo con las herramientas del navegador. `browse` abre una ruta de la referencia y te devuelve lo que muestra la página: cada elemento visible con su etiqueta, selector, posición y tamaño en píxeles (x,y ancho×alto), texto, tipografía (peso tamaño/interlineado familia), color, fondo, borde, radio, relleno y disposición (flex/grid). `click`, `fill`, `press` y `back` actúan sobre la página abierta y te devuelven cómo queda. Puedes mirarla a 1280 px (escritorio) y a 390 px (móvil) con `width`.

Qué cuenta:
- Las mismas rutas (los mismos nombres de fichero que enlaza la referencia; la portada es index.html).
- El mismo aspecto, píxel a píxel, a 1280 px y a 390 px: posiciones, tamaños, colores, tipografías, textos.
- El mismo comportamiento: botones, formularios, filtros, contadores y enlaces hacen lo mismo y muestran los mismos textos.
- Todo local: nada de CDN, fuentes web ni imágenes remotas. Usa las familias tipográficas que veas (Arial, Georgia...).

Cómo trabajar: explora todas las páginas y estados (también después de pulsar botones), escribe los ficheros con `write_file`, abre tus páginas con `preview` para comparar con la referencia, corrige y termina con `finish`. Unas pruebas ocultas recorren los dos sitios, comparan capturas de pantalla (SSIM) y comprueban textos y comportamiento."""

TOOLS = [
    ("browse", "Abre una ruta de la referencia y devuelve lo que muestra.", {"path": ("string", "Ruta, p. ej. / o /articulo-1.html"), "width": ("integer", "1280 o 390")}, ["path"]),
    ("click", "Pulsa un elemento de la página de referencia abierta.", {"target": ("string", "Selector de Playwright: CSS (#id, .clase, button) o text=Texto visible"), "width": ("integer", "1280 o 390")}, ["target"]),
    ("fill", "Escribe en un campo de la página de referencia abierta.", {"target": ("string", "Selector del campo"), "text": ("string", "Texto"), "width": ("integer", "1280 o 390")}, ["target", "text"]),
    ("press", "Pulsa una tecla en la página de referencia abierta.", {"key": ("string", "Enter, Tab, Escape..."), "target": ("string", "Selector donde pulsarla (opcional)"), "width": ("integer", "1280 o 390")}, ["key"]),
    ("back", "Vuelve a la página anterior de la referencia.", {"width": ("integer", "1280 o 390")}, []),
    ("write_file", "Escribe (o sustituye) un fichero de tu sitio.", {"path": ("string", "Ruta relativa: index.html, estilo.css, app.js..."), "content": ("string", "Contenido completo")}, ["path", "content"]),
    ("read_file", "Lee un fichero de tu sitio.", {"path": ("string", "Ruta relativa")}, ["path"]),
    ("list_files", "Lista los ficheros de tu sitio.", {}, []),
    ("preview", "Abre una ruta de TU sitio y devuelve lo que muestra, igual que browse.", {"path": ("string", "Ruta"), "width": ("integer", "1280 o 390")}, ["path"]),
    ("finish", "Termina: tu sitio está listo.", {"note": ("string", "Una frase sobre lo que has hecho")}, []),
]


def tool_specs() -> list[dict[str, Any]]:
    out = []
    for name, desc, props, required in TOOLS:
        out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": {
            "type": "object", "properties": {k: {"type": t, "description": d} for k, (t, d) in props.items()}, "required": required}}})
    return out


def snapshot_text(snap: dict[str, Any], limit: int = SNAPSHOT_CHARS) -> str:
    """The browser's picture of a page as compact lines the model can read."""
    if not snap:
        return "(la página no devolvió nada)"
    page = snap.get("page") or {}
    lines = [f"title: {snap.get('title', '')}", f"route: {snap.get('url', '')}",
             f"page: {page.get('width')}x{page.get('height')} bg={page.get('bg')} color={page.get('color')} font={page.get('font')}"]
    for it in snap.get("items") or []:
        parts = ["  " * min(int(it.get("d") or 0), 8) + it.get("tag", ""), it.get("sel", ""), f"@{it.get('x')},{it.get('y')} {it.get('w')}x{it.get('h')}"]
        if it.get("text"):
            parts.append(json.dumps(it["text"], ensure_ascii=False))
        for key in ("font", "color", "bg", "bgimage", "border", "radius", "shadow", "padding", "layout", "align", "transform", "spacing", "href", "type",
                    "placeholder", "value", "checked", "alt", "label"):
            if it.get(key) not in (None, ""):
                parts.append(f"{key}={it[key]}")
        lines.append(" ".join(str(p) for p in parts))
    text = "\n".join(lines)
    return text if len(text) <= limit else text[:limit] + "\n[... cortado]"


def safe_path(site: Path, rel: str) -> Optional[Path]:
    rel = str(rel or "").strip().replace("\\", "/").lstrip("/")
    if not rel or ".." in rel.split("/") or not rel.lower().endswith(SUFFIXES) or not re.match(r"^[A-Za-z0-9_./-]+$", rel):
        return None
    return site / rel


@dataclass
class AgentResult:
    steps: int = 0
    tokens_in: Optional[int] = 0
    tokens_out: Optional[int] = 0
    latency_ms: float = 0.0
    stop: str = ""
    note: str = ""
    error: str = ""
    files: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)


Chat = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Completion]
Explore = Callable[[str, int, dict[str, Any]], dict[str, Any]]     # (which: reference|candidate, width, action) -> worker reply


class RebuildAgent:
    def __init__(self, chat: Chat, explore: Explore, site: Path, *, api: str = "openai", max_steps: int = 30,
                 cancel: Callable[[], bool] = lambda: False):
        self.chat, self.explore, self.site, self.api, self.max_steps, self.cancel = chat, explore, Path(site), api, max_steps, cancel
        self.site.mkdir(parents=True, exist_ok=True)

    def files(self) -> list[str]:
        return sorted(p.relative_to(self.site).as_posix() for p in self.site.rglob("*") if p.is_file())

    # ------------------------------------------------------------------ tools
    def _browser(self, which: str, width: Any, action: dict[str, Any]) -> str:
        w = 390 if str(width or "").strip() == "390" else 1280
        reply = self.explore(which, w, action)
        head = f"[{'referencia' if which == 'reference' else 'tu sitio'} a {w} px]"
        if reply.get("error"):
            head += f" error: {reply['error']}"
        if reply.get("page_errors"):
            head += f" errores de la página: {'; '.join(reply['page_errors'])}"
        return head + "\n" + snapshot_text(reply.get("snapshot") or {})

    def run_tool(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """(the text the model gets back, whether the loop ends)."""
        if name == "browse":
            return self._browser("reference", args.get("width"), {"goto": str(args.get("path") or "/")}), False
        if name == "click":
            return self._browser("reference", args.get("width"), {"click": str(args.get("target") or "")}), False
        if name == "fill":
            return self._browser("reference", args.get("width"), {"fill": str(args.get("target") or ""), "text": str(args.get("text") or "")}), False
        if name == "press":
            return self._browser("reference", args.get("width"), {"press": str(args.get("key") or "Enter"), "on": str(args.get("target") or "")}), False
        if name == "back":
            return self._browser("reference", args.get("width"), {"back": True}), False
        if name == "preview":
            return self._browser("candidate", args.get("width"), {"goto": str(args.get("path") or "/")}), False
        if name == "write_file":
            path = safe_path(self.site, args.get("path"))
            content = str(args.get("content") or "")
            if path is None:
                return f"ruta no válida: {args.get('path')!r} (relativa, sin '..', con extensión {', '.join(SUFFIXES)})", False
            if len(content.encode("utf-8")) > MAX_FILE:
                return f"fichero demasiado grande (máximo {MAX_FILE // 1000} KB)", False
            if not path.exists() and len(self.files()) >= MAX_FILES:
                return f"demasiados ficheros (máximo {MAX_FILES})", False
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="")
            return f"escrito {path.relative_to(self.site).as_posix()} ({len(content)} caracteres)", False
        if name == "read_file":
            path = safe_path(self.site, args.get("path"))
            if path is None or not path.is_file():
                return f"no existe: {args.get('path')!r}", False
            return path.read_text(encoding="utf-8", errors="replace")[:60_000], False
        if name == "list_files":
            return "\n".join(self.files()) or "(vacía)", False
        if name == "finish":
            return "hecho", True
        return f"herramienta desconocida: {name}", False

    # ------------------------------------------------------------------ loop
    def _assistant(self, comp: Completion, ids: list[str]) -> dict[str, Any]:
        calls = []
        for cid, call in zip(ids, comp.tool_calls):
            args = call.get("arguments") or {}
            calls.append({"id": cid, "type": "function", "function": {"name": call["name"], "arguments": args if self.api == "ollama" else json.dumps(args, ensure_ascii=False)}})
        msg: dict[str, Any] = {"role": "assistant", "content": comp.text or ""}
        if calls:
            msg["tool_calls"] = calls
        return msg

    @staticmethod
    def _trimmed(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tool_idx = [i for i, m in enumerate(history) if m.get("role") == "tool"]
        old = set(tool_idx[:-KEEP_FULL]) if len(tool_idx) > KEEP_FULL else set()
        out = []
        for i, m in enumerate(history):
            if i in old:
                first = str(m.get("content") or "").splitlines()[:1]
                m = {**m, "content": (first[0] if first else "")[:200] + " [resultado antiguo omitido]"}
            out.append(m)
        return out

    def _fallback_html(self, text: str) -> bool:
        """A model that answers with a page in the text instead of calling write_file: the page is kept as index.html."""
        m = re.search(r"```(?:html)?\s*(<!doctype html.*?</html>)\s*```", text or "", re.I | re.S) or re.search(r"(<!doctype html.*</html>)", text or "", re.I | re.S)
        if not m or (self.site / "index.html").exists():
            return False
        (self.site / "index.html").write_text(m.group(1), encoding="utf-8", newline="")
        return True

    def run(self, intro: str) -> AgentResult:
        res = AgentResult()
        history: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": intro}]
        tools = tool_specs()
        idle = 0
        while res.steps < self.max_steps:
            if self.cancel():
                res.stop = "cancelled"
                break
            res.steps += 1
            comp = self.chat(self._trimmed(history), tools)
            res.latency_ms += comp.latency_ms or 0.0
            res.tokens_in = None if res.tokens_in is None or comp.prompt_tokens is None else res.tokens_in + comp.prompt_tokens
            res.tokens_out = None if res.tokens_out is None or comp.completion_tokens is None else res.tokens_out + comp.completion_tokens
            record: dict[str, Any] = {"step": res.steps, "text": (comp.text or "")[:300], "calls": []}
            res.trace.append(record)
            if comp.error:
                res.stop, res.error = "server_error", str(comp.error)[:300]
                record["error"] = res.error
                break
            if not comp.tool_calls:
                if self._fallback_html(comp.text):
                    record["calls"].append({"name": "write_file", "path": "index.html", "from": "text"})
                idle += 1
                if idle >= 2:
                    res.stop = "no_tools"
                    break
                history.append({"role": "assistant", "content": comp.text or ""})
                history.append({"role": "user", "content": "Usa las herramientas (browse, write_file, preview...) y termina con finish."})
                continue
            idle = 0
            ids = [f"call_{res.steps}_{i}" for i in range(len(comp.tool_calls))]
            history.append(self._assistant(comp, ids))
            done = False
            for cid, call in zip(ids, comp.tool_calls):
                args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                if "_unparsed" in args:
                    out, end = "los argumentos no son JSON válido", False
                else:
                    out, end = self.run_tool(call["name"], args)
                record["calls"].append({"name": call["name"], **({"path": str(args.get("path"))[:120]} if args.get("path") else {}),
                                        **({"target": str(args.get("target"))[:120]} if args.get("target") else {})})
                msg: dict[str, Any] = {"role": "tool", "tool_call_id": cid, "content": out}
                if self.api == "ollama":
                    msg["tool_name"] = call["name"]
                history.append(msg)
                if end:
                    done = True
                    res.note = str(args.get("note") or "")[:300]
            if done:
                res.stop = "finished"
                break
        else:
            res.stop = "steps"
        if not res.stop:
            res.stop = "steps"
        res.files = self.files()
        return res
