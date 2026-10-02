"""Where dataset records come from. Every source turns a small JSON description into raw records; nothing is trusted:
files and the answers of other apps are data, and the same cleaning (``normalize_record``) applies to all of them.

Source types: ``jsonl`` (pasted text or a file), ``csv`` (choose the columns), ``files`` (uploaded or listed text/markdown files),
``folder`` (an absolute path; system, credential and hidden folders are refused), ``family`` (a tool of another Hoard app, called
through the hub, with a mapping of result fields to record fields), ``synthetic`` (a teacher model writes items from source chunks).
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path, PurePath
from typing import Any, Callable, Iterator, Optional

from .. import paths
from ..errors import PygmalionError
from ..messages import text as msg
from .formats import content_text, detect_kind, normalize_record, with_meta

MAX_INLINE_CHARS = 25_000_000
MAX_FILE_BYTES = 200 * 1024 * 1024
TEXT_EXTENSIONS = (".txt", ".md", ".markdown", ".rst", ".text")
MAX_FOLDER_FILES = 20_000

FamilyCall = Callable[[str, str, dict[str, Any]], dict[str, Any]]
Teacher = Callable[[list[dict[str, str]], int], str]


# ------------------------------------------------------------------ text chunking
def chunk_text(text: str, chunk_chars: int = 1500) -> list[str]:
    """Cut text into pieces of about ``chunk_chars`` along paragraph boundaries. A paragraph longer than the limit is split at
    sentence ends, and a sentence longer than the limit is cut hard."""
    chunk_chars = max(100, int(chunk_chars))
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.replace("\r\n", "\n")) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        if len(p) <= chunk_chars:
            pieces.append(p)
            continue
        for sentence in re.split(r"(?<=[.!?…])\s+", p):
            while len(sentence) > chunk_chars:
                pieces.append(sentence[:chunk_chars])
                sentence = sentence[chunk_chars:]
            if sentence:
                pieces.append(sentence)
    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > chunk_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


# ------------------------------------------------------------------ small readers
def _read_text_file(path: str, data_dir: Optional[Path], allow_data_subdir: Optional[Path] = None) -> str:
    path = paths.clean_user_path(path)                   # a path pasted from the file manager comes with its quotes
    reason = paths.unsafe_file(path, data_dir, allow_data_subdir=allow_data_subdir)
    if reason:
        raise PygmalionError("forbidden", "file_forbidden", path=path, reason=reason)
    p = Path(path)
    if p.stat().st_size > MAX_FILE_BYTES:
        raise PygmalionError("too_large", "file_too_big", name=p.name, mb=MAX_FILE_BYTES // 1048576)
    return p.read_text(encoding="utf-8-sig", errors="replace")


def _inline_or_path(spec: dict[str, Any], data_dir: Optional[Path], allow: Optional[Path]) -> str:
    if isinstance(spec.get("text"), str):
        if len(spec["text"]) > MAX_INLINE_CHARS:
            raise PygmalionError("too_large", "pasted_too_large")
        return spec["text"]
    if spec.get("path"):
        return _read_text_file(str(spec["path"]), data_dir, allow)
    raise PygmalionError("invalid", "source_text_or_path", type=spec.get("type"))


# ------------------------------------------------------------------ source loaders
def load_jsonl(spec: dict[str, Any], ctx: "SourceContext") -> Iterator[dict[str, Any]]:
    text = _inline_or_path(spec, ctx.data_dir, ctx.allow_data_subdir)
    label = spec.get("label") or (Path(paths.clean_user_path(spec["path"])).name if spec.get("path") else "pasted jsonl")
    bad = 0
    stripped = text.strip()
    if stripped.startswith("["):                     # a JSON array of records is accepted too
        try:
            items = json.loads(stripped)
        except ValueError:
            items = None
        if isinstance(items, list):
            for item in items:
                yield with_meta(item, source=label) if isinstance(item, dict) else {}
            return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(item, dict):
            yield with_meta(item, source=label)
        else:
            bad += 1
    if bad:
        ctx.notes.append(msg("note_bad_lines", label=label, bad=bad))


def load_csv(spec: dict[str, Any], ctx: "SourceContext") -> Iterator[dict[str, Any]]:
    text = _inline_or_path(spec, ctx.data_dir, ctx.allow_data_subdir)
    label = spec.get("label") or (Path(paths.clean_user_path(spec["path"])).name if spec.get("path") else "pasted csv")
    columns = spec.get("columns") or {}
    if not columns:
        raise PygmalionError("invalid", "csv_columns_needed")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    missing = [c for c in columns.values() if c not in (reader.fieldnames or [])]
    if missing:
        raise PygmalionError("invalid", "csv_column_missing", missing=missing, columns=list(reader.fieldnames or []))
    for row in reader:
        rec = {field: (row.get(column) or "") for field, column in columns.items()}
        yield with_meta(rec, source=label)


def load_files(spec: dict[str, Any], ctx: "SourceContext") -> Iterator[dict[str, Any]]:
    chunk = int(spec.get("chunk_chars") or ctx.chunk_chars)
    items: list[tuple[str, str]] = []
    for item in spec.get("items") or []:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            items.append((str(item.get("name") or "file"), item["text"]))
    for path in spec.get("paths") or []:
        path = paths.clean_user_path(path)
        items.append((Path(path).name, _read_text_file(path, ctx.data_dir, ctx.allow_data_subdir)))
    if not items:
        raise PygmalionError("invalid", "files_source_needs")
    for name, text in items:
        for piece in chunk_text(text, chunk):
            yield with_meta({"text": piece}, source=name)


def skipped_in_folder(relative: PurePath) -> bool:
    """True for a file found below a source folder that must not be read: inside a hidden or configuration folder (``.git``,
    ``node_modules``...), named like a credential, or itself hidden. ``relative`` is the path below the chosen folder; only its
    parts are looked at (never a "/" in a string), so it is right for Windows paths too. The chosen folder itself was vetted by
    ``paths.unsafe_folder``."""
    return bool(paths.hidden_names(relative.parts[:-1]) or paths.SECRET_NAMES.search(relative.name) or relative.name.startswith("."))


def load_folder(spec: dict[str, Any], ctx: "SourceContext") -> Iterator[dict[str, Any]]:
    folder = paths.clean_user_path(spec.get("path") or "")
    reason = paths.unsafe_folder(folder, ctx.data_dir, allow_data_subdir=ctx.allow_data_subdir)
    if reason:
        raise PygmalionError("forbidden", "folder_forbidden", path=folder, reason=reason)
    extensions = tuple(e.lower() if e.startswith(".") else "." + e.lower() for e in (spec.get("extensions") or TEXT_EXTENSIONS))
    recursive = spec.get("recursive", True)
    root = Path(folder).resolve()
    candidates = root.rglob("*") if recursive else root.glob("*")
    count = 0
    for path in sorted(candidates):
        if path.suffix.lower() not in extensions or not path.is_file():
            continue
        if skipped_in_folder(path.relative_to(root)):
            continue
        count += 1
        if count > MAX_FOLDER_FILES:
            ctx.notes.append(msg("note_folder_many", max=MAX_FOLDER_FILES))
            return
        if path.stat().st_size > MAX_FILE_BYTES:
            ctx.notes.append(msg("note_file_big", name=path.name, mb=MAX_FILE_BYTES // 1048576))
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        rel = path.relative_to(root).as_posix()      # the same label on every OS
        for piece in chunk_text(text, int(spec.get("chunk_chars") or ctx.chunk_chars)):
            yield with_meta({"text": piece}, source=rel)


# ------------------------------------------------------------------ other apps
def dig(value: Any, path: str) -> Any:
    """``a.b.0.c`` into nested dicts and lists; None when a step is missing."""
    for part in [p for p in path.split(".") if p != ""]:
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value


def family_items(result: Any, items_path: str = "") -> list[Any]:
    """The list of items inside a tool result: at ``items_path`` when given, else the result itself or its first list."""
    target = dig(result, items_path) if items_path else result
    if isinstance(target, list):
        return target
    if isinstance(target, dict):
        for value in target.values():
            if isinstance(value, list):
                return value
    return []


def map_item(item: Any, mapping: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field, source in mapping.items():
        value = dig(item, source) if isinstance(item, (dict, list)) else None
        if value is None or value == "":
            return {}
        out[field] = value if isinstance(value, (list, dict)) else str(value)
    return out


def call_family(spec: dict[str, Any], ctx: "SourceContext") -> Any:
    if ctx.family_call is None:
        raise PygmalionError("offline", "family_offline")
    app, tool = str(spec.get("app") or "").strip(), str(spec.get("tool") or "").strip()
    if not app or not tool:
        raise PygmalionError("invalid", "family_source_needs")
    arguments = dict(spec.get("arguments") or {})
    key = json.dumps([app, tool, arguments], sort_keys=True, default=str)
    if key in ctx.family_cache:
        return ctx.family_cache[key]
    reply = ctx.family_call(app, tool, arguments)
    if not isinstance(reply, dict) or not reply.get("ok"):
        why = (reply or {}).get("error") if isinstance(reply, dict) else "no answer"
        raise PygmalionError("offline", "family_no_answer", app=app, tool=tool, why=why)
    ctx.family_cache[key] = reply.get("result")
    return ctx.family_cache[key]


def load_family(spec: dict[str, Any], ctx: "SourceContext") -> Iterator[dict[str, Any]]:
    result = call_family(spec, ctx)
    mapping = spec.get("mapping") or {}
    if not mapping:
        raise PygmalionError("invalid", "family_mapping_needed")
    label = f"{spec['app']}.{spec['tool']}"
    limit = int(spec.get("limit") or 100_000)
    skipped = 0
    for item in family_items(result, str(spec.get("items_path") or ""))[:limit]:
        rec = map_item(item, mapping)
        if not rec:
            skipped += 1
            continue
        yield with_meta(rec, source=label)
    if skipped:
        ctx.notes.append(msg("note_items_unmapped", label=label, skipped=skipped))


# ------------------------------------------------------------------ synthetic data
SYNTH_PROMPT_ES = (
    "Eres un generador de datos de entrenamiento. Tarea: {task}\n\nFragmento de origen:\n\"\"\"\n{chunk}\n\"\"\"\n\n"
    "Genera {n} elementos. Responde SOLO con una lista JSON de objetos con las claves \"prompt\" y \"response\", sin texto antes ni después. "
    "Todo lo que afirmes debe salir del fragmento.")


def parse_json_items(text: str) -> list[dict[str, Any]]:
    """The first JSON array (or object) in a model answer, even inside a ``` fence; [] when there is none."""
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fence:
        body = fence.group(1).strip()
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = body.find(opener), body.rfind(closer)
        if start != -1 and end > start:
            try:
                data = json.loads(body[start:end + 1])
            except ValueError:
                continue
            if isinstance(data, dict):
                data = data.get("items") or data.get("data") or [data]
            return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []
    return []


def synthetic_chunks(spec: dict[str, Any], ctx: "SourceContext") -> list[str]:
    chunks: list[str] = []
    for sub in spec.get("from") or []:
        for rec in load_source(sub, ctx):
            text = content_text(rec).strip() if isinstance(rec, dict) else ""
            if text:
                chunks.append(text)
    return chunks


def estimate_synthetic(spec: dict[str, Any], chunks: int, seconds_per_call: float = 25.0) -> dict[str, Any]:
    per_chunk = max(1, int(spec.get("per_chunk") or 3))
    cap = int(spec.get("max_items") or 200)
    calls = min(chunks, -(-cap // per_chunk))
    return {"chunks": chunks, "calls": calls, "max_items": cap, "estimated_seconds": int(calls * seconds_per_call),
            "note": msg("synthetic_note")}


def load_synthetic(spec: dict[str, Any], ctx: "SourceContext") -> Iterator[dict[str, Any]]:
    if ctx.teacher is None:
        raise PygmalionError("offline", "teacher_missing")
    task = str(spec.get("task") or "").strip()
    if not task:
        raise PygmalionError("invalid", "synthetic_task_needed")
    chunks = synthetic_chunks(spec, ctx)
    plan = estimate_synthetic(spec, len(chunks))
    per_chunk, cap = max(1, int(spec.get("per_chunk") or 3)), plan["max_items"]
    made = 0
    for number, chunk in enumerate(chunks):
        if made >= cap:
            break
        if ctx.cancelled():
            break
        ctx.progress(str(msg("progress_synthetic", n=number + 1, total=len(chunks))), made, cap)
        prompt = SYNTH_PROMPT_ES.format(task=task, chunk=chunk[:6000], n=per_chunk)
        try:
            answer = ctx.teacher([{"role": "user", "content": prompt}], 1500)
        except Exception as exc:  # noqa: BLE001 — one failed call must not lose the items already made
            ctx.notes.append(msg("note_teacher_failed", chunk=number + 1, detail=exc))
            continue
        for item in parse_json_items(answer)[:per_chunk]:
            rec = {"prompt": str(item.get("prompt") or item.get("question") or "").strip(),
                   "response": str(item.get("response") or item.get("answer") or "").strip()}
            if rec["prompt"] and rec["response"]:
                made += 1
                yield with_meta(rec, source=f"synthetic #{number + 1}", synthetic=True, status="pending")
                if made >= cap:
                    break


# ------------------------------------------------------------------ dispatcher
class SourceContext:
    """What the loaders may use: folders, the family call, the teacher model, a progress callback and a place for notes."""

    def __init__(self, *, data_dir: Optional[Path] = None, allow_data_subdir: Optional[Path] = None, chunk_chars: int = 1500,
                 family_call: Optional[FamilyCall] = None, teacher: Optional[Teacher] = None,
                 progress: Optional[Callable[[str, int, int], None]] = None, cancelled: Optional[Callable[[], bool]] = None):
        self.data_dir = data_dir
        self.allow_data_subdir = allow_data_subdir
        self.chunk_chars = chunk_chars
        self.family_call = family_call
        self.teacher = teacher
        self._progress = progress
        self._cancelled = cancelled
        self.notes: list[str] = []
        self.family_cache: dict[str, Any] = {}      # one call per (app, tool, arguments) for the life of a preview or build

    def progress(self, message: str, step: int, total: int) -> None:
        if self._progress:
            self._progress(message, step, total)

    def cancelled(self) -> bool:
        return bool(self._cancelled and self._cancelled())


LOADERS = {"jsonl": load_jsonl, "csv": load_csv, "files": load_files, "folder": load_folder, "family": load_family, "synthetic": load_synthetic}


def load_source(spec: dict[str, Any], ctx: SourceContext) -> Iterator[dict[str, Any]]:
    kind = str(spec.get("type") or "").strip()
    loader = LOADERS.get(kind)
    if loader is None:
        raise PygmalionError("invalid", "source_type_unknown", type=kind, options=list(LOADERS))
    return loader(spec, ctx)


def preview_source(spec: dict[str, Any], ctx: SourceContext, limit: int = 5) -> dict[str, Any]:
    """What a source would yield: the first ``limit`` records, the detected kind and any notes. For a synthetic source it only
    estimates the work (it does not call the teacher) unless ``spec["sample"]`` is true, which runs one call."""
    kind = str(spec.get("type") or "")
    if kind == "synthetic" and not spec.get("sample"):
        chunks = synthetic_chunks(spec, ctx)
        return {"type": kind, "items": [], "count": None, "kind": "instruction", "notes": ctx.notes, "estimate": estimate_synthetic(spec, len(chunks))}
    items, total_seen = [], 0
    if kind == "synthetic":
        spec = {**spec, "max_items": min(int(spec.get("max_items") or limit), limit)}
    for rec in load_source(spec, ctx):
        total_seen += 1
        norm = normalize_record(rec, str(spec.get("kind") or "") or None)
        if len(items) < limit and norm is not None:
            items.append(norm)
        if total_seen >= 20_000 and kind != "family":
            break
    detected = detect_kind(items[0]) if items else None
    raw = None
    if kind == "family":
        raw = call_family(spec, ctx)
        raw = {"first_items": family_items(raw, str(spec.get("items_path") or ""))[:limit]}
    return {"type": kind, "items": items, "count": total_seen, "kind": detected, "notes": ctx.notes, "raw": raw}
