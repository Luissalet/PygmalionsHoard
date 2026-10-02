"""Datasets as the rest of the app sees them: build from sources, apply operations, review, page through records, export for training.

A version is an immutable JSONL file (``datasets/<id>/v<N>.jsonl``) with its SHA-256, counts, split and the recipe that built it.
Changing anything (an operation, a review) writes a new version that points at its parent, so a training run can always name the
exact data it saw.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional

from ..errors import PygmalionError
from ..messages import text as msg
from ..settings import Settings
from ..store import Store
from ..util import est_tokens, sha256_file, sha256_text
from . import operations as ops
from .operations import preview_text
from .formats import KINDS, content_text, kind_of, map_text, meta_of, normalize_record, status_of, strip_meta, to_kind, with_meta
from .sources import SourceContext, FamilyCall, Teacher, load_source, preview_source

Progress = Callable[[str, int, int], None]
NAME_RE = re.compile(r"^[\w .\-áéíóúüñÁÉÍÓÚÜÑ]{1,80}$")


def sanitize_recipe(spec: dict[str, Any]) -> dict[str, Any]:
    """The build spec without the bulk of inline text (kept as a hash and a length), so recipes stay small and reproducible."""
    out = json.loads(json.dumps(spec, default=str))

    def clean(source: dict[str, Any]) -> None:
        if isinstance(source.get("text"), str):
            source["text_sha256"], source["text_chars"] = sha256_text(source["text"]), len(source["text"])
            source["text"] = None
        for item in source.get("items") or []:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                item["sha256"], item["chars"] = sha256_text(item["text"]), len(item["text"])
                item["text"] = None
        for sub in source.get("from") or []:
            if isinstance(sub, dict):
                clean(sub)

    for source in out.get("sources") or []:
        if isinstance(source, dict):
            clean(source)
    return out


class DatasetService:
    def __init__(self, store: Store, settings: Settings, datasets_dir: Path, *, data_dir: Optional[Path] = None,
                 family_call: Optional[FamilyCall] = None, teacher: Optional[Callable[[], Optional[Teacher]]] = None,
                 clock: Callable[[], float] = time.time):
        self.store = store
        self.settings = settings
        self.dir = Path(datasets_dir)
        self.data_dir = data_dir
        self.family_call = family_call
        self.teacher_factory = teacher
        self.clock = clock
        self._cache: "OrderedDict[tuple[str, int], list[dict[str, Any]]]" = OrderedDict()

    # ------------------------------------------------------------------ plumbing
    def context(self, progress: Optional[Progress] = None, cancelled: Optional[Callable[[], bool]] = None) -> SourceContext:
        return SourceContext(data_dir=self.data_dir, allow_data_subdir=None, chunk_chars=self.settings.int("dataset.chunk_chars"),
                             family_call=self.family_call, teacher=self.teacher_factory() if self.teacher_factory else None,
                             progress=progress, cancelled=cancelled)

    def read_records(self, version: dict[str, Any]) -> list[dict[str, Any]]:
        path = Path(version["path"])
        try:
            key = (str(path), path.stat().st_mtime_ns)
        except OSError as exc:
            raise PygmalionError("not_found", "dataset_file_missing", n=version["n"], path=path) from exc
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        records = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
        self._cache[key] = records
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        return records

    def _write_version(self, ds: dict[str, Any], records: list[dict[str, Any]], *, kind: str, recipe: dict[str, Any],
                       parent: Optional[str], eval_pct: Optional[float] = None, min_eval: Optional[int] = None,
                       seed: Optional[int] = None, report: Optional[list[dict[str, Any]]] = None,
                       notes: Optional[list[str]] = None) -> dict[str, Any]:
        n = self.store.next_version_number(ds["id"])
        folder = self.dir / ds["id"]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"v{n}.jsonl"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        usable = [i for i, r in enumerate(records) if status_of(r) == "ok"]
        splits = ops.make_split(usable, self.settings.float("train.eval_split") if eval_pct is None else eval_pct,
                                self.settings.int("train.eval_min") if min_eval is None else min_eval,
                                self.settings.int("train.seed") if seed is None else seed)
        st = ops.stats(records)
        st["report"] = report or []
        st["notes"] = notes or []
        st["pii"] = ops.pii_scan(records)
        return self.store.add_version(ds["id"], n, str(path), sha256_file(path), kind=kind, records=len(records), usable=len(usable),
                                      chars=st["chars"], tokens=st["tokens"], splits=splits, recipe=recipe, stats=st, parent=parent)

    # ------------------------------------------------------------------ views
    def version_view(self, v: dict[str, Any], detail: bool = False) -> dict[str, Any]:
        usable = v["usable"]
        evals = len(v["splits"].get("eval") or [])
        out = {"id": v["id"], "dataset_id": v["dataset_id"], "n": v["n"], "kind": v["kind"], "records": v["records"], "usable": usable,
               "train": usable - evals, "eval": evals, "chars": v["chars"], "tokens": v["tokens"], "sha256": v["sha256"],
               "parent": v["parent"], "created_ts": v["created_ts"], "path": v["path"]}
        out["pending"] = v["stats"].get("pending", 0)
        out["rejected"] = v["stats"].get("rejected", 0)
        if detail:
            out.update(stats=v["stats"], recipe=v["recipe"], splits={k: v["splits"].get(k) for k in ("seed", "eval_pct", "min_eval")})
        return out

    def dataset_view(self, ds: dict[str, Any], detail: bool = False) -> dict[str, Any]:
        versions = self.store.versions(ds["id"])
        latest = versions[-1] if versions else None
        out = {"id": ds["id"], "name": ds["name"], "description": ds["description"], "kind": ds["kind"], "notes": ds["notes"],
               "created_ts": ds["created_ts"], "updated_ts": ds["updated_ts"], "versions": len(versions),
               "latest": self.version_view(latest) if latest else None}
        if detail:
            out["version_list"] = [self.version_view(v) for v in versions]
        return out

    def list(self) -> list[dict[str, Any]]:
        return [self.dataset_view(d) for d in self.store.datasets()]

    def get(self, ref: str) -> dict[str, Any]:
        return self.dataset_view(self.store.dataset(ref), detail=True)

    def get_version(self, ref: str, n: Optional[int] = None) -> dict[str, Any]:
        return self.version_view(self.store.find_version(ref, n), detail=True)

    # ------------------------------------------------------------------ preview
    def preview(self, source: dict[str, Any], limit: int = 5) -> dict[str, Any]:
        return preview_source(source, self.context(), limit)

    # ------------------------------------------------------------------ build
    def build(self, spec: dict[str, Any], progress: Optional[Progress] = None, cancelled: Optional[Callable[[], bool]] = None) -> dict[str, Any]:
        """Run a build spec: ``{name | dataset, description, kind, sources: [...], operations: [...], split: {eval_pct, min_eval, seed}}``."""
        sources = spec.get("sources") or []
        if not sources:
            raise PygmalionError("invalid", "source_needed")
        ctx = self.context(progress, cancelled)
        raw: list[dict[str, Any]] = []
        for number, source in enumerate(sources):
            if progress:
                progress(str(msg("progress_source", n=number + 1, total=len(sources), type=source.get("type"))), number, len(sources))
            raw.extend(load_source(source, ctx))
            if ctx.cancelled():
                raise PygmalionError("failed", "cancelled")
        report: list[dict[str, Any]] = [{"op": "load", "before": 0, "after": len(raw), "changed": 0}]
        forced = str(spec.get("kind") or "").strip() or None
        if forced and forced not in KINDS:
            raise PygmalionError("invalid", "dataset_kind_unknown", kind=forced, options=list(KINDS))
        normalised = [normalize_record(r) for r in raw]
        valid = [r for r in normalised if r is not None]
        report.append({"op": "validate", "before": len(raw), "after": len(valid), "changed": len(raw) - len(valid)})
        if not valid:
            if ctx.notes:
                raise PygmalionError("invalid", "no_usable_records_notes", note=ctx.notes[0])
            raise PygmalionError("invalid", "no_usable_records")
        kind = forced or self._majority_kind(valid)
        converted = [to_kind(r, kind) for r in valid]
        records = [r for r in converted if r is not None]
        report.append({"op": "convert", "before": len(valid), "after": len(records), "changed": len(valid) - len(records), "kind": kind})
        records, op_report = ops.run_operations(records, spec.get("operations") or [], {"near_dup": self.settings.float("dataset.near_dup")})
        report.extend(op_report)
        records = [with_meta(r, status=status_of(r)) for r in records]
        if not records:
            raise PygmalionError("invalid", "ops_removed_all")
        ds = self._dataset_for(spec, kind)
        split = spec.get("split") or {}
        version = self._write_version(ds, records, kind=kind, recipe={"build": sanitize_recipe(spec)}, parent=None, eval_pct=split.get("eval_pct"),
                                      min_eval=split.get("min_eval"), seed=split.get("seed"), report=report, notes=ctx.notes)
        return {"dataset": self.dataset_view(self.store.dataset(ds["id"])), "version": self.version_view(version, detail=True), "notes": ctx.notes}

    @staticmethod
    def _majority_kind(records: list[dict[str, Any]]) -> str:
        counts: dict[str, int] = {}
        for r in records:
            counts[kind_of(r)] = counts.get(kind_of(r), 0) + 1
        # instruction converts to chat for free, so a mix of the two is a chat dataset
        if "instruction" in counts and "chat" in counts and "text" not in counts:
            return "chat"
        return max(counts, key=lambda k: counts[k])

    def _dataset_for(self, spec: dict[str, Any], kind: str) -> dict[str, Any]:
        if spec.get("dataset"):
            ds = self.store.dataset(str(spec["dataset"]))
            if ds["kind"] != kind and self.store.versions(ds["id"]):
                raise PygmalionError("invalid", "dataset_kind_mismatch", name=ds["name"], have=ds["kind"], kind=kind)
            return ds
        name = str(spec.get("name") or "").strip()
        if not name or not NAME_RE.match(name):
            raise PygmalionError("invalid", "dataset_name_needed")
        return self.store.create_dataset(name, kind, str(spec.get("description") or ""))

    # ------------------------------------------------------------------ operations on an existing version
    def apply_operations(self, ref: str, operations: list[dict[str, Any]], n: Optional[int] = None) -> dict[str, Any]:
        base = self.store.find_version(ref, n)
        ds = self.store.dataset(base["dataset_id"])
        records = list(self.read_records(base))
        records, report = ops.run_operations(records, operations, {"near_dup": self.settings.float("dataset.near_dup")})
        if not records:
            raise PygmalionError("invalid", "ops_removed_all")
        version = self._write_version(ds, records, kind=base["kind"], recipe={"operations": operations, "from_version": base["id"]},
                                      parent=base["id"], report=report)
        return {"version": self.version_view(version, detail=True), "report": report}

    # ------------------------------------------------------------------ records and review
    def records(self, ref: str, n: Optional[int] = None, *, offset: int = 0, limit: int = 50, status: str = "", text: str = "",
                source: str = "", synthetic: Optional[bool] = None, split: str = "", lang: str = "", min_chars: int = 0,
                max_chars: int = 0, has_pii: Optional[bool] = None) -> dict[str, Any]:
        version = self.store.find_version(ref, n)
        records = self.read_records(version)
        held = set(version["splits"].get("eval") or [])
        needle = text.lower()
        rows = []
        for i, rec in enumerate(records):
            st = status_of(rec)
            if status and st != status:
                continue
            if synthetic is not None and bool(meta_of(rec).get("synthetic")) != synthetic:
                continue
            if source and source.lower() not in str(meta_of(rec).get("source", "")).lower():
                continue
            if split:
                current = "none" if st != "ok" else ("eval" if i in held else "train")
                if split != current:
                    continue
            body = content_text(rec)
            if needle and needle not in body.lower():
                continue
            if min_chars and len(body) < min_chars or max_chars and len(body) > max_chars:
                continue
            if lang and ops.detect_language(body)[0] != lang:
                continue
            if has_pii is not None and bool(ops.pii.scan(ops.record_text(rec))) != has_pii:
                continue
            rows.append(i)
        page = rows[max(0, offset): max(0, offset) + max(1, min(limit, 500))]
        view = []
        for i in page:
            rec = records[i]
            st = status_of(rec)
            view.append({"index": i, "kind": kind_of(rec), "status": st, "source": meta_of(rec).get("source", ""),
                         "synthetic": bool(meta_of(rec).get("synthetic")), "chars": len(content_text(rec)),
                         "split": "none" if st != "ok" else ("eval" if i in held else "train"), "preview": preview_text(rec),
                         "record": strip_meta(rec)})
        return {"version": version["id"], "n": version["n"], "total": len(rows), "offset": offset, "limit": limit, "records": view}

    def review(self, ref: str, n: Optional[int] = None, *, accept: Optional[list[int]] = None, reject: Optional[list[int]] = None,
               edit: Optional[dict[Any, dict[str, Any]]] = None, accept_all_pending: bool = False) -> dict[str, Any]:
        """Accept, reject or edit records of a version; the result is a new version (the old one stays as it was)."""
        base = self.store.find_version(ref, n)
        ds = self.store.dataset(base["dataset_id"])
        records = [dict(r) for r in self.read_records(base)]
        total = len(records)
        edits = {int(k): v for k, v in (edit or {}).items()}
        for label, group in (("accept", accept or []), ("reject", reject or []), ("edit", list(edits))):
            bad = [i for i in group if not 0 <= int(i) < total]
            if bad:
                raise PygmalionError("invalid", "record_out_of_range", label=label, index=bad[0], last=total - 1)
        changed = {"accepted": 0, "rejected": 0, "edited": 0}
        for i in accept or []:
            records[i] = with_meta(records[i], status="ok")
            changed["accepted"] += 1
        if accept_all_pending:
            for i, r in enumerate(records):
                if status_of(r) == "pending":
                    records[i] = with_meta(r, status="ok")
                    changed["accepted"] += 1
        for i in reject or []:
            records[i] = with_meta(records[i], status="rejected")
            changed["rejected"] += 1
        for i, new in edits.items():
            clean = normalize_record({**new, "_meta": records[i].get("_meta") or {}}, kind_of(records[i]))
            if clean is None:
                raise PygmalionError("invalid", "record_edit_invalid", index=i, kind=kind_of(records[i]))
            records[i] = with_meta(clean, edited=True)
            changed["edited"] += 1
        version = self._write_version(ds, records, kind=base["kind"], recipe={"review": {k: v for k, v in changed.items()}, "from_version": base["id"]},
                                      parent=base["id"], eval_pct=base["splits"].get("eval_pct"), min_eval=base["splits"].get("min_eval"),
                                      seed=base["splits"].get("seed"))
        return {"version": self.version_view(version, detail=True), "changed": changed}

    # ------------------------------------------------------------------ export and calibration text
    def export_for_training(self, ref: str, dest: Path, n: Optional[int] = None) -> dict[str, Any]:
        """Write ``train.jsonl`` and ``eval.jsonl`` (usable records only, bookkeeping stripped) into ``dest``."""
        version = self.store.find_version(ref, n)
        records = self.read_records(version)
        usable = [i for i, r in enumerate(records) if status_of(r) == "ok"]
        train_idx, eval_idx = ops.split_indices(usable, version["splits"])
        dest.mkdir(parents=True, exist_ok=True)
        for name, idx in (("train", train_idx), ("eval", eval_idx)):
            with open(dest / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
                for i in idx:
                    fh.write(json.dumps(strip_meta(records[i]), ensure_ascii=False) + "\n")
        return {"kind": version["kind"], "version": version["id"], "n_train": len(train_idx), "n_eval": len(eval_idx),
                "train": str(dest / "train.jsonl"), "eval": str(dest / "eval.jsonl"), "tokens": version["tokens"]}

    def delete(self, ref: str) -> dict[str, Any]:
        ds = self.store.dataset(ref)
        in_use = [a["name"] for a in self.store.artifacts(limit=5000) if a.get("dataset_version") and
                  a["dataset_version"] in {v["id"] for v in self.store.versions(ds["id"])}]
        folder = self.dir / ds["id"]
        self.store.delete_dataset(ds["id"])
        if folder.is_dir():
            for f in folder.glob("*"):
                try:
                    f.unlink()
                except OSError:
                    pass
            try:
                folder.rmdir()
            except OSError:
                pass
        self._cache.clear()
        return {"deleted": ds["name"], "still_referenced_by": in_use}
