"""Artifacts and how they derive from each other: bases, adapters, merges, GGUF files, importance matrices, published tags and
context variants. Every artifact keeps its parents, the dataset version, the job and a recipe that is enough to reproduce it."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

from .errors import PygmalionError
from .gguf_meta import read_metadata, summarize
from .store import ARTIFACT_KINDS, Store
from .util import dir_size

COLUMNS = {"base": 0, "adapter": 1, "merged": 2, "ctx_variant": 2, "gguf": 3, "imatrix": 3, "ollama": 4}
KIND_LABELS = {"base": "Base", "adapter": "LoRA", "merged": "Fusión", "gguf": "GGUF", "imatrix": "Matriz de importancia",
               "ollama": "Ollama", "ctx_variant": "Contexto"}
NODE_W, NODE_H, COL_GAP, ROW_GAP = 178, 46, 52, 14
VERDICT_ORDER = ("better", "no_clear_difference", "worse")


def portable(value: Any) -> Any:
    """Recipe values without machine-specific absolute paths: a path becomes its last component."""
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(v) for v in value]
    if isinstance(value, str) and (re.match(r"^[A-Za-z]:[\\/]", value) or (value.startswith("/") and "/" in value[1:])) and len(value) > 3:
        return re.split(r"[\\/]", value.rstrip("\\/"))[-1]
    return value


def is_quantized(artifact: dict[str, Any]) -> bool:
    if artifact["kind"] != "gguf":
        return False
    qtype = ((artifact.get("recipe") or {}).get("params") or {}).get("qtype") or (artifact.get("metrics") or {}).get("quant")
    return bool(qtype)


def ppl_of(artifact: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The stored perplexity measurement of an artifact (``value``, ``error`` and the text, context and chunks it was measured with)."""
    ppl = (artifact.get("metrics") or {}).get("ppl")
    return ppl if isinstance(ppl, dict) and isinstance(ppl.get("value"), (int, float)) else None


def same_measure(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Two perplexity numbers can be compared only when they come from the same text, context and chunk count."""
    return (a.get("ctx"), a.get("chunks"), a.get("text_tag") or a.get("text")) == (b.get("ctx"), b.get("chunks"), b.get("text_tag") or b.get("text"))


def qtype_of(artifact: dict[str, Any]) -> str:
    return str(((artifact.get("recipe") or {}).get("params") or {}).get("qtype") or (artifact.get("metrics") or {}).get("quant") or "").upper()


def verdict_of(artifact: dict[str, Any]) -> Optional[str]:
    galton = (artifact.get("metrics") or {}).get("galton") or {}
    return galton.get("verdict")


class Lineage:
    def __init__(self, store: Store):
        self.store = store

    # ------------------------------------------------------------------ creating
    def ensure_base(self, path: str | Path, name: str = "", recipe: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """The ``base`` artifact of a Hugging Face folder, created on first use."""
        folder = Path(path)
        existing = self.store.find_artifact_by_path(str(folder))
        if existing:
            return existing
        if not (folder / "config.json").is_file():
            raise PygmalionError("invalid", "not_model_folder", folder=folder)
        return self.store.create_artifact("base", name or folder.name, path=str(folder), size=dir_size(folder),
                                          recipe=recipe or {"source": "local", "folder": folder.name})

    def resolve(self, ref: str, kinds: Optional[tuple[str, ...]] = None) -> dict[str, Any]:
        """An artifact from its id or name, or from a path (a local model folder becomes a base artifact on the fly)."""
        ref = (ref or "").strip()
        if not ref:
            raise PygmalionError("invalid", "artifact_ref_needed")
        try:
            found = self.store.artifact(ref)
        except PygmalionError:
            found = self.store.find_artifact_by_path(ref)
            if found is None and os.path.isabs(ref) and Path(ref).is_dir() and (Path(ref) / "config.json").is_file():
                found = self.ensure_base(ref)
            if found is None:
                raise
        if kinds and found["kind"] not in kinds:
            raise PygmalionError("invalid", "artifact_wrong_kind", name=found["name"], kind=found["kind"], options=list(kinds))
        return found

    def add_metrics(self, aid: str, key: str, value: Any) -> dict[str, Any]:
        return self.store.merge_metrics(aid, key, value)

    def mark_published(self, aid: str, entry: dict[str, Any]) -> dict[str, Any]:
        art = self.store.artifact(aid)
        published = [p for p in art["published"] if not (p.get("type") == entry.get("type") and p.get("name") == entry.get("name"))]
        published.append(entry)
        return self.store.update_artifact(aid, published=published)

    def unmark_published(self, aid: str, type_: str, name: str) -> dict[str, Any]:
        art = self.store.artifact(aid)
        return self.store.update_artifact(aid, published=[p for p in art["published"] if not (p.get("type") == type_ and p.get("name") == name)])

    # ------------------------------------------------------------------ walking
    def ancestors(self, aid: str) -> list[dict[str, Any]]:
        """Parents, grandparents... nearest first, each once."""
        seen: dict[str, dict[str, Any]] = {}
        queue = [aid]
        while queue:
            current = queue.pop(0)
            try:
                art = self.store.artifact(current)
            except PygmalionError:
                continue
            for parent in art["parents"]:
                if parent not in seen and parent != aid:
                    try:
                        seen[parent] = self.store.artifact(parent)
                    except PygmalionError:
                        continue
                    queue.append(parent)
        return list(seen.values())

    def descendants(self, aid: str) -> list[dict[str, Any]]:
        everything = self.store.artifacts(limit=5000)
        out: dict[str, dict[str, Any]] = {}
        frontier = {aid}
        while frontier:
            nxt = set()
            for art in everything:
                if art["id"] not in out and frontier & set(art["parents"]):
                    out[art["id"]] = art
                    nxt.add(art["id"])
            frontier = nxt
        return list(out.values())

    def nearest_comparable(self, aid: str) -> Optional[dict[str, Any]]:
        """The closest ancestor a evaluation can use as the reference: a GGUF file or a published Ollama tag."""
        for art in self.ancestors(aid):
            if art["kind"] in ("gguf", "ollama"):
                return art
        return None

    def gguf_summary(self, art: dict[str, Any]) -> Optional[dict[str, Any]]:
        if art["kind"] != "gguf" or not Path(art["path"]).is_file():
            return None
        try:
            return summarize(read_metadata(art["path"]))
        except PygmalionError:
            return None

    # ------------------------------------------------------------------ views
    def card(self, art: dict[str, Any]) -> dict[str, Any]:
        galton = (art.get("metrics") or {}).get("galton") or {}
        training = (art.get("metrics") or {}).get("training") or {}
        return {"id": art["id"], "kind": art["kind"], "name": art["name"], "path": art["path"], "size": art["size"], "parents": art["parents"],
                "dataset_version": art["dataset_version"], "job_id": art["job_id"], "pinned": art["pinned"], "notes": art["notes"],
                "created_ts": art["created_ts"], "published": art["published"], "exists": bool(art["path"]) and Path(art["path"]).exists() if art["kind"] != "ollama" else None,
                "verdict": galton.get("verdict"), "ppl": (art.get("metrics") or {}).get("ppl"), "quant": (art.get("metrics") or {}).get("quant"),
                "final_loss": training.get("final_loss"), "eval_loss": training.get("final_eval_loss")}

    # ------------------------------------------------------------------ perplexity tables
    @staticmethod
    def _plain_root(art: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> Optional[str]:
        """The base model's id when ``art`` is that base model's own GGUF (base, then a GGUF, then quantizations of it: no dataset, adapter, merge
        or context variant in between), else None."""
        current = art
        for _ in range(8):
            if current["kind"] == "base":
                return current["id"]
            if current["kind"] != "gguf" or current.get("dataset_version") or ((current.get("recipe") or {}).get("params") or {}).get("adapter"):
                return None
            parents = [by_id[p] for p in current["parents"] if p in by_id and by_id[p]["kind"] in ("gguf", "base")]
            if not parents:
                return None
            current = parents[0]
        return None

    def _root_base(self, art: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> Optional[str]:
        seen = set()
        stack = [art["id"]]
        while stack:
            cur = by_id.get(stack.pop())
            if cur is None or cur["id"] in seen:
                continue
            seen.add(cur["id"])
            if cur["kind"] == "base":
                return cur["id"]
            stack.extend(cur["parents"])
        return None

    def annotate_ppl(self, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add ``ppl_cmp`` to the cards that have a perplexity: the change against the nearest ancestor measured the same way (``parent``: how much
        the quantization or the merge moved it) and against the base model's own file with the same quantization (``base``: what the fine-tune
        did). Differences are ``value - other``, so a positive number is worse."""
        measured = [c for c in cards if c.get("ppl")]
        if not measured:
            return cards
        everything = {a["id"]: a for a in self.store.artifacts(limit=5000)}
        for card in measured:
            art = everything.get(card["id"])
            if art is None:
                continue
            mine = ppl_of(art)
            if mine is None:
                continue
            cmp: dict[str, Any] = {}
            queue, seen = list(art["parents"]), set()
            while queue:
                parent = everything.get(queue.pop(0))
                if parent is None or parent["id"] in seen:
                    continue
                seen.add(parent["id"])
                theirs = ppl_of(parent)
                if theirs is not None and same_measure(mine, theirs):
                    cmp["parent"] = {"id": parent["id"], "name": parent["name"], "value": theirs["value"], "delta": round(mine["value"] - theirs["value"], 4),
                                     "pct": round(100.0 * (mine["value"] - theirs["value"]) / theirs["value"], 2) if theirs["value"] else None}
                    break
                queue.extend(parent["parents"])
            root = self._root_base(art, everything)
            if root and self._plain_root(art, everything) is None:
                for other in everything.values():
                    if other["id"] == art["id"] or other["kind"] != "gguf" or qtype_of(other) != qtype_of(art):
                        continue
                    theirs = ppl_of(other)
                    if theirs is None or not same_measure(mine, theirs) or self._plain_root(other, everything) != root:
                        continue
                    cmp["base"] = {"id": other["id"], "name": other["name"], "value": theirs["value"], "delta": round(mine["value"] - theirs["value"], 4),
                                   "pct": round(100.0 * (mine["value"] - theirs["value"]) / theirs["value"], 2) if theirs["value"] else None}
                    break
            if cmp:
                card["ppl_cmp"] = cmp
        return cards

    def ppl_family(self, art: dict[str, Any]) -> list[dict[str, Any]]:
        """Every GGUF of the artifact's family that has a perplexity measured like its own, as cards with ``ppl_cmp``: the table that shows the
        f16 file, each quantization and the base's own files side by side."""
        everything = {a["id"]: a for a in self.store.artifacts(limit=5000)}
        root = self._root_base(art, everything)
        mine = ppl_of(art)
        rows = []
        for other in everything.values():
            if other["kind"] != "gguf" or ppl_of(other) is None:
                continue
            if root and self._root_base(other, everything) != root:
                continue
            if mine is not None and not same_measure(mine, ppl_of(other)):
                continue
            rows.append(self.card(other))
        rows.sort(key=lambda c: (bool(c["dataset_version"]), c["created_ts"]))
        return self.annotate_ppl(rows)

    def detail(self, ref: str) -> dict[str, Any]:
        art = self.resolve(ref)
        dataset = None
        if art.get("dataset_version"):
            try:
                v = self.store.version(art["dataset_version"])
                dataset = {"version": v["id"], "n": v["n"], "dataset": self.store.dataset(v["dataset_id"])["name"], "records": v["records"],
                           "sha256": v["sha256"], "tokens": v["tokens"]}
            except PygmalionError:
                dataset = {"version": art["dataset_version"], "missing": True}
        return {**self.card(art), "recipe": art["recipe"], "metrics": art["metrics"], "dataset": dataset,
                "ancestors": [self.card(a) for a in self.ancestors(art["id"])],
                "descendants": [self.card(a) for a in self.descendants(art["id"])],
                "comparable_parent": (lambda p: self.card(p) if p else None)(self.nearest_comparable(art["id"])),
                "ppl_family": self.ppl_family(art) if art["kind"] == "gguf" else []}

    def graph(self, root: Optional[str] = None, kinds: Optional[list[str]] = None) -> dict[str, Any]:
        """Nodes laid out in columns by kind, and the edges between them. ``root`` limits it to one artifact's family."""
        arts = self.store.artifacts(limit=5000)
        if root:
            focus = self.resolve(root)
            keep = {focus["id"]} | {a["id"] for a in self.ancestors(focus["id"])} | {a["id"] for a in self.descendants(focus["id"])}
            arts = [a for a in arts if a["id"] in keep]
        if kinds:
            arts = [a for a in arts if a["kind"] in kinds]
        arts.sort(key=lambda a: (COLUMNS.get(a["kind"], 5), a["created_ts"]))
        rows: dict[int, int] = {}
        nodes = []
        ids = {a["id"] for a in arts}
        for a in arts:
            col = COLUMNS.get(a["kind"], 5)
            row = rows.get(col, 0)
            rows[col] = row + 1
            nodes.append({**self.card(a), "col": col, "row": row, "x": col * (NODE_W + COL_GAP), "y": row * (NODE_H + ROW_GAP),
                          "w": NODE_W, "h": NODE_H, "label": KIND_LABELS.get(a["kind"], a["kind"])})
        edges = [{"from": p, "to": a["id"]} for a in arts for p in a["parents"] if p in ids]
        width = (max(COLUMNS.values()) + 1) * (NODE_W + COL_GAP) - COL_GAP
        height = max([n["y"] + NODE_H for n in nodes], default=NODE_H)
        return {"nodes": nodes, "edges": edges, "width": width, "height": height, "columns": [{"kind": k, "col": c, "label": KIND_LABELS.get(k, k)}
                                                                                       for k, c in sorted(COLUMNS.items(), key=lambda kv: kv[1])]}

    def recipe(self, ref: str) -> dict[str, Any]:
        """Everything needed to reproduce an artifact, oldest step first. Machine-specific paths are reduced to names."""
        art = self.resolve(ref)
        chain = self._oldest_first(art)
        steps = []
        for a in chain:
            step: dict[str, Any] = {"artifact": a["id"], "kind": a["kind"], "name": a["name"], "recipe": portable(a["recipe"])}
            if a.get("dataset_version"):
                try:
                    v = self.store.version(a["dataset_version"])
                    step["dataset"] = {"name": self.store.dataset(v["dataset_id"])["name"], "version": v["n"], "sha256": v["sha256"],
                                       "records": v["records"], "recipe": portable(v["recipe"])}
                except PygmalionError:
                    step["dataset"] = {"missing": a["dataset_version"]}
            metrics = a["metrics"]
            kept = {k: metrics[k] for k in ("training", "ppl", "quant", "galton") if k in metrics}
            if kept:
                step["metrics"] = portable(kept)
            steps.append(step)
        return {"format": "pygmalion-recipe/1", "target": art["id"], "name": art["name"], "steps": steps}

    def _oldest_first(self, art: dict[str, Any]) -> list[dict[str, Any]]:
        """The artifact and its ancestors with every parent before its children (a depth-first post-order over the parents)."""
        order: list[dict[str, Any]] = []
        seen: set[str] = set()

        def visit(current: dict[str, Any]) -> None:
            if current["id"] in seen:
                return
            seen.add(current["id"])
            for parent in current["parents"]:
                try:
                    visit(self.store.artifact(parent))
                except PygmalionError:
                    continue
            order.append(current)

        visit(art)
        return order

    # ------------------------------------------------------------------ editing
    def update(self, ref: str, *, name: Optional[str] = None, notes: Optional[str] = None, pinned: Optional[bool] = None) -> dict[str, Any]:
        art = self.resolve(ref)
        fields: dict[str, Any] = {}
        if name is not None:
            if not name.strip():
                raise PygmalionError("invalid", "name_empty")
            fields["name"] = name.strip()
        if notes is not None:
            fields["notes"] = notes
        if pinned is not None:
            fields["pinned"] = pinned
        return self.store.update_artifact(art["id"], **fields)

    def delete(self, ref: str, delete_files: bool = False, within: Optional[Path] = None) -> dict[str, Any]:
        """Remove the record; with ``delete_files`` also its files, but only inside ``within`` (the work folder) and never a base model."""
        art = self.resolve(ref)
        if art["published"]:
            raise PygmalionError("conflict", "artifact_published", name=art["name"], where=[p.get("name", "?") for p in art["published"]])
        children = self.store.children(art["id"])
        removed = 0
        if delete_files and art["path"] and art["kind"] not in ("base", "ollama"):
            target = Path(art["path"])
            if within is not None:
                try:
                    inside = target.resolve().is_relative_to(Path(within).resolve()) and target.resolve() != Path(within).resolve()
                except OSError:
                    inside = False
                if not inside:
                    raise PygmalionError("forbidden", "files_outside_work", path=target)
            try:
                if target.is_file():
                    target.unlink()
                    removed = 1
                elif target.is_dir():
                    import shutil
                    shutil.rmtree(target)
                    removed = 1
            except OSError as exc:
                raise PygmalionError("failed", "delete_failed", path=target, detail=str(exc)) from exc
        self.store.delete_artifact(art["id"])
        return {"deleted": art["name"], "id": art["id"], "files_removed": bool(removed), "children_left_orphaned": [c["id"] for c in children]}

    def storage(self) -> dict[str, Any]:
        total, by_kind = 0, {}
        for a in self.store.artifacts(limit=5000):
            total += a["size"] or 0
            by_kind[a["kind"]] = by_kind.get(a["kind"], 0) + (a["size"] or 0)
        return {"total": total, "by_kind": by_kind}


__all__ = ["Lineage", "ARTIFACT_KINDS", "is_quantized", "verdict_of", "portable"]
