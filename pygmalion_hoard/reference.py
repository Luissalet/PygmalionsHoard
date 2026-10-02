"""What a result is measured against.

A question such as "did the training help or hurt?" is answered by comparing the result with the model the training started from, written
in the same format: the same quantization type and, when the result used an importance matrix, a matrix computed on the same calibration
text. The nearest GGUF ancestor of a fine-tuned file is the f16 file of the *same* fine-tune, which measures the quantization and not the
training, so it is the reference only for a result that has no training behind it (a plain quantization) or when the caller asks for it.

========================  ==============================================================================================
intent                    reference
========================  ==============================================================================================
``against`` given         that artifact, always
``quant`` (the default    the nearest GGUF ancestor (the unquantized file the result was quantized from)
of a plain quantization)
``dataset``, ``style``,   the base model the training started from, as a GGUF with the result's quantization; built (and kept in
``writing``, ``code``,    the lineage, below the base model) when it does not exist yet
``general``, ``smoke``
``context``               the model before the context extension, built the same way
========================  ==============================================================================================

This module decides and describes; the steps it returns are queued by the caller (``Operations.evaluate_start`` or a training pipeline),
because only a job can convert and quantize.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from . import calib
from .errors import PygmalionError
from .lineage import Lineage, is_quantized, qtype_of
from .messages import CodedText, text
from .store import Store

BASE_INTENTS = ("dataset", "style", "writing", "code", "general", "smoke")
BASE_KINDS = ("base", "merged", "ctx_variant")


def short(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def params_of(art: dict[str, Any]) -> dict[str, Any]:
    return (art.get("recipe") or {}).get("params") or {}


def has_file(art: dict[str, Any]) -> bool:
    return bool(art.get("path")) and Path(art["path"]).is_file()


def quant_label(spec: dict[str, Any]) -> str:
    return str(spec.get("qtype") or spec.get("outtype") or "f16").upper()


class Reference:
    def __init__(self, store: Store, lineage: Lineage):
        self.store = store
        self.lineage = lineage

    # ------------------------------------------------------------------ what the result is made of
    def backing_gguf(self, art: dict[str, Any]) -> Optional[dict[str, Any]]:
        """The GGUF file behind an artifact: itself, or the file a published tag was made from."""
        if art["kind"] == "gguf":
            return art
        for parent in art["parents"]:
            try:
                found = self.store.artifact(parent)
            except PygmalionError:
                continue
            if found["kind"] == "gguf" and not params_of(found).get("adapter"):
                return found
        return None

    def matrix_of(self, quantized: dict[str, Any]) -> Optional[dict[str, Any]]:
        """The importance matrix a quantized file was made with, as ``{spec, tag, chunks}``."""
        for parent in quantized["parents"]:
            try:
                art = self.store.artifact(parent)
            except PygmalionError:
                continue
            if art["kind"] != "imatrix":
                continue
            recipe = params_of(art)
            spec = recipe.get("calibration") if isinstance(recipe.get("calibration"), dict) else {"source": "bundled"}
            info = (art.get("metrics") or {}).get("calibration") or {}
            chunks = recipe.get("chunks") or (art.get("metrics") or {}).get("chunks")
            return {"spec": spec, "tag": str(info.get("tag") or calib.tag_of(spec)), "chunks": int(chunks) if isinstance(chunks, (int, float)) and chunks else None}
        return None

    def spec_of(self, art: dict[str, Any]) -> dict[str, Any]:
        """How a result is written: ``{qtype, outtype, imatrix}``. A quantized file takes its ``outtype`` from the file it was quantized from."""
        gguf = self.backing_gguf(art)
        if gguf is None:
            return {"qtype": "", "outtype": "f16", "imatrix": None}
        if is_quantized(gguf):
            outtype = "f16"
            for ancestor in self.lineage.ancestors(gguf["id"]):
                if ancestor["kind"] == "gguf" and not is_quantized(ancestor):
                    outtype = str(params_of(ancestor).get("outtype") or "f16")
                    break
            return {"qtype": qtype_of(gguf), "outtype": outtype, "imatrix": self.matrix_of(gguf)}
        return {"qtype": "", "outtype": str(params_of(gguf).get("outtype") or "f16"), "imatrix": None}

    # ------------------------------------------------------------------ where the result came from
    def training_base(self, art: dict[str, Any]) -> tuple[bool, Optional[dict[str, Any]]]:
        """``(trained, base)``: whether the artifact has a training behind it, and the model the nearest training started from."""
        chain = [art, *self.lineage.ancestors(art["id"])]
        adapter = next((a for a in chain if a["kind"] == "adapter"), None)
        if adapter is not None:
            for parent in adapter["parents"]:
                try:
                    found = self.store.artifact(parent)
                except PygmalionError:
                    continue
                if found["kind"] in BASE_KINDS:
                    return True, found
            return True, None
        return any(a.get("dataset_version") for a in chain), None

    def extension_origin(self, art: dict[str, Any]) -> Optional[dict[str, Any]]:
        """The model the nearest context extension was made from."""
        for ancestor in [art, *self.lineage.ancestors(art["id"])]:
            if ancestor["kind"] == "ctx_variant":
                for parent in ancestor["parents"]:
                    try:
                        found = self.store.artifact(parent)
                    except PygmalionError:
                        continue
                    if found["kind"] in BASE_KINDS:
                        return found
                return None
        return None

    # ------------------------------------------------------------------ what already exists
    def _derived(self, model_id: str) -> list[dict[str, Any]]:
        """The plain GGUF files of a model (no adapter, no dataset): the base's own files."""
        return [a for a in self.store.children(model_id) if a["kind"] == "gguf" and not params_of(a).get("adapter") and not a.get("dataset_version")]

    def find_unquantized(self, model_id: str, outtype: str = "", must_exist: bool = True) -> Optional[dict[str, Any]]:
        """The model's unquantized GGUF; the one written with ``outtype`` when there are several."""
        found = [a for a in self._derived(model_id) if not is_quantized(a) and (not must_exist or has_file(a))]
        for art in found:
            if outtype and str(params_of(art).get("outtype") or "f16").lower() == outtype.lower():
                return art
        return found[0] if found else None

    def find_matrix(self, f16_id: str, wanted: dict[str, Any], must_exist: bool = True) -> Optional[dict[str, Any]]:
        for art in self.store.children(f16_id):
            if art["kind"] != "imatrix" or (must_exist and not has_file(art)):
                continue
            info = (art.get("metrics") or {}).get("calibration") or {}
            chunks = params_of(art).get("chunks") or (art.get("metrics") or {}).get("chunks")
            if str(info.get("tag") or "") == wanted["tag"] and (not wanted.get("chunks") or not chunks or int(chunks) == int(wanted["chunks"])):
                return art
        return None

    def find_quantized(self, model_id: str, qtype: str, matrix_tag: Optional[str], must_exist: bool = True) -> Optional[dict[str, Any]]:
        """The model's file quantized to ``qtype``, with an importance matrix of that calibration tag (``None``: without a matrix)."""
        for f16 in self._derived(model_id):
            if is_quantized(f16):
                continue
            for art in self.store.children(f16["id"]):
                if art["kind"] != "gguf" or not is_quantized(art) or qtype_of(art) != qtype.upper() or (must_exist and not has_file(art)):
                    continue
                matrix = self.matrix_of(art)
                if (matrix["tag"] if matrix else None) == matrix_tag:
                    return art
        return None

    def find(self, model_id: str, spec: dict[str, Any], must_exist: bool = True) -> Optional[dict[str, Any]]:
        """The base's own file written like ``spec``."""
        if spec.get("qtype"):
            matrix = spec.get("imatrix")
            return self.find_quantized(model_id, spec["qtype"], matrix["tag"] if matrix else None, must_exist)
        return self.find_unquantized(model_id, spec.get("outtype", ""), must_exist)

    # ------------------------------------------------------------------ what has to be built
    def chain(self, model_id: str, spec: dict[str, Any], *, build: bool = True) -> dict[str, Any]:
        """The base's file written like ``spec``: ``{artifact, ref, f16, steps}``. ``artifact`` is the file when it exists; otherwise ``steps`` are
        the pipeline steps that make it (``convert``, ``imatrix``, ``quantize``, reusing whatever already exists), named with ``_as`` so a later
        step reaches the pieces: ``ref`` is the id or ``$name`` of the final file and ``f16`` that of the unquantized one."""
        existing = self.find(model_id, spec)
        qtype = spec.get("qtype") or ""
        f16 = existing if (existing and not qtype) else self.find_unquantized(model_id, spec.get("outtype", ""))
        if existing:
            return {"artifact": existing, "ref": existing["id"], "f16": f16["id"] if f16 else None, "steps": []}
        if not build:
            return {"artifact": None, "ref": None, "f16": f16["id"] if f16 else None, "steps": []}
        steps: list[dict[str, Any]] = []
        if f16 is not None:
            f16_ref = f16["id"]
        else:
            f16_ref = "$baseline_gguf"
            steps.append({"kind": "convert", "title": text("title_baseline_convert"),
                          "params": {"model": model_id, "outtype": spec.get("outtype") or "f16", "_as": "baseline" if not qtype else "baseline_gguf"}})
            if not qtype:
                f16_ref = "$baseline"
        if not qtype:
            return {"artifact": None, "ref": f16_ref, "f16": f16_ref, "steps": steps}
        matrix_ref = None
        wanted = spec.get("imatrix")
        if wanted:
            have = self.find_matrix(f16["id"], wanted) if f16 is not None else None
            if have is not None:
                matrix_ref = have["id"]
            else:
                matrix_ref = "$baseline_imatrix"
                steps.append({"kind": "imatrix", "title": text("title_baseline_imatrix"),
                              "params": {"gguf": f16_ref, "calibration": wanted["spec"], "chunks": wanted.get("chunks"), "_as": "baseline_imatrix"}})
        steps.append({"kind": "quantize", "title": text("title_baseline_quantize", type=qtype),
                      "params": {"gguf": f16_ref, "type": qtype, "imatrix": matrix_ref, "_as": "baseline"}})
        return {"artifact": None, "ref": "$baseline", "f16": f16_ref, "steps": steps}

    # ------------------------------------------------------------------ the decision
    def decide(self, art: dict[str, Any], intent: str = "") -> tuple[str, str, Optional[dict[str, Any]], bool]:
        """``(effective intent, mode, origin model, trained)``: ``origin`` is the model the reference is a file of (``None``: the parent file)."""
        trained, base = self.training_base(art)
        extended = self.extension_origin(art)
        effective = intent or ("context" if extended is not None else "general" if trained else "quant")
        if effective == "context" and extended is not None:
            return effective, "context", extended, trained
        if effective in BASE_INTENTS and trained and base is not None:
            return effective, "base", base, trained
        return effective, "parent", None, trained

    def origin_model(self, art: dict[str, Any], intent: str = "") -> Optional[dict[str, Any]]:
        """The model a pipeline that ends in an evaluation should build its reference from (``None``: compare with the parent file)."""
        return self.decide(art, intent)[2]

    def plan(self, child: dict[str, Any], intent: str = "", against: str = "", *, build: bool = True) -> dict[str, Any]:
        """The reference of ``child`` for ``intent`` (the *effective* intent: ``quant`` when nothing says otherwise and the result has no training).

        Returns ``{mode, intent, model, quant, imatrix, artifact, ref, ready, steps, f16, warnings, message, notes}``. ``mode`` is ``explicit``,
        ``parent``, ``base`` or ``context``; ``artifact`` is the file to compare with when it exists, else ``None`` and ``steps`` make it;
        ``model`` is the name shown; ``origin`` the base model artifact (for ``base`` and ``context``)."""
        warnings: list[CodedText] = []
        if against:
            art = self.lineage.resolve(against, ("gguf", "ollama"))
            return self._answer("explicit", intent, art, art, self.spec_of(art), None, [], None, warnings, text("eval_ref_explicit", model=art["name"]))
        effective, mode, origin, trained = self.decide(child, intent)
        if origin is None and trained and effective in BASE_INTENTS:
            warnings.append(text("eval_ref_unknown_base"))
        if origin is None:
            parent = self.lineage.nearest_comparable(child["id"])
            if parent is None:
                raise PygmalionError("not_found", "no_comparable", name=child["name"])
            return self._answer("parent", effective, parent, parent, self.spec_of(parent), None, [], None, warnings, text("eval_ref_parent", model=parent["name"]))
        spec = self.spec_of(child)
        made = self.chain(origin["id"], spec, build=build)
        label = quant_label(spec)
        ready = made["artifact"] is not None
        key = f"eval_ref_{mode}" + ("" if ready else "_build")
        notes = [text("eval_ref_imatrix")] if spec.get("imatrix") else []
        return self._answer(mode, effective, made["artifact"], origin, spec, made["ref"], made["steps"], made["f16"], warnings,
                            text(key, model=origin["name"], quant=label), notes, label)

    @staticmethod
    def _answer(mode: str, intent: str, artifact: Optional[dict[str, Any]], origin: Optional[dict[str, Any]], spec: dict[str, Any], ref: Optional[str],
                steps: list[dict[str, Any]], f16: Optional[str], warnings: list[CodedText], message: CodedText, notes: Optional[list[CodedText]] = None,
                label: str = "") -> dict[str, Any]:
        if mode in ("explicit", "parent") and artifact is not None:
            ref = artifact["id"]
        return {"mode": mode, "intent": intent, "origin": origin if mode in ("base", "context") else None, "artifact": artifact, "ref": ref,
                "model": (origin or {}).get("name", "") if mode in ("base", "context") else (artifact or {}).get("name", ""),
                "quant": label or quant_label(spec), "imatrix": spec.get("imatrix"), "ready": artifact is not None, "steps": steps, "f16": f16,
                "warnings": warnings, "message": message, "notes": notes or []}

    @staticmethod
    def view(plan: dict[str, Any]) -> dict[str, Any]:
        """The part of a plan the interface and the assistants read."""
        artifact = plan["artifact"]
        return {"mode": plan["mode"], "intent": plan["intent"], "model": plan["model"], "quant": plan["quant"], "imatrix": bool(plan["imatrix"]),
                "ready": plan["ready"], "artifact": {"id": artifact["id"], "name": artifact["name"]} if artifact else None,
                "steps": [s["kind"] for s in plan["steps"]], "message": plan["message"], "notes": plan["notes"], "warnings": plan["warnings"]}
