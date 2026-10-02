"""Turning a request into jobs: the training plan (recommended parameters, memory and time estimates, which allowed GPU fits) and the
"full recipe" pipelines that follow a training, a merge, a context extension or a conversion with the next steps."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

from . import calib
from . import llama_tools as LT
from . import vram as V
from .datasets.operations import split_indices
from .errors import PygmalionError
from .gpus import GpuManager
from .lineage import Lineage
from .messages import text
from .reference import Reference
from .settings import Settings
from .store import Store
from .util import read_json

SEQ_STEPS = (4096, 2048, 1536, 1024, 768, 512, 384, 256)
TRAIN_KEYS = ("method", "rank", "alpha", "dropout", "lr", "seq_len", "batch", "grad_accum", "epochs", "max_steps", "warmup", "weight_decay",
              "eval_every", "save_every", "seed", "target_modules", "train_on", "gradient_checkpointing")


class Planner:
    def __init__(self, store: Store, lineage: Lineage, settings: Settings, gpus: GpuManager, references: Optional[Reference] = None):
        self.store = store
        self.lineage = lineage
        self.settings = settings
        self.gpus = gpus
        self.references = references or Reference(store, lineage)

    # ------------------------------------------------------------------ training plan
    def _defaults(self) -> dict[str, Any]:
        d = self.settings.train_defaults()
        d.update(target_modules=["all"], train_on="assistant", gradient_checkpointing=True)
        return d

    def train_plan(self, base_ref: str, dataset_ref: str, dataset_n: Optional[int] = None, overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        base = self.lineage.resolve(base_ref, ("base", "merged", "ctx_variant"))
        version = self.store.find_version(dataset_ref, dataset_n)
        config = read_json(Path(base["path"]) / "config.json", {}) or {}
        if not V.estimate_params(config):
            raise PygmalionError("invalid", "plan_no_config", name=base["name"])
        params = {**self._defaults(), **{k: v for k, v in (overrides or {}).items() if k in TRAIN_KEYS and v is not None}}
        if params["method"] not in ("qlora", "lora"):
            raise PygmalionError("invalid", "train_method_invalid")
        warnings: list[Any] = []
        model_ctx = V.text_config(config).get("max_position_embeddings")
        asked_seq = (overrides or {}).get("seq_len")
        inventory = self.gpus.allowed_inventory()
        best_total = max([g["total_mb"] for g in inventory], default=0)
        estimate = V.estimate_training_memory(config, method=params["method"], rank=params["rank"], batch=params["batch"], seq_len=params["seq_len"])
        if not asked_seq:
            if model_ctx and params["seq_len"] > model_ctx:
                params["seq_len"] = int(model_ctx)
            # shrink the sequence until the estimate fits the biggest allowed card (never when the caller chose it)
            while best_total and estimate["total_mb"] > best_total * 0.92 and params["seq_len"] > SEQ_STEPS[-1]:
                lower = [s for s in SEQ_STEPS if s < params["seq_len"]]
                if not lower:
                    break
                params["seq_len"] = lower[0]
                estimate = V.estimate_training_memory(config, method=params["method"], rank=params["rank"], batch=params["batch"], seq_len=params["seq_len"])
                warnings.append(text("plan_seq_lowered", seq_len=params["seq_len"], mb=best_total))
        estimate = V.estimate_training_memory(config, method=params["method"], rank=params["rank"], batch=params["batch"], seq_len=params["seq_len"])
        if params["method"] == "lora" and best_total and estimate["total_mb"] > best_total:
            warnings.append(text("plan_lora_big"))
        n_train = max(1, version["usable"] - len(version["splits"].get("eval") or []))
        per_step = params["batch"] * params["grad_accum"]
        steps_per_epoch = max(1, math.ceil(n_train / per_step))
        total_steps = params["max_steps"] if params["max_steps"] else max(1, math.ceil(steps_per_epoch * params["epochs"]))
        # tokens to process: the records the steps visit (a partial last epoch included) x the average record, which is cut at seq_len
        average = version["tokens"] / max(1, version["usable"])
        per_record = min(average, params["seq_len"])
        records_seen = total_steps * per_step
        tokens_per_epoch = int(n_train * per_record)
        fits = V.fit_on_gpus(estimate["total_mb"], inventory)
        pick = V.pick_gpus(estimate["total_mb"], inventory)
        gpu_name = next((g["name"] for g in inventory if pick.get("gpus") and g["index"] == pick["gpus"][0]), "")
        timing = V.estimate_time(estimate["params"], tokens_per_epoch, records_seen / n_train, params["method"], gpu_name, steps=total_steps)
        if total_steps < 20:
            warnings.append(text("plan_few_steps"))
        if n_train < 50:
            warnings.append(text("plan_few_records", n=n_train))
        if not inventory:
            warnings.append(text("plan_no_inventory"))
        elif not any(f["fits_total"] for f in fits) and not pick.get("fits"):
            warnings.append(pick.get("reason") or text("plan_no_fit"))
        elif not any(f["fits_free"] for f in fits):
            warnings.append(text("plan_wait_free"))
        return {"base": {"id": base["id"], "name": base["name"]}, "dataset": {"version": version["id"], "n": version["n"], "kind": version["kind"],
                                                                                 "records": version["usable"], "train": n_train, "tokens": version["tokens"]},
                "params": params, "vram": estimate, "fits": fits, "pick": pick, "steps": {"per_epoch": steps_per_epoch, "total": total_steps},
                "time": timing, "warnings": warnings, "allowed_gpus": self.gpus.allowed()}

    def train_params(self, base_ref: str, dataset_ref: str, dataset_n: Optional[int], overrides: dict[str, Any], name: str = "") -> dict[str, Any]:
        """The full, explicit parameters stored on the job (so it can be reproduced), after validating base and dataset."""
        plan = self.train_plan(base_ref, dataset_ref, dataset_n, overrides)
        t = plan["time"]
        out = {**plan["params"], "base": plan["base"]["id"], "dataset": plan["dataset"]["version"],
               "_estimate": {"seconds": t["seconds"], "tokens_per_s": t["tokens_per_s"], "tokens": t["tokens"]}}      # what the job view compares the measured speed with
        if name:
            out["name"] = name
        return out

    # ------------------------------------------------------------------ pipelines
    def normalise_after(self, after: Optional[dict[str, Any]]) -> dict[str, Any]:
        a = dict(after or {})
        types, warnings = LT.check_quant_types(a.get("quantize") or [], bool(a.get("imatrix"))) if a.get("quantize") else ([], [])
        a["quantize"] = types
        a["warnings"] = warnings
        a["imatrix"] = bool(a.get("imatrix")) if types else False
        a["outtype"] = (a.get("convert") if isinstance(a.get("convert"), str) else None) or "f16"
        a["convert"] = bool(a.get("convert") or types or a.get("publish") or a.get("evaluate") or a.get("perplexity"))
        return a

    def ppl_params(self, ref: str, a: dict[str, Any]) -> dict[str, Any]:
        """Parameters of a perplexity step: the text, context and chunk count are fixed here (not left to each job), so every file of one pipeline,
        its baseline included, is measured the same way."""
        return {"gguf": ref, "text": a.get("perplexity_text") or {"source": "bundled"},
                "ctx": int(a.get("perplexity_ctx") or self.settings.int("quant.ppl_ctx")),
                "chunks": int(a.get("perplexity_chunks") or self.settings.int("quant.ppl_chunks"))}

    @staticmethod
    def measured_like(art: dict[str, Any], params: dict[str, Any]) -> bool:
        """Has this GGUF already been measured with these perplexity parameters?"""
        done = (art.get("metrics") or {}).get("ppl") or {}
        return bool(done) and done.get("ctx") == params["ctx"] and done.get("chunks") == params["chunks"] and done.get("spec") == params["text"]

    def post_steps(self, *, hf: Optional[str], gguf: Optional[str], baseline_model: Optional[str], after: dict[str, Any], name: str) -> list[dict[str, Any]]:
        """Steps that follow a stage that produced an HF folder (``hf`` is its ``$ref``) or start from a GGUF (``gguf`` is its id)."""
        a = self.normalise_after(after)
        steps: list[dict[str, Any]] = []
        src = gguf
        if hf and a["convert"]:
            steps.append({"kind": "convert", "title": text("title_convert"), "params": {"model": hf, "outtype": a["outtype"], "name": f"{name}-{a['outtype']}"}})
            src = "$gguf"
        if src is None:
            return steps
        types = a["quantize"]
        if types:
            if a["imatrix"]:
                steps.append({"kind": "imatrix", "title": text("title_imatrix"), "params": {"gguf": src, "calibration": a.get("calibration") or {"source": "bundled"},
                                                                                           "chunks": a.get("chunks")}})
            for t in types:
                steps.append({"kind": "quantize", "title": text("title_quantize", type=t), "params": {"gguf": src, "type": t, "imatrix": "$imatrix" if a["imatrix"] else None,
                                                                                         "name": f"{name}-{t}"}})
        primary = f"$quant:{types[0]}" if types else src
        if a.get("perplexity"):
            # the unquantized file and every quantization on the same text, context and chunks, so the table compares them
            targets = ([(src, a["outtype"])] if types and src else []) + [(f"$quant:{t}", t) for t in types]
            for ref, what in targets or [(primary, "")]:
                steps.append({"kind": "perplexity", "title": text("title_perplexity_of", what=what) if what else text("title_perplexity_step"),
                              "params": self.ppl_params(ref, a)})
        pub = a.get("publish")
        if pub:
            pub = pub if isinstance(pub, dict) else {}
            steps.append({"kind": "publish", "title": text("title_publish_step"), "params": {"gguf": primary, "target": pub.get("target", "ollama"), "name": pub.get("name") or name,
                                                                              "tag": pub.get("tag", "latest"), "num_ctx": pub.get("num_ctx"), "template": pub.get("template", ""),
                                                                              "system": pub.get("system", "")}})
        ev = a.get("evaluate")
        if ev:
            ev = ev if isinstance(ev, dict) else {}
            steps.append({"kind": "evaluate", "title": text("title_evaluate_step"), "params": {"artifact": primary, "against": ev.get("against", ""),
                                                                                "intent": ev.get("intent", ""), "suites": ev.get("suites"),
                                                                                "regression": ev.get("regression", True)}})
        return steps

    def with_baseline(self, steps: list[dict[str, Any]], baseline_model: Optional[str], after: dict[str, Any]) -> list[dict[str, Any]]:
        """Put the base model's file in front of the pipeline: the steps that build it, written like the result (same quantization, an importance
        matrix on the same calibration text) and reusing whatever of it exists, when an evaluation needs it (and ``against`` is not given); and a
        perplexity step for it when perplexity is on, so the fine-tune's effect shows next to the base's own number. Perplexity only measures
        baseline files that exist or are being built here; it never converts the base just for that."""
        if not baseline_model:
            return steps
        a = self.normalise_after(after)
        ev = after.get("evaluate")
        wants_baseline = bool(ev) and not (isinstance(ev, dict) and ev.get("against"))
        build = wants_baseline and not (isinstance(ev, dict) and ev.get("baseline") is False)
        qtype = a["quantize"][0] if a["quantize"] else None
        calibration = a.get("calibration") or {"source": "bundled"}
        spec = {"qtype": qtype or "", "outtype": a["outtype"],
                "imatrix": {"spec": calibration, "tag": calib.tag_of(calibration), "chunks": a.get("chunks")} if a["imatrix"] else None}
        chain = self.references.chain(baseline_model, spec, build=build)
        evaluate = next((s for s in steps if s["kind"] == "evaluate"), None)
        if wants_baseline and evaluate and chain["ref"]:
            evaluate["params"]["against"] = chain["ref"]
        refs: dict[str, Optional[str]] = {"f16": chain["f16"], qtype or "": chain["ref"]}
        measure: list[dict[str, Any]] = []
        if a.get("perplexity"):
            for what, ref in ((("f16", refs["f16"]),) if qtype else ()) + ((qtype or "", refs.get(qtype or "")),):
                if not ref:
                    continue
                params = self.ppl_params(ref, a)
                if not ref.startswith("$"):
                    try:
                        if self.measured_like(self.store.artifact(ref), params):
                            continue
                    except PygmalionError:
                        continue
                measure.append({"kind": "perplexity", "title": text("title_baseline_perplexity", what=what or "f16"), "params": params})
        return chain["steps"] + measure + steps
