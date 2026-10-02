"""One function per job kind. Each receives a ``JobContext`` and returns the job's result; the artifacts they create carry their
parents, dataset version and a recipe, so the lineage is complete without any extra bookkeeping."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Optional

from . import calib as calibration
from . import llama_tools as LT
from . import vram as V
from .convert_detect import convertible
from .errors import PygmalionError
from .gguf_meta import read_metadata, summarize
from .hf import check_repo
from .jobs import Deps, JobContext, Runner
from .lineage import is_quantized, portable
from .messages import text
from .util import dir_size, read_json, slug
from .workers import _stio as stio

BASE_KINDS = ("base", "merged", "ctx_variant")


def short(name: str) -> str:
    """A model name without its organisation prefix (``acme/tiny`` becomes ``tiny``)."""
    return name.rsplit("/", 1)[-1]


def clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if not k.startswith("_") and v is not None}


def tool_failure(prefix: str, result: Any) -> PygmalionError:
    """The error for a program that failed: its last output lines, or its exit code when it printed none. ``prefix`` names the catalogue entries
    (``<prefix>_failed`` with the output, ``<prefix>_failed_code`` with the code)."""
    tail = " | ".join(x for x in result.tail[-3:] if x.strip())[:300]
    if tail:
        return PygmalionError("failed", f"{prefix}_failed", detail=tail)
    return PygmalionError("failed", f"{prefix}_failed_code", code=result.returncode)


def ensure_disk(folder: Path, need: int, what: Any) -> None:
    probe = folder
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return
    if free < need * 1.05:
        raise PygmalionError("failed", "disk_low", what=what, need_gb=round(need / 1e9, 1), free_gb=round(free / 1e9, 1))


def tool_or_raise(path: Optional[str], name: str, setting: str) -> str:
    if not path:
        raise PygmalionError("tool_missing", "tool_not_found", name=name, setting=setting)
    return path


class Runners:
    def __init__(self, deps: Deps):
        self.d = deps

    def table(self) -> dict[str, Runner]:
        return {"download": self.download, "dataset_build": self.dataset_build, "train": self.train, "merge_lora": self.merge_lora,
                "merge_models": self.merge_models, "convert": self.convert, "imatrix": self.imatrix, "quantize": self.quantize,
                "perplexity": self.perplexity, "ctx_extend": self.ctx_extend, "publish": self.publish, "evaluate": self.evaluate}

    # ------------------------------------------------------------------ helpers
    def _out_dir(self, ctx: JobContext) -> tuple[str, Path]:
        out_id = ctx.reserve_output_id()
        folder = self.d.work.output_dir(out_id)
        folder.mkdir(parents=True, exist_ok=True)
        return out_id, folder

    def _hf_model(self, ref: str) -> dict[str, Any]:
        art = self.d.lineage.resolve(ref, BASE_KINDS)
        if not (Path(art["path"]) / "config.json").is_file():
            raise PygmalionError("not_found", "model_folder_missing", name=art["name"], path=art["path"])
        return art

    def _create(self, ctx: JobContext, out_id: str, kind: str, name: str, path: Path, parents: list[str], recipe: dict[str, Any],
                metrics: Optional[dict[str, Any]] = None, dataset_version: Optional[str] = None,
                params_extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        existing = None
        try:
            existing = self.d.store.artifact(out_id)
        except PygmalionError:
            pass
        if existing:                       # a retry of the same job replaces the record it made before
            self.d.store.delete_artifact(out_id)
        art = self.d.store.create_artifact(kind, name, path=str(path), size=dir_size(path), parents=parents, dataset_version=dataset_version,
                                           job_id=ctx.id, recipe={"job_kind": ctx.job["kind"], "params": {**portable(clean_params(ctx.params)), **(params_extra or {})}, **recipe},
                                           metrics=metrics or {}, artifact_id=out_id)
        ctx.set_output(art["id"])
        return art

    # ------------------------------------------------------------------ download
    def download(self, ctx: JobContext) -> dict[str, Any]:
        repo = check_repo(ctx.params.get("repo_id", ""))
        work = self.d.work
        work.ensure()
        dest = work.base_dir(repo)
        if ctx.params.get("expected_bytes"):
            ensure_disk(work.root, int(ctx.params["expected_bytes"]), repo)
        token = self.d.config.secret("HF_TOKEN")
        outcome = ctx.run_worker("download", {"repo_id": repo, "local_dir": str(dest), "revision": ctx.params.get("revision") or None,
                                              "trust_remote_code": False}, gpus=[], env={"HF_TOKEN": token} if token else {})
        base = self.d.lineage.ensure_base(dest, repo, {"source": "huggingface", "repo_id": repo, "revision": outcome.result.get("revision")})
        self.d.store.update_artifact(base["id"], size=dir_size(dest))
        ctx.set_output(base["id"])
        return {"artifact": base["id"], "path": str(dest), **outcome.result}

    # ------------------------------------------------------------------ datasets
    def dataset_build(self, ctx: JobContext) -> dict[str, Any]:
        spec = ctx.params.get("spec") or {}

        def progress(message: str, step: int, total: int) -> None:
            ctx.progress(message=message, step=step, total=total)

        return self.d.datasets.build(spec, progress=progress, cancelled=ctx.cancel.is_set)

    # ------------------------------------------------------------------ training
    def train(self, ctx: JobContext) -> dict[str, Any]:
        p = ctx.params
        base = self._hf_model(p["base"])
        version = self.d.store.find_version(p["dataset"], p.get("dataset_n"))
        dataset_name = self.d.store.dataset(version["dataset_id"])["name"]
        name = p.get("name") or f"{short(base['name'])}-{slug(dataset_name, 24)}-{p.get('method', 'qlora')}"
        export = self.d.datasets.export_for_training(version["id"], ctx.dir / "data")
        if export["n_train"] < 1:
            raise PygmalionError("invalid", "dataset_no_train")
        config = read_json(Path(base["path"]) / "config.json", {}) or {}
        estimate = V.estimate_training_memory(config, method=p.get("method", "qlora"), rank=int(p.get("rank", 16)), batch=int(p.get("batch", 1)),
                                              seq_len=int(p.get("seq_len", 2048)))
        ctx.set_vram(estimate)
        out_id, out_dir = self._out_dir(ctx)
        grant = ctx.acquire_gpu(estimate["total_mb"], f"pygmalion train {name}")
        try:
            args = {"base_path": base["path"], "output_dir": str(out_dir), "train_path": export["train"], "eval_path": export["eval"] if export["n_eval"] else None,
                    "method": p.get("method", "qlora"), "rank": int(p.get("rank", 16)), "alpha": int(p.get("alpha", 32)), "dropout": float(p.get("dropout", 0.05)),
                    "target_modules": p.get("target_modules") or ["all"], "lr": float(p.get("lr", 2e-4)), "weight_decay": float(p.get("weight_decay", 0.0)),
                    "warmup": float(p.get("warmup", 0.03)), "epochs": float(p.get("epochs", 1)), "max_steps": int(p.get("max_steps", 0)),
                    "batch_size": int(p.get("batch", 1)), "grad_accum": int(p.get("grad_accum", 16)), "seq_len": int(p.get("seq_len", 2048)),
                    "eval_every": int(p.get("eval_every", 50)), "save_every": int(p.get("save_every", 100)), "train_on": p.get("train_on", "assistant"),
                    "gradient_checkpointing": bool(p.get("gradient_checkpointing", True)), "seed": int(p.get("seed", 42)), "resume": ctx.resuming}
            ctx.message(text("progress_training"), total=None)
            outcome = ctx.run_worker("train_lora", args, gpus=grant.gpus)
        finally:
            grant.release()
        recipe = {"base": base["name"], "dataset_version": version["id"]}
        art = self._create(ctx, out_id, "adapter", name, out_dir, [base["id"]], recipe, {"training": outcome.result, "vram_estimate": estimate},
                           dataset_version=version["id"])
        return {"artifact": art["id"], "summary": outcome.result, "vram_estimate_mb": estimate["total_mb"]}

    # ------------------------------------------------------------------ merges
    def merge_lora(self, ctx: JobContext) -> dict[str, Any]:
        p = ctx.params
        base = self._hf_model(p["base"])
        adapter = self.d.lineage.resolve(p["adapter"], ("adapter",))
        device = p.get("device") or self.d.settings.get("merge.device")
        out_id, out_dir = self._out_dir(ctx)
        ensure_disk(out_dir, int(base["size"] or 0), text("what_merged_model"))
        name = p.get("name") or f"{short(base['name'])}+{adapter['name']}"
        grant = ctx.acquire_gpu(int((base["size"] or 0) / 1048576 * 1.15) + 1500, f"pygmalion merge {name}") if device == "cuda" else None
        try:
            args = {"base_path": base["path"], "adapter_path": adapter["path"], "output_dir": str(out_dir), "device": "cuda:0" if grant else "cpu",
                    "shard_gb": 4, "mode": p.get("mode", "auto")}
            outcome = ctx.run_worker("merge_lora", args, gpus=grant.gpus if grant else [])
        finally:
            if grant:
                grant.release()
        art = self._create(ctx, out_id, "merged", name, out_dir, [base["id"], adapter["id"]], {"base": base["name"], "adapter": adapter["name"]},
                           {"merge": outcome.result}, dataset_version=adapter.get("dataset_version"))
        return {"artifact": art["id"], **outcome.result}

    def merge_models(self, ctx: JobContext) -> dict[str, Any]:
        p = ctx.params
        models = [self._hf_model(m) for m in p["models"]]
        base = self._hf_model(p["base"]) if p.get("base") else None
        method = p.get("method", "linear")
        out_id, out_dir = self._out_dir(ctx)
        ensure_disk(out_dir, int(models[0]["size"] or 0), text("what_merged_model"))
        name = p.get("name") or f"merge-{method}-" + "+".join(short(m["name"]) for m in models)[:60]
        args = {"models": [m["path"] for m in models], "output_dir": str(out_dir), "method": method, "weights": p.get("weights"),
                "base": base["path"] if base else None, "t": p.get("t", 0.5), "t_map": p.get("t_map") or {}, "density": p.get("density", 0.5),
                "lam": p.get("lam", 1.0), "seed": p.get("seed", 0), "consensus": p.get("consensus", "linear"), "normalize": p.get("normalize", True),
                "shard_gb": 4}
        outcome = ctx.run_worker("merge_models", args, gpus=[])
        parents = [m["id"] for m in models] + ([base["id"]] if base and base["id"] not in [m["id"] for m in models] else [])
        art = self._create(ctx, out_id, "merged", name, out_dir, parents, {"method": method}, {"merge": outcome.result})
        return {"artifact": art["id"], **outcome.result}

    def ctx_extend(self, ctx: JobContext) -> dict[str, Any]:
        p = ctx.params
        model = self._hf_model(p["model"])
        out_id, out_dir = self._out_dir(ctx)
        factor = float(p.get("factor", 4))
        name = p.get("name") or f"{short(model['name'])}-ctx{factor:g}x"
        outcome = ctx.run_worker("ctx_extend", {"model_path": model["path"], "output_dir": str(out_dir), "factor": factor,
                                                "original_max_position": p.get("original"), "link": p.get("link", "hardlink")}, gpus=[])
        art = self._create(ctx, out_id, "ctx_variant", name, out_dir, [model["id"]], {"factor": factor}, {"ctx": outcome.result},
                           dataset_version=model.get("dataset_version"))
        return {"artifact": art["id"], **outcome.result}

    # ------------------------------------------------------------------ GGUF
    def convert(self, ctx: JobContext) -> dict[str, Any]:
        p = ctx.params
        work = self.d.work
        python = work.python()
        if not python or not Path(python).is_file():
            raise PygmalionError("env_missing", "env_not_set")
        outtype = p.get("outtype", "f16")
        out_id, out_dir = self._out_dir(ctx)
        if p.get("adapter"):
            adapter = self.d.lineage.resolve(p["adapter"], ("adapter",))
            base = self._hf_model(p["base"])
            script = tool_or_raise(work.src_script("convert_lora_to_gguf.py"), "convert_lora_to_gguf.py", "llama.src_dir")
            out = out_dir / LT.gguf_name(adapter["name"], "lora", f"-{outtype}")
            argv = LT.convert_lora_argv(python, script, base["path"], adapter["path"], str(out), outtype)
            parents, source, label = [adapter["id"], base["id"]], adapter, "progress_convert_adapter"
            total = None
        else:
            source = self._hf_model(p["model"])
            script = tool_or_raise(work.src_script("convert_hf_to_gguf.py"), "convert_hf_to_gguf.py", "llama.src_dir")
            known = convertible(source["path"], script)
            if known["known"] is False:
                raise PygmalionError("unsupported", "convert_unsupported", note=known["note"])
            ensure_disk(out_dir, int((source["size"] or 0) * 1.05), text("what_gguf_file"))
            out = out_dir / LT.gguf_name(source["name"], outtype)
            argv = LT.convert_argv(python, script, source["path"], str(out), outtype)
            parents, label = [source["id"]], "progress_convert_model"
            try:
                total = len(stio.model_tensors(source["path"]))
            except stio.StioError:
                total = None
        done = {"n": 0}

        def on_line(line: str) -> None:
            if LT.parse_convert_line(line):
                done["n"] += 1
                if done["n"] % 5 == 0:
                    ctx.progress(step=done["n"], total=total)

        ctx.message(text(label, outtype=outtype))
        result = ctx.run_tool(argv, on_line=on_line, gpus=[], env={"PYTHONUTF8": "1"}, timeout_s=6 * 3600)
        if not result.ok or not out.is_file() or out.stat().st_size == 0:
            raise tool_failure("convert", result)
        metrics: dict[str, Any] = {}
        try:
            summary = summarize(read_metadata(out))
            metrics["gguf"] = {k: summary.get(k) for k in ("architecture", "context_length", "block_count", "file_type", "rope_scaling_type", "rope_scaling_factor", "has_chat_template")}
            if any(a["kind"] == "ctx_variant" for a in self.d.lineage.ancestors(source["id"]) + [source]) and not summary.get("rope_scaling_type"):
                metrics["note"] = text("note_rope_missing")
        except PygmalionError as exc:
            metrics["gguf_error"] = exc.message
        art = self._create(ctx, out_id, "gguf", p.get("name") or out.stem, out, parents, {"outtype": outtype}, metrics,
                           dataset_version=source.get("dataset_version"), params_extra={"outtype": outtype, **({"adapter": True} if p.get("adapter") else {})})
        return {"artifact": art["id"], "size": art["size"], "gguf": metrics.get("gguf")}

    def _calibration(self, ctx: JobContext, spec: dict[str, Any]) -> dict[str, Any]:
        records = None
        if spec.get("source") == "dataset" or spec.get("dataset"):
            version = self.d.store.find_version(spec["dataset"], spec.get("n"))
            records = self.d.datasets.read_records(version)
        work = self.d.work
        work.calib_dir.mkdir(parents=True, exist_ok=True)
        tag = calibration.tag_of(spec)
        info = calibration.write_calibration(work.calib_dir / f"calib-{tag}.txt", records)
        info["tag"] = tag
        return info

    def imatrix(self, ctx: JobContext) -> dict[str, Any]:
        p = ctx.params
        src = self.d.lineage.resolve(p["gguf"], ("gguf",))
        exe = tool_or_raise(self.d.work.llama_bin("llama-imatrix"), "llama-imatrix", "llama.bin_dir")
        calib_info = self._calibration(ctx, p.get("calibration") or {"source": "bundled"})
        chunks = int(p.get("chunks") or self.d.settings.int("quant.imatrix_chunks"))
        out_id, out_dir = self._out_dir(ctx)
        out = out_dir / "imatrix.dat"
        need = int((src["size"] or 0) / 1048576 * 1.08) + 1500
        grant = ctx.acquire_gpu(need, f"pygmalion imatrix {src['name']}")
        state = {"total": chunks}

        def on_line(line: str) -> None:
            parsed = LT.parse_imatrix_line(line)
            if "total" in parsed:
                state["total"] = parsed["total"]
            if "step" in parsed:
                ctx.progress(step=parsed["step"], total=state["total"])

        try:
            ctx.message(text("progress_imatrix"), total=chunks, step=0)
            result = ctx.run_tool(LT.imatrix_argv(exe, src["path"], calib_info["path"], str(out), chunks), on_line=on_line, gpus=grant.gpus,
                                  timeout_s=6 * 3600)
        finally:
            grant.release()
        if not result.ok or not out.is_file():
            raise tool_failure("imatrix", result)
        art = self._create(ctx, out_id, "imatrix", f"{src['name']}-imatrix", out, [src["id"]], {"calibration": calib_info["source"]},
                           {"calibration": calib_info, "chunks": chunks})
        return {"artifact": art["id"], "calibration": calib_info, "chunks": chunks}

    def quantize(self, ctx: JobContext) -> dict[str, Any]:
        p = ctx.params
        src = self.d.lineage.resolve(p["gguf"], ("gguf",))
        qtype = str(p["type"]).upper()
        _types, warnings = LT.check_quant_types([qtype], bool(p.get("imatrix")))
        exe = tool_or_raise(self.d.work.llama_bin("llama-quantize"), "llama-quantize", "llama.bin_dir")
        matrix = self.d.lineage.resolve(p["imatrix"], ("imatrix",)) if p.get("imatrix") else None
        out_id, out_dir = self._out_dir(ctx)
        ensure_disk(out_dir, int((src["size"] or 0) * 0.6), text("what_quantized_file"))
        out = out_dir / LT.gguf_name(src["name"].removesuffix(".gguf"), qtype)
        info: dict[str, Any] = {}

        def on_line(line: str) -> None:
            parsed = LT.parse_quantize_line(line)
            info.update({k: v for k, v in parsed.items() if k.endswith("bytes")})
            if "step" in parsed:
                ctx.progress(step=parsed["step"], total=parsed["total"])

        ctx.message(text("progress_quantize", type=qtype))
        result = ctx.run_tool(LT.quantize_argv(exe, src["path"], str(out), qtype, matrix["path"] if matrix else None), on_line=on_line, gpus=[],
                              timeout_s=6 * 3600)
        if not result.ok or not out.is_file() or out.stat().st_size == 0:
            raise tool_failure("quantize", result)
        size = out.stat().st_size
        art = self._create(ctx, out_id, "gguf", p.get("name") or out.stem, out, [src["id"]] + ([matrix["id"]] if matrix else []),
                           {"qtype": qtype, "imatrix": matrix["id"] if matrix else None},
                           {"quant": qtype, "imatrix": bool(matrix), "ratio": round(size / src["size"], 3) if src["size"] else None, "warnings": warnings},
                           dataset_version=src.get("dataset_version"), params_extra={"qtype": qtype})
        return {"artifact": art["id"], "size": size, "type": qtype, "warnings": warnings}

    def perplexity(self, ctx: JobContext) -> dict[str, Any]:
        p = ctx.params
        art = self.d.lineage.resolve(p["gguf"], ("gguf",))
        exe = tool_or_raise(self.d.work.llama_bin("llama-perplexity"), "llama-perplexity", "llama.bin_dir")
        spec = p.get("text") or {"source": "bundled"}
        text_info = self._calibration(ctx, spec)
        context = int(p.get("ctx") or self.d.settings.int("quant.ppl_ctx"))
        chunks = int(p.get("chunks") or self.d.settings.int("quant.ppl_chunks"))
        grant = ctx.acquire_gpu(int((art["size"] or 0) / 1048576 * 1.1) + 1200, f"pygmalion perplexity {art['name']}")
        found: dict[str, Any] = {}

        def on_line(line: str) -> None:
            parsed = LT.parse_perplexity_line(line)
            if parsed.get("final"):
                found.update(parsed)
            elif "chunk" in parsed:
                ctx.progress(step=parsed["chunk"], total=chunks)

        try:
            ctx.message(text("progress_perplexity"), step=0, total=chunks)
            result = ctx.run_tool(LT.perplexity_argv(exe, art["path"], text_info["path"], context, chunks), on_line=on_line, gpus=grant.gpus, timeout_s=3 * 3600)
        finally:
            grant.release()
        if not result.ok or "ppl" not in found:
            raise tool_failure("perplexity", result)
        value = {"value": found["ppl"], "error": found["ppl_error"], "ctx": context, "chunks": chunks, "text": text_info["source"],
                 "text_tag": text_info.get("tag"), "spec": spec, "ts": self.d.store.clock()}
        self.d.lineage.add_metrics(art["id"], "ppl", value)
        if (self.d.store.artifact(art["id"])["metrics"].get("ppl") or {}).get("value") != value["value"]:
            raise PygmalionError("failed", "ppl_not_stored", name=art["name"])      # never report a measurement that was not kept
        return {"artifact": art["id"], **value}

    # ------------------------------------------------------------------ publish and evaluate
    def publish(self, ctx: JobContext) -> dict[str, Any]:
        p = ctx.params
        target = p.get("target", "ollama")
        out: dict[str, Any] = {}
        if target in ("ollama", "both"):
            out["ollama"] = self.d.publisher.publish_ollama(p["gguf"], name=p.get("name", ""), tag=p.get("tag", "latest"), num_ctx=p.get("num_ctx"),
                                                            adapter=p.get("adapter", ""), template=p.get("template", ""), system=p.get("system", ""),
                                                            cancel=ctx.cancel, log_path=ctx.log_path, job_id=ctx.id)
            ctx.set_output(out["ollama"]["artifact"])
        if target in ("llama", "both"):
            out["llama"] = self.d.publisher.publish_llama(p["gguf"], name=p.get("name", ""), ctx=p.get("ctx") or p.get("num_ctx"), ngl=int(p.get("ngl", 99)),
                                                          gpu=p.get("gpu"), extra_args=p.get("extra_args"))
        return out

    def evaluate(self, ctx: JobContext) -> dict[str, Any]:
        p = ctx.params

        def progress(step: Any = None, state: str = "", run: Any = None) -> None:
            message = text("progress_eval_suite") if state == "suite" else text("progress_galton", run=run, state=state)
            ctx.progress(force=True, message=message, pct=step if isinstance(step, (int, float)) else None)

        return self.d.evaluator.run(p["artifact"], against=p.get("against", ""), intent=p.get("intent", ""), suites=p.get("suites"),
                                    settings=p.get("settings"), regression=bool(p.get("regression", True)), cancel=ctx.cancel, progress=progress,
                                    log=ctx.log, reference=p.get("reference"))
