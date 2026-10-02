"""What the tools and the interface do, one method each: they validate a request, queue the jobs and shape the answer. The business logic
lives in the modules they call; this is the seam between them and the two front doors (assistant tools and the web interface)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from . import llama_tools as LT
from . import vram as V
from .errors import PygmalionError
from .gguf_meta import fit_table, read_metadata, summarize
from .hf import check_repo
from .hoard_link.waiting import clamp_wait
from .jobs import KINDS
from .messages import text as msg
from .store import ACTIVE_STATES
from .util import slug
from .workers import _stio as stio

if TYPE_CHECKING:
    from .services import Services

CTX_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]


class Operations:
    def __init__(self, svc: "Services"):
        self.svc = svc
        self.jobs = svc.jobs
        self.store = svc.store
        self.lineage = svc.lineage

    # ------------------------------------------------------------------ job helpers
    def _answer(self, job: dict[str, Any], wait_s: float = 0.0, **extra: Any) -> dict[str, Any]:
        wait_s = clamp_wait(wait_s)                 # the shared limit: a call that waited longer than the bridge allows would only time out
        if wait_s > 0:
            job = self.jobs.wait(job["id"], wait_s)
        view = self.jobs.view(job, detail=job["state"] in ("done", "failed"))
        out = {"job": view, "done": job["state"] == "done", "state": job["state"], "result": job["result"] if job["state"] == "done" else None, **extra}
        if wait_s > 0 and job["state"] in ACTIVE_STATES:
            out["still_running"] = True             # the wait ended first: the job goes on, poll job_get
        return out

    def _pipeline(self, steps: list[dict[str, Any]], wait_s: float = 0.0, **extra: Any) -> dict[str, Any]:
        job = self.jobs.pipeline(steps)
        return self._answer(job, wait_s, steps=[s["kind"] for s in steps], **extra)

    def _require_env(self) -> None:
        if not self.svc.work.python_ok():
            raise PygmalionError("env_missing", "env_not_set")

    # ------------------------------------------------------------------ overview
    def overview(self) -> dict[str, Any]:
        dash = self.svc.dashboard()
        counts = dash["counts"]
        return {"summary": (f"{counts['bases']} bases, {counts['datasets']} datasets, {counts['artifacts']} artifacts, {counts['jobs_running']} running, "
                            f"{counts['jobs_active']} active jobs, {counts['jobs_failed']} failed or interrupted."),
                "environment": dash["env"], "galton": dash["galton"], "gpus": [{k: g.get(k) for k in ("index", "name", "total_mb", "free_mb", "allowed", "reserved", "leases")}
                                                                             for g in dash["gpus"]["gpus"]],
                "hub": dash["gpus"]["hub"], "allowed_gpus": dash["gpus"]["allowed"], "jobs": dash["jobs"], "failed_jobs": dash["failed"],
                "latest_artifacts": dash["artifacts"], "storage": dash["storage"], "work_dir": str(self.svc.work.root),
                "next_steps": self._next_steps(dash)}

    @staticmethod
    def _next_steps(dash: dict[str, Any]) -> list[str]:
        steps = []
        env = dash["env"]
        if not env.get("checked"):
            steps.append("Run env_check to see what the trainer environment and the llama.cpp tools can do.")
        elif not env.get("ok"):
            steps.append(f"env_check found {env.get('problems')} problem(s): follow the fix commands it lists.")
        if not dash["counts"]["bases"]:
            steps.append("Add a base model: hf_search then base_download, or put a Hugging Face folder in the work folder's hf/.")
        if not dash["counts"]["datasets"]:
            steps.append("Build a dataset with dataset_create from your own files, a family app or a teacher model.")
        return steps

    # ------------------------------------------------------------------ bases
    def base_get(self, ref: str) -> dict[str, Any]:
        art = self.lineage.resolve(ref, ("base", "merged", "ctx_variant"))
        card = next((c for c in self.svc.bases() if c["path"] == art["path"]), None)
        out: dict[str, Any] = {"artifact": self.lineage.card(art), "local": card}
        if self.svc.work.python_ok() and Path(art["path"]).is_dir():
            probe = self.svc.env.probe(model=art["path"])
            out["trainer_check"] = probe.get("model") if probe.get("ok") else {"error": probe.get("error"), "hint": probe.get("hint")}
        config = Path(art["path"]) / "config.json"
        if config.is_file():
            import json
            try:
                cfg = json.loads(config.read_text(encoding="utf-8"))
                out["estimate"] = V.estimate_training_memory(cfg, method="qlora")
            except (OSError, ValueError):
                pass
        out["children"] = [self.lineage.card(c) for c in self.store.children(art["id"])]
        return out

    def hf_search(self, query: str, pipeline: str, sort: str, limit: int) -> dict[str, Any]:
        return {"results": self.svc.hub.search(query, pipeline=pipeline, sort=sort, limit=limit), "query": query}

    def base_download(self, repo_id: str, revision: str = "", confirm: bool = False) -> dict[str, Any]:
        repo = check_repo(repo_id)
        info = self.svc.hub.info(repo)
        self.svc.work.ensure()
        free = shutil.disk_usage(self.svc.work.root).free
        summary = {"repo_id": repo, "download_bytes": info["download_bytes"], "download_gb": round(info["download_bytes"] / 1e9, 2), "gated": info["gated"],
                   "license": info["license"], "architecture": info["architecture"], "free_disk_gb": round(free / 1e9, 1),
                   "has_python_files": info["has_python_files"], "weights": info["weights"]}
        if not info["weights"]:
            raise PygmalionError("unsupported", "repo_no_safetensors", repo=repo)
        if info["download_bytes"] * 1.05 > free:
            raise PygmalionError("failed", "repo_too_big", repo=repo, need_gb=summary["download_gb"], free_gb=summary["free_disk_gb"])
        if not confirm:
            return {"started": False, **summary, "hint": msg("download_confirm", gb=summary["download_gb"], folder=self.svc.work.hf_dir)}
        job = self.jobs.submit("download", {"repo_id": repo, "revision": revision or None, "expected_bytes": info["download_bytes"]}, title=msg("title_download", name=repo))
        return {"started": True, **summary, "job": self.jobs.view(job)}

    # ------------------------------------------------------------------ datasets
    def dataset_create(self, spec: dict[str, Any], wait_s: float) -> dict[str, Any]:
        job = self.jobs.submit("dataset_build", {"spec": spec}, title=msg("title_dataset", name=spec.get("name") or spec.get("dataset") or "…"))
        return self._answer(job, wait_s)

    # ------------------------------------------------------------------ training
    def train_plan(self, base: str, dataset: str, dataset_n: Optional[int], overrides: dict[str, Any]) -> dict[str, Any]:
        return self.svc.planner.train_plan(base, dataset, dataset_n, overrides)

    def train_start(self, base: str, dataset: str, dataset_n: Optional[int], overrides: dict[str, Any], name: str, after: Optional[dict[str, Any]],
                    force: bool, wait_s: float) -> dict[str, Any]:
        self._require_env()
        plan = self.svc.planner.train_plan(base, dataset, dataset_n, overrides)
        if plan["pick"].get("fits") is False and not force:
            raise PygmalionError("gpu_unavailable", "train_no_fit", reason=plan["pick"]["reason"])
        params = self.svc.planner.train_params(base, dataset, dataset_n, overrides, name)
        steps: list[dict[str, Any]] = [{"kind": "train", "title": msg("title_train"), "params": params}]
        label = name or f"{plan['base']['name'].rsplit('/', 1)[-1]}-{slug(self.store.dataset(self.store.version(plan['dataset']['version'])['dataset_id'])['name'], 24)}"
        if after:
            a = self.svc.planner.normalise_after(after)
            if a["convert"] or after.get("merge"):
                steps.append({"kind": "merge_lora", "title": msg("title_merge"), "params": {"base": plan["base"]["id"], "adapter": "$adapter", "name": f"{label}-merged"}})
                steps += self.svc.planner.post_steps(hf="$merged", gguf=None, baseline_model=plan["base"]["id"], after=after, name=label)
                steps = self.svc.planner.with_baseline(steps, plan["base"]["id"], after)
        return self._pipeline(steps, wait_s, plan={"vram": plan["vram"], "time": plan["time"], "warnings": plan["warnings"], "pick": plan["pick"]})

    # ------------------------------------------------------------------ merges and context
    def merge_check(self, models: list[str], base: str = "") -> dict[str, Any]:
        arts = [self.lineage.resolve(m, ("base", "merged", "ctx_variant")) for m in models]
        if base:
            arts.append(self.lineage.resolve(base, ("base", "merged", "ctx_variant")))
        try:
            report = stio.check_compatible([a["path"] for a in arts])
        except stio.StioError as exc:
            raise PygmalionError("invalid", "models_unreadable", detail=str(exc)) from exc
        return {**report, "models": [{"id": a["id"], "name": a["name"], "size": a["size"]} for a in arts],
                "disk_needed_gb": round((arts[0]["size"] or 0) / 1e9, 2)}

    def merge_lora_start(self, base: str, adapter: str, name: str, device: str, after: Optional[dict[str, Any]], wait_s: float) -> dict[str, Any]:
        self._require_env()
        b = self.lineage.resolve(base, ("base", "merged", "ctx_variant"))
        a = self.lineage.resolve(adapter, ("adapter",))
        label = name or f"{b['name']}+{a['name']}"
        params = {"base": b["id"], "adapter": a["id"], "name": label, "device": device or self.svc.settings.get("merge.device")}
        steps = [{"kind": "merge_lora", "title": msg("title_merge"), "params": params}]
        if after:
            steps += self.svc.planner.post_steps(hf="$merged", gguf=None, baseline_model=b["id"], after=after, name=label)
            steps = self.svc.planner.with_baseline(steps, b["id"], after)
        return self._pipeline(steps, wait_s)

    def merge_models_start(self, params: dict[str, Any], after: Optional[dict[str, Any]], wait_s: float) -> dict[str, Any]:
        self._require_env()
        report = self.merge_check(params["models"], params.get("base") or "")
        if not report["ok"]:
            raise PygmalionError("invalid", "models_incompatible", problems="; ".join(report["problems"][:3]))
        resolved = {**params, "models": [self.lineage.resolve(m)["id"] for m in params["models"]]}
        if params.get("base"):
            resolved["base"] = self.lineage.resolve(params["base"])["id"]
        steps = [{"kind": "merge_models", "title": msg("title_merge_models"), "params": resolved}]
        if after:
            steps += self.svc.planner.post_steps(hf="$merged", gguf=None, baseline_model=None, after=after, name=params.get("name") or "merge")
        return self._pipeline(steps, wait_s, compatibility=report)

    def ctx_extend_start(self, model: str, factor: Optional[float], target: Optional[int], name: str, after: Optional[dict[str, Any]], wait_s: float) -> dict[str, Any]:
        self._require_env()
        art = self.lineage.resolve(model, ("base", "merged", "ctx_variant"))
        cfg = stio_config(art["path"])
        original = int((V.text_config(cfg).get("max_position_embeddings")) or 0)
        if target and original:
            factor = round(target / original, 4)
        if not factor or factor <= 1:
            raise PygmalionError("invalid", "ctx_factor_needed" if original else "ctx_factor_needed_unknown", original=original)
        steps = [{"kind": "ctx_extend", "title": msg("title_ctx_extend"), "params": {"model": art["id"], "factor": factor, "name": name or "", "original": original or None}}]
        if after:
            after = dict(after)
            if after.get("evaluate") is True or isinstance(after.get("evaluate"), dict) and not after["evaluate"].get("intent"):
                after["evaluate"] = {**(after["evaluate"] if isinstance(after["evaluate"], dict) else {}), "intent": "context"}
            steps += self.svc.planner.post_steps(hf="$ctx_variant", gguf=None, baseline_model=art["id"], after=after, name=name or f"{art['name']}-ctx{factor:g}x")
            steps = self.svc.planner.with_baseline(steps, art["id"], after)
        return self._pipeline(steps, wait_s, factor=factor, original=original, new_context=int(original * factor) if original else None)

    # ------------------------------------------------------------------ GGUF
    def convert_start(self, model: str, adapter: str, base: str, outtype: str, name: str, after: Optional[dict[str, Any]], wait_s: float) -> dict[str, Any]:
        self._require_env()
        if adapter:
            a = self.lineage.resolve(adapter, ("adapter",))
            b = self.lineage.resolve(base or (a["parents"][0] if a["parents"] else ""), ("base", "merged", "ctx_variant"))
            return self._pipeline([{"kind": "convert", "title": msg("title_convert_adapter"), "params": {"adapter": a["id"], "base": b["id"], "outtype": outtype, "name": name}}], wait_s)
        m = self.lineage.resolve(model, ("base", "merged", "ctx_variant"))
        if after:
            steps = self.svc.planner.post_steps(hf=m["id"], gguf=None, baseline_model=None, after={**after, "convert": outtype}, name=name or m["name"].rsplit("/", 1)[-1])
            if steps and steps[0]["kind"] == "convert":
                steps[0]["params"]["name"] = name or f"{m['name']}-{outtype}"
            ev = after.get("evaluate")
            if ev:       # a fine-tuned model is measured against the model its training started from, not against its own f16 file
                origin = self.svc.references.origin_model(m, ev.get("intent", "") if isinstance(ev, dict) else "")
                if origin is not None:
                    steps = self.svc.planner.with_baseline(steps, origin["id"], {**after, "convert": outtype})
        else:
            steps = [{"kind": "convert", "title": msg("title_convert"), "params": {"model": m["id"], "outtype": outtype, "name": name or None}}]
        return self._pipeline(steps, wait_s)

    def quantize_start(self, gguf: str, model: str, types: list[str], imatrix: bool, calibration: Optional[dict[str, Any]], chunks: Optional[int],
                       perplexity: bool, outtype: str, name: str, wait_s: float) -> dict[str, Any]:
        if not (gguf or model):
            raise PygmalionError("invalid", "quantize_needs_source")
        if gguf:
            src = self.lineage.resolve(gguf, ("gguf",))
            hf, gg, label = None, src["id"], name or src["name"].removesuffix(".gguf")
        else:
            m = self.lineage.resolve(model, ("base", "merged", "ctx_variant"))
            hf, gg, label = m["id"], None, name or m["name"].rsplit("/", 1)[-1]
            self._require_env()
        after = {"quantize": types, "imatrix": imatrix, "calibration": calibration, "chunks": chunks, "perplexity": perplexity, "convert": outtype}
        _types, warnings = LT.check_quant_types(types, imatrix)
        steps = self.svc.planner.post_steps(hf=hf, gguf=gg, baseline_model=None, after=after, name=label)
        return self._pipeline(steps, wait_s, warnings=warnings)

    def perplexity_start(self, gguf: str, text: Optional[dict[str, Any]], ctx: Optional[int], chunks: Optional[int], wait_s: float) -> dict[str, Any]:
        art = self.lineage.resolve(gguf, ("gguf",))
        job = self.jobs.submit("perplexity", {"gguf": art["id"], "text": text or {"source": "bundled"}, "ctx": ctx, "chunks": chunks}, title=msg("title_perplexity", name=art["name"]))
        return self._answer(job, wait_s)

    def ctx_fit(self, ref: str, contexts: Optional[list[int]], gpu: Optional[int]) -> dict[str, Any]:
        art = self.lineage.resolve(ref, ("gguf",))
        summary = summarize(read_metadata(art["path"]))
        inventory = self.svc.gpus.allowed_inventory()
        free = max([g["free_mb"] for g in inventory if gpu is None or g["index"] == gpu], default=None)
        total = max([g["total_mb"] for g in inventory if gpu is None or g["index"] == gpu], default=None)
        lengths = contexts or [c for c in CTX_LENGTHS if not summary.get("context_length") or c <= max(summary["context_length"] * 8, 8192)]
        return {"artifact": art["id"], "summary": summary, "free_mb": free, "total_mb": total,
                "rows": fit_table(summary, lengths, free_mb=total), "rows_free_now": fit_table(summary, lengths, free_mb=free),
                "note": msg("ctx_fit_note")}

    # ------------------------------------------------------------------ evaluation and publishing
    def evaluate_plan(self, artifact: str, against: str, intent: str, suites: Optional[list[str]], regression: bool = True) -> dict[str, Any]:
        return self.svc.evaluator.explain(artifact, against, intent, suites, regression)

    def evaluate_start(self, artifact: str, against: str, intent: str, suites: Optional[list[str]], wait_s: float, regression: bool = True) -> dict[str, Any]:
        """Queue an evaluation. The reference is the base model the training started from, written like the result, for every intent that asks
        whether a training helped (see ``reference.py``): when that file does not exist yet, the steps that make it are queued first, as one
        pipeline that ends in the evaluation."""
        plan = self.svc.evaluator.plan(artifact, against, intent, suites, regression)
        reference = plan["reference"]
        steps = list(reference["steps"])
        if any(s["kind"] == "convert" for s in steps):
            self._require_env()
            origin = reference["origin"]
            if origin is None or not (Path(origin["path"]) / "config.json").is_file():
                raise PygmalionError("not_found", "model_folder_missing", name=reference["model"], path=origin["path"] if origin else "")
        params: dict[str, Any] = {"artifact": plan["child"]["id"], "against": reference["ref"], "intent": plan["intent"],
                                  "reference": {"mode": reference["mode"], "model": reference["model"], "quant": reference["quant"], "imatrix": bool(reference["imatrix"])}}
        if plan["intent"] == "dataset":
            params["regression"] = plan["regression"]          # the suite is built from the dataset when the job runs
        else:
            params["suites"] = plan["suites"]
        title = msg("title_evaluate", name=plan["child"]["name"])
        shown = {"child": plan["child"]["name"], "parent": plan["parent"]["name"] if plan["parent"] else f"{reference['model']} ({reference['quant']})",
                 "suites": plan["suites"], "intent": plan["intent"], "dataset": plan["dataset"], "reference": self.svc.references.view(reference)}
        if not steps:
            return self._answer(self.jobs.submit("evaluate", params, title=title), wait_s, plan=shown)
        steps.append({"kind": "evaluate", "title": title, "params": params})
        return self._pipeline(steps, wait_s, plan=shown)

    def publish_start(self, gguf: str, target: str, params: dict[str, Any], wait_s: float) -> dict[str, Any]:
        art = self.lineage.resolve(gguf, ("gguf",))
        if target in ("ollama", "both"):
            self.svc.publisher.ollama_plan(art["id"], params.get("name", ""), params.get("tag", "latest"), params.get("num_ctx"), params.get("adapter", ""))
        job = self.jobs.submit("publish", {"gguf": art["id"], "target": target, **{k: v for k, v in params.items() if v not in (None, "")}}, title=msg("title_publish", name=art["name"]))
        return self._answer(job, wait_s)

    def jobs_list(self, states: Optional[list[str]], kind: str, limit: int) -> dict[str, Any]:
        if kind and kind not in KINDS:
            raise PygmalionError("invalid", "job_kind_unknown", kind=kind, options=list(KINDS))
        rows = self.store.jobs(states=states or None, kind=kind, limit=limit)
        return {"jobs": [self.jobs.view(j) for j in rows], "scheduler": self.jobs.status()}


def stio_config(path: str) -> dict[str, Any]:
    import json
    try:
        return json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PygmalionError("not_found", "config_unreadable", path=path) from exc
