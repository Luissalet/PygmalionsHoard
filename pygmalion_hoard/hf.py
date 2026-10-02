"""Base models: search on the Hugging Face Hub (the only network use besides downloads) and scan the work folder for what is
already on disk, with badges for what each model supports."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import httpx

from . import vram as V
from .convert_detect import convertible, model_architectures
from .errors import PygmalionError
from .util import dir_size, read_json
from .workers.download import expected_bytes, patterns_for

HUB = "https://huggingface.co"
PIPELINES = ("text-generation", "image-text-to-text")
SORTS = ("downloads", "likes", "lastModified", "trendingScore")
BYTES_PER = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "I4": 0.5}
REPO_RE = re.compile(r"^[\w.\-]{1,96}/[\w.\-]{1,96}$")


def check_repo(repo_id: str) -> str:
    repo = (repo_id or "").strip()
    if not REPO_RE.match(repo):
        raise PygmalionError("invalid", "repo_id_invalid", repo=repo_id)
    return repo


class HubClient:
    """The few Hub API calls the app makes. ``transport`` lets tests answer without a network."""

    def __init__(self, token: Callable[[], str] = lambda: "", transport: Optional[httpx.BaseTransport] = None, offline: bool = False,
                 timeout_s: float = 25.0):
        self.token = token
        self.transport = transport
        self.offline = offline
        self.timeout_s = timeout_s

    def _get(self, path: str, params: Optional[list[tuple[str, str]]] = None) -> Any:
        if self.offline:
            raise PygmalionError("offline", "offline_mode")
        headers = {"Accept": "application/json", "User-Agent": "pygmalion-hoard"}
        token = self.token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with httpx.Client(transport=self.transport, timeout=self.timeout_s, headers=headers, trust_env=self.transport is None) as client:
                response = client.get(HUB + path, params=params)
        except httpx.HTTPError as exc:
            raise PygmalionError("offline", "hf_unreachable", detail=type(exc).__name__) from exc
        if response.status_code == 404:
            raise PygmalionError("not_found", "hf_not_found")
        if response.status_code in (401, 403):
            raise PygmalionError("forbidden", "hf_refused")
        if response.status_code >= 400:
            raise PygmalionError("failed", "hf_status", status=response.status_code)
        try:
            return response.json()
        except ValueError as exc:
            raise PygmalionError("failed", "hf_unreadable") from exc

    @staticmethod
    def _license(tags: list[str]) -> str:
        return next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), "")

    @classmethod
    def _bytes_from_safetensors(cls, info: Any) -> Optional[int]:
        if not isinstance(info, dict):
            return None
        params = info.get("parameters")
        if isinstance(params, dict) and params:
            return int(sum(count * BYTES_PER.get(dtype, 2) for dtype, count in params.items()))
        total = info.get("total")
        return int(total * 2) if isinstance(total, (int, float)) else None

    def _card(self, m: dict[str, Any]) -> dict[str, Any]:
        tags = [str(t) for t in m.get("tags") or []]
        config = m.get("config") if isinstance(m.get("config"), dict) else {}
        archs = config.get("architectures") or []
        st = m.get("safetensors") if isinstance(m.get("safetensors"), dict) else {}
        return {"repo_id": m.get("id") or m.get("modelId") or "", "downloads": m.get("downloads"), "likes": m.get("likes"),
                "pipeline": m.get("pipeline_tag") or "", "gated": bool(m.get("gated")), "license": self._license(tags),
                "architecture": archs[0] if archs else "", "params": st.get("total"), "approx_bytes": self._bytes_from_safetensors(st),
                "has_safetensors": bool(st) or "safetensors" in tags, "updated": m.get("lastModified"), "private": bool(m.get("private"))}

    def search(self, text: str, *, pipeline: str = "text-generation", sort: str = "downloads", limit: int = 20) -> list[dict[str, Any]]:
        if pipeline and pipeline not in PIPELINES:
            raise PygmalionError("invalid", "hf_pipeline_unknown", pipeline=pipeline, options=list(PIPELINES))
        if sort not in SORTS:
            raise PygmalionError("invalid", "hf_sort_unknown", sort=sort, options=list(SORTS))
        params: list[tuple[str, str]] = [("search", text.strip()), ("sort", sort), ("direction", "-1"), ("limit", str(max(1, min(limit, 50))))]
        if pipeline:
            params.append(("pipeline_tag", pipeline))
        for field in ("downloads", "likes", "pipeline_tag", "tags", "gated", "safetensors", "config", "lastModified", "private"):
            params.append(("expand[]", field))
        data = self._get("/api/models", params)  # a list of pairs keeps the repeated expand[] keys
        return [self._card(m) for m in data if isinstance(m, dict)] if isinstance(data, list) else []

    def info(self, repo_id: str) -> dict[str, Any]:
        """The repository's files with sizes and what a download of it would fetch."""
        repo = check_repo(repo_id)
        data = self._get(f"/api/models/{repo}", [("blobs", "true")])
        siblings = [SimpleNamespace(rfilename=s.get("rfilename", ""), size=s.get("size")) for s in data.get("siblings") or []]
        names = [s.rfilename for s in siblings]
        weights = [n for n in names if n.endswith(".safetensors")]
        card = self._card({**data, "id": data.get("id", repo)})
        download_bytes = expected_bytes(siblings, patterns_for(False))
        return {**card, "files": len(names), "weights": len(weights), "download_bytes": download_bytes,
                "has_python_files": any(n.endswith(".py") for n in names), "sha": data.get("sha")}


# ------------------------------------------------------------------ local scan
def is_model_dir(folder: Path) -> bool:
    return (folder / "config.json").is_file() and any(folder.glob("*.safetensors"))


def repo_of_folder(name: str) -> str:
    """The Hugging Face repository a work-folder name stands for (``acme--tiny`` is ``acme/tiny``); empty for a plain folder name."""
    return name.replace("--", "/", 1) if "--" in name else ""


def local_card(folder: Path, convert_script: Optional[str] = None, trainable_archs: Optional[set[str]] = None) -> dict[str, Any]:
    config = read_json(folder / "config.json", {}) or {}
    arch = (model_architectures(folder) or [""])[0]
    params = V.estimate_params(config)
    dims = V.dims(config)
    text = V.text_config(config)
    conv = convertible(folder, convert_script)
    trainable: Optional[bool] = None if trainable_archs is None else arch in trainable_archs
    repo = repo_of_folder(folder.name)
    return {"name": folder.name, "repo_id": repo, "path": str(folder), "size": dir_size(folder), "architecture": arch,
            "model_type": config.get("model_type", ""), "params": params, "params_b": round(params / 1e9, 2) if params else None,
            "hidden": int(dims["hidden"]), "layers": int(dims["layers"]), "vocab": int(dims["vocab"]),
            "context": text.get("max_position_embeddings"), "multimodal": isinstance(config.get("text_config"), dict),
            "moe": bool(dims["experts"]), "trainable": trainable, "convertible": conv["known"], "convert_note": conv["note"],
            "dtype": config.get("torch_dtype") or text.get("torch_dtype") or config.get("dtype")}


def local_model_folders(hf_dir: Path) -> list[Path]:
    """The model folders in the work folder, sorted: the one definition of "a base on disk" for the list, the badges and the counts."""
    if not hf_dir.is_dir():
        return []
    try:
        return [f for f in sorted(hf_dir.iterdir()) if f.is_dir() and is_model_dir(f)]
    except OSError:
        return []


def scan_local(hf_dir: Path, convert_script: Optional[str] = None, trainable_archs: Optional[set[str]] = None) -> list[dict[str, Any]]:
    return [local_card(f, convert_script, trainable_archs) for f in local_model_folders(hf_dir)]
