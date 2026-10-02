"""Download a model from Hugging Face into the work folder, resumable, with progress.

Arguments: ``repo_id``, ``local_dir``, ``revision`` (optional), ``trust_remote_code`` (default false: Python files of the repository are
then not downloaded). The token comes from the environment (``HF_TOKEN``), never from the arguments file.

Only weights in safetensors, configs, tokenizer files and templates are fetched. Progress is the size of the folder (including the
``.incomplete`` files of the library) against the total the Hub reports, so it keeps moving during a single big shard.
"""

from __future__ import annotations

import fnmatch
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

BASE_PATTERNS = ["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt", "*.jinja"]


def patterns_for(trust_remote_code: bool) -> list[str]:
    return BASE_PATTERNS + (["*.py"] if trust_remote_code else [])


def expected_bytes(siblings: list[Any], patterns: list[str]) -> int:
    total = 0
    for s in siblings:
        name = getattr(s, "rfilename", "")
        if any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(os.path.basename(name), p) for p in patterns):
            total += getattr(s, "size", None) or 0
    return total


def folder_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def download(args: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download
    repo, local = args["repo_id"], Path(args["local_dir"])
    token = os.environ.get("HF_TOKEN") or None
    patterns = patterns_for(bool(args.get("trust_remote_code", False)))
    revision = args.get("revision") or None
    try:
        info = HfApi().model_info(repo, revision=revision, files_metadata=True, token=token)
    except Exception as exc:  # noqa: BLE001
        text = f"{type(exc).__name__}: {exc}"
        raise C.WorkerError("hf_read_failed", C.hint_for(text), repo=repo, detail=text[:300]) from exc
    total = expected_bytes(info.siblings or [], patterns)
    names = [getattr(s, "rfilename", "") for s in info.siblings or []]
    if not any(n.endswith(".safetensors") for n in names):
        raise C.WorkerError("repo_no_safetensors", repo=repo)
    local.mkdir(parents=True, exist_ok=True)
    C.log(f"Downloading {repo} ({total / 1e9:.2f} GB expected) to {local}")
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            snapshot_download(repo_id=repo, revision=revision, local_dir=str(local), allow_patterns=patterns, token=token)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    started, last = time.monotonic(), 0
    while thread.is_alive():
        thread.join(2.0)
        if C.STOP.is_set():
            raise C.WorkerError("download_stopped")
        done = folder_bytes(local)
        speed = (done - last) / 2.0 if last else 0
        last = done
        C.emit("progress", step=int(done / 1048576), total=int(total / 1048576) or None, eta_s=int((total - done) / speed) if speed > 0 and total > done else None,
               mb_per_s=round(speed / 1048576, 1))
    if "error" in box:
        text = f"{type(box['error']).__name__}: {box['error']}"
        raise C.WorkerError("download_failed", C.hint_for(text), detail=text[:300])
    if not (local / "config.json").is_file():
        raise C.WorkerError("download_no_config")
    result = {"repo_id": repo, "path": str(local), "bytes": folder_bytes(local), "seconds": int(time.monotonic() - started),
              "revision": info.sha}
    C.emit("artifact", path=str(local), kind="base")
    C.emit("result", data=result)
    return result


def main(args: dict[str, Any]) -> None:
    download(args)


if __name__ == "__main__":
    sys.exit(C.run_worker(main))
