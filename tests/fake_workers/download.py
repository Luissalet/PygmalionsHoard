"""Stand-in for workers/download.py: writes a tiny model folder into local_dir. FAKE_DOWNLOAD_FAIL makes it fail like a gated repository."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pygmalion_hoard" / "workers"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common as C  # noqa: E402
from models import make_model  # noqa: E402


def main(args):
    if os.environ.get("FAKE_DOWNLOAD_FAIL"):
        raise RuntimeError("401 Client Error: gated repo")
    local = Path(args["local_dir"])
    print("using token " + os.environ.get("HF_TOKEN", ""), flush=True)
    for i in range(1, 4):
        C.emit("progress", step=i * 10, total=30, mb_per_s=12.5, eta_s=3 - i)
    make_model(local, seed=7)
    (local / "token-seen.txt").write_text(os.environ.get("HF_TOKEN", ""), encoding="utf-8")
    C.emit("artifact", path=str(local), kind="base")
    C.emit("result", data={"repo_id": args["repo_id"], "path": str(local), "bytes": 1000, "seconds": 1, "revision": "abc123"})


if __name__ == "__main__":
    sys.exit(C.run_worker(main))
