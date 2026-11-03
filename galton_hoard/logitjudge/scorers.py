"""Where the option probabilities come from.

* ``ServerScorer``: an chat-completions server that returns ``logprobs`` with ``top_logprobs`` (llama.cpp server, vLLM, recent Ollama).
  One request per question: ``max_tokens=1``, ``temperature=0``, and the alternatives of the first generated token are read.
* ``HFRunner``: a Hugging Face model (and optionally a LoRA adapter) in the trainer environment. A worker process does one forward pass per
  question and reads the logits of the label tokens, for the base model and, in the same process, with the adapter on.

A server that ignores the logprobs request is reported as such (``lj_no_logprobs``): there is no fallback to a generated answer, because
that would not be the same measurement.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from ..errors import GaltonError

TOP_DEFAULT = 20
WORKER = Path(__file__).resolve().parents[2] / "pygmalion_hoard" / "workers" / "logit_judge.py"


def label_of(token: str) -> str:
    """The label a returned token stands for: surrounding spaces and the word-start marks of byte-pair and sentencepiece vocabularies removed."""
    return token.replace("▁", " ").replace("Ġ", " ").strip()


def labels_from_top(top: list[dict[str, Any]], labels: list[str]) -> dict[str, Optional[float]]:
    """Log-probability of each label from a list of ``{token, logprob}``. Two tokens of the same label (``A`` and `` A``) are summed in probability space.
    A label missing from the list (below the top N) is ``None``."""
    found: dict[str, list[float]] = {}
    for entry in top:
        if not isinstance(entry, dict) or not isinstance(entry.get("logprob"), (int, float)):
            continue
        label = label_of(str(entry.get("token", "")))
        if label in labels:
            found.setdefault(label, []).append(float(entry["logprob"]))
    out: dict[str, Optional[float]] = {}
    for label in labels:
        values = found.get(label)
        if not values:
            out[label] = None
        else:
            top_value = max(values)
            out[label] = top_value + math.log(sum(math.exp(v - top_value) for v in values))
    return out


def top_of_reply(data: Any) -> Optional[list[dict[str, Any]]]:
    """The alternatives for the first generated token, from a chat-completions reply (``logprobs.content[0].top_logprobs``) or a legacy completions reply
    (``logprobs.top_logprobs[0]`` as a mapping). None when the server sent no logprobs."""
    try:
        choice = data["choices"][0]
        lp = choice.get("logprobs")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    if not isinstance(lp, dict):
        return None
    content = lp.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        first = content[0]
        top = first.get("top_logprobs")
        if isinstance(top, list) and top:
            return top
        if isinstance(first.get("logprob"), (int, float)):                       # the sampled token only, no alternatives: marked, never read as a distribution
            return [{"token": first.get("token", ""), "logprob": first["logprob"], "sampled_only": True}]
    legacy = lp.get("top_logprobs")
    if isinstance(legacy, list) and legacy and isinstance(legacy[0], dict):
        return [{"token": k, "logprob": v} for k, v in legacy[0].items()]
    return None


class ServerScorer:
    def __init__(self, contestant: dict[str, Any], *, client_factory: Optional[Callable[[], httpx.Client]] = None, top_logprobs: int = TOP_DEFAULT,
                 timeout_s: float = 120.0, no_thinking: bool = True, api_key: str = ""):
        self.contestant, self.top, self.timeout, self.no_thinking, self.api_key = contestant, max(1, min(int(top_logprobs), 50)), timeout_s, no_thinking, api_key
        self.client_factory = client_factory or (lambda: httpx.Client(timeout=timeout_s))
        self.url = str(contestant.get("url") or "").rstrip("/")
        if self.url.endswith("/v1"):                                              # registered servers keep the OpenAI base (…/v1)
            self.url = self.url[:-3]
        self.model = str(contestant.get("model") or contestant.get("name") or "")

    def _post(self, client: httpx.Client, messages: list[dict[str, str]], *, thinking_flag: bool) -> httpx.Response:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": self.top, "stream": False}
        if thinking_flag:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return client.post(f"{self.url}/v1/chat/completions", json=body, headers=headers, timeout=self.timeout)

    def score(self, client: httpx.Client, messages: list[dict[str, str]], labels: list[str]) -> dict[str, Optional[float]]:
        try:
            resp = self._post(client, messages, thinking_flag=self.no_thinking)
            if resp.status_code in (400, 422) and self.no_thinking:             # a strict server that refuses the extra field
                resp = self._post(client, messages, thinking_flag=False)
        except httpx.HTTPError as exc:
            raise GaltonError("unavailable", "lj_server_unreachable", name=self.contestant.get("name", self.url), error=type(exc).__name__) from exc
        if resp.status_code >= 400:
            raise GaltonError("unavailable", "lj_server_error", name=self.contestant.get("name", self.url), http=resp.status_code, body=resp.text[:200])
        try:
            data = resp.json()
        except ValueError as exc:
            raise GaltonError("unavailable", "lj_server_error", name=self.contestant.get("name", self.url), http=resp.status_code, body=resp.text[:200]) from exc
        top = top_of_reply(data)
        if top is None:
            raise GaltonError("unsupported", "lj_no_logprobs", name=self.contestant.get("name", self.url))
        if len(top) == 1 and top[0].get("sampled_only"):                       # one token at probability 1 would fake a certain answer
            raise GaltonError("unsupported", "lj_no_alternatives", name=self.contestant.get("name", self.url))
        out = labels_from_top(top, labels)
        if all(v is None for v in out.values()):
            raise GaltonError("invalid", "lj_labels_not_in_top", name=self.contestant.get("name", self.url), top=len(top), labels=" ".join(labels))
        return out

    def probe(self, client: httpx.Client) -> dict[str, Any]:
        """Ask a question with two labels and report whether logprobs came back (never raises for an unsupported server)."""
        messages = [{"role": "user", "content": "Answer with the letter A or B only. Question: is water wet? A) yes B) no"}]
        try:
            self.score(client, messages, ["A", "B"])
            return {"supported": True, "name": self.contestant.get("name"), "url": self.url, "top_logprobs": self.top}
        except GaltonError as exc:
            info = {"name": self.contestant.get("name"), "url": self.url, "reachable": exc.key != "lj_server_unreachable"}
            if exc.key == "lj_labels_not_in_top":                      # logprobs came back, the model just did not put the probe's labels in its top tokens
                return {**info, "supported": True, "top_logprobs": self.top, "note": exc.item()}
            return {**info, "supported": False, "problem": exc.item()}


class HFRunner:
    """Runs ``pygmalion_hoard/workers/logit_judge.py`` under the trainer's Python. ``cancel`` kills the process."""

    def __init__(self, python: str, *, work_dir: Path, gpus: list[int], timeout_s: float = 1800.0, worker: Optional[Path] = None):
        self.python, self.work_dir, self.gpus, self.timeout, self.worker = python, Path(work_dir), gpus, timeout_s, Path(worker or WORKER)
        self.proc: Optional[subprocess.Popen] = None
        self.cancelled = threading.Event()
        self.timed_out = threading.Event()

    def kill(self) -> None:
        self.cancelled.set()
        self._stop()

    def _timeout(self) -> None:
        self.timed_out.set()
        self._stop()

    def _stop(self) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.kill()

    def run(self, args: dict[str, Any], on_log: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
        if not self.python or not Path(self.python).is_file():
            raise GaltonError("unavailable", "lj_no_python")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        args_path = self.work_dir / f"logit-args-{os.getpid()}-{threading.get_ident()}.json"
        args_path.write_text(json.dumps(args, ensure_ascii=False), encoding="utf-8")
        env = {**os.environ, "PYTHONUTF8": "1", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpus) if (self.gpus and args.get("device") != "cpu") else ""
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        result: dict[str, Any] = {}
        error: dict[str, Any] = {}
        timer = None
        try:
            self.proc = subprocess.Popen([self.python, str(self.worker), "--args", str(args_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace", env=env, creationflags=flags)
            timer = threading.Timer(self.timeout, self._timeout)
            timer.start()
            if self.cancelled.is_set():                                         # cancelled while the process was being started
                self._stop()
            for line in self.proc.stdout or []:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                kind = event.get("event") if isinstance(event, dict) else None
                if kind == "result":
                    result = event.get("data") or {}
                elif kind == "error":
                    error = event
                elif kind == "log":
                    on_log(str(event.get("msg", "")))
            code = self.proc.wait()
        finally:
            if timer is not None:
                timer.cancel()
            try:
                args_path.unlink()
            except OSError:
                pass
        if self.cancelled.is_set():
            raise GaltonError("conflict", "lj_cancelled")
        if self.timed_out.is_set():
            raise GaltonError("unavailable", "lj_worker_timeout", seconds=int(self.timeout))
        if error:
            raise GaltonError("unavailable", "lj_worker_failed", detail=str(error.get("msg", ""))[:300], hint_text=str(error.get("hint", ""))[:200])
        if code != 0 or not result:
            raise GaltonError("unavailable", "lj_worker_exit", exit_code=code)
        return result
