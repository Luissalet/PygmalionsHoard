"""The capture proxy: a chat-completions endpoint in front of a local model server that records every call with the token ids it ran on.

Two kinds of server are understood:

* ``vllm``: asked for ``logprobs`` and ``return_token_ids``; the answer carries ``prompt_token_ids`` (the prompt after the chat template),
  ``choices[].token_ids`` and the log-probability of each sampled token. Nothing else is needed.
* ``llama_server`` (llama.cpp): the answer carries the id and log-probability of each sampled token (``logprobs.content[]``) but not the ids of
  the prompt. They are rebuilt with the server's own ``POST /apply-template`` (messages and tools through the model's chat template) and
  ``POST /tokenize``. The rebuilt count is checked against ``usage.prompt_tokens``: when it does not match, the call is stored as *unverified*
  and left out of the exports unless asked for. ``/apply-template`` renders the tools only when the server runs with ``--jinja``; images and
  other non-text parts of a message make the count differ too, which is what the check is for.

The client always gets an ordinary chat-completions answer: the extra fields are removed, a streaming request is served from the complete answer
(the server is always asked without streaming, so a call is recorded whole) and the sampling parameters the client sent are not touched, unless
``capture.faithful_sampling`` is on: then ``top_k``, ``top_p`` and ``min_p`` are neutralised so the logged log-probabilities describe the
distribution the token was drawn from (the setting exists for reinforcement learning, where a truncated sampler biases the estimate).
"""

from __future__ import annotations

import copy
import json
import re
import time
from typing import Any, Callable, Iterator, Optional

import httpx

from ..settings import Settings
from .store import CaptureStore

UPSTREAM_TYPES = ("auto", "llama_server", "vllm")
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "max_tokens", "max_completion_tokens", "seed", "repetition_penalty", "frequency_penalty",
                 "presence_penalty", "stop", "n")
TEMPLATE_KEYS = ("tools", "tool_choice", "parallel_tool_calls", "chat_template_kwargs")
NAME_RE = re.compile(r"^[A-Za-z0-9_.:@\-]{1,80}$")
DUMMY_KEYS = {"", "none", "null", "empty", "no-key", "not-needed", "sk-no-key-required", "sk-none", "dummy", "local", "lm-studio", "ollama"}
DEFAULT_SESSION = "default"
TOKEN_ID = re.compile(r"^token_id:(\d+)$")
MAX_TOP = 20


class ProxyError(Exception):
    """A failure the proxy reports to its client as a chat-completions error body."""

    def __init__(self, status: int, message: str, kind: str = "capture_error"):
        super().__init__(message)
        self.status, self.message, self.kind = status, message, kind

    def body(self) -> dict[str, Any]:
        return {"error": {"message": self.message, "type": self.kind, "code": self.kind}}


class Reply:
    """What the router sends back: a status, a JSON body, headers, and (when the client asked to stream) the body to emit as events."""

    def __init__(self, status: int, body: Any, headers: Optional[dict[str, str]] = None, stream: bool = False, include_usage: bool = False,
                 client_logprobs: bool = False):
        self.status, self.body, self.headers = status, body, headers or {}
        self.stream, self.include_usage, self.client_logprobs = stream, include_usage, client_logprobs

    def events(self) -> Iterator[bytes]:
        return sse_events(self.body, include_usage=self.include_usage, with_logprobs=self.client_logprobs)


def clean_name(value: Optional[str]) -> str:
    value = (value or "").strip()
    return value if NAME_RE.match(value) else ""


def identify(path_harness: str, path_session: str, headers: Any) -> tuple[str, str]:
    """``(harness, session)`` of a request: the URL first, then the ``X-Capture-*`` headers, then the API key (``session`` or ``harness:session``)."""
    harness, session = clean_name(path_harness), clean_name(path_session)
    harness = harness or clean_name(headers.get("x-capture-harness"))
    session = session or clean_name(headers.get("x-capture-session"))
    if not session:
        auth = (headers.get("authorization") or "").strip()
        key = auth[7:].strip() if auth.lower().startswith("bearer ") else auth
        if key.lower() not in DUMMY_KEYS and NAME_RE.match(key):
            if ":" in key:
                head, _, tail = key.partition(":")
                harness, session = harness or clean_name(head), clean_name(tail)
            else:
                session = key
    return harness, session or DEFAULT_SESSION


class CaptureProxy:
    def __init__(self, store: CaptureStore, settings: Settings, *, clock: Callable[[], float] = time.time,
                 transport: Optional[httpx.BaseTransport] = None):
        self.store = store
        self.settings = settings
        self.clock = clock
        self._transport = transport
        self._client: Optional[httpx.Client] = None
        self._detected: dict[str, tuple[str, str]] = {}
        self._eos: dict[str, Optional[int]] = {}

    # ------------------------------------------------------------------ configuration
    def config(self) -> dict[str, Any]:
        s = self.settings
        return {"enabled": s.bool("capture.enabled"), "upstream_url": s.get("capture.upstream_url").strip(), "upstream_type": s.get("capture.upstream_type"),
                "top_logprobs": max(1, min(MAX_TOP, s.int("capture.top_logprobs"))), "faithful_sampling": s.bool("capture.faithful_sampling"),
                "timeout_s": max(5, s.int("capture.timeout_s"))}

    @staticmethod
    def base_of(url: str) -> str:
        url = (url or "").strip().rstrip("/")
        for suffix in ("/v1/chat/completions", "/v1"):
            if url.endswith(suffix):
                url = url[: -len(suffix)]
        return url.rstrip("/")

    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(transport=self._transport, timeout=httpx.Timeout(self.config()["timeout_s"], connect=10.0))
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def forget(self) -> None:
        """Drop what was learned about the upstream (its type, its end-of-turn token): after the address or the server changed."""
        self._detected.clear()
        self._eos.clear()

    # ------------------------------------------------------------------ the upstream
    def detect(self, base: str, force: bool = False) -> tuple[str, str]:
        """``(type, detail)`` of the server at ``base``: llama.cpp answers ``GET /props``, vLLM answers ``GET /version``."""
        if not force and base in self._detected:
            return self._detected[base]
        found = ("unknown", "no answer")
        try:
            r = self.client().get(f"{base}/props", timeout=8.0)
            data = _json(r)
            if r.status_code == 200 and isinstance(data, dict) and any(k in data for k in ("default_generation_settings", "build_info", "total_slots", "model_path")):
                found = ("llama_server", str(data.get("build_info") or "llama.cpp"))
            else:
                r = self.client().get(f"{base}/version", timeout=8.0)
                data = _json(r)
                if r.status_code == 200 and isinstance(data, dict) and "version" in data:
                    found = ("vllm", str(data["version"]))
                else:
                    found = ("unknown", f"/props {r.status_code}")
        except httpx.HTTPError as exc:
            found = ("unreachable", type(exc).__name__)
        if found[0] not in ("unreachable",):
            self._detected[base] = found
        return found

    def resolve_type(self, base: str, configured: str) -> str:
        if configured in ("llama_server", "vllm"):
            return configured
        kind = self.detect(base)[0]
        return kind if kind in ("llama_server", "vllm") else "unknown"

    def check(self) -> dict[str, Any]:
        """Is the upstream reachable, which kind is it and which models does it serve. Never raises."""
        cfg = self.config()
        base = self.base_of(cfg["upstream_url"])
        if not base:
            return {"configured": False}
        kind, detail = self.detect(base, force=True)
        out: dict[str, Any] = {"configured": True, "reachable": kind != "unreachable", "detected": kind, "detail": detail,
                               "type": cfg["upstream_type"] if cfg["upstream_type"] != "auto" else kind}
        try:
            r = self.client().get(f"{base}/v1/models", timeout=8.0)
            data = _json(r)
            if r.status_code == 200 and isinstance(data, dict):
                out["models"] = [str(m.get("id")) for m in (data.get("data") or []) if isinstance(m, dict)][:20]
                out["reachable"] = True
        except httpx.HTTPError:
            pass
        return out

    def models(self) -> Reply:
        cfg = self.config()
        base = self._require(cfg)
        try:
            r = self.client().get(f"{base}/v1/models", timeout=15.0)
        except httpx.HTTPError as exc:
            raise ProxyError(502, f"the upstream did not answer: {type(exc).__name__}", "upstream_unreachable") from exc
        body = _json(r)
        return Reply(r.status_code, body if body is not None else {"error": {"message": r.text[:300], "type": "upstream_error"}})

    @staticmethod
    def _require(cfg: dict[str, Any]) -> str:
        if not cfg["enabled"]:
            raise ProxyError(503, "the capture proxy is stopped; start it in Pygmalion's Hoard (Capture)", "capture_stopped")
        base = CaptureProxy.base_of(cfg["upstream_url"])
        if not base:
            raise ProxyError(503, "no upstream is configured for the capture proxy", "capture_not_configured")
        return base

    # ------------------------------------------------------------------ one call
    def chat(self, harness: str, session: str, body: dict[str, Any]) -> Reply:
        cfg = self.config()
        base = self._require(cfg)
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            raise ProxyError(400, "the request needs a messages list", "invalid_request_error")
        kind = self.resolve_type(base, cfg["upstream_type"])
        if kind == "unreachable":
            kind = "unknown"
        stream = bool(body.get("stream"))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage")) if isinstance(body.get("stream_options"), dict) else False
        client_logprobs = bool(body.get("logprobs"))
        up = self.upstream_request(body, kind, cfg)
        started = time.monotonic()
        try:
            r = self.client().post(f"{base}/v1/chat/completions", json=up)
        except httpx.HTTPError as exc:
            raise ProxyError(502, f"the upstream did not answer: {type(exc).__name__}: {str(exc)[:160]}", "upstream_unreachable") from exc
        latency = int((time.monotonic() - started) * 1000)
        data = _json(r)
        if r.status_code >= 400 or not isinstance(data, dict):
            return Reply(r.status_code if r.status_code >= 400 else 502,
                         data if isinstance(data, dict) else {"error": {"message": (r.text or "")[:300], "type": "upstream_error"}})
        if kind == "unknown":
            kind = "vllm" if "prompt_token_ids" in data else ("llama_server" if _has_ids(data) else "unknown")
        nodes = []
        for index, choice in enumerate(data.get("choices") or []):
            call = self.build_call(base, kind, body, data, choice, cfg, latency)
            call["upstream"] = base
            node = self.store.add_call(session, harness, call)
            nodes.append(node)
            if index == 0:
                first = node
        headers = {"X-Capture-Session": session}
        if nodes:
            headers.update({"X-Capture-Node": first["id"], "X-Capture-Verified": "1" if first["verified"] else "0"})
        return Reply(200, self.client_view(data, client_logprobs), headers, stream=stream, include_usage=include_usage, client_logprobs=client_logprobs)

    def upstream_request(self, body: dict[str, Any], kind: str, cfg: dict[str, Any]) -> dict[str, Any]:
        up = copy.deepcopy(body)
        up["stream"] = False
        up.pop("stream_options", None)
        wanted = cfg["top_logprobs"]
        asked = int(body.get("top_logprobs") or 0) if body.get("logprobs") else 0
        up["logprobs"] = True
        up["top_logprobs"] = min(MAX_TOP, max(wanted, asked))
        if kind == "vllm":
            up["return_token_ids"] = True
            if not body.get("logprobs"):
                up["return_tokens_as_token_ids"] = True      # the client did not ask for logprobs, so naming the tokens by id changes nothing it sees
        if cfg["faithful_sampling"]:
            if kind == "llama_server":
                up.update(top_k=0, top_p=1.0, min_p=0.0)     # the server's own defaults would truncate the distribution if the keys were just dropped
            else:
                for key in ("top_k", "top_p", "min_p"):
                    up.pop(key, None)
        return up

    # ------------------------------------------------------------------ what a call is made of
    def build_call(self, base: str, kind: str, body: dict[str, Any], data: dict[str, Any], choice: dict[str, Any], cfg: dict[str, Any],
                   latency_ms: int) -> dict[str, Any]:
        notes: list[str] = []
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        finish = str(choice.get("finish_reason") or "")
        content = ((choice.get("logprobs") or {}).get("content") or []) if isinstance(choice.get("logprobs"), dict) else []
        pieces: Optional[list[str]] = None
        top: Optional[list[Any]] = None
        completion: list[int] = []
        logprobs: list[Optional[float]] = []
        prompt: list[int] = []
        source = "none"
        if kind == "vllm":
            ids = choice.get("token_ids")
            if isinstance(ids, list) and all(isinstance(i, int) for i in ids):
                completion = list(ids)
            else:
                notes.append("no token_ids in the answer")
            logprobs = [_float(c.get("logprob")) for c in content]
            named = [TOKEN_ID.match(str(c.get("token", ""))) for c in content]
            if completion and content:
                if len(content) != len(completion):
                    notes.append(f"{len(content)} logprobs for {len(completion)} tokens")
                elif all(named) and [int(m.group(1)) for m in named if m] != completion:
                    notes.append("the logprob tokens do not match token_ids")
            elif content and not completion:
                completion = [int(m.group(1)) for m in named if m]
            if cfg["top_logprobs"] > 1:
                top = [[[_top_id(t), _float(t.get("logprob"))] for t in (c.get("top_logprobs") or [])] for c in content]
            pids = data.get("prompt_token_ids")
            if isinstance(pids, list) and all(isinstance(i, int) for i in pids):
                prompt, source = list(pids), "upstream"
            else:
                notes.append("no prompt_token_ids in the answer")
            if prompt and usage.get("prompt_tokens") is not None and int(usage["prompt_tokens"]) != len(prompt):
                notes.append(f"prompt ids {len(prompt)} but usage says {usage['prompt_tokens']}")
            if completion and usage.get("completion_tokens") is not None and int(usage["completion_tokens"]) != len(completion):
                notes.append(f"completion ids {len(completion)} but usage says {usage['completion_tokens']}")
        elif kind == "llama_server":
            if content and all(isinstance(c, dict) and isinstance(c.get("id"), int) for c in content):
                completion = [int(c["id"]) for c in content]
                logprobs = [_float(c.get("logprob")) for c in content]
                pieces = [str(c.get("token", "")) for c in content]
                if cfg["top_logprobs"] > 1:
                    top = [[[t.get("id"), _float(t.get("logprob"))] for t in (c.get("top_logprobs") or [])] for c in content]
            else:
                notes.append("no token ids in the answer's logprobs")
            prompt, source, found = self.llama_prompt(base, body, usage)
            notes += found
            if completion:
                extra = self.close_turn(base, completion, logprobs, usage, finish)
                if extra == "appended":
                    pieces = (pieces or []) + [""]
                    notes.append("end-of-turn token appended (the server did not report it)")
                elif extra:
                    notes.append(extra)
        else:
            notes.append("the upstream is neither llama.cpp nor vLLM: no token ids")
        logprobs = logprobs + [None] * (len(completion) - len(logprobs))
        known = [x for x in logprobs[: len(completion)] if x is not None]
        flags: dict[str, Any] = {"lp_known": len(known)}
        if known:
            flags.update(lp_mean=round(sum(known) / len(known), 4), lp_min=round(min(known), 4))
        if any("end-of-turn token appended" in n for n in notes):
            flags["eos_appended"] = True
        verified = bool(prompt) and bool(completion) and not [n for n in notes if _blocking(n)]
        message = choice.get("message") if isinstance(choice.get("message"), dict) else None
        return {
            "prompt_ids": prompt, "completion_ids": completion, "logprobs": logprobs[: len(completion)], "top": top, "pieces": pieces,
            "messages": list(body.get("messages") or []), "tools": body.get("tools"), "response_message": message, "finish_reason": finish,
            "sampling": {k: body[k] for k in SAMPLING_KEYS if k in body} | ({"faithful": True} if cfg["faithful_sampling"] else {}),
            "model": str(data.get("model") or body.get("model") or ""), "upstream_type": kind, "prompt_ids_source": source,
            "verified": verified, "verify_note": "; ".join(notes), "usage": usage, "latency_ms": latency_ms, "flags": flags,
        }

    def llama_prompt(self, base: str, body: dict[str, Any], usage: dict[str, Any]) -> tuple[list[int], str, list[str]]:
        """The prompt ids of a llama.cpp call: the model's chat template through ``/apply-template``, then ``/tokenize``, checked against the
        server's own count. Returns ``(ids, source, notes)``; a count that does not match is a blocking note."""
        asked = {"messages": body.get("messages")} | {k: body[k] for k in TEMPLATE_KEYS if k in body}
        try:
            r = self.client().post(f"{base}/apply-template", json=asked)
            data = _json(r)
            if r.status_code != 200 or not isinstance(data, dict) or not isinstance(data.get("prompt"), str):
                return [], "unavailable", [f"/apply-template answered {r.status_code}"]
            text = data["prompt"]
            best: list[int] = []
            wanted = usage.get("prompt_tokens")
            for add_special in (False, True):
                t = self.client().post(f"{base}/tokenize", json={"content": text, "add_special": add_special, "parse_special": True})
                tokens = _json(t)
                if t.status_code != 200 or not isinstance(tokens, dict):
                    return [], "unavailable", [f"/tokenize answered {t.status_code}"]
                ids = [x["id"] if isinstance(x, dict) else x for x in (tokens.get("tokens") or [])]
                if not all(isinstance(i, int) for i in ids):
                    return [], "unavailable", ["/tokenize returned something that is not token ids"]
                best = best or ids
                if wanted is not None and len(ids) == int(wanted):
                    return ids, "apply-template+tokenize", []
            if wanted is None:
                return best, "apply-template+tokenize", ["the answer has no usage.prompt_tokens to check the prompt ids against"]
            return best, "apply-template+tokenize", [f"rebuilt prompt has {len(best)} tokens but the server counted {wanted} (tools without --jinja, images, or a template the endpoint renders differently)"]
        except httpx.HTTPError as exc:
            return [], "unavailable", [f"could not rebuild the prompt ids: {type(exc).__name__}"]

    def close_turn(self, base: str, completion: list[int], logprobs: list[Optional[float]], usage: dict[str, Any], finish: str) -> str:
        """llama.cpp leaves the end-of-turn token out of the reported tokens. When the server counted one more token than it listed and the call
        ended on its own, the model's end-of-turn token is appended (its log-probability is unknown, so it is null) so the model is trained to stop.
        Returns ``"appended"``, a note when it could not be done, or ``""`` when nothing was missing."""
        counted = usage.get("completion_tokens")
        if counted is None or finish not in ("stop", "tool_calls") or int(counted) != len(completion) + 1:
            if counted is not None and int(counted) not in (len(completion), len(completion) + 1):
                return f"completion ids {len(completion)} but usage says {counted}"
            return ""
        eos = self.eos_id(base)
        if eos is None:
            return "the end-of-turn token was not reported and could not be found"
        completion.append(eos)
        logprobs.append(None)
        return "appended"

    def eos_id(self, base: str) -> Optional[int]:
        if base in self._eos:
            return self._eos[base]
        found: Optional[int] = None
        try:
            props = _json(self.client().get(f"{base}/props", timeout=8.0))
            piece = props.get("eos_token") if isinstance(props, dict) else None
            if isinstance(piece, str) and piece:
                t = _json(self.client().post(f"{base}/tokenize", json={"content": piece, "add_special": False, "parse_special": True}))
                ids = [x["id"] if isinstance(x, dict) else x for x in ((t or {}).get("tokens") or [])] if isinstance(t, dict) else []
                if len(ids) == 1 and isinstance(ids[0], int):
                    found = ids[0]
        except httpx.HTTPError:
            found = None
        self._eos[base] = found
        return found

    @staticmethod
    def client_view(data: dict[str, Any], client_logprobs: bool) -> dict[str, Any]:
        out = copy.deepcopy(data)
        out.pop("prompt_token_ids", None)
        out.pop("prompt_logprobs", None)
        for choice in out.get("choices") or []:
            choice.pop("token_ids", None)
            choice.pop("prompt_logprobs", None)
            if not client_logprobs:
                choice["logprobs"] = None
        return out


def _blocking(note: str) -> bool:
    return not (note.startswith("end-of-turn token appended") or note.startswith("the end-of-turn token was not reported"))


def _json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_id(entry: dict[str, Any]) -> Any:
    if isinstance(entry.get("id"), int):
        return entry["id"]
    m = TOKEN_ID.match(str(entry.get("token", "")))
    return int(m.group(1)) if m else None


def _has_ids(data: dict[str, Any]) -> bool:
    for choice in data.get("choices") or []:
        lp = choice.get("logprobs")
        if isinstance(lp, dict) and lp.get("content") and isinstance(lp["content"][0], dict) and "id" in lp["content"][0]:
            return True
    return False


# ------------------------------------------------------------------ streaming a complete answer
def _sse(obj: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n\n"


def sse_events(data: dict[str, Any], *, include_usage: bool = False, with_logprobs: bool = False) -> Iterator[bytes]:
    """A complete chat-completions answer as the event stream a streaming client expects: a role chunk, the reasoning, the text and each tool
    call in one chunk apiece, the finish chunk, optionally the usage chunk, then ``[DONE]``."""
    base = {"id": data.get("id") or "chatcmpl-capture", "object": "chat.completion.chunk", "created": data.get("created") or int(time.time()),
            "model": data.get("model") or ""}
    if data.get("system_fingerprint"):
        base["system_fingerprint"] = data["system_fingerprint"]

    def chunk(index: int, delta: dict[str, Any], finish: Optional[str] = None, logprobs: Any = None) -> bytes:
        return _sse({**base, "choices": [{"index": index, "delta": delta, "logprobs": logprobs, "finish_reason": finish}]})

    for choice in data.get("choices") or []:
        index = int(choice.get("index") or 0)
        message = choice.get("message") or {}
        yield chunk(index, {"role": "assistant", "content": ""})
        if message.get("reasoning_content"):
            yield chunk(index, {"reasoning_content": message["reasoning_content"]})
        if message.get("content"):
            yield chunk(index, {"content": message["content"]}, logprobs=choice.get("logprobs") if with_logprobs else None)
        calls = message.get("tool_calls") or []
        if calls:
            yield chunk(index, {"tool_calls": [{"index": k, "id": c.get("id"), "type": c.get("type") or "function",
                                                 "function": {"name": (c.get("function") or {}).get("name"), "arguments": (c.get("function") or {}).get("arguments") or ""}}
                                                for k, c in enumerate(calls)]})
        yield chunk(index, {}, finish=choice.get("finish_reason") or "stop")
    if include_usage and data.get("usage"):
        yield _sse({**base, "choices": [], "usage": data["usage"]})
    yield b"data: [DONE]\n\n"
