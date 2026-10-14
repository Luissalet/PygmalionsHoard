"""Fake model servers for the capture proxy: a llama.cpp style server (``/apply-template``, ``/tokenize``, ids and log-probabilities in the chat
answer) and a vLLM style server (``prompt_token_ids`` and ``token_ids`` in the chat answer). Both share a tiny deterministic tokenizer and a ChatML
template, so what the proxy rebuilds can be compared with what the server really ran on."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import httpx

SPECIAL = {"<|im_start|>": 1, "<|im_end|>": 2}
BOS = 0
TOKEN = re.compile(r"<\|im_(?:start|end)\|>|\n| ?[\w']+| ?[^\w\s]")


class Tokenizer:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = dict(SPECIAL)
        self.pieces: dict[int, str] = {v: k for k, v in SPECIAL.items()}

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids = [BOS] if add_special else []
        for piece in TOKEN.findall(text):
            if piece not in self.vocab:
                self.vocab[piece] = 100 + len(self.vocab)
                self.pieces[self.vocab[piece]] = piece
            ids.append(self.vocab[piece])
        return ids


def text_of(content: Any) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def tool_call_text(call: dict[str, Any]) -> str:
    fn = call["function"]
    return "<tool_call>" + json.dumps({"name": fn["name"], "arguments": json.loads(fn["arguments"])}, separators=(",", ":")) + "</tool_call>"


def render(messages: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]] = None, generation: bool = True) -> str:
    out = ""
    if tools:
        out += "<|im_start|>system\n# Tools\n" + json.dumps([t["function"]["name"] for t in tools]) + "<|im_end|>\n"
    for m in messages:
        body = text_of(m.get("content"))
        if m.get("tool_calls"):
            body += "".join(tool_call_text(c) for c in m["tool_calls"])
        out += f"<|im_start|>{m['role']}\n{body}<|im_end|>\n"
    if generation:
        out += "<|im_start|>assistant\n"
    return out


class FakeServer:
    """Common part: a script of replies, the requests received, and the generation of a reply as tokens."""

    kind = "base"

    def __init__(self) -> None:
        self.tok = Tokenizer()
        self.replies: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.calls: list[str] = []
        self.report_eos = True
        self.fail_with: Optional[int] = None

    def say(self, text: str = "", tool_calls: Optional[list[tuple[str, dict[str, Any]]]] = None) -> "FakeServer":
        self.replies.append({"text": text, "tool_calls": tool_calls})
        return self

    def next_reply(self) -> tuple[dict[str, Any], str, str]:
        reply = self.replies.pop(0) if self.replies else {"text": "ok", "tool_calls": None}
        message: dict[str, Any] = {"role": "assistant", "content": reply["text"] or None}
        raw = reply["text"] or ""
        if reply["tool_calls"]:
            message["tool_calls"] = [{"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                                     for i, (n, a) in enumerate(reply["tool_calls"])]
            raw += "".join(tool_call_text(c) for c in message["tool_calls"])
        return message, raw, "tool_calls" if reply["tool_calls"] else "stop"

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(f"{request.method} {path}")
        body = json.loads(request.content) if request.content else {}
        if path == "/v1/chat/completions":
            self.requests.append(body)
            if self.fail_with:
                return httpx.Response(self.fail_with, json={"error": {"message": "boom", "type": "server_error"}})
            return self.chat(body)
        if path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": "fake-model", "object": "model"}]})
        return self.other(request, path, body)

    def other(self, request: httpx.Request, path: str, body: dict[str, Any]) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    def chat(self, body: dict[str, Any]) -> httpx.Response:  # pragma: no cover - overridden
        raise NotImplementedError


class FakeLlama(FakeServer):
    kind = "llama_server"

    def __init__(self, jinja: bool = True) -> None:
        super().__init__()
        self.jinja = jinja
        self.special_bos = False          # does the tokenizer add a BOS when asked (the real count then includes it)

    def other(self, request, path, body):
        if path == "/props":
            return httpx.Response(200, json={"default_generation_settings": {}, "build_info": "b-test", "eos_token": "<|im_end|>", "bos_token": ""})
        if path == "/apply-template":
            tools = body.get("tools") if self.jinja else None
            return httpx.Response(200, json={"prompt": render(body["messages"], tools)})
        if path == "/tokenize":
            return httpx.Response(200, json={"tokens": self.tok.encode(body["content"], bool(body.get("add_special")))})
        return super().other(request, path, body)

    def chat(self, body):
        message, raw, finish = self.next_reply()
        prompt = self.tok.encode(render(body["messages"], body.get("tools")), add_special=self.special_bos)
        ids = self.tok.encode(raw)
        content = [{"id": i, "token": self.tok.pieces[i], "bytes": list(self.tok.pieces[i].encode()), "logprob": -0.1 * (n % 5 + 1),
                    "top_logprobs": [{"id": i, "token": self.tok.pieces[i], "logprob": -0.1 * (n % 5 + 1)}]} for n, i in enumerate(ids)]
        usage = {"prompt_tokens": len(prompt), "completion_tokens": len(ids) + 1, "total_tokens": len(prompt) + len(ids) + 1}
        if self.report_eos:
            content.append({"id": SPECIAL["<|im_end|>"], "token": "<|im_end|>", "bytes": [], "logprob": -0.05, "top_logprobs": []})
        return httpx.Response(200, json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "fake-llama",
            "choices": [{"index": 0, "finish_reason": finish, "message": message, "logprobs": {"content": content}}],
            "usage": usage, "timings": {"cache_n": 0}})


class FakeVllm(FakeServer):
    kind = "vllm"

    def other(self, request, path, body):
        if path == "/version":
            return httpx.Response(200, json={"version": "0.11.0"})
        return super().other(request, path, body)

    def chat(self, body):
        message, raw, finish = self.next_reply()
        prompt = self.tok.encode(render(body["messages"], body.get("tools")))
        ids = self.tok.encode(raw) + [SPECIAL["<|im_end|>"]]
        named = bool(body.get("return_tokens_as_token_ids"))
        content = [{"token": f"token_id:{i}" if named else self.tok.pieces[i], "logprob": -0.2 * (n % 3 + 1), "bytes": [], "top_logprobs": []}
                   for n, i in enumerate(ids)]
        choice = {"index": 0, "finish_reason": finish, "message": message, "logprobs": {"content": content}}
        if body.get("return_token_ids"):
            choice["token_ids"] = ids
        data = {"id": "chatcmpl-v", "object": "chat.completion", "created": 1, "model": "fake-glm", "choices": [choice],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(ids), "total_tokens": len(prompt) + len(ids)}}
        if body.get("return_token_ids"):
            data["prompt_token_ids"] = prompt
        return httpx.Response(200, json=data)


class Routed(httpx.BaseTransport):
    """Sends a request to the fake server that owns its host; any other host refuses the connection."""

    def __init__(self) -> None:
        self.hosts: dict[str, FakeServer] = {}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        server = self.hosts.get(request.url.host)
        if server is None:
            raise httpx.ConnectError("connection refused", request=request)
        return server.handle(request)
