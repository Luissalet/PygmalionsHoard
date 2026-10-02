"""Record formats. A record is a plain dict in one of three shapes:

* ``chat``: ``{"messages": [{"role": "system|user|assistant", "content": str}, ...]}``
* ``instruction``: ``{"prompt": str, "response": str}`` (an optional ``system``); converted to chat when training
* ``text``: ``{"text": str}`` for style or continued pretraining

Bookkeeping lives under ``_meta`` (``source``, ``synthetic``, ``status`` = ok | pending | rejected) and is never given to the trainer.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

KINDS = ("chat", "instruction", "text")
ROLES = {"system": "system", "user": "user", "assistant": "assistant", "human": "user", "gpt": "assistant", "bot": "assistant",
         "model": "assistant", "developer": "system"}
MAX_RECORD_CHARS = 400_000
STATUSES = ("ok", "pending", "rejected")


def detect_kind(rec: Any) -> Optional[str]:
    if not isinstance(rec, dict):
        return None
    if isinstance(rec.get("messages"), list):
        return "chat"
    if isinstance(rec.get("conversations"), list):
        return "chat"
    if "prompt" in rec and "response" in rec:
        return "instruction"
    if isinstance(rec.get("instruction"), str) and isinstance(rec.get("output"), str):
        return "instruction"
    if isinstance(rec.get("text"), str):
        return "text"
    return None


def _clean_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _messages(raw: list[Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in raw:
        if not isinstance(m, dict):
            return []
        role = ROLES.get(str(m.get("role", m.get("from", ""))).lower())
        content = m.get("content", m.get("value", m.get("text")))
        if role is None or not isinstance(content, str):
            return []
        out.append({"role": role, "content": content.strip()})
    return out


def normalize_record(rec: Any, kind: Optional[str] = None) -> Optional[dict[str, Any]]:
    """The clean record of ``kind`` (detected when None), or None when it is unusable (empty, malformed, no assistant turn)."""
    kind = kind or detect_kind(rec)
    if kind is None:
        return None
    meta = rec.get("_meta") if isinstance(rec.get("_meta"), dict) else None
    out: dict[str, Any]
    if kind == "chat":
        raw = rec.get("messages") if isinstance(rec.get("messages"), list) else rec.get("conversations")
        messages = _messages(raw or [])
        if not messages or not any(m["role"] == "assistant" and m["content"] for m in messages):
            return None
        if all(not m["content"] for m in messages):
            return None
        out = {"messages": messages}
    elif kind == "instruction":
        prompt = _clean_str(rec.get("prompt", rec.get("instruction")))
        extra = _clean_str(rec.get("input"))
        response = _clean_str(rec.get("response", rec.get("output")))
        if extra and "prompt" not in rec:
            prompt = f"{prompt}\n\n{extra}"
        if not prompt or not response:
            return None
        out = {"prompt": prompt, "response": response}
        if _clean_str(rec.get("system")):
            out["system"] = _clean_str(rec["system"])
    elif kind == "text":
        text = _clean_str(rec.get("text"))
        if not text:
            return None
        out = {"text": text}
    else:
        return None
    if len(record_text(out)) > MAX_RECORD_CHARS:
        return None
    if meta is not None:
        out["_meta"] = meta
    return out


def record_text(rec: dict[str, Any]) -> str:
    """All the text of a record in reading order (used for hashing, length, language and personal-data scans)."""
    if "messages" in rec:
        return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in rec["messages"])
    if "prompt" in rec:
        return (str(rec.get("system", "")) + "\n" if rec.get("system") else "") + f"{rec.get('prompt', '')}\n{rec.get('response', '')}"
    return str(rec.get("text", ""))


def content_text(rec: dict[str, Any]) -> str:
    """Only what a person wrote (no role labels), for near-duplicate detection."""
    if "messages" in rec:
        return "\n".join(str(m.get("content", "")) for m in rec["messages"])
    if "prompt" in rec:
        return f"{rec.get('prompt', '')}\n{rec.get('response', '')}"
    return str(rec.get("text", ""))


def kind_of(rec: dict[str, Any]) -> str:
    return detect_kind(rec) or "text"


def map_text(rec: dict[str, Any], fn: Callable[[str], str]) -> dict[str, Any]:
    """A copy of ``rec`` with ``fn`` applied to every text field."""
    out = dict(rec)
    if "messages" in rec:
        out["messages"] = [{**m, "content": fn(str(m.get("content", "")))} for m in rec["messages"]]
    elif "prompt" in rec:
        out["prompt"] = fn(str(rec.get("prompt", "")))
        out["response"] = fn(str(rec.get("response", "")))
        if rec.get("system"):
            out["system"] = fn(str(rec["system"]))
    elif "text" in rec:
        out["text"] = fn(str(rec["text"]))
    return out


def to_kind(rec: dict[str, Any], target: str) -> Optional[dict[str, Any]]:
    """Convert between kinds where it is lossless enough: instruction to chat, a one-exchange chat to instruction, anything to text."""
    have = kind_of(rec)
    if have == target:
        return rec
    meta = {"_meta": rec["_meta"]} if isinstance(rec.get("_meta"), dict) else {}
    if target == "chat" and have == "instruction":
        messages = ([{"role": "system", "content": rec["system"]}] if rec.get("system") else []) + [
            {"role": "user", "content": rec["prompt"]}, {"role": "assistant", "content": rec["response"]}]
        return {"messages": messages, **meta}
    if target == "instruction" and have == "chat":
        turns = [m for m in rec["messages"] if m["role"] != "system"]
        system = next((m["content"] for m in rec["messages"] if m["role"] == "system"), "")
        if len(turns) == 2 and turns[0]["role"] == "user" and turns[1]["role"] == "assistant":
            out = {"prompt": turns[0]["content"], "response": turns[1]["content"], **meta}
            if system:
                out["system"] = system
            return out
        return None
    if target == "text":
        if have == "chat":
            body = "\n\n".join(f"{m['content']}" for m in rec["messages"] if m["role"] != "system")
        else:
            body = f"{rec['prompt']}\n\n{rec['response']}"
        return {"text": body, **meta}
    return None


def meta_of(rec: dict[str, Any]) -> dict[str, Any]:
    m = rec.get("_meta")
    return m if isinstance(m, dict) else {}


def status_of(rec: dict[str, Any]) -> str:
    s = meta_of(rec).get("status", "ok")
    return s if s in STATUSES else "ok"


def with_meta(rec: dict[str, Any], **fields: Any) -> dict[str, Any]:
    out = dict(rec)
    out["_meta"] = {**meta_of(rec), **fields}
    return out


def strip_meta(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in rec.items() if k != "_meta"}
