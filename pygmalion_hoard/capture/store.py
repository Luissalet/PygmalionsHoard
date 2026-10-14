"""Storage of captured sessions and calls. A call is stored as a *delta* of its parent: the prompt ids and the messages that follow the parent's
whole sequence, so a long agent session costs the new tokens of each step and not the whole context again. Reading one back walks the chain."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Iterable, Optional, Sequence

from ..db import Database
from ..errors import PygmalionError
from ..util import dumps, loads, new_id
from . import tree as T

TAIL_CACHE = 16


def _load(value: Any, default: Any) -> Any:
    return loads(value, default)


class CaptureStore:
    def __init__(self, db: Database, clock: Callable[[], float] = time.time):
        self.db = db
        self.clock = clock
        self._lock = threading.RLock()
        self._tail: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()     # id -> conversation up to and including the model's answer

    # ------------------------------------------------------------------ sessions
    def ensure_session(self, session: str, harness: str = "") -> dict[str, Any]:
        with self._lock:
            row = self.db.one("SELECT * FROM capture_sessions WHERE id = ?", (session,))
            now = self.clock()
            if row is None:
                self.db.execute("INSERT INTO capture_sessions(id, harness, created_ts, updated_ts) VALUES (?, ?, ?, ?)", (session, harness, now, now))
            elif harness and not row["harness"]:
                self.db.execute("UPDATE capture_sessions SET harness = ? WHERE id = ?", (harness, session))
            return dict(self.db.one("SELECT * FROM capture_sessions WHERE id = ?", (session,)))

    def session(self, session: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM capture_sessions WHERE id = ?", (session,))
        if row is None:
            raise PygmalionError("not_found", "no_capture_session", ref=session)
        return dict(row)

    def sessions(self, *, harness: str = "", since: Optional[float] = None, until: Optional[float] = None, ids: Sequence[str] = (),
                 limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        where, params = ["1 = 1"], []
        if harness:
            where.append("s.harness = ?")
            params.append(harness)
        if since is not None:
            where.append("s.updated_ts >= ?")
            params.append(since)
        if until is not None:
            where.append("s.created_ts <= ?")
            params.append(until)
        if ids:
            where.append(f"s.id IN ({','.join('?' * len(ids))})")
            params += list(ids)
        sql = (
            "SELECT s.*, "
            "(SELECT COUNT(*) FROM capture_nodes n WHERE n.session_id = s.id) AS calls, "
            "(SELECT COALESCE(SUM(n.completion_len), 0) FROM capture_nodes n WHERE n.session_id = s.id) AS completion_tokens, "
            "(SELECT COUNT(*) FROM capture_nodes n WHERE n.session_id = s.id AND n.verified = 0) AS unverified, "
            "(SELECT COUNT(*) FROM capture_nodes n WHERE n.session_id = s.id AND n.parent_id IS NULL) AS roots, "
            "(SELECT COUNT(*) FROM capture_nodes n WHERE n.session_id = s.id AND NOT EXISTS "
            "(SELECT 1 FROM capture_nodes c WHERE c.parent_id = n.id)) AS leaves, "
            "(SELECT MAX(n.model) FROM capture_nodes n WHERE n.session_id = s.id) AS model "
            f"FROM capture_sessions s WHERE {' AND '.join(where)} ORDER BY s.updated_ts DESC LIMIT ? OFFSET ?")
        return [dict(r) for r in self.db.query(sql, [*params, int(limit), int(offset)])]

    def harnesses(self) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT harness, COUNT(*) AS sessions, MAX(updated_ts) AS last_ts FROM capture_sessions GROUP BY harness ORDER BY sessions DESC")
        return [dict(r) for r in rows]

    def counts(self) -> dict[str, int]:
        one = lambda sql: int((self.db.one(sql) or [0])[0] or 0)  # noqa: E731
        return {"sessions": one("SELECT COUNT(*) FROM capture_sessions"), "calls": one("SELECT COUNT(*) FROM capture_nodes"),
                "unverified": one("SELECT COUNT(*) FROM capture_nodes WHERE verified = 0")}

    def set_outcome(self, session: str, *, reward: Optional[float], outcome: Optional[str], note: str = "", node: str = "", clear: bool = False) -> dict[str, Any]:
        with self._lock:
            self.session(session)
            if node:
                row = self.db.one("SELECT id FROM capture_nodes WHERE id = ? AND session_id = ?", (node, session))
                if row is None:
                    raise PygmalionError("not_found", "no_capture_node", ref=node)
                self.db.execute("UPDATE capture_nodes SET reward = ? WHERE id = ?", (None if clear else reward, node))
            else:
                if clear:
                    self.db.execute("UPDATE capture_sessions SET reward = NULL, outcome = NULL, note = '', updated_ts = ? WHERE id = ?", (self.clock(), session))
                else:
                    self.db.execute("UPDATE capture_sessions SET reward = ?, outcome = ?, note = ?, updated_ts = ? WHERE id = ?",
                                    (reward, outcome, note, self.clock(), session))
            return self.session(session)

    def delete(self, sessions: Iterable[str]) -> int:
        ids = [s for s in sessions]
        if not ids:
            return 0
        with self._lock:
            before = self.counts()["sessions"]
            self.db.execute(f"DELETE FROM capture_sessions WHERE id IN ({','.join('?' * len(ids))})", ids)
            self._tail.clear()
            return before - self.counts()["sessions"]

    # ------------------------------------------------------------------ adding a call
    def add_call(self, session: str, harness: str, call: dict[str, Any]) -> dict[str, Any]:
        """Store one call and link it to the deepest earlier call of the session whose whole sequence is an exact prefix of its prompt.
        ``call``: prompt_ids, completion_ids, logprobs, top, messages, tools, response_message, finish_reason, sampling, model, upstream,
        upstream_type, prompt_ids_source, verified, verify_note, usage, latency_ms, flags. Returns the stored node (without the big arrays)."""
        prompt = list(call["prompt_ids"])
        completion = list(call["completion_ids"])
        messages = list(call.get("messages") or [])
        tools = call.get("tools")
        with self._lock:
            self.ensure_session(session, harness)
            candidates = [dict(r) for r in self.db.query(
                "SELECT id, full_len, full_sha, seq FROM capture_nodes WHERE session_id = ? AND full_len > 0 AND full_len <= ?", (session, len(prompt)))]
            parent = T.find_parent(prompt, candidates) if prompt else None
            seq = int((self.db.one("SELECT MAX(seq) FROM capture_nodes WHERE session_id = ?", (session,)) or [0])[0] or 0) + 1
            parent_len = int(parent["full_len"]) if parent else 0
            delta = prompt[parent_len:]
            skip = 0
            stored_messages = messages
            tools_json: Optional[str] = dumps(tools)
            if parent:
                before = self._conversation(parent["id"])
                skip = _common_messages(before, messages)
                stored_messages = messages[skip:]
                if self._tools_of(parent["id"]) == tools:
                    tools_json = None                                   # same tools as the parent: inherit them
            full = prompt + completion
            node_id = new_id("cn")
            now = self.clock()
            self.db.insert("capture_nodes", {
                "id": node_id, "session_id": session, "seq": seq, "ts": now, "harness": harness or "", "upstream": call.get("upstream") or "",
                "upstream_type": call.get("upstream_type") or "", "model": call.get("model") or "", "parent_id": parent["id"] if parent else None,
                "parent_len": parent_len, "prompt_len": len(prompt), "prompt_delta": dumps(delta), "completion_len": len(completion),
                "completion_ids": dumps(completion), "logprobs": dumps(call.get("logprobs") or []),
                "top": dumps(call["top"]) if call.get("top") else None, "pieces": dumps(call["pieces"]) if call.get("pieces") else None,
                "full_len": len(full), "full_sha": T.fingerprint(full), "msg_skip": skip, "messages": dumps(stored_messages), "tools": tools_json,
                "response_message": dumps(call.get("response_message")), "finish_reason": call.get("finish_reason") or "",
                "sampling": dumps(call.get("sampling") or {}), "prompt_ids_source": call.get("prompt_ids_source") or "",
                "verified": 1 if call.get("verified") else 0, "verify_note": call.get("verify_note") or "", "usage": dumps(call.get("usage") or {}),
                "latency_ms": int(call.get("latency_ms") or 0), "flags": dumps(call.get("flags") or {})})
            self.db.execute("UPDATE capture_sessions SET updated_ts = ? WHERE id = ?", (now, session))
            effective = messages + ([call["response_message"]] if call.get("response_message") else [])
            self._remember(node_id, effective)
            return {"id": node_id, "session": session, "seq": seq, "parent": parent["id"] if parent else None, "prompt_tokens": len(prompt),
                    "completion_tokens": len(completion), "verified": bool(call.get("verified")), "verify_note": call.get("verify_note") or ""}

    # ------------------------------------------------------------------ reading calls back
    def _remember(self, node_id: str, conversation: list[dict[str, Any]]) -> None:
        self._tail[node_id] = conversation
        self._tail.move_to_end(node_id)
        while len(self._tail) > TAIL_CACHE:
            self._tail.popitem(last=False)

    def _conversation(self, node_id: str) -> list[dict[str, Any]]:
        """The messages of a call plus the answer the model gave: what the next call of the chain starts from."""
        if node_id in self._tail:
            return self._tail[node_id]
        row = self.db.one("SELECT parent_id, msg_skip, messages, response_message FROM capture_nodes WHERE id = ?", (node_id,))
        if row is None:
            return []
        before = self._conversation(row["parent_id"])[: row["msg_skip"]] if row["parent_id"] else []
        conversation = before + _load(row["messages"], [])
        answer = _load(row["response_message"], None)
        if answer:
            conversation = conversation + [answer]
        self._remember(node_id, conversation)
        return conversation

    def _tools_of(self, node_id: str) -> Any:
        cursor: Optional[str] = node_id
        while cursor:
            row = self.db.one("SELECT parent_id, tools FROM capture_nodes WHERE id = ?", (cursor,))
            if row is None:
                return None
            if row["tools"] is not None:
                return _load(row["tools"], None)
            cursor = row["parent_id"]
        return None

    def nodes(self, session: str, *, heavy: bool = True) -> list[dict[str, Any]]:
        """Every call of a session, oldest first, with the ids rebuilt (full prompt) and the messages rebuilt. ``heavy=False`` leaves the id
        arrays out (lengths only), for listings."""
        self.session(session)
        rows = self.db.query("SELECT * FROM capture_nodes WHERE session_id = ? ORDER BY seq", (session,))
        done: dict[str, dict[str, Any]] = {}
        out: list[dict[str, Any]] = []
        sess = dict(self.db.one("SELECT * FROM capture_sessions WHERE id = ?", (session,)))
        for row in rows:
            parent = done.get(row["parent_id"]) if row["parent_id"] else None
            node = self._node(row, parent, sess, heavy)
            done[node["id"]] = node
            out.append(node)
        return out

    def node(self, node_id: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM capture_nodes WHERE id = ?", (node_id,))
        if row is None:
            raise PygmalionError("not_found", "no_capture_node", ref=node_id)
        chain = [row]
        while chain[-1]["parent_id"]:
            parent = self.db.one("SELECT * FROM capture_nodes WHERE id = ?", (chain[-1]["parent_id"],))
            if parent is None:
                break
            chain.append(parent)
        sess = dict(self.db.one("SELECT * FROM capture_sessions WHERE id = ?", (row["session_id"],)))
        built: Optional[dict[str, Any]] = None
        for item in reversed(chain):
            built = self._node(item, built, sess, True)
        assert built is not None
        return built

    def _node(self, row: Any, parent: Optional[dict[str, Any]], sess: dict[str, Any], heavy: bool) -> dict[str, Any]:
        delta = _load(row["prompt_delta"], [])
        prompt: list[int] = []
        if heavy:
            prompt = (list(parent["prompt_ids"]) + list(parent["completion_ids"]) if parent else []) + delta
        completion = _load(row["completion_ids"], [])
        messages_tail = _load(row["messages"], [])
        messages: list[dict[str, Any]] = []
        if heavy:
            before = (parent["_conversation"][: row["msg_skip"]]) if parent else []
            messages = before + messages_tail
        tools_raw = row["tools"]
        tools = _load(tools_raw, None) if tools_raw is not None else (parent["tools"] if parent else None)
        answer = _load(row["response_message"], None)
        node = {
            "id": row["id"], "session": row["session_id"], "seq": row["seq"], "ts": row["ts"], "harness": row["harness"] or sess.get("harness") or "",
            "upstream": row["upstream"], "upstream_type": row["upstream_type"], "model": row["model"], "parent": row["parent_id"],
            "prompt_len": row["prompt_len"], "prompt_new": len(delta), "completion_len": row["completion_len"], "finish_reason": row["finish_reason"],
            "prompt_ids_source": row["prompt_ids_source"], "verified": bool(row["verified"]), "verify_note": row["verify_note"],
            "usage": _load(row["usage"], {}), "latency_ms": row["latency_ms"], "sampling": _load(row["sampling"], {}), "flags": _load(row["flags"], {}),
            "reward": row["reward"] if row["reward"] is not None else sess.get("reward"), "outcome": sess.get("outcome"),
            "tools": tools, "response_message": answer, "new_messages": messages_tail,
        }
        if heavy:
            node.update(prompt_ids=prompt, completion_ids=completion, logprobs=_load(row["logprobs"], []), top=_load(row["top"], None), pieces=_load(row["pieces"], None), messages=messages,
                        _conversation=messages + ([answer] if answer else []))
        return node


def _common_messages(before: Sequence[dict[str, Any]], messages: Sequence[dict[str, Any]]) -> int:
    n = 0
    for a, b in zip(before, messages):
        if a != b:
            break
        n += 1
    return n
