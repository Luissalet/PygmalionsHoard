"""What the app does with captured calls: the proxy's state and address, the session views, the outcome of a session, the export of training
rows (a dataset for the built-in trainer plus token-level files) and the deletion of sessions."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..errors import PygmalionError
from ..hoard_link import atomic
from ..settings import Settings
from ..util import slug
from . import tree as T
from .proxy import UPSTREAM_TYPES, CaptureProxy
from .store import CaptureStore

FORMATS = ("dataset", "sft", "rl")
PREVIEW = 160
ALL_LIMIT = 100_000


def parse_when(value: Any) -> Optional[float]:
    """A time as seconds since the epoch: a number, a numeric string or an ISO date/time (UTC when it has no offset)."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PygmalionError("invalid", "capture_date_bad", value=text) from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") in (None, "text", "input_text"))
    return "" if content is None else str(content)


def flatten_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Chat messages as plain ``system|user|assistant`` text turns, the shape the built-in trainer reads. Tool results become user turns and tool
    calls become ``<tool_call>`` blocks in the assistant text: an approximation of what the model saw. The token files keep the exact tokens."""
    out: list[dict[str, str]] = []
    for m in messages:
        role = str(m.get("role") or "")
        body = text_of(m.get("content")).strip()
        if role in ("tool", "function"):
            role, body = "user", "Tool result:\n" + body
        elif role == "developer":
            role = "system"
        elif role == "assistant" and m.get("tool_calls"):
            blocks = []
            for call in m["tool_calls"]:
                fn = (call or {}).get("function") or {}
                args = fn.get("arguments")
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except ValueError:
                    pass
                blocks.append("<tool_call>\n" + json.dumps({"name": fn.get("name"), "arguments": args}, ensure_ascii=False) + "\n</tool_call>")
            body = (body + "\n" if body else "") + "\n".join(blocks)
        if role not in ("system", "user", "assistant"):
            continue
        if out and out[-1]["role"] == role and role != "system":
            out[-1]["content"] += "\n\n" + body
        else:
            out.append({"role": role, "content": body})
    return out


def preview(text: str, limit: int = PREVIEW) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class CaptureService:
    def __init__(self, store: CaptureStore, proxy: CaptureProxy, settings: Settings, *, exports_dir: Path, port: Callable[[], int], datasets: Any,
                 clock: Callable[[], float] = time.time):
        self.store = store
        self.proxy = proxy
        self.settings = settings
        self.exports_dir = Path(exports_dir)
        self.port = port
        self.datasets = datasets
        self.clock = clock

    # ------------------------------------------------------------------ the proxy
    def urls(self) -> dict[str, str]:
        root = f"http://127.0.0.1:{self.port()}"
        return {"base_url": f"{root}/capture/v1", "session_url": f"{root}/capture/<session>/v1", "harness_session_url": f"{root}/capture/<harness>/<session>/v1",
                "chat_completions": f"{root}/capture/v1/chat/completions"}

    def status(self, check: bool = False) -> dict[str, Any]:
        cfg = self.proxy.config()
        out: dict[str, Any] = {**cfg, **self.urls(), "counts": self.store.counts(), "harnesses": self.store.harnesses()}
        if check:
            out["upstream"] = self.proxy.check()
        return out

    def set_proxy(self, action: str, upstream_url: Optional[str] = None, upstream_type: Optional[str] = None, top_logprobs: Optional[int] = None,
                  faithful_sampling: Optional[bool] = None, timeout_s: Optional[int] = None) -> dict[str, Any]:
        values: dict[str, Any] = {}
        if upstream_url is not None:
            url = self.proxy.base_of(upstream_url)
            if url and not re.match(r"^https?://[^\s/]+", url):
                raise PygmalionError("invalid", "capture_upstream_bad", url=upstream_url)
            values["capture.upstream_url"] = url
        if upstream_type is not None:
            values["capture.upstream_type"] = upstream_type
        if top_logprobs is not None:
            values["capture.top_logprobs"] = top_logprobs
        if faithful_sampling is not None:
            values["capture.faithful_sampling"] = faithful_sampling
        if timeout_s is not None:
            values["capture.timeout_s"] = timeout_s
        if action == "start":
            values["capture.enabled"] = True
        elif action == "stop":
            values["capture.enabled"] = False
        if action == "start" and not (values.get("capture.upstream_url") or self.settings.get("capture.upstream_url")):
            raise PygmalionError("invalid", "capture_upstream_needed")
        if values:
            self.settings.set(values)
            self.proxy.forget()
        return self.status(check=action in ("start", "check"))

    # ------------------------------------------------------------------ sessions
    def sessions(self, *, harness: str = "", since: Any = None, until: Any = None, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        rows = self.store.sessions(harness=harness, since=parse_when(since), until=parse_when(until), limit=limit, offset=offset)
        return {"sessions": [self._session_row(r) for r in rows], "harnesses": self.store.harnesses(), "counts": self.store.counts()}

    @staticmethod
    def _session_row(r: dict[str, Any]) -> dict[str, Any]:
        return {"id": r["id"], "harness": r["harness"], "created_ts": r["created_ts"], "updated_ts": r["updated_ts"], "calls": r["calls"],
                "roots": r["roots"], "leaves": r["leaves"], "unverified": r["unverified"], "completion_tokens": r["completion_tokens"],
                "model": r.get("model") or "", "reward": r["reward"], "outcome": r["outcome"], "note": r["note"]}

    def session_view(self, session: str, node: str = "", max_tokens: int = 200, limit: int = 500) -> dict[str, Any]:
        sess = self.store.session(session)
        if node:
            return self.node_view(node, session, max_tokens)
        nodes = self.store.nodes(session, heavy=False)
        by_id = {n["id"]: n for n in nodes}
        depth: dict[str, int] = {}
        for n in nodes:                                         # oldest first, so a parent always has its depth already
            depth[n["id"]] = depth[n["parent"]] + 1 if n["parent"] in depth else 0
        kids: dict[str, int] = {}
        for n in nodes:
            if n["parent"] in by_id:
                kids[n["parent"]] = kids.get(n["parent"], 0) + 1
        shown = []
        for n in nodes[:limit]:
            flags = n["flags"]
            news = [m for m in n["new_messages"] if m.get("role") != "assistant"]
            shown.append({
                "id": n["id"], "seq": n["seq"], "parent": n["parent"], "depth": depth[n["id"]], "children": kids.get(n["id"], 0), "ts": n["ts"],
                "model": n["model"], "prompt_tokens": n["prompt_len"], "completion_tokens": n["completion_len"], "finish_reason": n["finish_reason"],
                "verified": n["verified"], "verify_note": n["verify_note"], "prompt_ids_source": n["prompt_ids_source"], "reward": n["reward"],
                "latency_ms": n["latency_ms"], "lp_mean": flags.get("lp_mean"), "lp_min": flags.get("lp_min"),
                "eos_appended": bool(flags.get("eos_appended")),
                "input": preview(text_of(news[-1].get("content")) if news else ""), "output": preview(_answer_text(n["response_message"])),
                "tool_calls": [((c or {}).get("function") or {}).get("name") for c in ((n["response_message"] or {}).get("tool_calls") or [])]})
        leaves = sum(1 for n in nodes if not kids.get(n["id"]))
        return {"session": self._session_row({**sess, "calls": len(nodes), "roots": sum(1 for n in nodes if n["parent"] is None), "leaves": leaves,
                                              "unverified": sum(1 for n in nodes if not n["verified"]),
                                              "completion_tokens": sum(n["completion_len"] for n in nodes),
                                              "model": nodes[-1]["model"] if nodes else ""}),
                "nodes": shown, "truncated": len(nodes) > limit}

    def node_view(self, node: str, session: str = "", max_tokens: int = 200) -> dict[str, Any]:
        n = self.store.node(node)
        if session and n["session"] != session:
            raise PygmalionError("not_found", "no_capture_node", ref=node)
        completion, lps, pieces = n["completion_ids"], n["logprobs"], n.get("pieces") or []
        rows = [{"i": i, "id": completion[i], "token": pieces[i] if i < len(pieces) else None, "logprob": lps[i] if i < len(lps) else None}
                for i in range(min(len(completion), max(0, max_tokens)))]
        known = sorted((i for i in range(len(completion)) if i < len(lps) and lps[i] is not None), key=lambda i: lps[i])[:10]
        view = {k: n[k] for k in ("id", "session", "seq", "ts", "harness", "upstream", "upstream_type", "model", "parent", "finish_reason", "prompt_ids_source",
                                  "verified", "verify_note", "usage", "latency_ms", "sampling", "flags", "reward", "outcome")}
        view.update(prompt_tokens=n["prompt_len"], completion_tokens=n["completion_len"], messages=n["messages"], tools=n["tools"], response=n["response_message"],
                    new_prompt_tokens=n["prompt_new"], new_prompt_ids=n["prompt_ids"][len(n["prompt_ids"]) - n["prompt_new"]:][:max(0, max_tokens)],
                    tokens=rows, lowest=[{"i": i, "id": completion[i], "token": pieces[i] if i < len(pieces) else None, "logprob": lps[i]} for i in known],
                    tokens_shown=len(rows))
        return {"node": view}

    def outcome(self, session: str, reward: Optional[float], outcome: Optional[str], note: str = "", node: str = "", clear: bool = False) -> dict[str, Any]:
        row = self.store.set_outcome(session, reward=reward, outcome=outcome, note=note, node=node, clear=clear)
        return {"session": self._session_row({**row, "calls": 0, "roots": 0, "leaves": 0, "unverified": 0, "completion_tokens": 0})}

    def delete(self, sessions: Sequence[str], delete_all: bool = False) -> dict[str, Any]:
        ids = list(sessions)
        if delete_all:
            ids = [r["id"] for r in self.store.sessions(limit=ALL_LIMIT)]
        if not ids:
            raise PygmalionError("invalid", "capture_delete_what")
        return {"deleted": self.store.delete(ids), "asked": len(ids)}

    # ------------------------------------------------------------------ export
    def select_rows(self, *, sessions: Sequence[str] = (), harness: Sequence[str] = (), since: Any = None, until: Any = None, require_reward: bool = False,
                    min_reward: Optional[float] = None, include_unverified: bool = False, loss_on: str = "all", dedupe_shared: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
        """``(rows, report, nodes_by_id)``: every root-to-leaf sequence that passes the filters, before balancing."""
        t0, t1 = parse_when(since), parse_when(until)
        wanted = {h for h in harness if h}
        listed = self.store.sessions(ids=list(sessions), limit=ALL_LIMIT)
        rows: list[dict[str, Any]] = []
        by_id: dict[str, dict[str, Any]] = {}
        report = {"sessions_considered": 0, "calls": 0, "unverified_excluded": 0, "no_loss_tokens": 0, "filtered_out": 0}
        for s in listed:
            if wanted and s["harness"] not in wanted:
                continue
            report["sessions_considered"] += 1
            nodes = self.store.nodes(s["id"])
            report["calls"] += len(nodes)
            usable = [n for n in nodes if include_unverified or n["verified"]]
            report["unverified_excluded"] += len(nodes) - len(usable)
            part, skipped = T.sequences_report(usable, loss_on=loss_on, dedupe_shared=dedupe_shared)
            report["no_loss_tokens"] += skipped["no_loss_tokens"]
            for n in usable:
                by_id[n["id"]] = n
            for row in part:
                reward = row["reward"]
                if (t0 is not None and (row["ts"] or 0) < t0) or (t1 is not None and (row["ts"] or 0) > t1) \
                        or (require_reward and reward is None) or (min_reward is not None and (reward is None or reward < min_reward)) \
                        or (wanted and row["harness"] not in wanted):
                    report["filtered_out"] += 1
                    continue
                rows.append(row)
        rows.sort(key=lambda r: (r["ts"] or 0, r["leaf"]))
        return rows, report, by_id

    def export(self, *, sessions: Sequence[str] = (), harness: Sequence[str] = (), since: Any = None, until: Any = None, require_reward: bool = False,
               min_reward: Optional[float] = None, include_unverified: bool = False, loss_on: str = "all", dedupe_shared: bool = True,
               balance_by_harness: Any = None, seed: int = 42, formats: Optional[Sequence[str]] = None, dataset_name: str = "",
               dataset: str = "", dry_run: bool = False) -> dict[str, Any]:
        """``formats`` is any of ``dataset`` (chat records registered as a dataset), ``sft`` and ``rl`` (token files); when omitted: both token
        files, plus the dataset when ``dataset_name`` or ``dataset`` is given."""
        if formats is None:
            formats = ["sft", "rl"] + (["dataset"] if (dataset_name or dataset) else [])
        formats = [f for f in formats if f in FORMATS]
        rows, report, by_id = self.select_rows(sessions=sessions, harness=harness, since=since, until=until, require_reward=require_reward, min_reward=min_reward,
                                               include_unverified=include_unverified, loss_on=loss_on, dedupe_shared=dedupe_shared)
        kept, per_harness = T.balance_by_harness(rows, balance_by_harness, seed)
        summary = {"rows_available": len(rows), "rows": len(kept), "loss_tokens": sum(r["loss_tokens"] for r in kept), "tokens": sum(r["tokens"] for r in kept),
                   "with_reward": sum(1 for r in kept if r["reward"] is not None), "per_harness": per_harness, "report": report,
                   "balance_by_harness": balance_by_harness, "loss_on": loss_on, "dedupe_shared": dedupe_shared, "include_unverified": include_unverified}
        if dry_run:
            return {"dry_run": True, **summary}
        if not kept:
            raise PygmalionError("invalid", "capture_nothing_to_export")
        if "dataset" in formats and not (dataset_name or dataset):
            raise PygmalionError("invalid", "capture_dataset_needed")
        if not formats:
            raise PygmalionError("invalid", "capture_formats_needed")
        stamp = datetime.fromtimestamp(self.clock(), timezone.utc).strftime("%Y%m%d-%H%M%S")
        folder = self.exports_dir / f"{stamp}-{slug(dataset_name or dataset or 'capture', 24)}"
        folder.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        if "sft" in formats:
            files["sft"] = str(_write_jsonl(folder / "sft.jsonl", (T.sft_row(r) for r in kept)))
        if "rl" in formats:
            files["rl"] = str(_write_jsonl(folder / "rl.jsonl", (T.rl_row(r) for r in kept)))
        result: dict[str, Any] = {"dry_run": False, **summary, "folder": str(folder), "files": files}
        if "dataset" in formats:
            records = [self._chat_record(r, by_id) for r in kept]
            spec = {"name": dataset_name, "dataset": dataset, "description": "Captured model calls (capture proxy)"}
            recipe = {"capture": {"harnesses": sorted(per_harness), "rows": len(kept), "balance_by_harness": balance_by_harness, "loss_on": loss_on,
                                  "include_unverified": include_unverified, "seed": seed, "exported_ts": self.clock()}}
            made = self.datasets.import_records(records, spec, recipe)
            result["dataset"] = made["dataset"]
            result["version"] = made["version"]
            result["train_on"] = "last" if loss_on == "last" else "assistant"
        manifest = {k: v for k, v in result.items() if k not in ("dataset", "version")} | {"created_ts": self.clock(), "formats": formats}
        atomic.write_bytes_atomic(folder / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
        result["manifest"] = str(folder / "manifest.json")
        return result

    @staticmethod
    def _chat_record(row: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
        leaf = by_id[row["leaf"]]
        turns = list(leaf["messages"]) + ([leaf["response_message"]] if leaf["response_message"] else [])
        meta = {"source": "capture", "harness": row["harness"], "session": row["session"], "leaf": row["leaf"], "nodes": len(row["nodes"]),
                "reward": row["reward"], "outcome": row["outcome"], "model": row["model"], "tools": len(leaf["tools"] or [])}
        return {"messages": flatten_messages(turns), "_meta": meta}


def _answer_text(message: Optional[dict[str, Any]]) -> str:
    if not message:
        return ""
    return text_of(message.get("content")) or text_of(message.get("reasoning_content"))


def _write_jsonl(path: Path, rows: Any) -> Path:
    tmp = atomic.tmp_path_for(path)
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        atomic.replace_with_retry(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


__all__ = ["CaptureService", "UPSTREAM_TYPES", "FORMATS", "flatten_messages", "parse_when"]
