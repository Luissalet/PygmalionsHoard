"""The prefix tree of captured calls and the training rows it yields. Pure Python (standard library only), so the tests run it without a
server or a tokenizer.

A *call* is one request to the model with the token ids that went in (``prompt_ids``, already through the chat template) and the ids that
came out (``completion_ids``, with the log-probability the sampler gave each). Call B is the *child* of call A when A's whole sequence
(prompt then completion) is an exact prefix of B's prompt: B continues the very tokens A sampled, so the two form one trajectory and the
tokens the model produced stay where it produced them. A call that continues no other one is a *root*; two calls that continue the same
one are siblings (a branch: a retry, a sampled alternative, a sub-agent). Every root-to-leaf path is one training sequence:

* the context (system prompt, user turns, tool results, the template's scaffolding) has mask 0;
* the tokens the model sampled have mask 1.

A token that sits in several paths (the sampled tokens of a call with two children) is trained on once: the later paths keep it as context
(``dedupe_shared``). The exact-prefix rule is strict on purpose: when the harness re-renders an earlier assistant turn in a way that
changes its tokens, the chain breaks there and the later call becomes a new root (its sampled tokens are still right; only the link is lost).
"""

from __future__ import annotations

import hashlib
import random
import struct
from typing import Any, Iterable, Optional, Sequence

IGNORE_INDEX = -100


# ------------------------------------------------------------------ fingerprints
def _pack(ids: Sequence[int]) -> bytes:
    return struct.pack(f"<{len(ids)}i", *ids) if ids else b""


def fingerprint(ids: Sequence[int]) -> str:
    """SHA-1 of the ids as little-endian int32 (the same on every machine)."""
    return hashlib.sha1(_pack(ids)).hexdigest()


def prefix_fingerprints(ids: Sequence[int], lengths: Iterable[int]) -> dict[int, str]:
    """``{length: fingerprint(ids[:length])}`` for every wanted length that fits, in one pass over ``ids``."""
    wanted = sorted({n for n in lengths if 0 <= n <= len(ids)})
    out: dict[int, str] = {}
    digest = hashlib.sha1()
    done = 0
    for n in wanted:
        if n > done:
            digest.update(_pack(ids[done:n]))
            done = n
        out[n] = digest.copy().hexdigest()
    return out


def find_parent(prompt_ids: Sequence[int], candidates: Sequence[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The deepest candidate whose full sequence is an exact prefix of ``prompt_ids``. Candidates carry ``full_len`` and ``full_sha``."""
    usable = [c for c in candidates if 0 < int(c["full_len"]) <= len(prompt_ids)]
    if not usable:
        return None
    prints = prefix_fingerprints(prompt_ids, (int(c["full_len"]) for c in usable))
    matching = [c for c in usable if prints.get(int(c["full_len"])) == c["full_sha"]]
    if not matching:
        return None
    return max(matching, key=lambda c: (int(c["full_len"]), int(c.get("seq", 0))))


def shared_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ------------------------------------------------------------------ the tree
def full_of(node: dict[str, Any]) -> list[int]:
    return list(node["prompt_ids"]) + list(node["completion_ids"])


def build(nodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Index a set of nodes (each with ``id``, ``parent``, ``seq``): children by parent, roots and leaves. A parent that is not in the set
    (filtered out, or unverified) leaves its child as a root."""
    by_id = {n["id"]: n for n in nodes}
    children: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    roots: list[str] = []
    for n in sorted(nodes, key=lambda x: (x.get("seq", 0), x["id"])):
        parent = n.get("parent")
        if parent in by_id:
            children[parent].append(n["id"])
        else:
            roots.append(n["id"])
    leaves = [i for i, kids in children.items() if not kids]
    return {"by_id": by_id, "children": children, "roots": roots, "leaves": leaves}


def path_to(tree: dict[str, Any], leaf: str) -> list[dict[str, Any]]:
    by_id = tree["by_id"]
    out: list[dict[str, Any]] = []
    cursor: Optional[str] = leaf
    while cursor is not None and cursor in by_id:
        node = by_id[cursor]
        out.append(node)
        cursor = node.get("parent") if node.get("parent") in by_id else None
    out.reverse()
    return out


def sequence_of(path: Sequence[dict[str, Any]], *, trained: set[str], loss_on: str = "all", dedupe_shared: bool = True) -> dict[str, Any]:
    """The training row of one root-to-leaf path: ``input_ids``, ``mask`` (1 for tokens the model sampled and may be trained on), aligned
    ``logprobs`` (None where mask is 0 or the server gave none) and the node ids. ``trained`` is updated with the nodes whose tokens this row
    trains on, so the next path through the same tree keeps them as context only."""
    ids: list[int] = []
    mask: list[int] = []
    logprobs: list[Optional[float]] = []
    last = path[-1]["id"]
    for index, node in enumerate(path):
        prompt = list(node["prompt_ids"])
        if index == 0:
            new_context = prompt
        else:
            new_context = prompt[len(ids):]                       # what the harness added after the previous sampled tokens
        ids += new_context
        mask += [0] * len(new_context)
        logprobs += [None] * len(new_context)
        completion = list(node["completion_ids"])
        lps = list(node.get("logprobs") or [])
        lps += [None] * (len(completion) - len(lps))
        train = (node["id"] == last) if loss_on == "last" else True
        if dedupe_shared and node["id"] in trained:
            train = False
        ids += completion
        mask += [1 if train else 0] * len(completion)
        logprobs += [(lps[i] if train else None) for i in range(len(completion))]
        if train:
            trained.add(node["id"])
    return {"input_ids": ids, "mask": mask, "logprobs": logprobs, "nodes": [n["id"] for n in path]}


def sequences(nodes: Sequence[dict[str, Any]], *, loss_on: str = "all", dedupe_shared: bool = True) -> list[dict[str, Any]]:
    """Every root-to-leaf sequence of the nodes, oldest first, with the session, harness and reward of the leaf."""
    return sequences_report(nodes, loss_on=loss_on, dedupe_shared=dedupe_shared)[0]


def sequences_report(nodes: Sequence[dict[str, Any]], *, loss_on: str = "all", dedupe_shared: bool = True) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """:func:`sequences` plus the count of paths left out because they had no trainable token (``no_loss_tokens``)."""
    tree = build(nodes)
    rows: list[dict[str, Any]] = []
    skipped = {"no_loss_tokens": 0}
    trained: set[str] = set()
    leaves = sorted(tree["leaves"], key=lambda i: (tree["by_id"][i].get("seq", 0), i))
    for leaf in leaves:
        path = path_to(tree, leaf)
        row = sequence_of(path, trained=trained, loss_on=loss_on, dedupe_shared=dedupe_shared)
        loss = sum(row["mask"])
        if loss == 0:
            skipped["no_loss_tokens"] += 1
            continue
        end = path[-1]
        first = path[0]
        row.update(session=end.get("session"), harness=end.get("harness") or "", reward=end.get("reward"), outcome=end.get("outcome"),
                   model=end.get("model") or "", leaf=end["id"], ts=end.get("ts"), loss_tokens=loss, tokens=len(row["input_ids"]),
                   root=first["id"], depth=len(path), sampling=end.get("sampling") or {})
        rows.append(row)
    return rows, skipped


# ------------------------------------------------------------------ balance by harness
def balance_by_harness(rows: Sequence[dict[str, Any]], mode: Any, seed: int = 42) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Limit the rows per harness. ``mode`` is None (keep all), ``"min"`` (as many as the smallest harness has) or an integer cap. The rows
    kept are a seeded random sample, returned in their original order. The report gives, per harness, how many rows and loss tokens were
    available and how many were kept."""
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row.get("harness") or "", []).append(index)
    if mode in (None, "", 0):
        cap: Optional[int] = None
    elif mode == "min":
        cap = min(len(v) for v in groups.values()) if groups else 0
    else:
        cap = max(1, int(mode))
    keep: set[int] = set()
    report: dict[str, dict[str, int]] = {}
    for harness, indexes in sorted(groups.items()):
        chosen = indexes
        if cap is not None and len(indexes) > cap:
            rng = random.Random(f"{seed}:{harness}")
            chosen = sorted(rng.sample(indexes, cap))
        keep.update(chosen)
        report[harness] = {"available": len(indexes), "kept": len(chosen),
                           "loss_tokens_available": sum(rows[i]["loss_tokens"] for i in indexes),
                           "loss_tokens_kept": sum(rows[i]["loss_tokens"] for i in chosen)}
    return [row for i, row in enumerate(rows) if i in keep], report


def sft_row(row: dict[str, Any]) -> dict[str, Any]:
    """Token-level SFT row: labels are the ids where the model sampled them and ``-100`` elsewhere."""
    labels = [tok if m else IGNORE_INDEX for tok, m in zip(row["input_ids"], row["mask"])]
    return {"input_ids": row["input_ids"], "labels": labels, "harness": row["harness"], "session": row["session"], "reward": row["reward"],
            "model": row["model"], "nodes": row["nodes"], "loss_tokens": row["loss_tokens"]}


def rl_row(row: dict[str, Any]) -> dict[str, Any]:
    """Token-level row for reinforcement learning: ids, loss mask, the log-probability of each sampled token under the sampler (null for context)
    and the reward (null when nothing verified the outcome)."""
    return {"input_ids": row["input_ids"], "loss_mask": row["mask"], "logprobs": row["logprobs"], "reward": row["reward"], "outcome": row["outcome"],
            "harness": row["harness"], "session": row["session"], "model": row["model"], "nodes": row["nodes"], "sampling": row["sampling"],
            "loss_tokens": row["loss_tokens"]}
