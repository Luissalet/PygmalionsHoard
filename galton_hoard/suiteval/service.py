"""Suit evaluations: define, record, score and export.

An evaluation belongs to one suit of Darwin's Hoard (by its slug) and has cases, each with executable acceptance checks, and three arms:
``generic`` (the plain tools), ``current`` (the suit as it is) and ``candidate`` (the improvement under test). For every case and arm a
record keeps whether the deliverable passed its checks, the minutes a person spent correcting it (entered by a person, optional), the total
time, the defects found and the tokens when known. Cases the candidate was built from are marked and left out of the score.

Export: the results go to Darwin's Hoard through its API when it answers (``suiteval.darwin_url``); otherwise, or when asked, they are
written to a JSON file that ``darwin_suit_import_eval`` reads. Either way the suit becomes «evaluated» there only if at least one case was scored.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from .. import messages as msgs
from ..errors import GaltonError
from ..hoard_link.atomic import write_json_atomic
from ..util import new_id
from . import checks as C, stats as S, verdict as V
from .store import SuitEvalStore

FORMAT = "pygmalion-suit-eval/1"
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,59}$")
MAX_CASES = 200
MODES = ("auto", "api", "file")


def _post(url: str, payload: dict[str, Any]) -> tuple[int, Any]:
    with httpx.Client(trust_env=False, timeout=8.0) as client:
        r = client.post(url, json=payload)
        try:
            body = r.json()
        except ValueError:
            body = r.text[:300]
        return r.status_code, body


def _get(url: str) -> tuple[int, Any]:
    with httpx.Client(trust_env=False, timeout=5.0) as client:
        r = client.get(url)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, r.text[:300]


class SuitEvals:
    def __init__(self, store: SuitEvalStore, settings: Any, *, data_dir: Path, clock: Callable[[], float] = time.time,
                 post_fn: Optional[Callable[[str, dict[str, Any]], tuple[int, Any]]] = None,
                 get_fn: Optional[Callable[[str], tuple[int, Any]]] = None):
        self.store, self.settings, self.data_dir, self.clock = store, settings, Path(data_dir), clock
        self.post_fn = post_fn or _post
        self.get_fn = get_fn or _get

    # ------------------------------------------------------------------ definition
    def _clean(self, spec: dict[str, Any]) -> dict[str, Any]:
        cases_in = spec.get("cases") or []
        if not cases_in:
            raise GaltonError("invalid", "se_no_cases")
        cases, seen = [], set()
        for i, raw in enumerate(cases_in):
            cid = str(raw.get("id") or f"k{i + 1}")
            if not CASE_ID.match(cid):
                raise GaltonError("invalid", "se_case_dup", id=cid)
            if cid in seen:
                raise GaltonError("invalid", "se_case_dup", id=cid)
            seen.add(cid)
            cases.append({"id": cid, "title": str(raw.get("title") or cid)[:200], "input": raw.get("input", ""),
                          "used_for_building": bool(raw.get("used_for_building")), "notes": str(raw.get("notes") or "")[:1000],
                          "checks": C.normalize(raw.get("checks"), cid)})
        cand = spec.get("candidate") or {}
        built = [str(x) for x in cand.get("built_from_cases") or []]
        for cid in built:
            if cid not in seen:
                raise GaltonError("invalid", "se_build_unknown", case=cid)
        return {"cases": cases, "candidate": {"description": str(cand.get("description") or "")[:2000], "built_from_cases": sorted(set(built))},
                "notes": str(spec.get("notes") or "")[:2000]}

    def define(self, *, id: str = "", suit: str, name: str = "", cases: list[dict[str, Any]], candidate: Optional[dict[str, Any]] = None, notes: str = "") -> dict[str, Any]:
        suit = suit.strip().lower()
        if not SLUG.match(suit):
            raise GaltonError("invalid", "se_bad_suit", slug=suit)
        definition = self._clean({"cases": cases, "candidate": candidate, "notes": notes})
        existing = self.store.eval(id) if id else None
        if id and existing is None:
            raise GaltonError("not_found", "se_no_eval", id=id)
        if existing:
            old = {c["id"]: c for c in existing["definition"]["cases"]}
            new_ids = {c["id"] for c in definition["cases"]}
            for cid in old:
                if cid not in new_ids and self.store.count_for_case(id, cid):
                    raise GaltonError("conflict", "se_case_has_records", case=cid)
        eval_id = id or new_id("se", self.clock())
        saved = self.store.put_eval(eval_id, suit, name.strip()[:200] or suit, definition)
        if existing:
            self._rescore_changed(saved, {c["id"]: c for c in existing["definition"]["cases"]})
        return self.get(eval_id)

    def _rescore_changed(self, ev: dict[str, Any], old: dict[str, dict[str, Any]]) -> None:
        for case in ev["definition"]["cases"]:
            if old.get(case["id"], {}).get("checks") == case["checks"]:
                continue
            for r in self.store.records(ev["id"]):
                if r["case_id"] == case["id"]:
                    self._run_checks(ev["id"], case, r["arm"], r["run"], r["inputs"], keep=r)

    # ------------------------------------------------------------------ records
    def _run_checks(self, eval_id: str, case: dict[str, Any], arm: str, run: int, inputs: dict[str, Any], *, keep: Optional[dict[str, Any]] = None,
                    extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        root = Path(inputs["deliverable"]) if inputs.get("deliverable") else None
        success, results = C.run_checks(case["checks"], root=root, answer=str(inputs.get("answer") or ""), values=inputs.get("values"),
                                        judgements=inputs.get("judgements"))
        fields: dict[str, Any] = {"success": success, "checks": results, "inputs": inputs}
        if not inputs.get("defects_manual"):
            fields["defects"] = sum(1 for r in results if r["ok"] is False)     # counted from the checks until a person enters the number
        fields.update(extra or {})
        return self.store.put_record(eval_id, case["id"], arm, run, fields)

    def record(self, *, eval: str, case: str, arm: str, run: int = 1, deliverable: str = "", answer: Optional[str] = None, values: Optional[dict[str, Any]] = None,
               judgements: Optional[dict[str, Any]] = None, correction_minutes: Optional[float] = None, total_time_s: Optional[float] = None,
               defects: Optional[int] = None, tokens: Optional[int] = None, notes: Optional[str] = None) -> dict[str, Any]:
        ev = self._eval(eval)
        if arm not in S.ARMS:
            raise GaltonError("invalid", "se_bad_arm", arm=arm)
        spec = next((c for c in ev["definition"]["cases"] if c["id"] == case), None)
        if spec is None:
            raise GaltonError("not_found", "se_no_case", case=case)
        old = self.store.record(eval, case, arm, run)
        inputs = dict((old or {}).get("inputs") or {})
        touched = old is None
        if deliverable:
            if not Path(deliverable).is_dir():
                raise GaltonError("invalid", "se_bad_deliverable", path=deliverable)
            inputs["deliverable"], touched = str(deliverable), True
        if answer is not None:
            inputs["answer"], touched = answer[:C.MAX_TEXT], True
        if values:
            inputs["values"], touched = {**(inputs.get("values") or {}), **values}, True
        if judgements:
            inputs["judgements"], touched = {**(inputs.get("judgements") or {}), **judgements}, True
        if defects is not None:
            inputs["defects_manual"] = True
        extra: dict[str, Any] = {}
        for key, val in (("correction_minutes", correction_minutes), ("total_time_s", total_time_s), ("tokens", tokens), ("notes", notes), ("defects", defects)):
            if val is not None:
                extra[key] = val
        if touched:
            self._run_checks(eval, spec, arm, run, inputs, keep=old, extra=extra)
        elif extra:
            self.store.put_record(eval, case, arm, run, extra)
        return self.store.record(eval, case, arm, run)

    def check(self, *, checks: list[dict[str, Any]], deliverable: str = "", answer: str = "", values: Optional[dict[str, Any]] = None,
              judgements: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Try checks against a folder without saving anything (to write a check and see what it says)."""
        specs = C.normalize(checks, "draft")
        if deliverable and not Path(deliverable).is_dir():
            raise GaltonError("invalid", "se_bad_deliverable", path=deliverable)
        success, results = C.run_checks(specs, root=Path(deliverable) if deliverable else None, answer=answer, values=values, judgements=judgements)
        return {"success": success, "checks": results}

    # ------------------------------------------------------------------ reading
    def _eval(self, eval_id: str) -> dict[str, Any]:
        ev = self.store.eval(eval_id)
        if ev is None:
            raise GaltonError("not_found", "se_no_eval", id=eval_id)
        return ev

    def get(self, eval_id: str) -> dict[str, Any]:
        ev = self._eval(eval_id)
        records = self.store.records(eval_id)
        return {"evaluation": ev, "records": records, "score": self.score(eval_id)}

    def list(self, suit: str = "") -> dict[str, Any]:
        rows = []
        for ev in self.store.evals(suit.strip().lower()):
            recs = self.store.records(ev["id"])
            sc = S.aggregate(ev["definition"], recs)
            vd = V.verdicts(sc)
            rows.append({"id": ev["id"], "suit": ev["suit"], "name": ev["name"], "cases": sc["cases_total"], "cases_scored": sc["cases_scored"],
                         "decisions": {k: v["decision"] for k, v in vd.items()},
                         "records": len(recs), "excluded": [e["case"] for e in sc["excluded"]], "exported": ev["exported"], "updated_ts": ev["updated_ts"]})
        return {"evaluations": rows, "count": len(rows)}

    def score(self, eval_id: str) -> dict[str, Any]:
        ev = self._eval(eval_id)
        agg = S.aggregate(ev["definition"], self.store.records(eval_id))
        return {"id": ev["id"], "suit": ev["suit"], "name": ev["name"], "arms": list(S.ARMS), "candidate": ev["definition"]["candidate"], **agg,
                "verdict": V.verdicts(agg)}

    def library(self) -> dict[str, Any]:
        """The suit library of Darwin's Hoard (one suit per kind of engagement) next to the evaluations kept here and their decisions.

        Darwin owns the suits; this only reads them. When Darwin does not answer, the suits known from local evaluations are listed."""
        base = str(self.settings.get("suiteval.darwin_url") or "").strip().rstrip("/")
        darwin: dict[str, Any] = {"url": base, "reachable": False, "reason": ""}
        suits: list[dict[str, Any]] = []
        try:
            status, body = self.get_fn(f"{base}/api/suits")
            if 200 <= status < 300 and isinstance(body, dict):
                darwin["reachable"] = True
                suits = [s for s in body.get("suits") or [] if isinstance(s, dict) and s.get("slug")]
            else:
                darwin["reason"] = f"HTTP {status}"
        except Exception as exc:  # noqa: BLE001 - Darwin not running is normal
            darwin["reason"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        local = self.list()["evaluations"]
        by_suit: dict[str, list[dict[str, Any]]] = {}
        for ev in local:
            by_suit.setdefault(ev["suit"], []).append(ev)
        items = []
        for s in suits:
            evs = by_suit.pop(s["slug"], [])
            items.append({"slug": s["slug"], "name": s.get("name") or s["slug"], "objective": s.get("objective") or "", "status": s.get("status") or "",
                          "backing": s.get("backing") or "", "missing": s.get("missing") or [], "unproven": s.get("unproven") or [],
                          "in_darwin": True, "evaluations": evs, "latest": evs[0] if evs else None})
        for slug, evs in sorted(by_suit.items()):
            items.append({"slug": slug, "name": slug, "objective": "", "status": "", "backing": "", "missing": [], "unproven": [],
                          "in_darwin": False, "evaluations": evs, "latest": evs[0] if evs else None})
        return {"darwin": darwin, "suits": items, "count": len(items), "evaluated_here": sum(1 for i in items if i["evaluations"])}

    def delete(self, eval_id: str) -> dict[str, Any]:
        self._eval(eval_id)
        self.store.delete_eval(eval_id)
        return {"deleted": eval_id}

    # ------------------------------------------------------------------ export
    def payload(self, eval_id: str) -> dict[str, Any]:
        sc = self.score(eval_id)
        return {"format": FORMAT, "suit": sc["suit"], "evaluation_id": sc["id"], "name": sc["name"], "generated_at": self.clock(), "source": "pygmalion-hoard",
                "arms": sc["arms"], "candidate": sc["candidate"], "cases_total": sc["cases_total"], "cases_scored": sc["cases_scored"], "excluded": sc["excluded"],
                "aggregate": sc["aggregate"], "paired": sc["paired"], "per_case": sc["per_case"],
                "warnings": [{"key": w.key, "text": str(w)} for w in sc["warnings"]],
                "verdict": {k: {**v, "reasons": [{"key": r.key, "text": str(r)} for r in v["reasons"]]} for k, v in sc["verdict"].items()}}

    def _folder(self) -> Path:
        configured = str(self.settings.get("suiteval.export_folder") or "").strip()
        return Path(configured) if configured else self.data_dir / "exports" / "suit-evals"

    def export(self, eval_id: str, *, mode: str = "auto", url: str = "") -> dict[str, Any]:
        if mode not in MODES:
            raise GaltonError("invalid", "se_bad_arm", arm=mode)
        payload = self.payload(eval_id)
        if payload["cases_scored"] < 1:
            raise GaltonError("invalid", "se_nothing_scored")
        base = (url or str(self.settings.get("suiteval.darwin_url") or "")).strip().rstrip("/")
        info: dict[str, Any] = {"at": self.clock(), "mode": mode, "delivered": False, "file": "", "url": "", "reason": ""}
        if mode in ("auto", "api"):
            if not re.match(r"^https?://", base, re.I):
                if mode == "api":
                    raise GaltonError("invalid", "se_export_url", url=base)
                info["reason"] = "no address"
            else:
                target = f"{base}/api/evaluations"
                try:
                    status, body = self.post_fn(target, payload)
                    if 200 <= status < 300:
                        info.update(delivered=True, url=target, status=status)
                    else:
                        info.update(reason=f"HTTP {status}: {str(body.get('error') if isinstance(body, dict) else body)[:200]}", status=status, url=target)
                        if mode == "api":
                            raise GaltonError("unavailable", "se_export_failed", reason=info["reason"])
                except GaltonError:
                    raise
                except Exception as exc:  # noqa: BLE001 - Darwin not running is the normal case for the fallback
                    info.update(reason=f"{type(exc).__name__}: {str(exc)[:160]}", url=target)
                    if mode == "api":
                        raise GaltonError("unavailable", "se_export_failed", reason=info["reason"]) from exc
        if not info["delivered"]:
            path = self._folder() / f"{payload['suit']}--{eval_id}.json"
            write_json_atomic(path, payload)
            info["file"] = str(path)
        self.store.set_exported(eval_id, info)
        return {"exported": info, "payload_summary": {"suit": payload["suit"], "cases_scored": payload["cases_scored"], "excluded": payload["excluded"],
                                                      "warnings": payload["warnings"]}}
