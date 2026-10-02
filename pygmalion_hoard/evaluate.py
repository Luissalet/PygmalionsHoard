"""Measure a result against its parent with Galton's Hoard.

The conversation goes through the family hub's proxy (``family.call("galton", tool, args)``). When the hub or Galton's entry in it
is not available, the same tools are called directly on Galton's own port with the token from its data folder. Galton owns the
GPU for the test run (it leases what its servers need), so this module never holds a lease.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from .errors import PygmalionError
from .hoard_link import _hubclient
from .lineage import Lineage
from .messages import text
from .reference import Reference
from .settings import Settings
from .store import Store
from .util import slug

REGRESSION_SUITES = ["rapida"]
SUITES_BY_INTENT = {
    "style": ["escritura-es", "instrucciones"],
    "writing": ["escritura-es", "instrucciones"],
    "context": ["contexto-largo"],
    "code": ["codigo-python"],
    "general": ["razonamiento", "instrucciones", "escritura-es"],
    "smoke": ["rapida"],
    "dataset": REGRESSION_SUITES,        # the held-out suite is built from the dataset; this is the general suite that runs beside it
    "quant": ["rapida", "razonamiento"],  # a plain quantization: did the smaller file keep the answers? (compared with its f16 parent)
}
DONE_STATES = ("done", "completed", "finished")
FAILED_STATES = ("failed", "error", "cancelled", "canceled")
VERDICTS = {"better": "better", "worse": "worse", "no clear difference": "no_clear_difference", "no_clear_difference": "no_clear_difference",
            "same": "no_clear_difference", "tie": "no_clear_difference", "inconclusive": "no_clear_difference", "no data": "no_data", "no_data": "no_data"}
SIBLING_TOKEN = Path(__file__).resolve().parents[2] / "Galton's Hoard" / "data" / "mcp-token"  # the usual layout: both apps side by side
LENGTH_RE = re.compile(r"(\d+)\s*k\s+tokens", re.I)
NEEDLE_SUITE = "contexto-largo"
RUBRIC = "¿Responde correctamente y con el mismo contenido que la referencia? Penaliza datos inventados."
RUBRIC_TEXT = "¿La continuación mantiene el contenido y el estilo de la referencia? Penaliza datos inventados."
CONTINUE_PROMPT = "Continúa el texto siguiente sin repetirlo:\n\n"
MAX_PROMPT_CHARS = 12000
MAX_REFERENCE_CHARS = 6000

FamilyCall = Callable[..., dict[str, Any]]


def normalise_verdict(value: Any) -> str:
    return VERDICTS.get(str(value or "").strip().lower().replace("-", " "), "unknown")


def first_key(data: Any, *names: str) -> Any:
    """The first of ``names`` found in a dict, looking one level into ``result``, ``run`` and ``data`` wrappers."""
    if not isinstance(data, dict):
        return None
    for name in names:
        if data.get(name) not in (None, ""):
            return data[name]
    for wrapper in ("result", "run", "data", "suite"):
        inner = data.get(wrapper)
        if isinstance(inner, dict):
            found = first_key(inner, *names)
            if found is not None:
                return found
    return None


def pick_suites(intent: str = "", suites: Optional[list[str]] = None, dataset_kind: str = "", context_variant: bool = False) -> list[str]:
    if suites:
        return [str(s) for s in suites]
    if intent:
        if intent not in SUITES_BY_INTENT:
            raise PygmalionError("invalid", "intent_unknown", intent=intent, options=list(SUITES_BY_INTENT))
        return list(SUITES_BY_INTENT[intent])
    if context_variant:
        return list(SUITES_BY_INTENT["context"])
    return list(SUITES_BY_INTENT["style" if dataset_kind in ("text", "instruction", "chat", "") else "general"])


def contestant_spec(artifact: dict[str, Any]) -> dict[str, Any]:
    """How another app is told to run an artifact: a GGUF by path, a published tag by name."""
    if artifact["kind"] == "gguf":
        return {"kind": "gguf", "path": artifact["path"], "name": artifact["name"]}
    if artifact["kind"] == "ollama":
        return {"kind": "ollama", "model": artifact["path"], "name": artifact["name"]}
    raise PygmalionError("invalid", "evaluate_kind", name=artifact["name"], kind=artifact["kind"])


def summarise_compare(raw: Any) -> dict[str, Any]:
    """Galton's comparison reduced to what we store: verdict, p-value, shared cases, score difference with its interval, counts."""
    warnings = first_key(raw, "warnings")
    out: dict[str, Any] = {"verdict": normalise_verdict(first_key(raw, "verdict")), "p_value": first_key(raw, "p_value"), "n": first_key(raw, "n"),
                           "warnings": [str(w) for w in warnings] if isinstance(warnings, list) else []}
    for key in ("diff", "diff_ci", "wins", "losses", "ties", "scope", "sentence"):
        value = first_key(raw, key)
        if value is not None:
            out[key] = value
    return out


# ------------------------------------------------------------------------------------------------- the dataset's own held-out suite
def suite_name(dataset: str, n: int) -> str:
    return f"pyg-{slug(dataset, 40)}-v{n}-eval"


def _cut_head(value: str, limit: int) -> str:
    """The end of ``value`` when it is too long (the answer follows the end of a prompt)."""
    return value if len(value) <= limit else value[-limit:]


def record_case(record: dict[str, Any], position: int) -> Optional[dict[str, Any]]:
    """One held-out record as a row for Galton's ``cases_import``: what to ask, the judge checker with its rubric and the reference answer.

    Chat records ask everything before the last assistant turn (one user turn with an optional system message becomes a plain prompt, a longer
    exchange keeps its turns); instruction records ask the prompt; text records ask for the continuation of their first part."""
    row: dict[str, Any]
    reference = ""
    rubric = RUBRIC
    if isinstance(record.get("messages"), list):
        turns = [m for m in record["messages"] if isinstance(m, dict) and m.get("role") in ("system", "user", "assistant")]
        last = max((i for i, m in enumerate(turns) if m["role"] == "assistant" and str(m.get("content") or "").strip()), default=None)
        if last is None:
            return None
        reference = str(turns[last]["content"]).strip()
        asked = turns[:last]
        users = [m for m in asked if m["role"] == "user"]
        if not users:
            return None
        system = next((str(m["content"]) for m in asked if m["role"] == "system"), "")
        if len(users) == 1 and all(m["role"] in ("system", "user") for m in asked):
            row = {"prompt": _cut_head(str(users[0]["content"]), MAX_PROMPT_CHARS)}
            if system:
                row["system"] = system
        else:
            row = {"prompt": {"text": _cut_head(str(users[-1]["content"]), MAX_PROMPT_CHARS),
                              "messages": [{"role": m["role"], "content": str(m.get("content") or "")} for m in asked]}}
    elif isinstance(record.get("prompt"), str) and isinstance(record.get("response"), str):
        reference = record["response"].strip()
        row = {"prompt": _cut_head(record["prompt"], MAX_PROMPT_CHARS)}
        if str(record.get("system") or "").strip():
            row["system"] = str(record["system"])
    elif isinstance(record.get("text"), str):
        body = record["text"].strip()
        if len(body) < 40:
            return None
        cut = body.rfind(" ", 0, max(20, int(len(body) * 0.6)))
        cut = cut if cut > 0 else int(len(body) * 0.6)
        head, reference = body[:cut].strip(), body[cut:].strip()
        row = {"prompt": CONTINUE_PROMPT + _cut_head(head, 1500)}
        reference = reference[:800]
        rubric = RUBRIC_TEXT
    else:
        return None
    reference = reference[:MAX_REFERENCE_CHARS]
    if not reference:
        return None
    asked_text = row["prompt"] if isinstance(row["prompt"], str) else row["prompt"]["text"]
    title = " ".join(asked_text.split())[:60]
    # the reference travels twice: as the case's reference answer (``expected`` in an import) and inside the checker, which is what the judge reads
    row.update({"title": f"eval-{position:03d} {title}".strip(), "expected": reference, "weight": 1,
                "checker": {"type": "judge", "rubric": rubric, "reference": reference}})
    return row


def eval_rows(records: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], int]:
    """The import rows for the held-out records (at most ``limit``) and how many records could not be turned into a case."""
    rows: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        if len(rows) >= limit:
            break
        row = record_case(record, len(rows) + 1)
        if row is None:
            skipped += 1
        else:
            rows.append(row)
    return rows, skipped


def answer_tokens(rows: list[dict[str, Any]]) -> int:
    """A generous answer length for the suite: the longest reference in characters over three, plus room, within 256 and 2048."""
    longest = max((len(r["expected"]) for r in rows), default=0)
    return max(256, min(2048, int(math.ceil(longest / 3.0)) + 128))


class Galton:
    def __init__(self, settings: Settings, *, family_call: Optional[FamilyCall] = None, transport: Optional[httpx.BaseTransport] = None,
                 offline: bool = False, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time):
        self.settings = settings
        self.family_call = family_call
        self.transport = transport
        self.offline = offline
        self.sleep = sleep
        self.clock = clock
        self._token_path = ""

    # ------------------------------------------------------------------ transport
    def token(self) -> str:
        """Galton's token: the setting, else the file the hub's registry names, else Galton's folder next to this one."""
        candidates = [self.settings.get("galton.token_file").strip()]
        if not candidates[0]:
            if not self._token_path:
                self._token_path = self._registry_token_file()
            candidates = [self._token_path, str(SIBLING_TOKEN)]
            if self._token_path and not Path(self._token_path).is_file():
                self._token_path = ""                       # moved or gone: ask the registry again next time
        for path in candidates:
            if not path:
                continue
            try:
                value = Path(path).read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if value:
                return value
        return ""

    def _registry_token_file(self) -> str:
        """Where Galton keeps its token, as the hub's registry knows it (best effort)."""
        if self.offline:
            return ""
        try:
            status, body = _hubclient.fetch(_hubclient.hub_url() + "/api/apps", timeout=8.0)  # the hub checks every app: seconds, not ms
        except Exception:  # noqa: BLE001
            return ""
        apps = body if isinstance(body, list) else (body or {}).get("apps") if isinstance(body, dict) else None
        for app in apps or []:
            if isinstance(app, dict) and (app.get("id") or app.get("app")) == "galton":
                direct = app.get("token_file")
                data_dir = app.get("data_dir")
                return str(direct or (Path(data_dir) / "mcp-token" if data_dir else ""))
        return ""

    def _via_hub(self, tool: str, args: dict[str, Any]) -> Optional[Any]:
        if self.family_call is None or self.offline:
            return None
        reply = None
        for attempt in range(2):  # one retry: a busy hub or a restart of Galton must not drop a long evaluation to the fallback
            try:
                reply = self.family_call("galton", tool, args, timeout=120.0)
                break
            except Exception:  # noqa: BLE001
                if attempt == 0:
                    self.sleep(2.0)
        if reply is None:
            return None
        if isinstance(reply, dict) and reply.get("ok") and "result" in reply:
            return reply["result"]
        if isinstance(reply, dict) and reply.get("status") in (400, 404, 409, 422) and reply.get("error"):
            raise PygmalionError("galton_unavailable", "galton_refused", tool=tool, detail=str(reply["error"])[:300])
        return None

    def _direct(self, tool: str, args: dict[str, Any]) -> Any:
        if self.offline:
            raise PygmalionError("offline", "galton_offline")
        token = self.token()
        if not token:
            raise PygmalionError("galton_unavailable", "galton_token_missing")
        url = self.settings.get("galton.url").rstrip("/") + "/api/agent/call"
        try:
            with httpx.Client(transport=self.transport, timeout=120.0, trust_env=self.transport is None) as client:
                response = client.post(url, json={"name": tool, "arguments": args, "caller": "pygmalion"}, headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError as exc:
            raise PygmalionError("galton_unavailable", "galton_unreachable", detail=type(exc).__name__) from exc
        try:
            body = response.json()
        except ValueError:
            body = {"error": response.text[:200]}
        if response.status_code >= 400:
            message = body.get("error") if isinstance(body, dict) else str(body)
            raise PygmalionError("galton_unavailable", "galton_status", status=response.status_code, detail=str(message)[:300])
        return body

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        hub = self._via_hub(tool, args)
        if hub is not None:
            return hub
        return self._direct(tool, args)

    def available(self) -> dict[str, Any]:
        """Is Galton reachable (hub proxy first, then directly)? Never raises."""
        if self.offline:
            return {"ok": False, "via": None, "detail": text("detail_offline")}
        try:
            if self._via_hub("galton_overview", {}) is not None:
                return {"ok": True, "via": "hub", "detail": ""}
        except PygmalionError as exc:
            return {"ok": False, "via": "hub", "detail": exc.coded()}
        try:
            self._direct("galton_overview", {})
            return {"ok": True, "via": "direct", "detail": ""}
        except PygmalionError as exc:
            return {"ok": False, "via": None, "detail": exc.coded()}


class Evaluator:
    def __init__(self, store: Store, lineage: Lineage, galton: Galton, settings: Settings, datasets: Any = None, references: Optional[Reference] = None):
        self.store = store
        self.lineage = lineage
        self.galton = galton
        self.settings = settings
        self.datasets = datasets
        self.references = references or Reference(store, lineage)

    # ------------------------------------------------------------------ planning
    def dataset_info(self, child: dict[str, Any]) -> Optional[dict[str, Any]]:
        """The dataset version a result was trained on (its own or its nearest ancestor's) when that version holds out records."""
        for art in [child, *self.lineage.ancestors(child["id"])]:
            vid = art.get("dataset_version")
            if not vid:
                continue
            try:
                version = self.store.version(vid)
                dataset = self.store.dataset(version["dataset_id"])
            except PygmalionError:
                return None
            held = list((version.get("splits") or {}).get("eval") or [])
            return {"version": version["id"], "n": version["n"], "dataset": dataset["name"], "kind": version["kind"], "records": len(held),
                    "sha": version["sha256"], "suite": suite_name(dataset["name"], version["n"])}
        return None

    def plan(self, ref: str, against: str = "", intent: str = "", suites: Optional[list[str]] = None, regression: bool = True) -> dict[str, Any]:
        """What an evaluation will do: the suites, and the reference (``plan["reference"]``, see ``reference.py``). ``plan["parent"]`` is the file
        to compare with when it exists, ``None`` when it has to be built first."""
        child = self.lineage.resolve(ref, ("gguf", "ollama"))
        context_variant = any(a["kind"] == "ctx_variant" for a in [child, *self.lineage.ancestors(child["id"])])
        dataset = self.dataset_info(child)
        dataset_kind = ""
        if child.get("dataset_version"):
            try:
                dataset_kind = self.store.version(child["dataset_version"])["kind"]
            except PygmalionError:
                dataset_kind = ""
        # no suites and no intent given: a result trained on a dataset that holds out records is measured on those records
        if not suites and (intent == "dataset" or (not intent and not context_variant and dataset and dataset["records"] > 0)):
            if not dataset or dataset["records"] < 1:
                raise PygmalionError("invalid", "eval_no_records", name=child["name"])
            chosen = [dataset["suite"], *(REGRESSION_SUITES if regression else [])]
            reference = self.references.plan(child, "dataset", against)
            return {"child": child, "parent": reference["artifact"], "reference": reference, "suites": chosen, "context_variant": context_variant,
                    "intent": "dataset", "dataset": dataset, "regression": bool(regression)}
        chosen = pick_suites(intent, suites, dataset_kind, context_variant)
        reference = self.references.plan(child, intent, against)
        return {"child": child, "parent": reference["artifact"], "reference": reference, "suites": chosen, "context_variant": context_variant,
                "intent": intent, "dataset": None, "regression": False}

    def explain(self, ref: str, against: str = "", intent: str = "", suites: Optional[list[str]] = None, regression: bool = True) -> dict[str, Any]:
        """The plan as a read-only answer: which reference an evaluation would use and whether it has to be prepared first."""
        plan = self.plan(ref, against, intent, suites, regression)
        return {"artifact": plan["child"]["id"], "name": plan["child"]["name"], "intent": plan["reference"]["intent"], "suites": plan["suites"],
                "dataset": plan["dataset"], "reference": self.references.view(plan["reference"])}

    # ------------------------------------------------------------------ the held-out suite in Galton
    def _judge_check(self) -> None:
        """Refuse early when Galton has no judge model: the held-out suite is graded by one, so a run without it measures nothing."""
        status = self.galton.call("galton_status", {})
        value = None
        if isinstance(status, dict):
            if "judge" in status:
                value = status["judge"]
            elif isinstance(status.get("settings"), dict):
                value = status["settings"].get("judge.contestant")
        if value is not None and not str(value).strip():
            raise PygmalionError("galton_unavailable", "galton_judge_missing")

    def ensure_suite(self, info: dict[str, Any], log: Callable[[str], None] = lambda m: None) -> dict[str, Any]:
        """Galton's suite with the held-out records of a dataset version: reused when one built from the same version (same content hash and
        number of cases) exists, built otherwise. Returns ``{id, name, cases, created, skipped, capped}``."""
        if self.datasets is None:
            raise PygmalionError("invalid", "eval_no_records", name=info["dataset"])
        version = self.store.version(info["version"])
        records = self.datasets.read_records(version)
        held = [records[i] for i in sorted((version.get("splits") or {}).get("eval") or []) if 0 <= i < len(records)]
        limit = max(1, self.settings.int("galton.eval_cases"))
        rows, skipped = eval_rows(held, limit)
        if not rows:
            raise PygmalionError("invalid", "eval_no_records", name=info["dataset"])
        key = f"sha256:{info['sha']}:{len(rows)}"
        listing = self.galton.call("suites_list", {"builtin": False})
        cards = {str(c.get("name")): c for c in (listing.get("suites") if isinstance(listing, dict) else None) or [] if isinstance(c, dict)}
        base = info["suite"]
        name = base
        for candidate in (base, f"{base}-{info['sha'][:8]}", f"{base}-{info['sha'][:16]}"):
            name = candidate
            card = cards.get(candidate)
            if card is None:
                break
            if key in str(card.get("description") or ""):
                if int(card.get("cases") or 0) == len(rows):
                    log(str(text("eval_suite_reused", name=candidate, n=len(rows))))
                    return {"id": card.get("id") or candidate, "name": candidate, "cases": len(rows), "created": False, "skipped": skipped,
                            "capped": len(held) > len(rows)}
                # built from this very version but left incomplete (an import that failed half way): it is ours, so replace it
                self.galton.call("suite_remove", {"suite": card.get("id") or candidate, "confirm": True})
                break
        created = self.galton.call("suite_create", {"name": name, "category": "custom", "max_tokens": answer_tokens(rows),
                                                    "description": f"Pygmalion: held-out records of {info['dataset']} v{info['n']}. {key}"})
        suite_id = first_key(created, "id") or name
        imported = self.galton.call("cases_import", {"suite": suite_id, "format": "jsonl", "text": "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)})
        added = int(first_key(imported, "added") or 0)
        if added < 1:
            errors = first_key(imported, "errors")
            raise PygmalionError("galton_unavailable", "galton_suite_empty", name=name, detail=json.dumps(errors, ensure_ascii=False, default=str)[:300] if errors else "")
        log(str(text("eval_suite_created", name=name, n=added)))
        return {"id": suite_id, "name": name, "cases": added, "created": True, "skipped": skipped, "capped": len(held) > len(rows)}

    # ------------------------------------------------------------------ running
    def run(self, ref: str, *, against: str = "", intent: str = "", suites: Optional[list[str]] = None, settings: Optional[dict[str, Any]] = None,
            regression: bool = True, cancel: Optional[threading.Event] = None, progress: Callable[..., None] = lambda **kw: None,
            log: Callable[[str], None] = lambda m: None, reference: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """``reference`` is what ``evaluate_start`` decided (``{mode, model, quant, imatrix}``) when it passed the file to compare with as ``against``:
        it is only recorded, so the verdict says which kind of reference it rests on."""
        plan = self.plan(ref, against, intent, suites, regression)
        child, parent = plan["child"], plan["parent"]
        if parent is None:        # the base model's file in this quantization does not exist and nothing queued it
            raise PygmalionError("not_found", "eval_reference_missing", model=plan["reference"]["model"], quant=plan["reference"]["quant"])
        used = reference or {k: plan["reference"][k] for k in ("mode", "model", "quant")} | {"imatrix": bool(plan["reference"]["imatrix"])}
        specs = [contestant_spec(parent), contestant_spec(child)]
        dataset_suite: Optional[dict[str, Any]] = None
        run_suites = plan["suites"]
        if plan["intent"] == "dataset":
            self._judge_check()
            progress(state="suite")
            dataset_suite = self.ensure_suite(plan["dataset"], log)
            run_suites = [dataset_suite["id"], *(REGRESSION_SUITES if plan["regression"] else [])]
        log(f"Evaluating {child['name']} against {parent['name']} on {', '.join(plan['suites'])}.")
        run_settings = dict(settings) if settings else {"temperature": 0, "repeats": 1}
        if plan["intent"] == "dataset" and "effort" not in run_settings:
            # the records answer directly: thinking first only spends the budget (and hours on a CPU) without
            # measuring what the training taught, so both models answer without it unless the caller asks
            run_settings["effort"] = "off"
        if plan["context_variant"] and "context" not in run_settings:
            window = self.child_context(child)
            if window:
                run_settings["context"] = window
        started = self.galton.call("run_start", {"suites": run_suites, "contestants": specs, "settings": run_settings,
                                                 "label": f"pygmalion: {child['name']}"[:120]})
        run_id = first_key(started, "id", "run_id")
        if isinstance(run_id, dict):
            run_id = run_id.get("id")
        if not run_id:
            raise PygmalionError("galton_unavailable", "galton_no_run", answer=json.dumps(started, default=str)[:200])
        run_card = self._wait(run_id, cancel, progress)
        ids = self._contestant_ids(run_card)
        if len(ids) != 2:
            raise PygmalionError("galton_unavailable", "galton_run_models", run=run_id, n=len(ids))
        # Galton calls a run "done" when at least one model finished; a comparison with a model that could not be run (no GPU room,
        # a file llama-server refused) would rest on old results or none, so that is an error, not a verdict
        for role, item in zip(("parent", "child"), run_card.get("contestants") or []):
            if isinstance(item, dict) and str(item.get("state") or "").lower() in FAILED_STATES:
                name = parent["name"] if role == "parent" else child["name"]
                raise PygmalionError("failed", f"galton_{role}_failed", name=name, run=run_id, detail=str(item.get("error") or item.get("state"))[:200])
        # a = the new model, b = its parent: "better" then means the new model is better.
        compare_args: dict[str, Any] = {"a": ids[1], "b": ids[0], "include_stale": False}
        record_extra: dict[str, Any] = {}
        if dataset_suite is not None:
            comparison = summarise_compare(self.galton.call("compare", {**compare_args, "suite": dataset_suite["id"]}))
            judged = self._judge_pending(run_id, run_card)
            if judged and comparison["verdict"] in ("no_data", "unknown"):
                raise PygmalionError("galton_unavailable", "galton_judge_unavailable", run=run_id, n=judged)
            warnings = list(comparison["warnings"])
            if judged:
                warnings.append(text("eval_pending_judge", n=judged))
            if dataset_suite["capped"]:
                warnings.append(text("eval_records_capped", n=dataset_suite["cases"], total=plan["dataset"]["records"]))
            comparison["warnings"] = warnings
            record_extra = {"intent": "dataset", "dataset": {**{k: plan["dataset"][k] for k in ("version", "n", "dataset", "records")},
                                                              "suite": dataset_suite["name"], "suite_id": dataset_suite["id"], "cases": dataset_suite["cases"],
                                                              "created": dataset_suite["created"], "skipped": dataset_suite["skipped"],
                                                              "verdict": comparison["verdict"]}}
            if plan["regression"]:
                record_extra["regression"] = self._regression(compare_args, REGRESSION_SUITES[0])
        else:
            if len(plan["suites"]) == 1:
                compare_args["suite"] = plan["suites"][0]
            comparison = summarise_compare(self.galton.call("compare", compare_args))
        if plan["reference"]["warnings"]:
            comparison["warnings"] = [*comparison["warnings"], *plan["reference"]["warnings"]]
        record = {"run": run_id, "suites": plan["suites"], "parent": parent["id"], "parent_name": parent["name"], "reference": used, "ts": self.galton.clock(),
                  "contestants": {"parent": ids[0], "child": ids[1]}, **comparison, **record_extra}
        if plan["context_variant"]:
            record["needle"] = self._needle(run_id)
        self.lineage.add_metrics(child["id"], "galton", record)
        # only a measured verdict that is not "worse": no shared cases ("no_data") or an unreadable answer is no reason to promote; a general suite that
        # got worse is no reason either, whatever the held-out records say
        regressed = (record_extra.get("regression") or {}).get("verdict") == "worse"
        record["promote_suggested"] = comparison["verdict"] in ("better", "no_clear_difference") and not regressed
        return record

    def _wait(self, run_id: Any, cancel: Optional[threading.Event], progress: Callable[..., None]) -> dict[str, Any]:
        deadline = self.galton.clock() + self.settings.int("galton.timeout_s")
        while True:
            if cancel is not None and cancel.is_set():
                try:
                    self.galton.call("run_cancel", {"run": run_id})
                except PygmalionError:
                    pass
                raise PygmalionError("failed", "cancelled")
            status = self.galton.call("run_status", {"run": run_id})
            run_card = status.get("run") if isinstance(status, dict) and isinstance(status.get("run"), dict) else status
            state = str(first_key(run_card, "state") or "").lower()
            pct = (run_card.get("progress") or {}).get("pct") if isinstance(run_card, dict) and isinstance(run_card.get("progress"), dict) else None
            progress(step=pct if isinstance(pct, (int, float)) and not isinstance(pct, bool) else None, state=state, run=run_id)
            if state in DONE_STATES:
                return run_card if isinstance(run_card, dict) else {}
            if state in FAILED_STATES:
                raise PygmalionError("failed", "galton_run_ended", run=run_id, state=state, detail=str(first_key(run_card, "error") or "")[:200])
            if self.galton.clock() > deadline:
                raise PygmalionError("failed", "galton_run_timeout", run=run_id)
            self.galton.sleep(3.0)

    @staticmethod
    def _judge_pending(run_id: Any, run_card: dict[str, Any]) -> int:
        """Answers of the run still waiting for the judge when Galton reports the run done (its judge could not be started)."""
        value = first_key(run_card, "pending_judge")
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0

    def _regression(self, compare_args: dict[str, Any], suite: str) -> dict[str, Any]:
        try:
            summary = summarise_compare(self.galton.call("compare", {**compare_args, "suite": suite}))
        except PygmalionError as exc:
            return {"suite": suite, "verdict": "no_data", "warnings": [exc.coded()]}
        return {"suite": suite, **{k: v for k, v in summary.items() if k != "sentence"}}

    def child_context(self, child: dict[str, Any]) -> Optional[int]:
        """The context window a context variant was built for, read from the GGUF's metadata kept with the artifact."""
        for art in [child, *self.lineage.ancestors(child["id"])]:
            value = ((art.get("metrics") or {}).get("gguf") or {}).get("context_length")
            if isinstance(value, int) and value > 0:
                return value
        return None

    @staticmethod
    def _contestant_ids(run_card: Any) -> list[str]:
        items = run_card.get("contestants") if isinstance(run_card, dict) else None
        ids = []
        for item in items if isinstance(items, list) else []:
            cid = item.get("id") if isinstance(item, dict) else item
            if cid:
                ids.append(str(cid))
        return ids

    def _needle(self, run_id: Any) -> Any:
        """Mean score per context length (``{"8k": {model id: score}}``), read from the long-context suite's case titles."""
        try:
            raw = self.galton.call("run_results", {"run": run_id, "suite": NEEDLE_SUITE, "include_output": False, "limit": 500})
        except PygmalionError:
            return None
        rows = raw.get("results") if isinstance(raw, dict) else None
        by_length: dict[str, dict[str, list[float]]] = {}
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            found = LENGTH_RE.search(str(row.get("title") or ""))
            who = str(row.get("contestant") or "")
            if found and who and isinstance(row.get("score"), (int, float)) and not isinstance(row.get("score"), bool):
                by_length.setdefault(f"{int(found.group(1))}k", {}).setdefault(who, []).append(float(row["score"]))
        return {length: {who: round(sum(v) / len(v), 3) for who, v in per.items()} for length, per in by_length.items()} or None
