"""Executable acceptance checks for the deliverable of one case.

Kinds: ``file_exists``, ``json_schema``, ``numeric``, ``regex`` and ``judge``. The first four read the deliverable folder (or the
answer text, or numbers handed in with the record) and give ``ok`` true or false with the reason. ``judge`` is a rubric: its verdict
(from a judge model, the logit judge or a person) is handed in with the record; until then it is ``pending`` (``ok`` is ``None``) and the
case is not counted as passed or failed.

Everything read here was written by a model or a person: files are read, never executed; paths cannot leave the deliverable folder (no
absolute path, no drive letter, no ``..``, symbolic links resolved); sizes are capped. The JSON-schema subset is checked by a small
validator that refuses a keyword it does not understand instead of silently passing it."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from ..errors import GaltonError

KINDS = ("file_exists", "json_schema", "numeric", "regex", "judge")
MAX_BYTES = 5_000_000
MAX_TEXT = 1_000_000
_ANNOTATIONS = {"$schema", "$id", "title", "description", "default", "examples", "$comment"}
_TYPES = {"string": str, "boolean": bool, "array": list, "object": dict, "null": type(None)}


# ------------------------------------------------------------------ paths
def safe_path(root: Optional[Path], rel: str) -> Path:
    """``rel`` inside ``root``, or an error. Accepts both separators; refuses absolute paths, drive letters and ``..``."""
    text = str(rel or "").strip().replace("\\", "/")
    bad = not text or text.startswith(("/", "~")) or bool(re.match(r"^[A-Za-z]:", text)) or any(part in ("..", "") for part in text.split("/"))
    if bad or root is None:
        raise GaltonError("invalid", "se_bad_path", path=text or "(empty)")
    base = Path(root).resolve()
    target = (base / PurePosixPath(text)).resolve()
    if base != target and base not in target.parents:
        raise GaltonError("invalid", "se_bad_path", path=text)
    return target


def _read_text(path: Path, limit: int = MAX_TEXT) -> str:
    if path.stat().st_size > limit:
        raise ValueError(f"larger than {limit} bytes")
    return path.read_bytes().decode("utf-8", errors="replace")


# ------------------------------------------------------------------ JSON schema subset
def validate_schema(value: Any, schema: Any, path: str = "$") -> list[str]:
    """Errors of ``value`` against a JSON-schema subset (type, enum, const, required, properties, additionalProperties, items, minimum,
    maximum, minLength, maxLength, pattern, minItems, maxItems). An unknown keyword raises ``se_schema_keyword``."""
    if schema is True or schema == {}:
        return []
    if schema is False:
        return [f"{path}: nothing is allowed here"]
    if not isinstance(schema, dict):
        raise GaltonError("invalid", "se_schema_keyword", keyword="schema")
    errors: list[str] = []
    for key, rule in schema.items():
        if key in _ANNOTATIONS:
            continue
        if key == "type":
            names = rule if isinstance(rule, list) else [rule]
            if not any(_is_type(value, n) for n in names):
                errors.append(f"{path}: expected {'/'.join(map(str, names))}, found {_type_name(value)}")
        elif key == "enum":
            if value not in rule:
                errors.append(f"{path}: not one of the allowed values")
        elif key == "const":
            if value != rule:
                errors.append(f"{path}: not the expected constant")
        elif key == "required":
            if isinstance(value, dict):
                errors += [f"{path}: missing «{k}»" for k in rule if k not in value]
        elif key == "properties":
            if isinstance(value, dict):
                for k, sub in rule.items():
                    if k in value:
                        errors += validate_schema(value[k], sub, f"{path}.{k}")
        elif key == "additionalProperties":
            if isinstance(value, dict):
                extra = [k for k in value if k not in (schema.get("properties") or {})]
                if rule is False and extra:
                    errors.append(f"{path}: unexpected «{extra[0]}»")
                elif isinstance(rule, dict):
                    for k in extra:
                        errors += validate_schema(value[k], rule, f"{path}.{k}")
        elif key == "items":
            if isinstance(value, list):
                for i, item in enumerate(value):
                    errors += validate_schema(item, rule, f"{path}[{i}]")
        elif key in ("minimum", "maximum"):
            if _is_type(value, "number"):
                if (key == "minimum" and value < rule) or (key == "maximum" and value > rule):
                    errors.append(f"{path}: {value} violates {key} {rule}")
        elif key in ("minLength", "maxLength"):
            if isinstance(value, str) and ((key == "minLength" and len(value) < rule) or (key == "maxLength" and len(value) > rule)):
                errors.append(f"{path}: length {len(value)} violates {key} {rule}")
        elif key == "pattern":
            if isinstance(value, str) and not re.search(rule, value):
                errors.append(f"{path}: does not match the pattern")
        elif key in ("minItems", "maxItems"):
            if isinstance(value, list) and ((key == "minItems" and len(value) < rule) or (key == "maxItems" and len(value) > rule)):
                errors.append(f"{path}: {len(value)} items violates {key} {rule}")
        else:
            raise GaltonError("invalid", "se_schema_keyword", keyword=key)
    return errors


def _type_name(value: Any) -> str:
    for name in ("null", "boolean", "string", "array", "object"):
        if _is_type(value, name):
            return name
    return "integer" if isinstance(value, int) else "number"


def _is_type(value: Any, name: str) -> bool:
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool) or (isinstance(value, float) and value.is_integer())
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    t = _TYPES.get(name)
    if t is None:
        raise GaltonError("invalid", "se_schema_keyword", keyword=f"type {name}")
    if name == "boolean":
        return isinstance(value, bool)
    return isinstance(value, t)


# ------------------------------------------------------------------ definition checks
def normalize(checks: Any, case: str = "") -> list[dict[str, Any]]:
    """Validate the checks of a case at definition time (so a bad spec fails when written, not when scored)."""
    if not isinstance(checks, list) or not checks:
        raise GaltonError("invalid", "se_case_no_checks", case=case)
    out, seen = [], set()
    for i, raw in enumerate(checks):
        if not isinstance(raw, dict) or raw.get("kind") not in KINDS:
            raise GaltonError("invalid", "se_check_kind", case=case, n=i + 1, kinds=", ".join(KINDS))
        spec = dict(raw)
        spec["id"] = str(spec.get("id") or f"c{i + 1}")[:60]
        if spec["id"] in seen:
            raise GaltonError("invalid", "se_check_dup", case=case, id=spec["id"])
        seen.add(spec["id"])
        kind = spec["kind"]
        need = {"file_exists": ["path"], "json_schema": ["path", "schema"], "numeric": ["expected"], "regex": ["pattern"], "judge": ["rubric"]}[kind]
        miss = [k for k in need if spec.get(k) in (None, "")]
        if miss:
            raise GaltonError("invalid", "se_check_field", case=case, id=spec["id"], field=miss[0], kind=kind)
        if kind == "json_schema":
            validate_schema(None, spec["schema"])      # an unsupported keyword fails here, with the keyword named
        if kind == "numeric":
            if not _num(spec["expected"]) or not _num(spec.get("tolerance", 0)) or float(spec.get("tolerance", 0)) < 0:
                raise GaltonError("invalid", "se_check_field", case=case, id=spec["id"], field="expected/tolerance", kind=kind)
            if not spec.get("path") and not spec.get("value"):
                raise GaltonError("invalid", "se_check_field", case=case, id=spec["id"], field="path or value", kind=kind)
        if kind == "regex":
            if len(str(spec["pattern"])) > 500:
                raise GaltonError("invalid", "se_check_field", case=case, id=spec["id"], field="pattern", kind=kind)
            try:
                re.compile(spec["pattern"], _flags(spec))
            except re.error as exc:
                raise GaltonError("invalid", "se_check_regex", case=case, id=spec["id"], error=str(exc)) from exc
            if not spec.get("path") and spec.get("target", "answer") != "answer":
                raise GaltonError("invalid", "se_check_field", case=case, id=spec["id"], field="path or target", kind=kind)
        out.append(spec)
    return out


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _flags(spec: dict[str, Any]) -> int:
    return sum({"i": re.I, "m": re.M, "s": re.S}.get(c, 0) for c in str(spec.get("flags", "")))


# ------------------------------------------------------------------ running
def run_check(spec: dict[str, Any], *, root: Optional[Path], answer: str = "", values: Optional[dict[str, Any]] = None,
              judgements: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Result ``{id, kind, ok, detail}``; ``ok`` is ``None`` for a judge check without verdict. Never raises for a bad deliverable: that is a failed check."""
    kind, cid = spec["kind"], spec["id"]
    out = {"id": cid, "kind": kind, "ok": False, "detail": ""}
    try:
        if kind == "judge":
            verdict = (judgements or {}).get(cid)
            if not isinstance(verdict, dict) or not isinstance(verdict.get("ok"), bool):
                out.update(ok=None, detail="waiting for a verdict")
            else:
                out.update(ok=verdict["ok"], detail=str(verdict.get("note") or verdict.get("judge") or "")[:300], judge=str(verdict.get("judge") or "")[:80])
                if _num(verdict.get("score")):
                    out["score"] = float(verdict["score"])
            return out
        if kind == "file_exists":
            p = safe_path(root, spec["path"])
            if not p.is_file():
                out["detail"] = "file not found"
            elif p.stat().st_size < int(spec.get("min_bytes", 1 if spec.get("nonempty", True) else 0)):
                out["detail"] = f"file smaller than {int(spec.get('min_bytes', 1))} bytes"
            else:
                out.update(ok=True, detail=f"{p.stat().st_size} bytes")
        elif kind == "json_schema":
            p = safe_path(root, spec["path"])
            if not p.is_file():
                out["detail"] = "file not found"
            else:
                errors = validate_schema(json.loads(_read_text(p, MAX_BYTES)), spec["schema"])
                out.update(ok=not errors, detail="; ".join(errors[:3]) if errors else "valid")
        elif kind == "numeric":
            actual = _actual(spec, root, values)
            tol = float(spec.get("tolerance", 0))
            expected = float(spec["expected"])
            allowed = abs(expected) * tol if spec.get("relative") else tol
            out.update(ok=abs(actual - expected) <= allowed + 1e-12, detail=f"found {actual:g}, expected {expected:g} ± {allowed:g}")
        elif kind == "regex":
            text = answer if (not spec.get("path") and spec.get("target", "answer") == "answer") else _read_text(safe_path(root, spec["path"]))
            hits = len(re.findall(spec["pattern"], text[:MAX_TEXT], _flags(spec)))
            if spec.get("forbid"):
                out.update(ok=hits == 0, detail=f"{hits} forbidden match(es)")
            else:
                low, high = int(spec.get("min_matches", 1)), spec.get("max_matches")
                out.update(ok=hits >= low and (high is None or hits <= int(high)), detail=f"{hits} match(es)")
    except GaltonError as exc:
        out["detail"] = exc.message
    except (OSError, ValueError, KeyError, TypeError, re.error) as exc:
        out["detail"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return out


def _actual(spec: dict[str, Any], root: Optional[Path], values: Optional[dict[str, Any]]) -> float:
    if spec.get("value"):
        v = (values or {}).get(spec["value"])
        if not _num(v):
            raise ValueError(f"value «{spec['value']}» not given")
        return float(v)
    text = _read_text(safe_path(root, spec["path"]), MAX_BYTES)
    data: Any = json.loads(text) if str(spec["path"]).lower().endswith(".json") or spec.get("pointer") else float(text.strip())
    for part in [p for p in re.split(r"[./]", str(spec.get("pointer") or "")) if p != ""]:
        data = data[int(part)] if isinstance(data, list) else data[part]
    if not _num(data):
        raise ValueError("not a number")
    return float(data)


def run_checks(checks: list[dict[str, Any]], *, root: Optional[Path], answer: str = "", values: Optional[dict[str, Any]] = None,
               judgements: Optional[dict[str, Any]] = None) -> tuple[Optional[bool], list[dict[str, Any]]]:
    """All checks of a case and the success they imply: false if any failed, true if all passed, ``None`` while a judge verdict is missing."""
    results = [run_check(c, root=root, answer=answer, values=values, judgements=judgements) for c in checks]
    if any(r["ok"] is False for r in results):
        return False, results
    if any(r["ok"] is None for r in results):
        return None, results
    return True, results
