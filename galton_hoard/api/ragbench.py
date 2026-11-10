"""REST for the RAG pipeline bench (the same handlers as the ``rag_*`` tools)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ..messages import wire
from .deps import tool

router = APIRouter()


@router.get("/api/rag/sets")
def sets(request: Request):
    return wire(tool(request, "rag_sets", {}))


@router.post("/api/rag/sets")
def set_save(request: Request, body: dict[str, Any]):
    return wire(tool(request, "rag_set_save", body))


@router.get("/api/rag/sets/{set_id}")
def set_get(request: Request, set_id: str, status: str = "", with_text: bool = False):
    return wire(tool(request, "rag_set_get", {"set": set_id, "status": status, "with_text": with_text}))


@router.delete("/api/rag/sets/{set_id}")
def set_delete(request: Request, set_id: str, confirm: bool = False):
    return wire(tool(request, "rag_set_delete", {"set": set_id, "confirm": confirm}))


@router.post("/api/rag/sets/{set_id}/generate")
def generate(request: Request, set_id: str, body: dict[str, Any]):
    return wire(tool(request, "rag_cases_generate", {**body, "set": set_id}))


@router.post("/api/rag/sets/{set_id}/cases")
def case_save(request: Request, set_id: str, body: dict[str, Any]):
    return wire(tool(request, "rag_case_save", {**body, "set": set_id}))


@router.post("/api/rag/sets/{set_id}/review")
def review(request: Request, set_id: str, body: dict[str, Any]):
    return wire(tool(request, "rag_cases_review", {**body, "set": set_id}))


@router.delete("/api/rag/cases/{case_id}")
def case_remove(request: Request, case_id: str, confirm: bool = False):
    return wire(tool(request, "rag_case_remove", {"id": case_id, "confirm": confirm}))


@router.post("/api/rag/probe")
def probe(request: Request, body: dict[str, Any]):
    return wire(tool(request, "rag_probe", body))


@router.get("/api/rag/runs")
def runs(request: Request, set: str = "", limit: int = 30):
    return wire(tool(request, "rag_run_list", {"set": set, "limit": limit}))


@router.post("/api/rag/runs")
def run_start(request: Request, body: dict[str, Any]):
    return wire(tool(request, "rag_run_start", body))


@router.get("/api/rag/compare")
def compare(request: Request, a: str, b: str):
    return wire(tool(request, "rag_compare", {"a": a, "b": b}))


@router.get("/api/rag/runs/{run_id}")
def run_result(request: Request, run_id: str, case: str = "", only_failed: bool = False):
    return wire(tool(request, "rag_run_result", {"run": run_id, "case": case, "only_failed": only_failed}))


@router.get("/api/rag/runs/{run_id}/status")
def run_status(request: Request, run_id: str):
    return wire(tool(request, "rag_run_status", {"run": run_id}))


@router.post("/api/rag/runs/{run_id}/cancel")
def run_cancel(request: Request, run_id: str):
    return wire(tool(request, "rag_run_cancel", {"run": run_id}))


@router.delete("/api/rag/runs/{run_id}")
def run_delete(request: Request, run_id: str, confirm: bool = False):
    return wire(tool(request, "rag_run_delete", {"run": run_id, "confirm": confirm}))
