"""REST for the evaluation of suits (the same handlers as the ``pygmalion_suit_eval_*`` tools)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ..messages import wire
from .deps import tool

router = APIRouter()


@router.get("/api/suit-evals")
def evals(request: Request, suit: str = ""):
    return wire(tool(request, "pygmalion_suit_eval_list", {"suit": suit}))


@router.post("/api/suit-evals")
def define(request: Request, body: dict[str, Any]):
    return wire(tool(request, "pygmalion_suit_eval_define", body))


@router.get("/api/suit-evals/{eval_id}")
def get(request: Request, eval_id: str):
    return wire(tool(request, "pygmalion_suit_eval_get", {"id": eval_id}))


@router.get("/api/suit-evals/{eval_id}/score")
def score(request: Request, eval_id: str):
    return wire(tool(request, "pygmalion_suit_eval_score", {"id": eval_id}))


@router.post("/api/suit-evals/{eval_id}/records")
def record(request: Request, eval_id: str, body: dict[str, Any]):
    return wire(tool(request, "pygmalion_suit_eval_record", {**body, "id": eval_id}))


@router.post("/api/suit-evals/{eval_id}/export")
def export(request: Request, eval_id: str, body: dict[str, Any] | None = None):
    return wire(tool(request, "pygmalion_suit_eval_export", {**(body or {}), "id": eval_id}))


@router.delete("/api/suit-evals/{eval_id}")
def delete(request: Request, eval_id: str, confirm: bool = False):
    return wire(tool(request, "pygmalion_suit_eval_delete", {"id": eval_id, "confirm": confirm}))
