"""The capture proxy's routes (``/capture/...``, a chat-completions endpoint for any harness) and thin REST routes over the capture tools.

The proxy answers in the format of the API it stands in for: its errors are ``{"error": {"message", "type", "code"}}``, not this app's envelope.
The session and the harness come from the URL (``/capture/<session>/v1`` or ``/capture/<harness>/<session>/v1``), from the ``X-Capture-Session`` and
``X-Capture-Harness`` headers, or from the API key (``session`` or ``harness:session``); without any of them the session is ``default``.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..capture.proxy import ProxyError, Reply, identify
from .deps import services, tool

router = APIRouter()


def _send(reply: Reply):
    if reply.stream and reply.status == 200:
        return StreamingResponse(reply.events(), media_type="text/event-stream", headers={**reply.headers, "Cache-Control": "no-cache"})
    return JSONResponse(reply.body, status_code=reply.status, headers=reply.headers)


def _error(exc: ProxyError):
    return JSONResponse(exc.body(), status_code=exc.status)


async def _chat(request: Request, harness: str, session: str):
    try:
        body = await request.json()
    except ValueError:
        return _error(ProxyError(400, "the body is not JSON", "invalid_request_error"))
    who_harness, who_session = identify(harness, session, request.headers)
    svc = services(request)
    try:
        reply = await run_in_threadpool(svc.capture_proxy.chat, who_harness, who_session, body)
    except ProxyError as exc:
        return _error(exc)
    return _send(reply)


async def _models(request: Request):
    try:
        return _send(await run_in_threadpool(services(request).capture_proxy.models))
    except ProxyError as exc:
        return _error(exc)


@router.post("/capture/v1/chat/completions")
async def chat_default(request: Request):
    return await _chat(request, "", "")


@router.post("/capture/{session}/v1/chat/completions")
async def chat_session(request: Request, session: str):
    return await _chat(request, "", session)


@router.post("/capture/{harness}/{session}/v1/chat/completions")
async def chat_harness(request: Request, harness: str, session: str):
    return await _chat(request, harness, session)


@router.get("/capture/v1/models")
async def models_default(request: Request):
    return await _models(request)


@router.get("/capture/{session}/v1/models")
async def models_session(request: Request, session: str):
    return await _models(request)


@router.get("/capture/{harness}/{session}/v1/models")
async def models_harness(request: Request, harness: str, session: str):
    return await _models(request)


# ------------------------------------------------------------------------------------------------- REST over the tools
class ProxyBody(BaseModel):
    action: str = Field("status", description="status | check | start | stop | configure")
    upstream_url: Optional[str] = None
    upstream_type: Optional[str] = None
    top_logprobs: Optional[int] = None
    faithful_sampling: Optional[bool] = None


class OutcomeBody(BaseModel):
    reward: Optional[float] = None
    outcome: Optional[str] = None
    note: str = ""
    node: str = ""
    clear: bool = False


class ExportBody(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


rest = APIRouter(prefix="/api/capture")


@rest.get("")
def capture_status(request: Request, check: bool = False):
    return tool(request, "capture_proxy", {"action": "check" if check else "status"})


@rest.post("/proxy")
def capture_proxy(request: Request, body: ProxyBody):
    return tool(request, "capture_proxy", body.model_dump(exclude_none=True))


@rest.get("/sessions")
def capture_sessions(request: Request, harness: str = "", since: str = "", until: str = "", limit: int = 50, offset: int = 0):
    return tool(request, "capture_sessions", {"harness": harness, "since": since or None, "until": until or None, "limit": limit, "offset": offset})


@rest.get("/sessions/{session}")
def capture_session(request: Request, session: str, node: str = "", max_tokens: int = 200):
    return tool(request, "capture_session", {"session": session, "node": node, "max_tokens": max_tokens})


@rest.post("/sessions/{session}/outcome")
def capture_outcome(request: Request, session: str, body: OutcomeBody):
    return tool(request, "capture_outcome", {"session": session, **body.model_dump(exclude_none=True)})


@rest.post("/export")
def capture_export(request: Request, body: ExportBody):
    return tool(request, "capture_export", body.arguments)


@rest.delete("/sessions/{session}")
def capture_delete(request: Request, session: str, confirm: bool = False):
    return tool(request, "capture_delete", {"sessions": [session], "confirm": confirm})
