"""One error type for every expected failure, so the API, the agent tools and the UI report it the same way."""

from __future__ import annotations

from typing import Any

from .hoard_link.agentkit import AppError
from .messages import ERRORS, KEY, CodedText, fields_of, fill, hint_text, recognise, scalar


class PygmalionError(AppError):
    """An expected, explainable failure: a stable ``code``, a human ``message`` and an actionable ``hint``.

    An :class:`~hoard_link.agentkit.AppError`: the shared error handlers and the agent router answer it with its own status and body (this app's
    handler adds the message parts the bundled interface translates)."""

    STATUS = {
        "not_found": 404,
        "confirm_required": 400,
        "invalid": 400,
        "not_configured": 400,
        "unsupported": 415,
        "too_large": 413,
        "forbidden": 403,
        "conflict": 409,
        "offline": 503,
        "env_missing": 503,
        "gpu_unavailable": 503,
        "galton_unavailable": 503,
        "tool_missing": 503,
        "failed": 500,
    }

    def __init__(self, code: str, message: str, hint: str = "", /, *, http_status: int | None = None, **details: Any):
        """``PygmalionError("invalid", "no_dataset", ref=x)`` takes the message and hint of the catalogue entry ``no_dataset``
        (``messages.ERRORS``) and fills them with the keyword arguments; the UI translates them from the ``key`` and ``params``.
        ``PygmalionError(code, "free text", hint)`` is for a message that cannot be known in advance (a text a library gave us); it has no key."""
        self.key = ""
        if KEY.match(message) and message in ERRORS:
            self.key = message
            template, hint_template = ERRORS[message]
            used = fields_of(template) | fields_of(hint_template)
            self.params = {k: scalar(v) for k, v in details.items() if k in used}
            details = {k: v for k, v in details.items() if k not in used}        # what is left travels with the error as it is (lists, ids, a path)
            message, hint = fill(template, self.params), fill(hint_template, self.params)
        elif KEY.match(message) and " " not in message and "_" in message:
            raise KeyError(f"{message!r} is not in messages.ERRORS")
        else:
            self.params = {}
        super().__init__(code, message, hint=hint, status=http_status, details=details)

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {**self.details, "error": self.message, "code": self.code}
        if self.hint:
            body["hint"] = self.coded_hint()
        if self.key:
            body["key"] = self.key
            body["params"] = self.params
        return body

    def coded_hint(self) -> str:
        """The hint as a ``CodedText`` (a ``str``) when it is the catalogue's hint or another sentence of the catalogue, else as it is."""
        if self.key and self.hint == hint_text(self.key, **self.params):
            return hint_text(self.key, **self.params)
        return recognise(self.hint)

    def coded(self) -> CodedText:
        """The message as a ``CodedText``: a ``str`` that keeps its key and parameters, for plan lines and warnings the UI translates."""
        return CodedText(self.message, self.key, self.params)
