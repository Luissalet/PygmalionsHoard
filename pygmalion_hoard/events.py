"""Events for the family bus, sent by one thread that the app owns and stops.

``family.emit`` starts a new daemon thread for every event. Threads like that are still running when the interpreter shuts down (a job
that finishes in the last second of a run, a test that ends right after a job) and can be in the middle of an HTTP call or of a write to
a stream while the interpreter closes them: the process then dies with a fatal error after the work was done. Here the events go
through a queue to a single worker; ``close`` drops what is still waiting, lets the one in flight end (it has a short timeout) and joins the
worker. Events are hints: after ``close`` they are dropped, the database is the truth.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable

log = logging.getLogger("pygmalion.events")

_STOP = object()
MAX_WAITING = 200


class EventPump:
    """A queue and one worker thread, started on the first event and joined by :meth:`close`."""

    def __init__(self, send: Callable[[str, dict[str, Any]], Any], *, wanted: Callable[[], bool] = lambda: True, name: str = "pygmalion-events"):
        self._send, self._wanted, self._name = send, wanted, name
        self._queue: queue.Queue = queue.Queue(MAX_WAITING)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False
        self.sent = 0
        self.dropped = 0

    def emit(self, type_: str, data: dict[str, Any]) -> bool:
        """Queue an event. ``False`` when it was dropped (closed, nobody listening, queue full)."""
        if not self._wanted():
            return False
        with self._lock:
            if self._closed:
                self.dropped += 1
                return False
            try:
                self._queue.put_nowait((type_, data))
            except queue.Full:
                self.dropped += 1
                return False
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
                self._thread.start()
        return True

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            try:
                self._send(*item)
                self.sent += 1
            except Exception:  # noqa: BLE001 — an event that cannot be sent is dropped
                self.dropped += 1

    def alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def close(self, timeout: float = 5.0) -> bool:
        """Stop for good: drop what still waits, wait for the event in flight. ``True`` when the worker has ended."""
        with self._lock:
            self._closed = True
            thread = self._thread
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is not _STOP:
                    self.dropped += 1
            if thread is not None:
                self._queue.put_nowait(_STOP)
        if thread is None:
            return True
        if thread is not threading.current_thread():
            thread.join(timeout)
        if thread.is_alive():
            log.warning("the event worker did not stop within %.1f s", timeout)
            return False
        return True
