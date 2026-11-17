"""A site folder served on 127.0.0.1 at a free port for as long as a run needs it: the reference sites of the bench and the sites the models write."""
from __future__ import annotations

import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:  # noqa: D401 - no access log on stderr
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class Site:
    """``with Site(folder) as base_url: ...`` serves ``folder`` until the block ends."""

    def __init__(self, folder: Path, name: str = "site"):
        self.folder, self.name = Path(folder), name
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self) -> str:
        handler = functools.partial(_Quiet, directory=str(self.folder))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.2}, name=f"webrebuild-serve-{self.name}", daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __exit__(self, *exc) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(5)
        self.server, self.thread = None, None
