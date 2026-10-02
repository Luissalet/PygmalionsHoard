"""Process-level configuration read from the environment (never from the DB)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .hoard_link.appconfig import AppPaths, env_flag, env_float, env_int, env_str, load_dotenv as _read_dotenv
from .hoard_link.guard import parse_allowed_hosts

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT = 5202
PREFIX = "PYGMALION_"


def load_dotenv(path: Path) -> dict[str, str]:
    """The pairs of a ``.env`` file (the shared reader). Unlike the shared reader's default they are **not** left in ``os.environ``: these are secrets
    for this process and the trainer and llama.cpp children it starts would inherit them. A variable the real environment already has stays."""
    before = set(os.environ)
    values = _read_dotenv(path)
    for key in values:
        if key not in before:
            os.environ.pop(key, None)
    return values


@dataclass
class Config:
    """Everything the process needs before the database exists."""

    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")
    port: int = DEFAULT_PORT
    port_strict: bool = False
    allowed_hosts: tuple[str, ...] = ()
    data_dir_configured: bool = False
    http_timeout_s: float = 25.0
    offline: bool = False  # never touch the network (tests): Hugging Face, the hub and Galton answer "offline"
    scheduler: bool = True  # background job lanes and the housekeeping tick; tests and the MCP-only mode switch it off
    workers_dir: Optional[Path] = None  # replaces pygmalion_hoard/workers (tests point it at fake workers)
    secrets: dict[str, str] = field(default_factory=dict)  # from .env (+ environment); never written back

    @property
    def paths(self) -> AppPaths:
        """The shared data-folder layout (``pygmalion.db``, ``mcp-token``, ``url``, ``logs/``)."""
        return AppPaths("pygmalion", REPO_ROOT, self.data_dir, self.data_dir_configured)

    @property
    def db_path(self) -> Path:
        return self.paths.db_path

    @property
    def token_path(self) -> Path:
        return self.paths.token_path

    @property
    def url_path(self) -> Path:
        return self.paths.url_path

    @property
    def logs_dir(self) -> Path:
        return self.paths.logs_dir

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def secrets_path(self) -> Path:
        """Write-only secrets (the Hugging Face token) saved from Settings: KEY=VALUE lines, owner-only permissions."""
        return self.data_dir / "secrets.env"

    @property
    def default_work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def package_workers_dir(self) -> Path:
        return Path(__file__).resolve().parent / "workers"

    def effective_workers_dir(self) -> Path:
        return self.workers_dir or self.package_workers_dir

    def secret(self, name: str) -> str:
        """Environment first, then .env, then data/secrets.env. ``name`` without the PYGMALION_ prefix (e.g. HF_TOKEN)."""
        key = f"{PREFIX}{name}"
        value = os.environ.get(key) or self.secrets.get(key)
        if value:
            return value.strip()
        return load_dotenv(self.secrets_path).get(key, "").strip()

    @classmethod
    def from_env(cls) -> "Config":
        raw_dir = env_str("PYGMALION_DATA_DIR") or ""
        raw_workers = env_str("PYGMALION_WORKERS_DIR") or ""
        port = env_int("PYGMALION_PORT", "PORT", default=DEFAULT_PORT)
        if not 1 <= port <= 65535:
            port = DEFAULT_PORT
        timeout = env_float("PYGMALION_HTTP_TIMEOUT_S", default=25.0)
        if not 2.0 <= timeout <= 120.0:
            timeout = 25.0
        return cls(
            data_dir=Path(raw_dir).expanduser() if raw_dir else REPO_ROOT / "data",
            port=port,
            port_strict=env_flag("PORT_STRICT"),
            allowed_hosts=parse_allowed_hosts(env_str("PYGMALION_ALLOWED_HOSTS")),
            data_dir_configured=bool(raw_dir),
            http_timeout_s=timeout,
            offline=env_flag("PYGMALION_OFFLINE"),
            scheduler=env_flag("PYGMALION_SCHEDULER", True),
            workers_dir=Path(raw_workers).expanduser() if raw_workers else None,
            secrets=load_dotenv(REPO_ROOT / ".env"),
        )
