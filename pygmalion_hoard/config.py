"""Process-level configuration read from the environment (never from the DB)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .guard import parse_allowed_hosts

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT = 5202
PREFIX = "PYGMALION_"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _int(raw: str, default: int, low: int, high: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if low <= value <= high else default


def _float(raw: str, default: float, low: float, high: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if low <= value <= high else default


def load_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env reader (KEY=VALUE, # comments, optional quotes). Never overrides the real environment."""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
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
    def db_path(self) -> Path:
        return self.data_dir / "pygmalion.db"

    @property
    def token_path(self) -> Path:
        return self.data_dir / "mcp-token"

    @property
    def url_path(self) -> Path:
        return self.data_dir / "url"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

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
        raw_dir = _env("PYGMALION_DATA_DIR")
        raw_workers = _env("PYGMALION_WORKERS_DIR")
        port = _int(_env("PYGMALION_PORT") or _env("PORT") or str(DEFAULT_PORT), DEFAULT_PORT, 1, 65535)
        return cls(
            data_dir=Path(raw_dir).expanduser() if raw_dir else REPO_ROOT / "data",
            port=port,
            port_strict=_env("PORT_STRICT") == "1",
            allowed_hosts=parse_allowed_hosts(_env("PYGMALION_ALLOWED_HOSTS")),
            data_dir_configured=bool(raw_dir),
            http_timeout_s=_float(_env("PYGMALION_HTTP_TIMEOUT_S"), 25.0, 2.0, 120.0),
            offline=_env("PYGMALION_OFFLINE") == "1",
            scheduler=_env("PYGMALION_SCHEDULER", "1") != "0",
            workers_dir=Path(raw_workers).expanduser() if raw_workers else None,
            secrets=load_dotenv(REPO_ROOT / ".env"),
        )
