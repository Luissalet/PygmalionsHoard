"""SQLite connection (WAL) and ordered schema migrations."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

MIGRATIONS: list[str] = [
    # 1: settings, datasets and their immutable versions, artifacts (lineage), jobs, runs
    """
    CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE datasets (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      description TEXT NOT NULL DEFAULT '',
      kind TEXT NOT NULL DEFAULT 'chat',
      notes TEXT NOT NULL DEFAULT '',
      created_ts REAL NOT NULL,
      updated_ts REAL NOT NULL
    );
    CREATE UNIQUE INDEX datasets_name ON datasets(name);
    CREATE TABLE dataset_versions (
      id TEXT PRIMARY KEY,
      dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
      n INTEGER NOT NULL,
      path TEXT NOT NULL,
      sha256 TEXT NOT NULL,
      kind TEXT NOT NULL DEFAULT 'chat',
      records INTEGER NOT NULL DEFAULT 0,
      usable INTEGER NOT NULL DEFAULT 0,
      chars INTEGER NOT NULL DEFAULT 0,
      tokens INTEGER NOT NULL DEFAULT 0,
      splits TEXT NOT NULL DEFAULT '{}',
      recipe TEXT NOT NULL DEFAULT '{}',
      stats TEXT NOT NULL DEFAULT '{}',
      parent TEXT,
      created_ts REAL NOT NULL,
      UNIQUE (dataset_id, n)
    );
    CREATE TABLE artifacts (
      id TEXT PRIMARY KEY,
      kind TEXT NOT NULL,
      name TEXT NOT NULL,
      path TEXT NOT NULL DEFAULT '',
      size INTEGER NOT NULL DEFAULT 0,
      parents TEXT NOT NULL DEFAULT '[]',
      dataset_version TEXT,
      job_id TEXT,
      recipe TEXT NOT NULL DEFAULT '{}',
      metrics TEXT NOT NULL DEFAULT '{}',
      published TEXT NOT NULL DEFAULT '[]',
      notes TEXT NOT NULL DEFAULT '',
      pinned INTEGER NOT NULL DEFAULT 0,
      created_ts REAL NOT NULL,
      updated_ts REAL NOT NULL
    );
    CREATE INDEX artifacts_kind ON artifacts(kind, created_ts);
    CREATE TABLE jobs (
      id TEXT PRIMARY KEY,
      kind TEXT NOT NULL,
      title TEXT NOT NULL DEFAULT '',
      state TEXT NOT NULL DEFAULT 'queued',
      lane TEXT NOT NULL DEFAULT 'cpu',
      params TEXT NOT NULL DEFAULT '{}',
      then_steps TEXT NOT NULL DEFAULT '[]',
      pipeline_id TEXT,
      step INTEGER NOT NULL DEFAULT 0,
      created_ts REAL NOT NULL,
      started_ts REAL,
      finished_ts REAL,
      progress TEXT NOT NULL DEFAULT '{}',
      result TEXT NOT NULL DEFAULT '{}',
      error TEXT NOT NULL DEFAULT '',
      hint TEXT NOT NULL DEFAULT '',
      job_dir TEXT NOT NULL DEFAULT '',
      vram TEXT NOT NULL DEFAULT '{}',
      gpus TEXT NOT NULL DEFAULT '[]',
      queue_position INTEGER,
      attempts INTEGER NOT NULL DEFAULT 0,
      cancel_requested INTEGER NOT NULL DEFAULT 0,
      out_artifact TEXT
    );
    CREATE INDEX jobs_state ON jobs(state, lane, created_ts);
    CREATE INDEX jobs_pipeline ON jobs(pipeline_id, step);
    CREATE TABLE runs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts REAL NOT NULL,
      kind TEXT NOT NULL,
      ref TEXT NOT NULL DEFAULT '',
      ok INTEGER NOT NULL DEFAULT 1,
      duration_ms INTEGER NOT NULL DEFAULT 0,
      detail TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX runs_ts ON runs(ts);
    """,
]


class Database:
    """One connection shared by every thread, guarded by a re-entrant lock.

    The app is the only writer; the MCP bridge never opens this file.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.migrate()

    def migrate(self) -> None:
        with self.lock:
            self.conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = self.conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            current = row["v"] or 0
            for index, sql in enumerate(MIGRATIONS, start=1):
                if index <= current:
                    continue
                script = f"BEGIN;\n{sql}\nINSERT INTO schema_version(version) VALUES ({index});\nCOMMIT;"
                try:
                    self.conn.executescript(script)
                except Exception:
                    if self.conn.in_transaction:
                        self.conn.execute("ROLLBACK")
                    raise

    def version(self) -> int:
        row = self.one("SELECT MAX(version) AS v FROM schema_version")
        return int(row["v"] or 0)

    def query(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple | list = ()) -> sqlite3.Row | None:
        with self.lock:
            return self.conn.execute(sql, params).fetchone()

    def execute(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        with self.lock:
            return self.conn.execute(sql, params)

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        row = self.one("SELECT value FROM settings WHERE key = ?", (key,))
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        self.execute("INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    def transaction(self):
        """`with db.transaction():` — BEGIN IMMEDIATE / COMMIT (ROLLBACK on error) under the lock."""
        return _Transaction(self)

    def close(self) -> None:
        with self.lock:
            try:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            self.conn.close()


class _Transaction:
    def __init__(self, db: Database):
        self.db = db

    def __enter__(self):
        self.db.lock.acquire()
        self.db.conn.execute("BEGIN IMMEDIATE")
        return self.db.conn

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.db.conn.execute("COMMIT")
            else:
                self.db.conn.execute("ROLLBACK")
        finally:
            self.db.lock.release()
        return False
