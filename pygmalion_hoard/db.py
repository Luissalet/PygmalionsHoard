"""Ordered schema migrations; the SQLite connection itself (WAL, busy timeout, re-entrant transactions, foreign keys) is the shared ``sqlkit.Database``."""

from __future__ import annotations

import functools

from .hoard_link import sqlkit

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


Database = functools.partial(sqlkit.Database, migrations=MIGRATIONS, foreign_keys=True)
