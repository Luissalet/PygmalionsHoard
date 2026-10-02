"""Stdio MCP bridge for Pygmalion's Hoard.

It never opens the database: every tool call is proxied to the running app (`POST /api/agent/call`) with the Bearer token from
`<DATA_DIR>/mcp-token`. The tool list is fetched from `GET /api/agent/tools` (refreshed while the bridge runs), so the bridge and the app can
never disagree. When nothing answers, the bridge starts the app itself (`python -m pygmalion_hoard`, detached, on the port of PYGMALION_URL) and
waits for it; PYGMALION_BRIDGE_AUTOSTART=0 turns that off. The bridge itself is the shared catalogue bridge of Hoard Link. The tools that wait
for a job (`wait_s`) wait at most the shared limit (150 s) and then say the job is still running; poll `job_get`.
"""

from __future__ import annotations

import sys

from pygmalion_hoard.hoard_link.bridge import CatalogBridge


def main() -> int:
    CatalogBridge(app="pygmalion", service="pygmalion-hoard", package="pygmalion_hoard", default_port=5202, data_dir_env="PYGMALION_DATA_DIR",
                  title="Pygmalion's Hoard", root=__file__, default_timeout=90.0).run_bridge()
    return 0


if __name__ == "__main__":
    sys.exit(main())
