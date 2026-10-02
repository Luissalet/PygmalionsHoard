"""`python -m pygmalion_hoard` — run the app with uvicorn on 127.0.0.1 (the shared Hoard Link launcher)."""

from __future__ import annotations

from .hoard_link.service import run_main


def main() -> int:
    return run_main(service="pygmalion-hoard", package="pygmalion_hoard", default_port=5202, app_factory="pygmalion_hoard.main:create_app",
                    data_dir_env="PYGMALION_DATA_DIR", port_env="PYGMALION_PORT", open_browser_default=False, title="Pygmalion's Hoard")


if __name__ == "__main__":
    raise SystemExit(main())
