"""API routers: one per area and the agent contract (health, PWA and the SPA come from Hoard Link in main.py)."""

from .agent import router as agent_router
from .health import router as status_router
from .ui import router as ui_router

ROUTERS = [status_router, ui_router, agent_router]
