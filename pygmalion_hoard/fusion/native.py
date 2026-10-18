"""Native training/evaluation engines, owned by one process and independent lifetimes."""
from contextlib import asynccontextmanager
import time
from dataclasses import replace
from fastapi import FastAPI
from pygmalion_hoard.hoard_link.guard import check_request
from pygmalion_hoard.hoard_link.copilot import install_copilot

SPECS = [
 {"id": "training", "name": "Entrenamiento", "origin": "Pygmalion", "description": "Datasets, LoRA, fusiones, cuantización, contexto, captura, linaje y publicación."},
 {"id": "evaluation", "name": "Evaluación", "origin": "Galton", "description": "Pruebas, ejecuciones, estadísticas, arena, arneses, usuarios simulados y rutas."},
 {"id": "automation", "name": "Evaluaciones automáticas", "origin": "Galton Watch", "description": "Permite refrescos y pruebas periódicas cuando Evaluación está activada. Desactivado por defecto."},
]

def configure_modules(modules, port):
    # Bound by the app factory, never by discovery or package import.
    for module in modules.values():
        module.factory = lambda key, path, modules=modules: factory(key, path, modules=modules, port=port)

class NativeGalton:
    def __init__(self, module):
        self.module = module
        self.clock = time.time
        self.sleep = time.sleep
    def call(self, name, arguments):
        return self.module.native_call(name, arguments)
    def available(self):
        return {"ok": self.module.app is not None and not self.module.stopping and not self.module.cleanup_pending, "via": "native", "detail": ""}

def factory(key, path, *, modules, port):
    if key == "automation":
        return FastAPI()
    if key == "training":
        from pygmalion_hoard.config import Config
        from pygmalion_hoard.main import create_app
    else:
        from galton_hoard.config import Config, load_dotenv
        from galton_hoard.main import create_app
    config = replace(Config.from_env(), data_dir=path, port=port, data_dir_configured=True)
    if key == "training":
        config.defer_existing_jobs = True
    else:
        config.secrets.update(load_dotenv(path.parent / ".env"))
        config.resume_pending = False
        config.automatic_work = lambda: modules["automation"].app is not None
    config.url_prefix = "/" + key
    app = create_app(config)
    original = app.router.lifespan_context
    @asynccontextmanager
    async def lifespan(instance):
        async with original(instance):
            svc = instance.state.services
            if key == "training":
                svc.galton = NativeGalton(modules["evaluation"])
                svc.evaluator.galton = svc.galton
            yield
    app.router.lifespan_context = lifespan
    return app
