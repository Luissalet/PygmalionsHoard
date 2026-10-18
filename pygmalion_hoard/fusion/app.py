"""Pygmalion: independent training, evaluation and automatic evaluation controls."""
from pathlib import Path
from pygmalion_hoard.config import Config
from .native import SPECS
from .runtime import create_runtime

def create_app(port=None, *, config=None, data_dir=None):
    config = config or Config.from_env()
    return create_runtime(name="Pygmalion’s Hoard", product="pygmalion", root=Path(__file__).parent,
        data_dir=Path(data_dir or config.data_dir), specs=SPECS, port=port or config.port,
        factory=lambda *args: None)
