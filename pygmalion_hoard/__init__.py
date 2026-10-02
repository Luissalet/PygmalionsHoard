"""Pygmalion's Hoard: a local studio to adapt language models. It builds datasets from your own material, fine-tunes with LoRA and
QLoRA, merges adapters and models, quantizes to GGUF with an importance matrix, extends the context window, measures every result
against its parent and publishes the good ones to Ollama and llama.cpp."""

__version__ = "0.1.0"
SERVICE = "pygmalion-hoard"
APP_ID = "pygmalion"
