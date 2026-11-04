"""A speech corpus: audio files with an independent reference transcript each.

Either a folder where every audio file has its reference next to it with the same name (``clip01.wav`` + ``clip01.txt``),
or a ``corpus.jsonl`` in the folder with one ``{"audio": path, "text": reference, "id"?: name}`` per line (paths relative to
the folder). Subfolders are walked. Files without a reference are listed as problems, never scored."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any, Callable, Optional

AUDIO = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".webm", ".mp4"}
MAX_ITEMS = 2000


def _wav_seconds(path: Path) -> Optional[float]:
    try:
        with wave.open(str(path), "rb") as fh:
            return round(fh.getnframes() / float(fh.getframerate()), 3)
    except (wave.Error, OSError, EOFError, ZeroDivisionError):
        return None


def _probe_seconds(path: Path) -> Optional[float]:
    try:
        from ..hoard_link.media.ffmpeg import FFmpeg

        return round(float(FFmpeg().duration(path)), 3)
    except Exception:  # noqa: BLE001 - no ffprobe or an unreadable file: duration unknown, RTF not computed
        return None


def duration_of(path: Path) -> Optional[float]:
    return _wav_seconds(path) if path.suffix.lower() == ".wav" else _probe_seconds(path)


def load(folder: str | Path, *, limit: int = MAX_ITEMS, duration: Callable[[Path], Optional[float]] = duration_of) -> dict[str, Any]:
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ValueError(f"Not a folder: {root}")
    items: list[dict[str, Any]] = []
    problems: list[dict[str, str]] = []
    manifest = root / "corpus.jsonl"
    if manifest.is_file():
        for n, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                audio = (root / str(row["audio"])).resolve()
                text = str(row["text"])
            except (ValueError, KeyError, TypeError) as error:
                problems.append({"file": f"corpus.jsonl:{n}", "problem": f"bad line: {error}"})
                continue
            if not audio.is_file():
                problems.append({"file": str(row.get("audio")), "problem": "audio file not found"})
                continue
            items.append({"id": str(row.get("id") or audio.stem), "audio": str(audio), "reference": text})
    else:
        for audio in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO):
            ref = audio.with_suffix(".txt")
            if not ref.is_file():
                problems.append({"file": str(audio.relative_to(root)), "problem": "no reference (.txt with the same name)"})
                continue
            text = ref.read_text(encoding="utf-8-sig").strip()
            if not text:
                problems.append({"file": str(audio.relative_to(root)), "problem": "empty reference"})
                continue
            items.append({"id": str(audio.relative_to(root).with_suffix("")).replace("\\", "/"), "audio": str(audio), "reference": text})
    items = items[:limit]
    for item in items:
        item["duration_s"] = duration(Path(item["audio"]))
    total = sum(i["duration_s"] or 0 for i in items)
    return {"folder": str(root), "items": items, "problems": problems, "files": len(items),
            "audio_s": round(total, 2), "unknown_duration": sum(1 for i in items if i["duration_s"] is None),
            "source": "corpus.jsonl" if manifest.is_file() else "pairs"}
