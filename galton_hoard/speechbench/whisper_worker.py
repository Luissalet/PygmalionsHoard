"""Runs in a Python that has faster-whisper (not imported by Galton): one warm model for the whole corpus.

stdin: {size, device, compute, language, models_dir, beam_size, files: [{id, audio}]}. stdout: JSON lines
{"event": "loaded", "load_s"}, then per file {"event": "file", "id", "text", "first_text_s", "decode_s", "language"} or
{"event": "error", "id", "error"}, and {"event": "done"}. ``first_text_s`` is the time from the start of decoding to
the first segment the decoder yields (its first window), the closest a file transcription gets to a first partial."""
import json
import sys
import time


def emit(**event):
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    spec = json.loads(sys.stdin.readline() or "{}")
    from faster_whisper import WhisperModel

    started = time.perf_counter()
    model = WhisperModel(spec.get("size") or "small", device=spec.get("device") or "cpu",
                         compute_type=spec.get("compute") or "int8", download_root=spec.get("models_dir") or None)
    emit(event="loaded", load_s=round(time.perf_counter() - started, 3))
    for item in spec.get("files") or []:
        try:
            t0 = time.perf_counter()
            segments, info = model.transcribe(item["audio"], language=spec.get("language") or None,
                                              beam_size=int(spec.get("beam_size") or 5), vad_filter=False)
            first, parts = None, []
            for segment in segments:
                if first is None:
                    first = time.perf_counter() - t0
                parts.append(segment.text.strip())
            emit(event="file", id=item["id"], text=" ".join(p for p in parts if p), language=info.language,
                 first_text_s=round(first, 3) if first is not None else None, decode_s=round(time.perf_counter() - t0, 3))
        except Exception as error:  # noqa: BLE001 - one bad file must not end the run
            emit(event="error", id=item.get("id"), error=f"{type(error).__name__}: {error}")
    emit(event="done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
