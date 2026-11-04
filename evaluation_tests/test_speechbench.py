"""Speech bench: WER/CER alignment, corpus loading, command engines, runs with faked engines through the tools and the UI route."""

import json
import sys
import time
import wave
from pathlib import Path

import pytest

from evaluation_tests.conftest import tool
from galton_hoard.speechbench import corpus, engines, metrics
from galton_hoard.speechbench.service import SpeechBench


def _wav(path: Path, seconds: float = 1.0) -> Path:
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16000)
        fh.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


def _corpus(folder: Path) -> Path:
    folder.mkdir()
    _wav(folder / "uno.wav", 2.0)
    (folder / "uno.txt").write_text("Hola, ¿qué tal estás?", encoding="utf-8")
    _wav(folder / "dos.wav", 1.0)
    (folder / "dos.txt").write_text("La canción empieza ya.", encoding="utf-8")
    _wav(folder / "sin_ref.wav", 0.5)
    return folder


def test_alignment_counts_and_rates():
    s = metrics.score("Hola, ¿qué tal estás?", "hola que tal estas hoy", accents=False)
    assert s["words"] == {"errors": 1, "substitutions": 0, "deletions": 0, "insertions": 1, "reference": 4} and s["wer"] == 0.25
    s = metrics.score("la canción empieza ya", "la cancion empieza")
    assert s["words"]["substitutions"] == 1 and s["words"]["deletions"] == 1 and s["wer"] == 0.5
    assert metrics.score("", "algo")["wer"] is None
    total = metrics.corpus_rates([{"score": metrics.score("a b c d", "a b c d")}, {"score": metrics.score("e f", "x")}])
    assert total["wer"] == round(2 / 6, 4) and total["reference_words"] == 6


def test_corpus_pairs_and_manifest(tmp_path):
    found = corpus.load(_corpus(tmp_path / "c"))
    assert [i["id"] for i in found["items"]] == ["dos", "uno"] and found["audio_s"] == 3.0
    assert found["problems"] == [{"file": "sin_ref.wav", "problem": "no reference (.txt with the same name)"}]
    (tmp_path / "c" / "corpus.jsonl").write_text(json.dumps({"audio": "uno.wav", "text": "hola"}) + "\n{bad\n", encoding="utf-8")
    manifest = corpus.load(tmp_path / "c")
    assert manifest["source"] == "corpus.jsonl" and manifest["files"] == 1 and manifest["problems"][0]["file"] == "corpus.jsonl:2"
    with pytest.raises(ValueError):
        corpus.load(tmp_path / "missing")


def test_command_engine_text_and_json(tmp_path):
    script = tmp_path / "fake_stt.py"
    script.write_text("import sys, json\nprint('noise')\nprint(json.dumps({'text': 'hola ' + sys.argv[2], 'first_text_s': 0.2}))\n", encoding="utf-8")
    commands: dict = {}
    engines.save_command(commands, "fake", [sys.executable, str(script), "{audio}", "{language}"], timeout_s=30)
    with pytest.raises(ValueError):
        engines.save_command(commands, "bad", [sys.executable, "x.py"])
    spec = engines.parse_engine("cmd:fake", commands)
    out = engines.run_command(spec, {"audio": "a.wav"}, language="es")
    assert out["text"] == "hola es" and out["first_text_s"] == 0.2 and out["decode_s"] >= 0
    assert engines.parse_engine("faster-whisper:tiny", {})["size"] == "tiny"
    with pytest.raises(ValueError):
        engines.parse_engine("faster-whisper:huge", {})


def _fake_whisper(python, engine, items, *, language, device, compute, models_dir, on_event, cancel):
    on_event({"event": "loaded", "load_s": 1.5})
    texts = {"uno": "hola que tal estas", "dos": "la cancion empieza"}
    for item in items:
        on_event({"event": "file", "id": item["id"], "text": texts[item["id"]], "first_text_s": 0.3, "decode_s": 0.5})


def _wait(bench, run_id):
    for _ in range(200):
        out = bench.get(run_id)
        if out["run"]["state"] not in ("queued", "running"):
            return out
        time.sleep(0.02)
    raise AssertionError("the run did not finish")


def test_service_run_scores_and_times(tmp_path, monkeypatch):
    monkeypatch.setattr(engines, "find_whisper_python", lambda *a: sys.executable)
    bench = SpeechBench(tmp_path / "data", runner={"whisper": _fake_whisper, "command": engines.run_command})
    folder = _corpus(tmp_path / "c")
    plan = bench.start(str(folder), ["faster-whisper:small"], language="es", dry_run=True)
    assert plan["dry_run"] and plan["files"] == 2 and plan["engines"] == ["faster-whisper:small"]
    card = bench.start(str(folder), ["faster-whisper:small"], language="es", accents=False)
    done = _wait(bench, card["id"])
    s = done["run"]["summary"]["faster-whisper:small"]
    assert done["run"]["state"] == "done" and s["files_done"] == 2 and s["load_s"] == 1.5
    assert s["wer"] == round(1 / 8, 4) and s["deletions"] == 1 and s["rtf"] == round(1.0 / 3.0, 3) and s["first_text_s_median"] == 0.3
    items = bench.get(card["id"], items=True)["items"]
    assert {i["id"]: i["wer"] for i in items} == {"uno": 0.0, "dos": 0.25}
    assert bench.list()["runs"][0]["id"] == card["id"]
    with pytest.raises(LookupError):
        bench.get("nothere")
    with pytest.raises(ValueError):
        bench.start(str(folder), ["cmd:none"])


def test_tools_and_ui_route(svc, client, tmp_path, monkeypatch):
    monkeypatch.setattr(engines, "find_whisper_python", lambda *a: None)
    listed = tool(svc, "speechbench_engines")
    assert listed["available"] == [] and listed["whisper"]["note"]
    script = tmp_path / "echo_stt.py"
    script.write_text("import sys\nprint('hola que tal estas' if 'uno' in sys.argv[1] else 'la cancion empieza ya')\n", encoding="utf-8")
    saved = tool(svc, "speechbench_engines", name="echo", command=[sys.executable, str(script), "{audio}"], timeout_s=30)
    assert "cmd:echo" in saved["available"]
    folder = _corpus(tmp_path / "c")
    assert tool(svc, "speechbench_corpus", folder=str(folder))["files"] == 2
    card = tool(svc, "speechbench_start", folder=str(folder), engines=["cmd:echo"], accents=False)
    done = _wait(svc.speechbench, card["id"])
    assert done["run"]["summary"]["cmd:echo"]["wer"] == 0.0
    assert tool(svc, "speechbench_result", run=card["id"], items=True)["items"][0]["text"]
    r = client.post("/api/ui/call", json={"name": "speechbench_list", "arguments": {}})
    assert r.status_code == 200 and "runs" in r.json()
    with pytest.raises(Exception):
        tool(svc, "speechbench_start", folder=str(folder), engines=["faster-whisper:small"])
