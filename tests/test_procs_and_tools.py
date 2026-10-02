"""Running programs (logging, redaction, cancel, timeouts), llama.cpp command lines and parsers, folders, calibration text, ports."""

import json
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from pygmalion_hoard import calib, llama_tools as LT, port as PORT, procs
from pygmalion_hoard.errors import PygmalionError
from pygmalion_hoard.workdir import EXE


def py(code):
    return [sys.executable, "-c", code]


# ------------------------------------------------------------------------------------------------ procs
def test_streaming_collects_lines_logs_them_and_reports_the_exit_code(tmp_path):
    seen = []
    log = tmp_path / "logs" / "job.log"
    r = procs.run_streaming(py("import sys; print('uno'); print('dos', file=sys.stderr); sys.exit(3)"), log_path=log, on_line=seen.append)
    assert r.returncode == 3 and r.ok is False and sorted(seen) == ["dos", "uno"] and sorted(r.tail) == ["dos", "uno"]
    text = log.read_text(encoding="utf-8")
    assert text.startswith("$ ") and "uno" in text and "dos" in text


def test_secrets_never_reach_the_log_the_callback_or_the_tail(tmp_path):
    seen = []
    log = tmp_path / "job.log"
    r = procs.run_streaming(py("print('token=hf_supersecret99')"), log_path=log, on_line=seen.append, secrets=["hf_supersecret99"])
    assert seen == ["token=***"] and r.tail == ["token=***"] and "hf_supersecret99" not in log.read_text(encoding="utf-8")
    r = procs.run_streaming(py("print(1)"), log_path=log, secrets=["1"])
    assert "$ " in log.read_text(encoding="utf-8")
    assert procs.redact("abc abc", ["abc"]) == "abc abc", "very short secrets are not masked (they would garble the log)"
    assert procs.redact("xx hf_abcdef yy", ["hf_abcdef"]) == "xx *** yy"


def test_a_failing_callback_does_not_stop_the_reader():
    def bad(line):
        raise ValueError("parser bug")
    r = procs.run_streaming(py("print('a'); print('b')"), on_line=bad)
    assert r.ok and r.tail == ["a", "b"]


def test_cancel_stops_the_process_and_its_children(tmp_path):
    marker = tmp_path / "child-alive"
    child = tmp_path / "child.py"
    child.write_text("import pathlib, time\nwhile True:\n    pathlib.Path(%r).write_text('x')\n    time.sleep(0.1)\n" % str(marker), encoding="utf-8")
    code = ("import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, {str(child)!r}])\n"
            "print('started', flush=True)\ntime.sleep(60)\n")
    cancel = threading.Event()
    started = threading.Event()

    def on_line(line):
        if line == "started":
            started.set()
    timer = threading.Thread(target=lambda: (started.wait(15), time.sleep(0.5), cancel.set()))
    timer.start()
    t0 = time.monotonic()
    r = procs.run_streaming(py(code), cancel=cancel, on_line=on_line, grace_s=1.0)
    timer.join()
    assert r.cancelled is True and r.ok is False and time.monotonic() - t0 < 20
    time.sleep(0.5)
    marker.unlink(missing_ok=True)
    time.sleep(0.6)
    assert not marker.exists(), "the grandchild was killed with the process group"


def test_a_stop_request_lets_the_worker_clean_up_first(tmp_path):
    flag = tmp_path / "saved"
    code = ("import signal, sys, time, pathlib\n"
            f"signal.signal(signal.SIGTERM, lambda *a: (pathlib.Path({str(flag)!r}).write_text('ok'), sys.exit(0)))\n"
            "print('ready', flush=True)\ntime.sleep(60)\n")
    if os.name == "nt":
        pytest.skip("SIGTERM handlers are POSIX; Windows uses CTRL_BREAK")
    cancel = threading.Event()
    threading.Thread(target=lambda: (time.sleep(1.0), cancel.set())).start()
    r = procs.run_streaming(py(code), cancel=cancel, grace_s=10.0)
    assert r.cancelled and flag.read_text(encoding="utf-8") == "ok"


def test_timeout_stops_a_stuck_program():
    r = procs.run_streaming(py("import time; time.sleep(60)"), timeout_s=0.5, grace_s=1.0)
    assert r.timed_out is True and r.ok is False and r.duration_s < 20


def test_a_program_that_cannot_start_raises_oserror():
    with pytest.raises(OSError):
        procs.run_streaming(["/definitely/not/here"])


def test_gpu_visibility_is_always_explicit():
    assert procs.build_env(gpus=[2, 3])["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert procs.build_env(gpus=[])["CUDA_VISIBLE_DEVICES"] == "", "no lease, no GPU"
    assert procs.build_env(extra={"A": 1}, gpus=None)["A"] == "1"
    assert procs.build_env()["PYTHONIOENCODING"] == "utf-8"


def test_run_capture():
    code, out, err = procs.run_capture(py("import sys; print('hola'); print('e', file=sys.stderr)"))
    assert (code, out.strip(), err.strip()) == (0, "hola", "e")
    assert procs.run_capture(py("import time; time.sleep(5)"), timeout_s=0.3)[0] is None
    assert procs.run_capture(["/nope/nothing"])[0] is None


# ------------------------------------------------------------------------------------------------ llama tools
def test_quant_types_are_validated_and_normalised():
    assert LT.check_quant_types(["q4_k_m", " Q8_0 ", "Q4_K_M"], False) == (["Q4_K_M", "Q8_0"], [])
    with pytest.raises(PygmalionError):
        LT.check_quant_types(["Q9"], False)
    with pytest.raises(PygmalionError):
        LT.check_quant_types([], False)
    assert LT.IMATRIX_RECOMMENDED == ("IQ4_XS", "IQ3_M")


def test_expected_sizes_follow_the_bits_per_weight():
    assert LT.expected_size(8_000_000_000, "Q4_K_M") < LT.expected_size(8_000_000_000, "Q8_0") < LT.expected_size(8_000_000_000, "f16")


def test_command_lines_are_argument_lists():
    assert LT.convert_argv("py", "c.py", "/hf", "/o.gguf", "bf16") == ["py", "c.py", "/hf", "--outfile", "/o.gguf", "--outtype", "bf16"]
    with pytest.raises(PygmalionError):
        LT.convert_argv("py", "c.py", "/hf", "/o", "q2")
    assert LT.convert_lora_argv("py", "l.py", "/b", "/a", "/o.gguf")[:5] == ["py", "l.py", "--base", "/b", "/a"]
    assert LT.imatrix_argv("imx", "m", "c.txt", "o.dat", 50) == ["imx", "-m", "m", "-f", "c.txt", "-o", "o.dat", "-ngl", "99", "--chunks", "50"]
    assert LT.quantize_argv("q", "a", "b", "Q4_K_M") == ["q", "a", "b", "Q4_K_M"]
    assert LT.quantize_argv("q", "a", "b", "Q4_K_M", "m.dat") == ["q", "--imatrix", "m.dat", "a", "b", "Q4_K_M"]
    assert LT.perplexity_argv("p", "m", "t", 512, 20)[-4:] == ["--chunks", "20", "-ngl", "99"]


def test_converter_progress_lines():
    assert LT.parse_convert_line("INFO:hf-to-gguf:blk.0.attn_q.weight,\ttorch.bfloat16 --> F16, shape = {4096, 4096}") == "blk.0.attn_q.weight"
    assert LT.parse_convert_line("INFO:hf-to-gguf:Set model parameters") is None


def test_imatrix_progress_lines():
    assert LT.parse_imatrix_line("compute_imatrix: computing over 100 chunks, n_ctx=512") == {"total": 100}
    assert LT.parse_imatrix_line("[1]6.1,[2]5.9,[3]5.7,") == {"step": 3}
    assert LT.parse_imatrix_line("save_imatrix: stored") == {}


def test_quantize_progress_and_sizes():
    line = "[  12/ 291]  blk.0.attn_q.weight - [ 4096,  4096,     1,     1], type =    f16, converting to q4_k .. size =    32.00 MiB ->     9.00 MiB"
    assert LT.parse_quantize_line(line) == {"step": 12, "total": 291}
    out = LT.parse_quantize_line("llama_model_quantize_internal: model size  =  7500.00 MB")
    assert out["model_bytes"] == 7500 * 1048576
    assert LT.parse_quantize_line("llama_model_quantize_internal: quant size  =     4.2 GiB")["quant_bytes"] == int(4.2 * (1 << 30))


def test_perplexity_lines():
    assert LT.parse_perplexity_line("Final estimate: PPL = 7.1234 +/- 0.04321") == {"ppl": 7.1234, "ppl_error": 0.04321, "final": True}
    assert LT.parse_perplexity_line("[3]7.4000,") == {"chunk": 3, "ppl_running": 7.4}
    assert LT.parse_perplexity_line("[1]8.1200,[2]7.9000,[3]7.4000,") == {"chunk": 3, "ppl_running": 7.4}, "the newest of a growing line"
    assert LT.parse_perplexity_line("llama_perf: load time = 1 ms") == {}


def test_gguf_names_are_portable():
    assert LT.gguf_name("acme/tiny: base?", "Q4_K_M") == "acme-tiny-base-Q4_K_M.gguf"
    assert LT.gguf_name("x", "lora", "-f16") == "x-lora-f16.gguf"
    assert LT.is_sharded_gguf("a-00001-of-00003.gguf") and not LT.is_sharded_gguf("a.gguf")


# ------------------------------------------------------------------------------------------------ work folders and programs
def test_work_paths_follow_the_setting(ctx):
    w = ctx.svc.work
    assert w.root == ctx.work and w.hf_dir == ctx.work / "hf" and w.base_dir("acme/x") == ctx.work / "hf" / "acme--x"
    assert w.output_dir("a_1") == ctx.work / "outputs" / "a_1" and w.job_dir("j_1") == ctx.work / "jobs" / "j_1"
    assert w.hf_dir.is_dir() and w.calib_dir.is_dir()


def test_default_work_folder_is_inside_the_data_folder(tmp_path):
    from conftest import build_services
    c = build_services(tmp_path / "z")
    try:
        c.svc.settings.set({"paths.work": ""})
        assert c.svc.work.root == c.data / "work"
    finally:
        c.svc.stop()


def test_programs_are_resolved_from_the_settings(ctx):
    w = ctx.svc.work
    assert w.python_ok() and w.python() == sys.executable
    assert Path(w.llama_bin("llama-quantize")).name == f"llama-quantize{EXE}" and Path(w.src_script("convert_hf_to_gguf.py")).name == "convert_hf_to_gguf.py"
    assert w.src_script("missing.py") is None and Path(w.ollama_exe()).name == f"ollama{EXE}"
    ctx.svc.settings.set({"llama.bin_dir": str(ctx.tmp / "empty"), "env.python": str(ctx.tmp / "nope")})
    assert w.llama_bin("llama-nonexistent-program") is None and not w.python_ok()


def test_a_replacement_workers_folder_falls_back_to_the_packaged_ones(ctx):
    assert ctx.svc.work.worker("train_lora").parent.name == "fake_workers"
    assert ctx.svc.work.worker("merge_lora").parent.name == "workers" and ctx.svc.work.worker("merge_lora").is_file()


# ------------------------------------------------------------------------------------------------ calibration text and ports
def test_the_bundled_calibration_text_has_both_languages_and_enough_material():
    text = calib.bundled_text()
    assert len(text) > 150_000 and sum(text.count(w) for w in (" the ", " and ", " of ")) > 500 and sum(text.count(w) for w in (" de ", " que ", " la ")) > 500
    assert "def " in text or "```" in text or "function" in text, "it includes some code, which importance matrices benefit from"


def test_calibration_from_a_dataset_is_padded_when_short(tmp_path):
    records = [{"messages": [{"role": "user", "content": "Pregunta"}, {"role": "assistant", "content": "Respuesta de la abuela"}]}] * 3
    out = calib.write_calibration(tmp_path / "c.txt", records)
    text = Path(out["path"]).read_text(encoding="utf-8")
    assert out["source"] == "dataset+bundled" and "Respuesta de la abuela" in text and "Pregunta" in text and "user:" not in text and out["chars"] == len(text)
    big = [{"text": "Un párrafo de texto corrido sin más. " * 30}] * 60
    assert calib.write_calibration(tmp_path / "d.txt", big)["source"] == "dataset"
    assert calib.write_calibration(tmp_path / "e.txt")["source"] == "bundled"


def test_rejected_and_pending_records_stay_out_of_the_calibration_text():
    recs = [{"text": "bueno", "_meta": {"status": "ok"}}, {"text": "malo", "_meta": {"status": "rejected"}}, {"text": "dudoso", "_meta": {"status": "pending"}}]
    assert calib.text_from_records(recs) == "bueno"
    assert len(calib.text_from_records([{"text": "x" * 1000}] * 10, max_chars=2500)) < 3200


def test_ports():
    free = PORT.free_port()
    assert PORT.can_listen(free) and PORT.find_available_port(free) == free
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        taken = s.getsockname()[1]
        assert PORT.can_listen(taken) is False and PORT.find_available_port(taken) > taken
    with pytest.raises(RuntimeError):
        PORT.find_available_port(70000)


def test_the_stop_file_reaches_a_worker_that_gets_no_signal(tmp_path):
    """On Windows a worker started without a console never receives Ctrl+Break; the stop file is what lets it save a checkpoint."""
    flag = tmp_path / "saved"
    workers = Path(procs.__file__).resolve().parent / "workers"
    code = ("import signal, sys, time, pathlib\n"
            f"sys.path.insert(0, {str(workers)!r})\n"
            "import _common as C\n"
            "C.install_stop_handlers()\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)   # like Windows: the polite signal never arrives\n"
            "print('ready', flush=True)\n"
            "deadline = time.time() + 60\n"
            "while not C.STOP.is_set() and time.time() < deadline:\n    time.sleep(0.05)\n"
            f"pathlib.Path({str(flag)!r}).write_text('checkpoint' if C.STOP.is_set() else 'killed')\n")
    stop_file = tmp_path / "STOP"
    stop_file.write_text("left over from an earlier stop", encoding="utf-8")   # must not stop the new run at once
    cancel = threading.Event()
    ready = threading.Event()
    threading.Thread(target=lambda: (ready.wait(15), time.sleep(0.5), cancel.set())).start()
    r = procs.run_streaming(py(code), cancel=cancel, grace_s=15.0, stop_file=stop_file, on_line=lambda line: ready.set() if line == "ready" else None)
    assert r.cancelled and flag.read_text(encoding="utf-8") == "checkpoint" and r.duration_s < 14


def test_children_number_gpus_like_nvidia_smi():
    assert procs.build_env(gpus=[2])["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert procs.build_env(extra={"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, gpus=[])["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
