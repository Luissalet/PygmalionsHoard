"""Program output made fit for a log: escape sequences stripped, spinner and progress frames collapsed to the last state."""

import sys

from pygmalion_hoard import logclean as L, procs

SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def test_ansi_and_osc_sequences_are_stripped():
    raw = "\x1b[?2026h\x1b[?25l\x1b[1Gpulling manifest \x1b[K\x1b]0;title\x07\x1b[?25h\x1b[?2026l"
    assert L.strip_ansi(raw) == "pulling manifest "
    assert L.strip_ansi("\x1b[31mred\x1b[0m") == "red" and L.strip_ansi("a\x00b\x7fc") == "abc"


def test_a_carriage_return_keeps_what_was_drawn_last():
    assert L.clean_line("10%\r50%\r100% done") == ("100% done", True)
    assert L.clean_line("plain text") == ("plain text", False)


def test_spinner_glyphs_and_bars_mark_progress_and_are_removed():
    text, progress = L.clean_line(f"\x1b[1G{SPIN[0]} gathering model components ")
    assert text == "gathering model components" and progress
    assert L.clean_line("copying file 45% ▕██████    ▏")[1] is True


def test_spinner_frames_collapse_into_one_line():
    frames = [f"\x1b[?2026h\x1b[1G{c} gathering model components \x1b[K\x1b[?2026l" for c in SPIN]
    out = L.clean_lines(["pulling manifest", *frames, "success"])
    assert out == ["pulling manifest", "gathering model components", "success"]


def test_progress_of_different_activities_stays_apart_and_percentages_advance_in_place():
    lines = ["copying file sha256:aa 10%", "copying file sha256:aa 55%", "copying file sha256:aa 100%", "writing manifest", "writing manifest",
             "success"]
    assert L.clean_lines(lines) == ["copying file sha256:aa 100%", "writing manifest", "writing manifest", "success"]


def test_ordinary_repeated_lines_are_never_merged():
    assert L.clean_lines(["epoch done", "epoch done", "epoch done"]) == ["epoch done"] * 3


def test_lines_with_only_escape_codes_disappear_but_blank_lines_stay():
    assert L.clean_lines(["a", "\x1b[?25h", "", "b"]) == ["a", "", "b"]


def test_a_streamed_spinner_is_one_line_in_the_log_and_the_tail(tmp_path):
    code = ("import sys\n"
            "for c in 'ABCDEFGH':\n"
            "    sys.stdout.write('\\r\\x1b[?25l' + chr(0x280b) + ' gathering model components ' + c + '\\x1b[K'); sys.stdout.flush()\n"
            "sys.stdout.write('\\n'); print('success')\n")
    log = tmp_path / "job.log"
    seen = []
    result = procs.run_streaming([sys.executable, "-c", code], log_path=log, on_line=seen.append)
    body = log.read_text(encoding="utf-8").split("print('success')\n", 1)[1]   # what follows the command line
    assert "\x1b" not in body and "\r" not in body and not any(c in body for c in SPIN)
    assert body.count("gathering model components") == 1 and body.rstrip().endswith("success")
    assert [x for x in body.splitlines() if x] == ["gathering model components H", "success"]
    assert [x for x in result.tail if x] == ["gathering model components H", "success"]


def test_old_logs_are_cleaned_when_read(ctx):
    job = ctx.svc.jobs.submit("convert", {"name": "x"})
    folder = ctx.svc.work.job_dir(job["id"])
    (folder / "job.log").write_text("start\n" + "".join(f"\x1b[1G{c} working hard \x1b[K\n" for c in SPIN) + "end\n", encoding="utf-8")
    assert ctx.svc.jobs.log_tail(job["id"], 50) == ["start", "working hard", "end"]
