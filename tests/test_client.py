"""The web client's sources stay consistent: every translation key used exists, both languages are filled, identity markers are in place."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "client" / "src"
I18N = (SRC / "i18n.js").read_text(encoding="utf-8")
ENTRIES = dict(re.findall(r'^\s*([a-z0-9_]+): \[(".*")\],?\s*$', I18N, re.M))
SOURCES = [p for p in sorted(SRC.rglob("*")) if p.suffix in (".js", ".jsx") and p.name != "i18n.js"]
PAGES = ["Panel", "Modelos", "Datasets", "Entrenar", "Trabajos", "Fusionar", "Cuantizar", "Contexto", "Linaje", "Ajustes"]


def used_keys():
    literal, prefixes = {}, set()
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for call in re.finditer(r"\bt\(([^()]*(?:\([^()]*\)[^()]*)*)\)", text):
            for q in re.finditer(r'"([a-z][a-z0-9_]*)"', call.group(1)):
                literal.setdefault(q.group(1), path.name)
            for q in re.finditer(r"`([a-z0-9_]+)\$\{", call.group(1)):
                prefixes.add(q.group(1))
        for m in re.finditer(r'key: "([a-z0-9_]+)"', text):
            literal.setdefault(m.group(1), path.name)
    return literal, prefixes


def test_every_key_used_in_the_client_is_translated():
    literal, _ = used_keys()
    missing = {k: f for k, f in literal.items() if k not in ENTRIES}
    assert not missing, missing


def test_dynamic_key_families_exist():
    _, prefixes = used_keys()
    for prefix in prefixes:
        assert any(k.startswith(prefix) for k in ENTRIES), prefix


def test_both_languages_are_filled_and_placeholders_match():
    assert len(ENTRIES) > 400
    for key, raw in ENTRIES.items():
        parts = re.findall(r'"((?:[^"\\]|\\.)*)"', raw)
        assert len(parts) == 2 and all(p.strip() for p in parts), key
        assert sorted(re.findall(r"\{(\w+)\}", parts[0])) == sorted(re.findall(r"\{(\w+)\}", parts[1])), key


def test_no_emojis_or_banned_words_in_the_client():
    banned = ("chatgpt", "claude", "openai", "anthropic", "lm studio", "gemini", "copilot")
    for path in [*SOURCES, SRC / "i18n.js"]:
        text = path.read_text(encoding="utf-8")
        assert not re.search("[\U0001F300-\U0001FAFF☀-➿]", text), path.name
        assert not any(w in text.lower() for w in banned), path.name


def test_ten_pages_and_identity():
    for page in PAGES:
        assert (SRC / "pages" / f"{page}.jsx").is_file(), page
    app = (SRC / "App.jsx").read_text(encoding="utf-8")
    assert all(f"./pages/{p}.jsx" in app for p in PAGES)
    html = (ROOT / "client" / "index.html").read_text(encoding="utf-8")
    assert 'data-hoard-app="pygmalion"' in html and "Pygmalion's Hoard" in html
    css = (SRC / "index.css").read_text(encoding="utf-8")
    assert 'html[data-hoard-app="pygmalion"]' in css and "#c46a8a" in css
    assert "pygmalion-lang" in I18N


def test_icons_and_manifest_assets_exist():
    for name in ("icon-192.png", "icon-512.png", "favicon.ico"):
        assert (ROOT / "client" / "public" / name).is_file(), name


def test_polling_ticks_when_the_page_becomes_visible():
    hooks = (SRC / "components" / "hooks.js").read_text(encoding="utf-8")
    assert 'addEventListener("visibilitychange"' in hooks and 'removeEventListener("visibilitychange"' in hooks
    assert "if (immediate) run();" in hooks, "the first call must not wait for the tab to be visible"
    assert "if (immediate) run();" in hooks and "document.hidden" in hooks.split("const visible")[1].split("\n")[0]


def test_the_app_bumps_the_version_when_a_count_moves():
    app = (SRC / "App.jsx").read_text(encoding="utf-8")
    assert "lastCounts" in app and "setVersion((v) => v + 1)" in app.split("const refreshDash")[1].split("const changed")[0]


def test_pages_that_poll_for_their_first_data_ask_for_the_immediate_tick():
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for call in re.finditer(r"usePoll\(([^;]*)\);", text):
            # every page with a poll also loads through useLoad on mount, or says so with the immediate flag
            assert "useLoad(" in text or "true" in call.group(1), path.name


def test_the_brand_mark_is_the_app_icon_not_a_letter():
    app = (SRC / "App.jsx").read_text(encoding="utf-8")
    brand = app[app.index("<aside"):app.index("</aside>")]
    assert re.search(r'<img src="/icon-192\.png"', brand) and ">P<" not in brand and "brand-mark" not in brand
    assert (ROOT / "client" / "public" / "icon-192.png").is_file() and (ROOT / "pygmalion_hoard" / "static" / "icon-192.png").is_file()


def test_backend_messages_are_never_printed_raw():
    """Fields the backend sends as `{key, params, text}` items go through t.msg / t.error; printing one directly would crash React."""
    raw = re.compile(r"\{(?:job|p|galton|fit|check|item|pick|plan\.time|preview|stats|local|b)\.(?:error|hint|message|detail|note|notes|warnings|problem|what|fix|reason|formula|convert_note|title)\}")
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        assert not raw.search(text), (path.name, raw.search(text).group(0))


def test_every_setting_has_a_label_in_both_languages():
    from pygmalion_hoard.settings import SPECS
    own_controls = {"scheduler.paused", "gpus.allowed", "gpus.reserved"}       # these have their own control and wording
    for name in SPECS:
        if name in own_controls:
            continue
        assert f"set_{name.replace('.', '_')}" in ENTRIES, name


def test_the_jobs_badge_counts_only_what_runs_or_waits():
    app = (SRC / "App.jsx").read_text(encoding="utf-8")
    assert "jobs: dash?.counts?.jobs_active || 0" in app and "jobs_failed" not in app.split("const badges")[1].split("\n")[0]


def test_the_dashboard_signature_changes_when_a_pipeline_moves_to_its_next_step():
    app = (SRC / "App.jsx").read_text(encoding="utf-8")
    assert "scheduler?.finished" in app.split("const refreshDash")[1].split("const changed")[0]


def test_the_job_page_keeps_polling_while_the_chain_is_live():
    jobs = (SRC / "pages" / "Trabajos.jsx").read_text(encoding="utf-8")
    assert "chainLive" in jobs and 'state === "planned"' in jobs and "usePoll(reload, active || chainLive" in jobs


def test_the_service_worker_is_named_by_the_build_and_never_caches_the_page():
    from pygmalion_hoard.hoard_link import service
    source = Path(service.__file__).read_text(encoding="utf-8")
    assert "_build_id" in source and "index.html" in source and 'request.mode === "navigate"' in source
