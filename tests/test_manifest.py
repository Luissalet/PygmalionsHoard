"""faustus-plugin.json, the README and the docs stay in sync with the code."""

import json
import re
import subprocess
import sys
from pathlib import Path

from pygmalion_hoard import SERVICE, __version__
from pygmalion_hoard.agent_tools import TOOLS
from pygmalion_hoard.config import DEFAULT_PORT

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", ROOT / "README.es.md", ROOT / "AGENTS.md", *sorted((ROOT / "docs").glob("*.md"))]
BANNED = ("chatgpt", "claude", "openai", "anthropic", "lm studio", "odysseus", "gemini", "copilot", "notion", "evernote", "paperless")
SLOGAN_HINTS = ("tagline", "slogan")


def test_manifest_matches_code():
    manifest = json.loads((ROOT / "faustus-plugin.json").read_text(encoding="utf-8"))
    assert manifest["id"] == "pygmalion" and manifest["name"] == "Pygmalion's Hoard"
    assert manifest["app"]["health"]["expect"]["service"] == SERVICE == "pygmalion-hoard"
    assert manifest["defaults"]["APP_URL"].endswith(f":{DEFAULT_PORT}") and DEFAULT_PORT == 5202
    assert manifest["app"]["launch_hint"]["env"]["PYGMALION_PORT"] == str(DEFAULT_PORT)


def test_every_tool_is_documented_in_both_readmes():
    for name in ("README.md", "README.es.md"):
        readme = (ROOT / name).read_text(encoding="utf-8")
        for tool in TOOLS:
            assert f"`{tool.name}`" in readme, f"{tool.name} missing in {name}"
        assert f"({len(TOOLS)})" in readme, name
    api = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    for tool in TOOLS:
        assert f"## `{tool.name}`" in api, tool.name


def test_first_lines_are_short():
    for tool in TOOLS:
        assert len(tool.description.splitlines()[0]) <= 110, tool.name


def test_no_other_products_or_slogans_in_docs():
    for path in DOCS:
        text = path.read_text(encoding="utf-8").lower()
        for word in BANNED + SLOGAN_HINTS:
            assert word not in text, f"{word} in {path.name}"


def test_no_placeholder_lines_in_docs():
    for path in DOCS:
        for line in path.read_text(encoding="utf-8").splitlines():
            assert not re.search(r"\b(TODO|TBD|FIXME)\b", line) and not re.search(r"(?i)lorem ipsum", line), f"{path.name}: {line}"


def test_version_matches():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(rf'version = "{re.escape(__version__)}"', pyproject)
    assert json.loads((ROOT / "package.json").read_text(encoding="utf-8"))["version"] == __version__


def test_the_api_doc_is_up_to_date():
    before = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_api_doc.py"), "--check"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert before == (ROOT / "docs" / "API.md").read_text(encoding="utf-8")


def test_the_bundled_calibration_text_has_both_languages():
    text = (ROOT / "pygmalion_hoard" / "calib" / "es-en-general.txt").read_text(encoding="utf-8")
    assert len(text) > 150_000
    assert " el " in text and " the " in text and " de " in text and " of " in text
