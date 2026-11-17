"""The browser of the website-rebuild bench. Run by path with any Python that has Playwright: it imports nothing from the app.

Started once per site and model; it reads one JSON command per line on stdin and answers one JSON line on stdout:

* ``{"cmd": "explore", "base", "width", "height", "action": {"goto": "/"} | {"click": sel} | {"fill": sel, "text"} | {"press": key} | {"back": true}}``:
  one action on a page that keeps its state between commands (one page per site), then a text picture of what the page shows.
* ``{"cmd": "case", "base", "width", "height", "case": {...}, "out": folder}``: one case of ``hoard_link.webfidelity`` in a fresh browser
  context (frozen clock, no animations); the checkpoint pictures go to ``out``.
* ``{"cmd": "reset"}``: close the exploration pages; ``{"cmd": "quit"}``.

The first line of stdin is the configuration: ``{"fidelity": path of webfidelity.py}``. Every request that leaves the site is aborted, so a
page cannot reach the network and two loads draw the same thing.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

SNAPSHOT_JS = r"""
() => {
  const out = [];
  const seen = new Set();
  const vis = (el) => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none' && Number(s.opacity) > 0.02; };
  const own = (el) => Array.from(el.childNodes).filter((n) => n.nodeType === 3).map((n) => n.textContent).join(' ').replace(/\s+/g, ' ').trim();
  const sel = (el) => { if (el.id) return '#' + el.id; const c = (el.getAttribute('class') || '').trim().split(/\s+/).filter(Boolean).slice(0, 2);
    return el.tagName.toLowerCase() + (c.length ? '.' + c.join('.') : ''); };
  const KEEP = new Set(['A','BUTTON','INPUT','TEXTAREA','SELECT','IMG','SVG','CANVAS','VIDEO','H1','H2','H3','H4','H5','H6','P','LI','LABEL','SPAN','DIV','SECTION','HEADER','FOOTER','NAV','MAIN','ARTICLE','ASIDE','BLOCKQUOTE','TD','TH','FORM','UL','OL','TABLE','FIGURE','STRONG','EM','SMALL']);
  const walk = (el, depth) => {
    if (out.length >= 260 || !el || el.nodeType !== 1) return;
    if (!KEEP.has(el.tagName.toUpperCase()) && !['BODY','HTML'].includes(el.tagName)) { for (const c of el.children) walk(c, depth); return; }
    if (el.tagName !== 'BODY' && el.tagName !== 'HTML') {
      if (!vis(el)) return;
      const r = el.getBoundingClientRect(); const s = getComputedStyle(el); const text = own(el).slice(0, 100);
      const boxy = s.backgroundColor !== 'rgba(0, 0, 0, 0)' || s.borderTopWidth !== '0px' || s.backgroundImage !== 'none' || s.boxShadow !== 'none';
      const tag = el.tagName.toLowerCase();
      const interactive = ['a','button','input','textarea','select','img','svg','canvas','video'].includes(tag) || /^h[1-6]$/.test(tag);
      if (text || boxy || interactive) {
        const item = { d: depth, tag, sel: sel(el), x: Math.round(r.x + scrollX), y: Math.round(r.y + scrollY), w: Math.round(r.width), h: Math.round(r.height) };
        if (text) item.text = text;
        if (text) { item.font = `${s.fontWeight} ${s.fontSize}/${s.lineHeight} ${s.fontFamily.split(',')[0].replace(/"/g, '')}`; item.color = s.color; if (s.textAlign !== 'start') item.align = s.textAlign; if (s.textTransform !== 'none') item.transform = s.textTransform; if (s.letterSpacing !== 'normal') item.spacing = s.letterSpacing; }
        if (s.backgroundColor !== 'rgba(0, 0, 0, 0)') item.bg = s.backgroundColor;
        if (s.backgroundImage !== 'none') item.bgimage = s.backgroundImage.slice(0, 160);
        if (s.borderTopWidth !== '0px') item.border = `${s.borderTopWidth} ${s.borderTopStyle} ${s.borderTopColor}`;
        if (s.borderRadius !== '0px') item.radius = s.borderRadius;
        if (s.boxShadow !== 'none') item.shadow = s.boxShadow.slice(0, 80);
        if (s.display === 'flex' || s.display === 'grid') item.layout = s.display + (s.display === 'grid' ? ' ' + s.gridTemplateColumns.slice(0, 60) : ' ' + s.flexDirection) + ' gap ' + s.gap;
        if (s.padding !== '0px') item.padding = s.padding;
        if (tag === 'a') item.href = el.getAttribute('href') || '';
        if (['input','textarea','select'].includes(tag)) { item.type = el.getAttribute('type') || tag; item.placeholder = el.getAttribute('placeholder') || ''; item.value = el.value || ''; if (el.type === 'checkbox') item.checked = el.checked; }
        if (tag === 'img') item.alt = el.getAttribute('alt') || '';
        if (el.getAttribute('aria-label')) item.label = el.getAttribute('aria-label');
        out.push(item);
      }
    }
    for (const c of el.children) walk(c, depth + 1);
  };
  walk(document.body, 0);
  const b = getComputedStyle(document.body);
  return { title: document.title, lang: document.documentElement.lang || '', url: location.pathname + location.search,
           page: { width: document.documentElement.scrollWidth, height: document.documentElement.scrollHeight, bg: b.backgroundColor, color: b.color,
                   font: b.fontFamily, margin: b.margin }, items: out };
}
"""


def load_fidelity(path: str):
    spec = importlib.util.spec_from_file_location("webfidelity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:120] or "cp"


class Worker:
    def __init__(self, playwright, fidelity):
        self.fid = fidelity
        self.browser = playwright.chromium.launch(headless=True, args=["--disable-gpu", "--font-render-hinting=none"])
        self.pages = {}      # (base, width) -> (context, page) of exploration

    def _context(self, base: str, width: int, height: int, *, frozen: bool):
        ctx = self.browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=1, locale="es-ES",
                                       timezone_id="UTC", color_scheme="light", reduced_motion="reduce")
        origin = re.match(r"^[a-z]+://[^/]+", base).group(0)
        blocked = []

        def gate(route):
            url = route.request.url
            if url.startswith(origin) or url.startswith(("data:", "blob:", "about:")):
                route.continue_()
            else:
                blocked.append(url[:200])
                route.abort()
        ctx.route("**/*", gate)
        if frozen:
            ctx.add_init_script(self.fid.FREEZE_JS)
        return ctx, blocked

    def explore(self, cmd):
        base, width, height = cmd["base"].rstrip("/"), int(cmd.get("width") or 1280), int(cmd.get("height") or 800)
        key = (base, width)
        if key not in self.pages:
            ctx, blocked = self._context(base, width, height, frozen=False)
            page = ctx.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)[:200]))
            page.on("console", lambda m: errors.append(m.text[:200]) if m.type == "error" else None)
            self.pages[key] = (ctx, page, errors, blocked)
        ctx, page, errors, blocked = self.pages[key]
        page.set_default_timeout(int(cmd.get("timeout_ms") or 8000))
        action = cmd.get("action") or {"goto": "/"}
        error = ""
        try:
            if "goto" in action:
                page.goto(base + "/" + str(action["goto"] or "").lstrip("/"), wait_until="load")
            elif "click" in action:
                page.locator(str(action["click"])).first.click()
            elif "fill" in action:
                page.locator(str(action["fill"])).first.fill(str(action.get("text", "")))
            elif "press" in action:
                page.locator(str(action.get("on") or "body")).first.press(str(action["press"]))
            elif "back" in action:
                page.go_back(wait_until="load")
            page.wait_for_timeout(250)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:240] if str(exc) else ''}"
        snap = {}
        try:
            snap = page.evaluate(SNAPSHOT_JS)
        except Exception as exc:  # noqa: BLE001
            error = error or f"{type(exc).__name__}: {str(exc)[:200]}"
        if cmd.get("shot"):
            try:
                page.screenshot(path=cmd["shot"], type="png")
            except Exception:  # noqa: BLE001
                pass
        out = {"ok": not error, "error": error, "snapshot": snap, "page_errors": errors[-5:], "blocked": blocked[-5:]}
        errors.clear()
        return out

    def case(self, cmd):
        base, width, height = cmd["base"].rstrip("/"), int(cmd["width"]), int(cmd.get("height") or 800)
        out_dir = Path(cmd["out"])
        out_dir.mkdir(parents=True, exist_ok=True)
        ctx, blocked = self._context(base, width, height, frozen=True)
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)[:200]))

        def shoot(route, w, name):
            path = out_dir / f"{safe_name(cmd['case']['id'])}--{safe_name(name)}--{w}.png"
            page.screenshot(path=str(path), type="png")
            return str(path)
        try:
            result = self.fid.run_case(page, base, cmd["case"], width=width, shoot=shoot, timeout_ms=int(cmd.get("timeout_ms") or 8000))
        finally:
            ctx.close()
        result.update(page_errors=errors[-5:], blocked=blocked[-5:])
        return result

    def reset(self):
        """Close the exploration pages (between two sites)."""
        for ctx, *_ in self.pages.values():
            try:
                ctx.close()
            except Exception:  # noqa: BLE001
                pass
        self.pages = {}

    def close(self):
        for ctx, *_ in self.pages.values():
            try:
                ctx.close()
            except Exception:  # noqa: BLE001
                pass
        self.browser.close()


def main() -> None:
    cfg = json.loads(sys.stdin.readline())
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"playwright is not available in this Python: {exc}"}), flush=True)
        return
    fidelity = load_fidelity(cfg["fidelity"])
    with sync_playwright() as p:
        worker = Worker(p, fidelity)
        print(json.dumps({"ok": True, "ready": True}), flush=True)
        try:
            for line in sys.stdin:
                if not line.strip():
                    continue
                cmd = json.loads(line)
                if cmd.get("cmd") == "quit":
                    break
                try:
                    if cmd.get("cmd") == "reset":
                        worker.reset()
                        reply = {"ok": True}
                    else:
                        reply = worker.explore(cmd) if cmd.get("cmd") == "explore" else worker.case(cmd)
                except Exception as exc:  # noqa: BLE001
                    reply = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                print(json.dumps(reply), flush=True)
        finally:
            worker.close()


if __name__ == "__main__":
    main()
