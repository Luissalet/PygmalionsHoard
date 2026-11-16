"""Takes screenshots of an HTML page in a headless browser. Run by path with any Python that has Playwright: it imports nothing from the app.

Input on stdin, one JSON object: {"html": path, "out": folder, "width", "height", "settle_ms": wait after load, "motion_ms": wait before the second picture, "timeout_s"}.
Output on stdout, one JSON line: {"ok": true, ...} or {"ok": false, "error": "..."}. The pictures are ``shot.png`` (after the page settled) and ``shot2.png``
(``motion_ms`` later): the difference between them tells whether the page moves by itself. Nothing leaves the machine: every request that is not the page itself is aborted and counted.
WebGL runs on the software rasteriser of the browser, so a page that draws 3D never needs the graphics card.
"""
import json
import sys
import time
from pathlib import Path


def main() -> None:
    cfg = json.loads(sys.stdin.read())
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    width, height = int(cfg["width"]), int(cfg["height"])
    timeout = float(cfg.get("timeout_s") or 30) * 1000
    result: dict = {"ok": False, "console_errors": [], "page_errors": [], "blocked_requests": []}
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"playwright is not available in this Python: {exc}"}))
        return
    html_url = Path(cfg["html"]).resolve().as_uri()
    started = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--disable-gpu"])
        try:
            context = browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=1)
            page = context.new_page()
            page.set_default_timeout(timeout)
            page.on("console", lambda m: result["console_errors"].append(m.text[:300]) if m.type == "error" else None)
            page.on("pageerror", lambda e: result["page_errors"].append(str(e)[:300]))

            def gate(route):
                url = route.request.url
                if url == html_url or url.startswith("data:") or url.startswith("blob:") or url.startswith("about:"):
                    route.continue_()
                else:
                    result["blocked_requests"].append(url[:200])
                    route.abort()
            page.route("**/*", gate)
            page.goto(html_url, wait_until="load")
            page.wait_for_timeout(int(cfg.get("settle_ms") or 0))
            page.screenshot(path=str(out / "shot.png"), type="png")
            info = page.evaluate("""() => ({title: document.title || '', text: (document.body ? document.body.innerText : '').trim().length,
                                           elements: document.getElementsByTagName('*').length, canvases: document.getElementsByTagName('canvas').length,
                                           svgs: document.getElementsByTagName('svg').length, scrollHeight: document.documentElement.scrollHeight,
                                           scrollWidth: document.documentElement.scrollWidth})""")
            result.update(title=info["title"][:120], text_chars=info["text"], elements=info["elements"], canvases=info["canvases"], svgs=info["svgs"],
                          page_height=info["scrollHeight"], page_width=info["scrollWidth"])
            motion_ms = int(cfg.get("motion_ms") or 0)
            files = ["shot.png"]
            if motion_ms > 0:
                page.wait_for_timeout(motion_ms)
                page.screenshot(path=str(out / "shot2.png"), type="png")
                files.append("shot2.png")
            result.update(files=files, ok=True)
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        finally:
            browser.close()
    result["render_s"] = round(time.time() - started, 2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
