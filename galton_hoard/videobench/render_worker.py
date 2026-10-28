"""Renders an HTML animation to frames in a headless browser. Run by path with any Python that has Playwright: it imports nothing from the app.

Input on stdin, one JSON object: {"html": path, "out": folder, "width", "height", "frames": N, "duration_cap": seconds, "default_duration": seconds, "timeout_s": per step}.
Output on stdout, one JSON line: {"ok": true, ...measures} or {"ok": false, "error": "..."}. Frames are written to ``out`` as PNG files named ``f00.png``...
Nothing leaves the machine: every request that is not the page itself is aborted and counted.
"""
import hashlib
import json
import sys
import time
from pathlib import Path


def main() -> None:
    cfg = json.loads(sys.stdin.read())
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    width, height = int(cfg["width"]), int(cfg["height"])
    timeout = float(cfg.get("timeout_s") or 20) * 1000
    result: dict = {"ok": False, "console_errors": [], "page_errors": [], "blocked_requests": []}
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"playwright is not available in this Python: {exc}"}))
        return
    html_url = Path(cfg["html"]).resolve().as_uri()
    started = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-gpu-vsync", "--autoplay-policy=no-user-gesture-required"])
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
            # The page may play itself with a requestAnimationFrame loop that calls seek(clock): that would move the picture between our seek and our screenshot.
            # The renderer is the only clock, so the page's own loop is switched off.
            page.add_init_script("window.requestAnimationFrame = function () { return 0; }; window.cancelAnimationFrame = function () {};")
            page.goto(html_url, wait_until="load")
            has_seek = page.evaluate("typeof window.seek === 'function'")
            result["has_seek"] = bool(has_seek)
            if not has_seek:
                result["error"] = "window.seek is not a function"
                print(json.dumps(result))
                return
            declared = page.evaluate("typeof window.DURATION === 'number' && isFinite(window.DURATION) ? window.DURATION : null")
            duration = float(declared) if declared else float(cfg.get("default_duration") or 10)
            result["declared_duration"] = declared
            duration = max(1.0, min(duration, float(cfg.get("duration_cap") or 30)))
            result["duration"] = duration
            n = max(2, int(cfg.get("frames") or 12))
            times = [round(duration * i / (n - 1), 4) for i in range(n)]
            result["times"] = times
            texts, files, ms = [], [], []
            for i, t in enumerate(times):
                t0 = time.time()
                page.evaluate("async t => { await window.seek(t); if (document.fonts) await document.fonts.ready; }", t)
                page.wait_for_timeout(30)
                path = out / f"f{i:02d}.png"
                page.screenshot(path=str(path), type="png")
                ms.append(round((time.time() - t0) * 1000))
                files.append(path.name)
                try:
                    texts.append(" ".join(page.evaluate("document.body ? document.body.innerText : ''").split())[:300])
                except Exception:  # noqa: BLE001
                    texts.append("")
            result.update(files=files, texts=texts, frame_ms=ms)
            # Is seek a pure function of t? Draw a middle time, go elsewhere, come back: the picture must be the same.
            mid = times[len(times) // 2]
            probe = []
            for t in (mid, 0, mid):
                page.evaluate("async t => { await window.seek(t); }", t)
                page.wait_for_timeout(30)
                probe.append(hashlib.sha256(page.screenshot(type="png")).hexdigest())
            result["deterministic"] = probe[0] == probe[2]
            result["ok"] = True
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        finally:
            browser.close()
    result["render_s"] = round(time.time() - started, 2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
