"""Render the brain viewer frame by frame and encode an MP4 for X/Twitter.

  python viz/record.py --episode trump --from-bar 120 --bars 90 --out out/trump.mp4

Needs the viz server running (python -m http.server 8765 --directory viz) and ffmpeg.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="demo")
    ap.add_argument("--url", default="http://localhost:8765/index.html")
    ap.add_argument("--from-bar", type=int, default=0)
    ap.add_argument("--bars", type=int, default=60)
    ap.add_argument("--fpb", type=int, default=14, help="video frames per market minute")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--out", default="out/flytrader.mp4")
    ap.add_argument("--hold", type=float, default=0, help="extra seconds at the end (shows the end card)")
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="flyframes_"))
    n = a.bars * a.fpb + int(a.hold * 30)
    with sync_playwright() as p:
        exe = next(Path.home().glob("Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/*"), None)
        browser = p.chromium.launch(executable_path=str(exe) if exe else None, headless=True,
                                    args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport={"width": a.width, "height": a.height}, device_scale_factor=1)
        page.goto(f"{a.url}?episode={a.episode}&record=1&fpb={a.fpb}&from={a.from_bar}&end={a.from_bar + a.bars - 1}")
        page.wait_for_function("window.ready === true", timeout=120_000)
        page.evaluate("document.fonts.ready")
        start = a.from_bar * a.fpb
        for i in range(n):
            page.evaluate(f"window.renderFrame({start + i})")
            page.screenshot(path=str(tmp / f"f{i:05d}.png"))
            if i % 100 == 0:
                print(f"  frame {i}/{n}", flush=True)
        browser.close()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(tmp / "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", a.out],
                   check=True)
    shutil.rmtree(tmp)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
