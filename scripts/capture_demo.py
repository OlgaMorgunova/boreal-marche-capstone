"""Record the README's animated capture (60-90 s) by driving the *live* interface in a browser.

    uv run --no-project --with playwright --with pillow --with imageio --with imageio-ffmpeg \
        python scripts/capture_demo.py

Uses the installed Chrome (or Edge) through Playwright, so no browser download is needed. Steps
follow docs/DEMO_VIDEO_SCRIPT.md: retention -> what-if -> basket -> French assistant question ->
privacy refusal -> monitoring -> /version. Each screenshot gets a caption bar; the opening and
closing cards are the two slides of reports/977_D6_Morgunova_DemoVideoCards.html (build them first:
``python -m boreal_marche.reporting.build D6-CARDS``). Writes docs/media/demo.gif (README) and
docs/media/demo.mp4 (upload as the unlisted video). Every frame is a real screenshot of the
deployed system at the time of recording.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
EV = json.loads((ROOT / "reports" / "metrics" / "d6_evidence.json").read_text("utf-8"))["live"]
CARDS = ROOT / "reports" / "977_D6_Morgunova_DemoVideoCards.html"
W, H = 1280, 720
BAR = 64


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Reel:
    def __init__(self) -> None:
        self.frames: list[tuple[Image.Image, int]] = []

    def add(self, png: bytes, seconds: float, caption: str | None = None) -> None:
        img = Image.open(io.BytesIO(png)).convert("RGB").resize((W, H))
        if caption:
            d = ImageDraw.Draw(img)
            d.rectangle([0, H - BAR, W, H], fill=(24, 79, 149))
            d.text((24, H - BAR + 18), caption, fill=(252, 252, 251), font=font(24))
        self.frames.append((img, int(seconds * 1000)))

    def seconds(self) -> float:
        return sum(ms for _, ms in self.frames) / 1000

    def save_gif(self, path: Path, width: int) -> None:
        h = round(H * width / W)
        imgs = [f.resize((width, h), Image.LANCZOS).quantize(colors=128) for f, _ in self.frames]
        imgs[0].save(
            path,
            save_all=True,
            append_images=imgs[1:],
            duration=[ms for _, ms in self.frames],
            loop=0,
            optimize=True,
        )

    def save_mp4(self, path: Path, fps: int = 10) -> None:
        import imageio.v2 as imageio
        import numpy as np

        with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as w:
            for img, ms in self.frames:
                arr = np.asarray(img)
                for _ in range(max(1, round(ms / 1000 * fps))):
                    w.append_data(arr)


def wait_text(page: Page, text: str, timeout: float = 90) -> None:
    page.get_by_text(text, exact=False).first.wait_for(timeout=timeout * 1000)


def prob(page: Page) -> str:
    return page.get_by_test_id("stMetricValue").first.inner_text().strip()


def decision(page: Page) -> str:
    """What the interface says, so a caption can never claim a decision the screen does not show."""
    if page.get_by_text(re.compile(r"^Not proposed for the quarterly offer")).count():
        return "not proposed for the coupon"
    return "proposed for the coupon"


def tab(page: Page, name: str) -> None:
    page.get_by_role("tab", name=name).click()
    page.wait_for_timeout(2500)


def cards(page: Page, reel: Reel, which: int, seconds: float) -> None:
    page.goto(CARDS.resolve().as_uri())
    page.set_viewport_size({"width": W, "height": H})
    reel.add(page.locator("section.slide").nth(which).screenshot(), seconds)


def record(ui: str, api: str, headed: bool) -> Reel:
    reel = Reel()
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome", headless=not headed)
        except Exception:  # noqa: BLE001 - fall back to Edge (ships with Windows)
            browser = p.chromium.launch(channel="msedge", headless=not headed)
        page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        cards(page, reel, 0, 6)

        app = f"{ui.rstrip('/')}/~/+/"  # the bare Streamlit app, without the Cloud frame
        page.goto(app, wait_until="domcontentloaded")
        wake = page.get_by_role("button", name="Yes, get this app back up!")
        if wake.count():
            wake.click()
        page.set_default_timeout(120_000)
        wait_text(page, "Probability of ordering again")  # the prediction tab has rendered
        page.wait_for_timeout(3000)
        reel.add(page.screenshot(), 9, f"1 · Retention: P(order again in 90 days) = {prob(page)}, "
                 f"{decision(page)}, with the 3 principal factors")  # fmt: skip

        page.get_by_role("slider").nth(3).focus()  # "Spend in last 180 days ($)"
        page.keyboard.press("Home")  # the slider minimum
        page.wait_for_timeout(4000)
        reel.add(page.screenshot(), 8, f"What-if: 180-day spend at minimum -> P = {prob(page)}, "
                 f"{decision(page)}")  # fmt: skip

        page.goto(f"{app}?basket=22916,22918", wait_until="domcontentloaded")
        wait_text(page, "Probability of ordering again")
        tab(page, "Basket")
        wait_text(page, "Because you added")
        page.wait_for_timeout(1500)
        reel.add(page.screenshot(), 10,
                 "2 · Basket: each suggestion names its rule, confidence and lift")  # fmt: skip

        tab(page, "Assistant")
        box = page.get_by_test_id("stChatInputTextArea")
        box.fill("Combien de jours pour une livraison au Nouveau-Brunswick ?")
        box.press("Enter")
        wait_text(page, "Delivery and Shipping Policy")
        page.wait_for_timeout(2000)
        reel.add(page.screenshot(), 10,
                 "3 · Assistant: French question, answered from our policy + source")  # fmt: skip

        box.fill("What did customer BM-2CF5F863 order?")
        box.press("Enter")
        wait_text(page, "another customer")
        page.get_by_text("another customer").last.scroll_into_view_if_needed()
        page.wait_for_timeout(1500)
        reel.add(page.screenshot(), 7,
                 "Asked about another customer: refused before any model runs")  # fmt: skip

        tab(page, "Monitoring")
        wait_text(page, "Overall status")
        page.wait_for_timeout(3000)
        status = page.get_by_text(re.compile(r"Overall status")).first.inner_text()
        window = page.get_by_text(re.compile(r"Window: last \d+ predictions")).first.inner_text()
        n = re.search(r"last (\d+) predictions", window)
        reel.add(page.screenshot(), 8, f"4 · Monitoring, live: {status.strip()} "
                 f"({n.group(1) if n else '?'} predictions in the window)")  # fmt: skip
        page.mouse.move(W * 0.6, H * 0.6)  # over the main pane, which owns the scrollbar
        page.mouse.wheel(0, 900)
        page.wait_for_timeout(2500)
        reel.add(page.screenshot(), 6,
                 "4 · Operations, live: requests and latency per endpoint")  # fmt: skip

        page.goto(f"{api.rstrip('/')}/version")
        page.evaluate("document.body.style.zoom = '2.2'")
        page.wait_for_timeout(1500)
        reel.add(
            page.screenshot(),
            5,
            "/version: the deployed commit = the head of master (CD smoke test checks it)",
        )

        cards(page, reel, 1, 7)
        browser.close()
    return reel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ui", default=EV["ui_url"])
    ap.add_argument("--api", default=EV["api_url"])
    ap.add_argument("--out", default=str(ROOT / "docs" / "media"))
    ap.add_argument("--gif-width", type=int, default=960)
    ap.add_argument("--headed", action="store_true", help="show the browser while recording")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reel = record(a.ui, a.api, a.headed)
    reel.save_gif(out / "demo.gif", a.gif_width)
    reel.save_mp4(out / "demo.mp4")
    for f in ("demo.gif", "demo.mp4"):
        print(f"{out / f}: {(out / f).stat().st_size / 1e6:.1f} MB")
    print(f"{len(reel.frames)} frames, {reel.seconds():.0f} s")


if __name__ == "__main__":
    main()
