"""Assemble a raw-image-only sheet for the unresolved Cam3 label review."""

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/black_next_stage_manifest_2026-10-27.csv"
OUT = ROOT / "docs/research/images/black_101155_raw_review_2026-10-27.jpg"


def main():
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "development_single_fault"
                and "10-11-55-" in r["id"]]
    assert len(rows) == 6 and {r["cam"] for r in rows} == {f"Cam{i}" for i in range(1, 7)}
    rows.sort(key=lambda r: r["cam"])
    cell_w, crop_h, title_h = 1600, 234, 34
    canvas = Image.new("RGB", (cell_w * 2, (crop_h + title_h) * 3), "#f1f1f1")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("arial.ttf", 22)
    for idx, row in enumerate(rows):
        with Image.open(row["source_path"]) as source:
            image = source.convert("RGB")
            h, w = image.height, image.width
            band = image.crop((0, int(h * .34), w, int(h * .60)))
            band = band.resize((cell_w, crop_h), Image.Resampling.LANCZOS)
        x = (idx % 2) * cell_w
        y = (idx // 2) * (crop_h + title_h)
        draw.text((x + 12, y + 5), f"{row['cam']}  {row['captured_at']}", fill="#111111", font=font)
        canvas.paste(band, (x, y + title_h))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, quality=90, subsampling=0)
    print(f"Wrote {OUT}: six raw bands, no labels or model predictions")


if __name__ == "__main__":
    main()
