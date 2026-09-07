"""
Builds ONE side-by-side comparison image per building from a multi-model
outputs/annotations.csv (produced by llm_annotate.py --models a,b), so the
two models' calls can be visually eyeballed against each other.

SIMPLIFIED (per direct instruction): no bounding boxes, no filled bands --
llm_annotate.py no longer requests or saves anything spatial. This script
just shows every image a model actually used (1-3, not only the first --
a model answering off something only visible in image 2/3 would otherwise
look like it "made something up" not in the picture shown), with a header
bar above each model's images stating its classification (color-coded),
confidence, and one-line reasoning. Purely a side-by-side reading aid.

Usage: python3 annotate/visualize_comparison.py --annotations outputs/annotations.csv
Writes outputs/comparison/<building_id>.jpg
"""
import argparse
import ast
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CLASS_COLOR = {
    "residential": (60, 130, 255),
    "commercial": (230, 60, 230),
    "shophouse": (255, 190, 40),
    "unsure": (150, 150, 150),
}
PANEL_W = 500
HEADER_H = 90
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def _parse_json_list(v):
    if not v or (isinstance(v, float) and pd.isna(v)):
        return []
    try:
        return json.loads(v)
    except Exception:
        try:
            return ast.literal_eval(v)
        except Exception:
            return []


def build_image_tile(image_path: str, image_index: int) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((PANEL_W, PANEL_W * 3))  # crops are tall (portrait) -- cap width, keep aspect
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 60, 26], fill=(0, 0, 0))
    draw.text((6, 3), f"img {image_index}", fill=(255, 255, 255), font=_font(16))
    return img


def build_model_group(row: dict, model: str) -> Image.Image:
    """ALL images a model actually used (1-3), side by side, with ONE header
    above them showing the model's classification."""
    image_paths = _parse_json_list(row.get("images_used"))
    cls = row.get("classification", "?")
    color = CLASS_COLOR.get(cls, (255, 255, 255))
    tiles = [build_image_tile(p, i) for i, p in enumerate(image_paths, start=1)]
    if not tiles:
        tiles = [Image.new("RGB", (PANEL_W, PANEL_W), (40, 40, 40))]

    tiles_w = sum(t.width for t in tiles) + 4 * (len(tiles) - 1)
    tiles_h = max(t.height for t in tiles)
    group = Image.new("RGB", (tiles_w, tiles_h + HEADER_H), (20, 20, 20))
    x = 0
    for t in tiles:
        group.paste(t, (x, HEADER_H))
        x += t.width + 4

    draw = ImageDraw.Draw(group)
    conf = row.get("confidence", "?")
    draw.text((8, 6), f"{model}  ({len(tiles)} image(s) used)", fill=(255, 255, 255), font=_font(18))
    draw.text((8, 30), f"{cls}  (conf={conf})", fill=color, font=_font(24))
    reasoning = str(row.get("reasoning", ""))[:100]
    draw.text((8, 62), reasoning, fill=(200, 200, 200), font=_font(14))
    return group


def build_comparison_image(building_id: str, rows_by_model: dict, out_path: Path):
    groups = []
    for model, row in rows_by_model.items():
        if not _parse_json_list(row.get("images_used")):
            continue
        groups.append(build_model_group(row, model))
    if not groups:
        return False
    total_w = sum(g.width for g in groups) + 10 * (len(groups) - 1)
    max_h = max(g.height for g in groups)
    title_h = 34
    canvas = Image.new("RGB", (total_w, max_h + title_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), building_id, fill=(255, 255, 0), font=_font(22))
    x = 0
    for g in groups:
        canvas.paste(g, (x, title_h))
        x += g.width + 10
        if x < total_w:
            draw.line([x - 5, title_h, x - 5, canvas.height], fill=(100, 100, 100), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", default="outputs/annotations.csv")
    ap.add_argument("--out-dir", default="outputs/comparison")
    args = ap.parse_args()

    df = pd.read_csv(args.annotations, dtype=str)
    df = df[df["status"] == "ok"]
    models = sorted(df["model"].unique())
    print(f"{len(df)} ok rows across models: {models}")

    written = 0
    for bid, grp in df.groupby("building_id"):
        rows_by_model = {r["model"]: r.to_dict() for _, r in grp.iterrows()}
        out_path = Path(args.out_dir) / f"{bid}.jpg"
        if build_comparison_image(bid, rows_by_model, out_path):
            written += 1
    print(f"Wrote {written} comparison images to {args.out_dir}/")


if __name__ == "__main__":
    main()
