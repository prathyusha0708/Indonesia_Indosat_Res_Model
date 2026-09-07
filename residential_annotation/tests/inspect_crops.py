"""
Manual visual check for build_crops.py output. Groups the manifest by
building_id and lays its (up to 3) marked panorama views side-by-side in one
image, each labeled with rank/distance/occluded -- so a human can tell in one
glance whether the 3 independent vantage points agree on a real, recognizable
building (good footprint, just check framing) or don't (likely bad building
footprint data), per the reason this per-building-multi-view setup exists.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, out.width, 40], fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(255, 255, 0), font=font)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="outputs/crops_manifest.csv")
    p.add_argument("--out-dir", default="outputs/inspect")
    p.add_argument("--limit", type=int, default=5, help="Number of BUILDINGS (not rows) to render")
    p.add_argument("--thumb-width", type=int, default=500)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        all_rows = list(csv.DictReader(f))

    by_building = defaultdict(list)
    for row in all_rows:
        by_building[row["building_id"]].append(row)

    building_ids = list(by_building.keys())[: args.limit]

    for bid in building_ids:
        views = [r for r in by_building[bid] if r["status"] == "ok"]
        if not views:
            print(f"{bid}: no usable views, skipping")
            continue
        views.sort(key=lambda r: int(r["rank"]))

        thumbs = []
        for r in views:
            img = Image.open(r["image_path"]).convert("RGB")
            th = int(img.height * args.thumb_width / img.width)
            thumb = img.resize((args.thumb_width, th))
            tag = (f"#{r['rank']} dist={r['distance_m']}m "
                   f"{'OCCLUDED' if r['occluded'] == 'True' else 'clear'}"
                   f"{' (forced)' if r.get('forced_clear') == 'True' else ''}")
            thumbs.append(label(thumb, tag))

        max_h = max(t.height for t in thumbs)
        combo = Image.new("RGB", (sum(t.width for t in thumbs) + 10 * (len(thumbs) - 1), max_h), (30, 30, 30))
        x = 0
        for t in thumbs:
            combo.paste(t, (x, 0))
            x += t.width + 10

        out_path = Path(args.out_dir) / f"{bid}.jpg"
        combo.save(out_path, "JPEG", quality=88)
        h = views[0]
        print(f"{bid}  height_m={h.get('height_m', ''):>6s} height_quality={h.get('height_quality', ''):10s} "
              f"({len(views)} views) -> {out_path}")


if __name__ == "__main__":
    main()
