"""
THROWAWAY exploration script -- not part of the pipeline yet.

v3: two-stage SAM3 pass, per your instruction to skip vertical camera-geometry
cropping and instead let SAM3 isolate the building itself first:
  Stage 1 -- detect "a building" in the full (horizontally-aimed, full-height)
             crop. Pick the box nearest the image's horizontal center (our
             crop is already aimed at the target building's bearing, so its
             box should sit near the middle, not off to the side).
  Stage 2 -- crop to JUST that building box (small padding), re-embed, and
             run the residential/commercial "story" prompts ONLY inside it.
             This throws out sky and street-level pavement/pedestrians
             automatically -- no need to estimate building height/pitch
             ourselves; SAM3's own building detection does that framing.

v2 fixes folded in:
  - Prompts say "story" not "floor" (v2's "residential floor" was matching
    ground-level PAVEMENT texture -- "floor" is ambiguous between a
    building's level and the ground surface itself; both bad crops you
    flagged were literal pavement/crosswalk hits).
  - Adjudication margin tightened 0.15 -> 0.05 (0.15 let a single weak,
    wrong commercial hit veto a clearly-stronger residential one -- the
    row-house/laundry case).
  - Commercial boxes are no longer drawn in the output -- only used
    internally to adjudicate; you only want to see the residential result.

Still reuses the tower pipeline's generic SAM3 engine (model loading,
run_concepts, nms_dedup, suppress_by_negatives, containment_ratio) -- only
the prompts, the two-stage framing, and the verdict logic are new.
"""
import argparse
import glob
import os
import sys
import time

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                 "Tower_Identification_Pipeline"))
from bearings.sam3_detect import run_concepts, suppress_by_negatives  # noqa: E402
# Imported directly (not vendored) -- this script is still throwaway; the
# real labeling/sam3_residential_detect.py will vendor properly later.

BUILDING_PROMPTS = ["building facade", "front of a building"]
POSITIVE_PROMPTS = [
    "an upper story of a residential apartment building with windows and balconies",
    "a residential building story with air conditioning units and balconies",
    "a row of apartment housing units on one story of a building",
]
NEGATIVE_PROMPTS = [
    "a commercial building story with shop signage and storefronts",
    "an office building story with a glass curtain wall facade",
    "a ground-level row of shops with storefront displays",
]
MIN_SCORE = 0.40
BUILDING_PAD_RATIO = 0.05  # small padding around the detected building box
FULLY_RESIDENTIAL_HEIGHT_FRAC = 0.85
ADJUDICATION_MARGIN = 0.05  # tightened from 0.15 -- see v3 docstring note

CROPS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "crops")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "sam3_quick_test")


def pick_building_box(building_dets, img_w):
    """Of all detected 'building' boxes, pick the one nearest the image's
    horizontal center -- the crop is already aimed at the target building's
    bearing, so the RIGHT building should sit near the middle, not a
    neighbor caught at the edge of the frame."""
    if not building_dets:
        return None
    center_x = img_w / 2.0

    def box_center_dist(det):
        x1, _, x2, _ = det[0]
        return abs((x1 + x2) / 2.0 - center_x)

    return min(building_dets, key=box_center_dist)[0]


def pad_box(box, img_w, img_h, pad_ratio):
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * pad_ratio, (y2 - y1) * pad_ratio
    return (max(0, x1 - pw), max(0, y1 - ph), min(img_w, x2 + pw), min(img_h, y2 + ph))


def union_y_range(dets):
    if not dets:
        return None
    y1 = min(d[0][1] for d in dets)
    y2 = max(d[0][3] for d in dets)
    return y1, y2


def classify(res_dets, com_dets, img_h, iou_thresh=0.15, margin=ADJUDICATION_MARGIN):
    kept_res, dropped_res = suppress_by_negatives(res_dets, com_dets, iou_thresh=iou_thresh, margin=margin)
    res_range = union_y_range(kept_res)
    if res_range:
        frac = (res_range[1] - res_range[0]) / img_h
        if frac >= FULLY_RESIDENTIAL_HEIGHT_FRAC:
            return "FULLY_RESIDENTIAL", (0.0, 1.0), kept_res, dropped_res
        return "PARTIAL_RESIDENTIAL", (res_range[0] / img_h, res_range[1] / img_h), kept_res, dropped_res
    if com_dets:
        return "NOT_RESIDENTIAL", None, kept_res, dropped_res
    return "UNCLEAR", None, kept_res, dropped_res


def draw_result(image, building_box, res_dets_full_coords):
    """Draws the stage-1 building box thin/blue (so we can sanity-check
    stage 1 too) and only the surviving residential boxes in green -- no
    commercial boxes, per your ask: you only want to see residential."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
    except Exception:
        font = ImageFont.load_default()
    if building_box:
        draw.rectangle(building_box, outline=(60, 120, 255), width=2)
    for box, score, prompt in res_dets_full_coords:
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=4)
        draw.text((x1 + 4, y1 + 2), f"RES {score:.2f}", fill=(0, 200, 0), font=font)
    return out


def embed(model, processor, image):
    img_inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)
    return vision_embeds, img_inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="Only process the first N crops")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    crop_paths = sorted(glob.glob(os.path.join(CROPS_DIR, "*.jpg")))
    if args.limit:
        crop_paths = crop_paths[: args.limit]
    print(f"Found {len(crop_paths)} crops in {CROPS_DIR} (processing {len(crop_paths)})")

    print("Loading facebook/sam3 ...")
    from transformers import Sam3Model, Sam3Processor
    t0 = time.monotonic()
    model = Sam3Model.from_pretrained("facebook/sam3", device_map="auto")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    print(f"  loaded in {time.monotonic() - t0:.1f}s on {model.device}")

    verdict_counts = {}
    for path in crop_paths:
        name = os.path.basename(path)
        image = Image.open(path).convert("RGB")
        img_w, img_h = image.size
        t0 = time.monotonic()

        # --- Stage 1: find the building itself, discard sky/road framing ---
        vision_embeds, img_inputs = embed(model, processor, image)
        building_dets = run_concepts(model, processor, vision_embeds, img_inputs, BUILDING_PROMPTS, MIN_SCORE)
        building_box = pick_building_box(building_dets, img_w)

        if building_box is None:
            # Fallback: no building detected at all -- behave like before (whole crop).
            sub_image, offset = image, (0, 0)
        else:
            building_box = pad_box(building_box, img_w, img_h, BUILDING_PAD_RATIO)
            x1, y1, x2, y2 = [int(v) for v in building_box]
            sub_image = image.crop((x1, y1, x2, y2))
            offset = (x1, y1)

        # --- Stage 2: residential/commercial story prompts, INSIDE the building only ---
        sub_w, sub_h = sub_image.size
        sub_vision_embeds, sub_img_inputs = embed(model, processor, sub_image)
        res_dets = run_concepts(model, processor, sub_vision_embeds, sub_img_inputs, POSITIVE_PROMPTS, MIN_SCORE)
        com_dets = run_concepts(model, processor, sub_vision_embeds, sub_img_inputs, NEGATIVE_PROMPTS, MIN_SCORE)
        elapsed = time.monotonic() - t0

        verdict, band, kept_res, dropped_res = classify(res_dets, com_dets, sub_h)
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        band_str = f"band={band[0]:.2f}-{band[1]:.2f}" if band else "band=None"

        print(f"{name}: {verdict} ({band_str}) | building_box={'yes' if building_box else 'NONE(fallback=full)'} | "
              f"res_raw={len(res_dets)}({[round(d[1],2) for d in res_dets]}) "
              f"res_kept={len(kept_res)} res_dropped_by_com={len(dropped_res)} "
              f"com={len(com_dets)}({[round(d[1],2) for d in com_dets]}) | {elapsed:.2f}s")

        # Translate kept_res boxes from sub-image coords back to full-image coords for drawing.
        ox, oy = offset
        kept_res_full = [([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy], s, pr) for (b, s, pr) in kept_res]

        annotated = draw_result(image, building_box, kept_res_full)
        annotated.save(os.path.join(OUT_DIR, name))

    print(f"\nAnnotated images written to {OUT_DIR}/")
    print("Verdict counts:", verdict_counts)


if __name__ == "__main__":
    main()
