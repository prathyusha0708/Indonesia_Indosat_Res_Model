# Residential Facade Annotation Pipeline — Plan

> **UPDATE (post-initial-testing): buildings source switched from Overture to Google Open
> Buildings.** After building/testing the pipeline against the original Overture-sourced
> `buildings_aoi.geojson` (7,916 buildings, ~4% with a height value), we switched to a local
> Jakarta-wide extract of Google Open Buildings 2.5D Temporal
> (`jakarta_buildings_2023_heights.geojson`, via `pull_buildings_openbuildings.py`), because 96%+
> of buildings there have a `height_m` estimate (+ `height_quality` flag) vs. Overture's ~4% --
> directly useful, and this pipeline never actually relied on Overture's sparse `class`/`subtype`
> tags for anything beyond soft stratification signal. New canonical `buildings_aoi.geojson`/`.csv`
> has **9,506 buildings** for the same 5 km² AOI. Schema changed: `id` -> `building_id` (both
> present in the new file, so old code didn't break), `class`/`subtype`/`num_floors` -> gone,
> replaced with `height_m`/`height_quality`/`height_valid`/`confidence`/`full_plus_code`.
> `crops/build_crops.py` and `tests/inspect_crops.py` updated accordingly. Old Overture output
> backed up in `output_overture_backup/`. Every mention of "Overture" below this line reflects
> the *original* design reasoning (still valid as reasoning, e.g. why we don't hard-filter on
> tags) -- just not the current literal data source anymore. See `README.md` for the up-to-date
> pull instructions.

## Context

We already have (from `Tower Identification_exp/output/buildings_aoi.geojson`) 7,916 building
polygons + centroids covering a 5 km² circle around the Indosat point in central Jakarta. The
goal now is: **for every building that has any residential use, draw a bounding box around the
residential portion of its street-facing facade** (full box if fully residential, a partial
"floor band" box if it's a mixed-use building like a shophouse, no box at all if it's not
residential). This is a new, genuinely different task from tower-finding, but the user
explicitly asked to reuse the engineering patterns already proven in
`Tower_Identification_Pipeline/` — reusing patterns and vendored code where they truly apply,
and diverging with clear reasoning where this problem is different.

**Answering the open question from the discussion** — what does the existing tower pipeline
actually do with SAM3/YOLO, precisely: it runs **SAM3 zero-shot directly in production** (no
training data at all — it's a text-prompted foundation model) as its main, "trusted" detector.
Separately, it also has a **fine-tuned YOLO model** that is a drop-in cheaper alternative — but
that YOLO model was fine-tuned **externally, through an undocumented process not present in the
repo** (confirmed by reading its full git history — only 3 commits, none about dataset/training).
So the tower repo does **not** contain an example of "use SAM3 to bootstrap a YOLO training set" —
that idea is entirely the user's own proposal here, not something to copy from the existing repo.
It's a sound, standard approach (teacher/student pseudo-labeling), just new work, not reuse.

**Also confirmed in this sandbox** (relevant to planning, not just theory): we already have
working GPU access (**NVIDIA A40, 46 GB VRAM** — notably stronger than the tower repo's own T4,
14.6 GB) and an already-authorized Hugging Face token with **verified gated access to
`facebook/sam3`** (test download of `config.json` succeeded). So SAM3 can run directly in this
environment for the seed-labeling step — no Modal/cloud GPU detour needed for a pilot this size.

## What's genuinely reusable vs genuinely new

| Tower pipeline piece | Reuse plan | Why |
|---|---|---|
| `coverage/streetview_coverage.py` | **Vendor-copy as-is** into the new pipeline | Pure, self-contained, no tower-specific logic — it just discovers Street View panoramas in a region. Exactly what we need too. |
| `detection/direction_view.py` (`slice_direction`) | **Vendor-copy as-is** | Already accepts an *arbitrary* `heading_deg`, not just cardinal N/E/S/W — fully generic equirect column-slicer. No modification needed. |
| `bearings/sam3_detect.py` (model loading, `nms_dedup`, `suppress_by_negatives/hard_negatives`, `run_concepts_batched`, `precompute_text_embeds`) | **Vendor-copy the generic detection engine**, swap only the prompt lists + add a new post-processing step | This is a fully generic "zero-shot detect with positive/soft-negative/hard-negative text prompts + suppression" engine — nothing tower-specific in the mechanics. |
| `bearings/yolo_direct_scan.py` (batched `model.predict(list_of_crops, imgsz=..., conf=...)`, `--num-shards`/`--shard-index`) | **Reuse the batching/sharding skeleton** for our production YOLO inference driver | Generic Ultralytics batching pattern; only the post-box-to-output math changes. |
| `validation/tower_llm_validate.py` (`image_content()`, `extract_json()`, `call_verdict()` retry loop, litellm usage) | **Reuse the generic LLM-vision-JSON-verdict engine** for an optional QA spot-check | Generic pattern; only the prompt + parsed schema change. API key sourcing changes (see below). |
| Point-to-point lat/lon **bearing** helper | **Does not exist anywhere in the tower repo** (confirmed by direct code read — every "bearing" there is pixel-position-in-crop → compass angle, never lat/lon-pair → compass angle) | Must write this fresh. Small, ~10 lines. |
| Triangulation (`sequential_ray_fit_v2.py`, `tile_and_fit.py`, `cluster_and_fit.py`) | **Not needed at all** | Triangulation exists to find an *unknown* tower location from multiple rays. We already know the exact building location (Overture centroid+polygon) — nothing to triangulate. |
| Hardcoded absolute host paths (`/home/azureuser/manoj/...`, sibling-app DB lookups) | **Explicitly not copied** — the tower repo's own README lists this as a flagged wart | Replace with CLI args / env vars so the new pipeline is portable and self-contained. |

Vendored files get a one-line header comment noting they're copied from
`Tower_Identification_Pipeline/<path>` for provenance; a shared installable package is a
reasonable *future* improvement but is over-engineering for this pilot.

## Key design decisions (and why not the alternatives)

**1. One aimed crop per building, not 4 generic cardinal crops per panorama.**
The tower pipeline scans all 4 cardinal directions per pano because the tower's location is
*unknown* — it needs omnidirectional coverage to later triangulate it. We already know each
building's exact centroid and full footprint polygon. So instead: for each building, find its
nearest usable panorama, compute the exact compass bearing from that panorama to the building,
and take **one tightly-framed crop aimed exactly at that building**. This is the efficiency the
user described ("a clear way of stopping to scan whole panoramic view") — confirmed as the right
call, not just a suggestion to consider.

**2. Use the full building polygon (not just centroid) to auto-size the crop's field of view.**
We already downloaded full Overture footprint polygons, not just centroids. For each building,
project every polygon vertex's bearing from the chosen panorama; the angular spread
(max − min, wraparound-safe) plus a fixed padding (e.g. +25%, clamped to [40°, 100°]) becomes the
crop's field of view. This auto-frames each building tightly regardless of its size/shape/distance
— no manual per-building tuning, and it's basically free since we already have the geometry.

**3. Cheap occlusion pre-filter using the buildings we already have.**
Before spending any GPU time on a building, check whether another building's polygon crosses the
straight line from the chosen panorama to this building's centroid (shapely line-intersection
against a spatial index of the same buildings GeoDataFrame). If blocked, flag
`occluded=True` and skip (or fall back to the 2nd-nearest panorama). Cheap, no new data needed,
avoids wasting SAM3/YOLO calls on facades that can't actually be seen.

**4. Residential region = a single vertical "floor band" on the facade, not a free-form mask.**
Per your confirmation that shophouses split by floor (not left/right), we don't need SAM3's raw
segmentation mask — we reduce it to two numbers: `y_top_frac`, `y_bottom_frac` (0–1, fraction of
crop height) marking where the residential band starts/ends. This directly encodes:
- fully residential → `(0.0, 1.0)`
- ground-floor shop / upper-floor housing (the common shophouse case) → e.g. `(0.0, 0.5)`
  (band anchored at the top)
- not residential at all → no row at all (per your rule: don't annotate)

We keep both edges as free variables (not hardcoded "always top-anchored") so the same schema
also covers the rarer inverse case (e.g. residential ground floor, commercial mezzanine above).
This single-rectangle target is also *exactly* the kind of large, simple, high-signal object a
small fine-tuned YOLO detector converges on quickly and cheaply — much easier than the tower
model's task (tiny, distant objects, hence its `imgsz=1600`); our objects fill most of the frame,
so a much smaller `imgsz` (~896) and a nano-sized model are enough, which matters directly for
your "don't want to spend a lot" constraint.

**5. Why street view, not the tower repo's satellite sub-project.**
The satellite/ sub-project's own analysis in the tower repo already found that even coarse
tower-shape visibility from top-down satellite imagery is capped by canopy/shadow limitations.
Residential-vs-commercial cues (curtains, laundry, balconies, shop signage, shutters) are
facade-level, street-height details — essentially invisible from directly overhead. Street View
is the only viable image source for this specific task; satellite is not a fallback option here.

**6. Why SAM3-seed → fine-tune YOLO, not "just run SAM3 on everything" or "just hand-label everything."**
- *Not* "run SAM3 on everything": this is exactly the cost you said you want to avoid — SAM3 is a
  large foundation model, much slower per image than a small fine-tuned detector, and you were
  explicit about not wanting to pay that cost at full scale.
- *Not* "hand-label everything from scratch": slow, and throws away a genuinely useful zero-shot
  prior. SAM3's proposed band only needs a human to *correct* (drag one line, or click
  "not residential"/"fully residential"), which is much faster than drawing boxes from nothing —
  the same "trusted zero-shot base, humans only fix mistakes" philosophy the tower repo itself
  uses.
- A human review pass on the seed set (not skipped) matters because training on SAM3's raw,
  uncorrected output would cap YOLO's accuracy at (or below) SAM3's zero-shot accuracy, and bake
  in any systematic SAM3 mistakes.

**7. Overture's own `class`/`subtype` fields as a hard pre-filter — considered and rejected.**
In our own pulled data, the vast majority of buildings have null `class`/`subtype` (only a small
minority are tagged at all) — too sparse to use as a hard residential/non-residential filter.
We do use it as a **soft signal**: for *stratifying* the SAM3 seed sample (so the ~800 seed images
aren't accidentally all one building type) and as an extra output column for later cross-checking.

## Pipeline stages

Proposed location: `Tower Identification_exp/residential_annotation/` (a new subfolder next to
`pull_buildings_5km.py`/`output/`, since this directly consumes and will directly extend that
folder's output — not a new top-level sibling to `Tower_Identification_Pipeline`, to avoid
scattering related work). Layout:

```
residential_annotation/
  coverage/streetview_coverage.py        # vendored as-is
  geometry/
    direction_view.py                    # vendored as-is (slice_direction)
    bearing.py                           # NEW: point-to-point bearing + polygon angular-span/FOV calc
    line_of_sight.py                     # NEW: occlusion check
  crops/build_crops.py                   # NEW: match building -> nearest usable pano, slice one crop each
                                          #      -> crops_manifest.csv + crops/<building_id>.jpg
  labeling/
    sam3_residential_detect.py           # adapted from bearings/sam3_detect.py: new prompts + band post-processing
    sample_seed_set.py                   # NEW: stratified sample selection for SAM3 seeding
    review_ui.py                         # adapted from validation/tower_validate_ui.py: FastAPI, one draggable band-line per image
  dataset/build_yolo_dataset.py          # NEW: reviewed_labels.csv -> YOLO images/labels/data.yaml
  train/train_yolo.py                    # NEW: ultralytics fine-tune driver (nano model, imgsz~896, single class)
  inference/yolo_residential_scan.py     # adapted from bearings/yolo_direct_scan.py: batched production inference
  postprocess/merge_into_buildings.py    # NEW: writes residential_fraction/bbox back into buildings_aoi.geojson/csv
  validation/llm_spotcheck.py            # adapted from validation/tower_llm_validate.py: optional cheap QA sample
  run_pipeline.py                        # NEW: orchestrator mirroring Tower_Identification_Pipeline/run_pipeline.py
  README.md
```

### Step-by-step

1. **Coverage** — run vendored `streetview_coverage.py --circle-center <lat> <lon> --circle-radius-m 1361.6` — **not** the raw 1261.6m AOI radius: padded by the 80m max pano-to-building match distance (below) plus a 20m safety margin, so a building sitting right at the AOI's edge can still be matched to a legitimate nearby panorama that happens to fall just outside the buildings circle. Gets every panorama in that padded area: `pano_id, lat, lon, heading_deg, ...`.
2. **Match + aim + crop** (`crops/build_crops.py`):
   - Load `buildings_aoi.geojson` + the coverage CSV, project both to the same UTM CRS already used in `pull_buildings_5km.py` (EPSG:32748).
   - For each building: find the nearest panorama within a max distance (default **80 m** — tighter than the tower repo's 400m sight limit, since we need a legible facade, not just tower visibility; configurable).
   - Compute the bearing from that pano to the building centroid (new `geometry/bearing.py`, planar/UTM formula — exact at these short distances, consistent with the repo's own `to_local_xy` planar-math style).
   - Compute FOV from the building polygon's angular spread as seen from the pano (design decision #2 above).
   - Occlusion check against other buildings (design decision #3); if blocked, try the 2nd-nearest pano, else mark `occluded=True` and skip.
   - Call vendored `slice_direction(pano_img, heading_deg=bearing, pano_heading_deg=pano.heading, fov_h_deg=computed_fov)`, save crop, append a row to `crops_manifest.csv` (`building_id, pano_id, bearing_deg, fov_deg, distance_m, occluded, class, subtype, num_floors, crop_path`).
3. **Stratified seed sample** (`labeling/sample_seed_set.py`) — pick **~800** crops (default, adjustable) stratified by Overture `class`/`subtype` (where present), footprint-area tercile, and distance tercile, so the seed set isn't skewed to one building type.
4. **SAM3 zero-shot seeding** (`labeling/sam3_residential_detect.py`), run locally on the confirmed A40:
   - Positive prompts (residential cues): e.g. "residential windows with curtains", "apartment balcony", "laundry hanging on a balcony", "house window with awning".
   - Negative/hard-negative prompts (commercial cues): e.g. "shop signage and storefront", "roller shutter door", "commercial glass storefront".
   - If nothing scores above threshold → no residential region (skip, matches your "don't annotate" rule) — cheap early exit, no wasted downstream work.
   - Else: take the union of surviving positive mask(s), compute `y_top_frac`/`y_bottom_frac` from the mask's vertical extent → this is the seed label.
5. **Human review UI** (`labeling/review_ui.py`, FastAPI, adapted from `tower_validate_ui.py`'s serving pattern) — each seed image shown with SAM3's proposed band overlaid; reviewer drags the top/bottom edge (or clicks "fully residential" / "not residential") — much faster than drawing from scratch. Writes `reviewed_labels.csv`.
6. **Build YOLO dataset** (`dataset/build_yolo_dataset.py`) — convert `reviewed_labels.csv` into standard Ultralytics format: one full-width box per image (`x_center=0.5, width=1.0, y_center=(top+bottom)/2, height=bottom-top`), single class `residential`; stratified 85/15 train/val split; write `data.yaml`.
7. **Fine-tune YOLO** (`train/train_yolo.py`) — small pretrained checkpoint (`yolo11n`/`yolov8n`), `imgsz=896` (facade crops are close-range, large-in-frame — no need for the tower model's `imgsz=1600`), single class. Report precision/recall/mAP on the held-out val split. Explicitly flagged in the README as "validated only within this pilot AOI" until tested on a second region — same caution the tower repo itself documents about its own YOLO model.
8. **Production inference at scale** (`inference/yolo_residential_scan.py`) — batched `model.predict()` over all building crops (reusing the batching/sharding skeleton from `yolo_direct_scan.py`), convert the predicted box back to `residential_fraction = height_frac`, no detection ⇒ not annotated.
9. **Merge back into the buildings dataset** (`postprocess/merge_into_buildings.py`) — join results into `buildings_aoi.geojson`/`.csv` as new columns: `residential_fraction, residential_bbox_norm, pano_id_used, bearing_deg, confidence, model_version` — this is the final deliverable, directly extending the dataset from `Tower Identification_exp`.
10. **Optional cheap QA spot-check** (`validation/llm_spotcheck.py`) — on a small re-sample (~150 buildings), reuse the litellm/JSON-verdict pattern from `tower_llm_validate.py` to sanity-check the fraction estimates independently. API key via `--api-key`/env var (not the sibling-DB lookup the tower repo uses, since that app doesn't exist here) — kept manual/optional since it costs money per call, same philosophy as the tower repo's own LLM validation step.

## Defaults (all overridable via CLI, called out explicitly since precision matters)

- Max pano-to-building distance: **80 m**
- FOV padding: **+25%** of angular span, clamped to **[40°, 100°]**
- SAM3 seed sample size: **~800 images**
- SAM3 score threshold: start at **0.40** (matches tower repo's own default, reusable starting point)
- YOLO model: **yolo11n** (nano), `imgsz=896`, single class `residential`
- Train/val split: **85/15**, stratified like the seed sample
- Production confidence threshold: start **0.35**, tune after validation
- QA spot-check sample: **~150 buildings**

## Verification plan

- After step 2: sanity-check a handful of crops visually (do they actually frame the right building, not a neighbor?) — spot check ~20 by eye.
- After step 4 (SAM3 seeding): print score-distribution histogram + a contact-sheet of ~30 random seed crops with their proposed bands overlaid.
- After step 7 (YOLO training): report precision/recall/mAP50 on held-out val split; require it to be in the same ballpark as the tower repo's own YOLO (~90%+) before trusting it at scale — if far below, seed set needs to grow or SAM3 prompts need tuning, not immediately blame YOLO.
- After step 9: recompute summary stats (residential building count, average residential_fraction, distribution by Overture class where tagged) and diff against a manual spot-check of ~30 buildings picked by eye from the map.
- Full run should be runnable end-to-end via `run_pipeline.py` on the existing 5 km² pilot AOI, matching the way `Tower_Identification_Pipeline/run_pipeline.py` chains its own stages.
