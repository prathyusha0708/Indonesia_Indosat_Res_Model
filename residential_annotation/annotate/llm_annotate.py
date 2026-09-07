"""
NEW pipeline stage -- residential/commercial/shophouse classification via a
cheap multimodal LLM (Gemini 2.5 Flash by default, via OpenRouter), run on
the crops/build_crops.py output.

For each building, ALL of its clear (non-occluded) view crops are sent
TOGETHER in a single request (so the model can cross-reference angles), and
many (building, model) pairs are processed concurrently (bounded by
--concurrency). Supports --models (comma-separated) to run the SAME
buildings through several models side-by-side for comparison, without
needing a separate script.

Per crops/build_crops.py's own convention, each crop is aimed at the target
building via a RED center line (mark_center()) -- the model is told to
annotate the FRONTMOST building that line passes through, not anything
behind it. A second, ORANGE line (mark_occluder()) may also appear marking a
*different*, potentially-blocking building -- the model is told to ignore it.

SIMPLIFIED (per direct instruction): no floor-band split, no bounding boxes
at all -- just ONE classification label per building, a confidence, and a
one-line reasoning. Nothing spatial is requested or saved.

Classification rule (explicit, per direct instruction -- false "commercial"
flags are costly, so the model is told to prefer "unsure" over guessing).
The only distinction that matters is RESIDENTIAL vs NON-RESIDENTIAL -- WHAT
KIND of non-residential use it is does not matter (a shop, office, mosque,
school, warehouse, market, government building, etc. are all equally
"non-residential"):
  - "residential" -- entire visible facade is residential.
  - "shophouse"   -- clearly a MIX of residential and non-residential use in
                      the same building (most commonly a non-residential
                      ground floor with residential floor(s) above, or the
                      reverse).
  - "commercial"  -- ONLY when confident the whole building is non-residential.
  - "unsure"      -- can't confidently tell -- never coerced into a guess.

Exception: a non-commercial UTILITY ground floor (cellar, garage, carport,
plain storage/structural base -- not living space, but also not any other
use either) does not by itself make a building "shophouse" or "commercial" --
a building with a plain garage ground floor and purely residential floor(s)
above is still "residential".

Reads crops_manifest.csv (the authoritative source for which SPECIFIC views
were clear -- a has_clear_view/<building_id>/ folder can still contain some
occluded images for that same building, so folder placement alone is not
enough) and only sends occluded=="False" rows per building.

Also joins the FULL building geometry/attributes (from buildings_aoi.geojson,
the same file crops/build_crops.py itself reads) into the final CSV, keyed by
building_id -- every column that file has (footprint polygon as WKT, height,
height_quality, confidence, area, etc.), prefixed "bldg_" so nothing collides
with the LLM's own same-named fields (its own "confidence" means something
completely different from the source data's "confidence"). This join is done
directly here, not as a separate post-processing script -- see
load_building_geometry()/write_csv().

Writes outputs/annotations.csv, one row per (building_id, model).
"""
import argparse
import asyncio
import base64
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import geopandas as gpd
import httpx
import pandas as pd

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_BUILDINGS_GEOJSON = "../output/buildings_aoi.geojson"  # same file crops/build_crops.py reads

VALID_CLASSES = {"residential", "shophouse", "commercial", "unsure"}
BASE_FIELDNAMES = ["building_id", "model", "status", "classification", "confidence", "target_visible",
                   "reasoning", "cost_usd", "images_used", "error"]


def load_dotenv(path: Path = Path(__file__).resolve().parent.parent / ".env") -> None:
    """Tiny manual .env loader (no python-dotenv dependency) -- only sets a
    variable if it isn't already present in the real environment, so an
    explicit `export` always wins over the file. Silently does nothing if
    the file doesn't exist."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


load_dotenv()

SYSTEM_PROMPT = """You are analyzing Google Street View crops of a single TARGET building in \
Jakarta, Indonesia, for a residential-facade annotation pipeline. You will be shown 1-3 images, \
all aimed at the SAME target building from different nearby camera positions.

In every image, a RED vertical line marks the exact compass bearing toward the target building's \
centroid. The target building is the FRONTMOST building that red line passes through -- i.e. the \
building closest to the camera at that bearing. If the red line visually appears to pass through \
MORE THAN ONE building at different depths (e.g. a nearer building in the foreground, and a \
further/taller one visible behind it, through a gap, over a rooftop, or through trees) -- and this \
happens often -- you MUST pick the NEARER one as the target, every time, even if the further one is \
larger, more visually prominent, or easier to describe. Never analyze or describe the further \
building. If an image also has an ORANGE vertical line, that marks a DIFFERENT building that may be \
partially blocking the view -- it is NOT the target, ignore it.

Classify the target building's use, using ONLY these four labels. The only distinction that \
matters is RESIDENTIAL vs NON-RESIDENTIAL -- what KIND of non-residential use it is does not \
matter and does not need to be identified (a shop, office, mosque, school, warehouse, market, \
government building, etc. are all equally "non-residential" for this task):
- "residential": the entire (counted, see exception below) visible facade is residential in \
character (windows with curtains/blinds, balconies, laundry hanging, no storefront/signage/ \
roller-shutter/institutional signage).
- "shophouse": the building is CLEARLY a MIX of residential and non-residential use -- most \
commonly a non-residential ground floor with residential floor(s) above, or the reverse. The \
non-residential part can be any use (shop, office, etc.), not only a shop.
- "commercial": ONLY when you are CONFIDENT the entire visible building is non-residential -- ANY \
non-residential use counts (an office tower, warehouse, standalone shop/showroom, market, mosque/ \
church/temple, school, government building, etc.; a full-glass curtain-wall facade covering the \
whole building with no balconies/curtains/laundry anywhere, or a large flexi banner/billboard \
covering all or most of the facade, are also strong non-residential cues on their own).
- "unsure": the facade isn't legible enough, too occluded/distant/dark, or you genuinely cannot \
confidently tell residential vs non-residential. IMPORTANT: false "commercial" labels are much \
more costly to this project than a missed classification -- if you are not confident it is purely \
non-residential, prefer "unsure" over "commercial". Never guess.

EXCEPTION -- non-commercial utility ground floors (a cellar, garage, carport, or plain storage/ \
structural base): if the ground floor is NOT itself living space but is also NOT any other kind of \
use (no commercial signage, shutter, storefront glazing, or institutional signage -- just a utility \
space nobody lives or works in), do NOT treat this as making the building a "shophouse" or \
"commercial" -- a building with a plain garage/cellar ground floor and purely residential floor(s) \
above is still "residential".

Return STRICT JSON only -- no markdown fences, no prose outside the JSON object -- matching \
EXACTLY this schema:
{
  "target_visible": true or false,
  "classification": "residential" | "shophouse" | "commercial" | "unsure",
  "confidence": 0.0-1.0,
  "reasoning": "one short sentence"
}"""


def encode_image_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def extract_json(text: str) -> dict:
    """Model responses sometimes wrap JSON in ```json fences despite instructions --
    strip those before parsing. Raises on failure -- caller stores the raw text
    for debugging rather than crashing the whole batch."""
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.DOTALL)
    if m:
        t = m.group(1)
    else:
        m2 = re.search(r"\{.*\}", t, re.DOTALL)  # first {...} block, in case of stray prose
        if m2:
            t = m2.group(0)
    return json.loads(t)


def _enforce_schema(parsed: dict) -> dict:
    """Validate classification -- raises on anything invalid so the caller
    retries rather than silently saving garbage."""
    cls = parsed.get("classification")
    if cls not in VALID_CLASSES:
        raise ValueError(f"model returned invalid classification: {cls!r}")
    return parsed


async def annotate_building(client: httpx.AsyncClient, api_key: str, model: str,
                             building_id: str, image_paths: list, sem: asyncio.Semaphore,
                             max_retries: int = 3) -> dict:
    content = [{"type": "text",
                "text": f"Target building_id: {building_id}. {len(image_paths)} image(s) follow, "
                        f"labeled Image 1..Image {len(image_paths)} in that order."}]
    for i, p in enumerate(image_paths, start=1):
        content.append({"type": "text", "text": f"Image {i}:"})
        content.append({"type": "image_url", "image_url": {"url": encode_image_data_url(p)}})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with sem:
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90.0)
                resp.raise_for_status()
                data = resp.json()
                raw_text = data["choices"][0]["message"]["content"]
                parsed = _enforce_schema(extract_json(raw_text))
                cost = data.get("usage", {}).get("cost")  # OpenRouter reports real $ cost per call
                return {"building_id": building_id, "model": model, "status": "ok", "cost_usd": cost,
                        "images_used": [str(p) for p in image_paths], **parsed}
            except Exception as e:  # noqa: BLE001 -- retry on anything transient, log the rest
                last_err = e
                if attempt < max_retries:
                    await asyncio.sleep(2.0 * attempt)
        return {"building_id": building_id, "model": model, "status": "error", "error": str(last_err),
                "images_used": [str(p) for p in image_paths]}


def load_clear_views_by_building(manifest_csv: str) -> dict:
    """building_id -> ordered list of image_path (str) for rows with
    status=="ok" and occluded=="False" (both stored as strings by csv.DictReader)."""
    df = pd.read_csv(manifest_csv, dtype=str)
    df = df[(df["status"] == "ok") & (df["occluded"] == "False")]
    out = {}
    for bid, grp in df.groupby("building_id"):
        out[bid] = list(grp.sort_values("rank")["image_path"])
    return out


def load_building_geometry(path: str) -> dict:
    """building_id -> dict of EVERY other column in buildings_aoi.geojson
    (footprint polygon, height, height_quality, confidence, area, etc.), all
    prefixed "bldg_" -- including the polygon itself, serialized to WKT since
    a CSV cell can't hold a Shapely object (load back with
    shapely.wkt.loads() or geopandas.GeoSeries.from_wkt()). Prefixing avoids
    any collision with the LLM's own same-named fields -- e.g. the source
    data's "confidence" (Open Buildings detection confidence) is completely
    different from the LLM's "confidence" (classification confidence)."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    other_cols = [c for c in gdf.columns if c not in ("building_id", "geometry")]
    out = {}
    for _, row in gdf.iterrows():
        rec = {f"bldg_{c}": row[c] for c in other_cols}
        rec["bldg_geometry_wkt"] = row.geometry.wkt if row.geometry is not None else ""
        out[row["building_id"]] = rec
    return out


def write_csv(results: list, out_path: Path, building_geo: dict = None):
    extra_fields = []
    if building_geo:
        sample = next(iter(building_geo.values()), {})
        extra_fields = list(sample.keys())
    fieldnames = BASE_FIELDNAMES + extra_fields

    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing_geo = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in BASE_FIELDNAMES}
            row["images_used"] = json.dumps(row["images_used"]) if row["images_used"] else ""
            if building_geo is not None:
                geo = building_geo.get(r["building_id"])
                if geo is None:
                    missing_geo += 1
                    geo = {}
                for k in extra_fields:
                    row[k] = geo.get(k, "")
            writer.writerow(row)
    if building_geo is not None and missing_geo:
        print(f"WARNING: {missing_geo}/{len(results)} rows had no matching building_id "
              f"in the buildings geojson -- geometry columns left blank for those.")


def print_comparison(results: list, models: list):
    """When >1 model was run, print a quick agreement summary + the specific
    buildings where classifications disagree -- the actual point of a
    side-by-side test, so it's visible without opening the CSV."""
    if len(models) < 2:
        return
    by_building = {}
    for r in results:
        by_building.setdefault(r["building_id"], {})[r["model"]] = r
    agree = disagree = 0
    diffs = []
    for bid, by_model in by_building.items():
        classes = {m: by_model[m].get("classification", "ERROR") for m in models if m in by_model}
        if len(set(classes.values())) == 1:
            agree += 1
        else:
            disagree += 1
            diffs.append((bid, classes))
    total = agree + disagree
    print(f"\n=== Model comparison: {models} ===")
    print(f"Agreement: {agree}/{total} buildings ({100*agree/total:.0f}%)" if total else "no buildings")
    if diffs:
        print(f"\nDisagreements ({len(diffs)}):")
        for bid, classes in diffs:
            print(f"  {bid}: " + ", ".join(f"{m}={c}" for m, c in classes.items()))


async def main_async(args):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set -- put it in residential_annotation/.env "
                  "(see llm_annotate.load_dotenv) or export it.")

    models = [m.strip() for m in args.models.split(",")] if args.models else [args.model]

    clear_by_building = load_clear_views_by_building(args.manifest)
    building_ids = list(clear_by_building.keys())
    if args.sample_n:
        import random
        random.Random(args.seed).shuffle(building_ids)
        building_ids = building_ids[: args.sample_n]
    print(f"{len(clear_by_building)} buildings have >=1 clear view in the manifest; "
          f"processing {len(building_ids)} of them x {len(models)} model(s) "
          f"= {len(building_ids) * len(models)} calls.", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    total_calls = len(building_ids) * len(models)
    t0 = time.monotonic()

    async with httpx.AsyncClient() as client:
        async def run_one(bid, model):
            nonlocal done
            paths = [Path(p) for p in clear_by_building[bid]]
            r = await annotate_building(client, api_key, model, bid, paths, sem)
            done += 1
            if done % args.progress_every == 0 or done == total_calls:
                elapsed = time.monotonic() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  {done}/{total_calls} calls done ({rate:.2f}/s, {elapsed:.0f}s elapsed)", flush=True)
            return r

        tasks = [run_one(bid, model) for bid in building_ids for model in models]
        results = await asyncio.gather(*tasks)

    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] == "error"]
    grand_total_cost = 0.0
    for model in models:
        counts = {}
        model_cost = 0.0
        for r in ok:
            if r["model"] == model:
                counts[r["classification"]] = counts.get(r["classification"], 0) + 1
                model_cost += r.get("cost_usd") or 0.0
        grand_total_cost += model_cost
        print(f"\n[{model}] {sum(counts.values())} ok, "
              f"{sum(1 for r in err if r['model']==model)} errors. Counts: {counts}. "
              f"Real cost (OpenRouter-reported): ${model_cost:.4f}")

    if len(models) > 1:
        print(f"\nTotal real cost across all models: ${grand_total_cost:.4f}")

    print_comparison(results, models)

    print(f"\nLoading building geometry from {args.buildings_geojson} for the join ...")
    building_geo = load_building_geometry(args.buildings_geojson)
    print(f"  -> {len(building_geo)} buildings loaded")

    out_path = Path(args.out_csv)
    write_csv(results, out_path, building_geo)
    print(f"\nWrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="outputs/crops_manifest.csv")
    p.add_argument("--buildings-geojson", default=DEFAULT_BUILDINGS_GEOJSON,
                    help="joined into the final CSV (full geometry + all attributes, bldg_-prefixed)")
    p.add_argument("--out-csv", default="outputs/annotations.csv")
    p.add_argument("--model", default=DEFAULT_MODEL, help="single model to run (ignored if --models given)")
    p.add_argument("--models", default=None,
                    help="comma-separated model slugs to run side-by-side for comparison, "
                         "e.g. google/gemini-2.5-flash,qwen/qwen3-vl-235b-a22b-instruct")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--sample-n", type=int, default=None, help="process a random subset, for testing")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress-every", type=int, default=20)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
