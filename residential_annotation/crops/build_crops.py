"""
NEW module. For every building, takes its 3 NEAREST Street View panoramas
(not just the single "best" one) and, for each, crops a FIXED 120-degree
(60 either side of the aim line) window around the building's centroid
bearing -- saved into their own per-building subfolder. See PLAN.md "Match +
aim + crop" (step 2, revised three times now).

v3 (per direct feedback after looking at real crops): for some buildings it
was genuinely unclear whether the marked line was pointing at a real
building or landing on nothing recognizable -- possibly bad footprint data
(a building_id that doesn't correspond to anything real/visible), possibly
just a bad single vantage point (blocked, too far, wrong angle). One view
can't tell those apart. Three independent views can: if all 3 land on the
same recognizable structure, that's strong evidence it's a real, well-placed
building; if they're inconsistent or none show anything, that's evidence of
a bad footprint, not just a bad viewpoint.

v4: briefly tried "don't crop at all, just mark the line on the full 360
panorama" (per feedback that the earlier per-building DYNAMIC FOV -- computed
from the footprint polygon's angular span -- was clamping down and cutting
buildings off). That's swung back to cropping, but with a FIXED fov_h_deg
(120 deg) instead of a per-building computed one -- generous and simple,
without needing to trust a polygon-derived width at all.

v5: back to a per-building SUBFOLDER (crops/<building_id>/<rank>_<pano_id>.jpg)
-- a brief flat-filename experiment made it hard to eyeball a building's 3
views together in a file browser. Each view also gets a companion top-down
map PNG (crops/<building_id>/<rank>_<pano_id>_map.png, via
geometry/occlusion_plot.py) showing the target/occluder footprints, nearby
buildings for context, the camera point, and the sightline -- because
judging occlusion from the street photo alone (which building is actually
blocking, and by how much) was too hard without seeing the geometry.

This REPLACES the single-best-match "find one usable view, skip if occluded"
logic (still available in git history if needed later) with: take the 3
nearest panoramas by plain distance, crop+mark each one, done -- occlusion is
still computed (via geometry/line_of_sight.py's polygon-based check) and
recorded per-view as information for a human reviewer, but no longer used to
filter/skip a candidate. Ties together:
  - geometry/bearing.py       (bearing toward the building's centroid)
  - geometry/line_of_sight.py (informational: is this view's line-of-sight
                                blocked by another building's polygon?)
  - geometry/direction_view.py (slice_direction() -- the fixed-FOV crop;
                                 mark_center() -- draws the aim line, which
                                 lands exactly at the crop's horizontal
                                 center by construction)

Nearest-neighbor search is plain vectorized numpy (per-building distance to
every panorama), not a KD-tree/scipy dependency -- at this AOI's scale
(thousands of buildings x hundreds-to-low-thousands of panoramas) that's
comfortably fast without adding a dependency.
"""
import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # residential_annotation/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))     # Tower Identification_exp/

from geometry.bearing import bearing_deg, bearing_xy, project
from geometry.line_of_sight import build_index, is_occluded, find_occluders
from geometry.direction_view import slice_direction, mark_center, mark_occluder, label_image, heading_offset_to_crop_x
from geometry.occlusion_plot import plot_occlusion
from coverage.streetview_coverage import download_panorama_image
from pull_buildings_5km import utm_epsg_for  # reuse existing helper, don't re-derive the UTM zone

MANIFEST_FIELDS = [
    "building_id", "rank", "status", "pano_id", "bearing_deg", "distance_m",
    "occluded", "occluder_building_id", "occluder_distance_m", "occluder_fraction", "forced_clear",
    "height_m", "height_quality", "confidence", "image_path", "map_path",
]
NEARBY_CONTEXT_DEG = 0.0012  # ~130m -- how far around the camera to pull "context" buildings for the map plot


def load_buildings(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    return gdf


def load_coverage(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    return df


def nearest_k_panoramas(building_xy: np.ndarray, pano_xy: np.ndarray, pano_ids: list, k: int) -> list:
    """Returns up to k (pano_id, distance_m) pairs, nearest-first. Plain
    distance ranking -- no occlusion filtering here; that's computed
    separately per-candidate as informational metadata (see main())."""
    dists = np.linalg.norm(pano_xy - building_xy[None, :], axis=1)
    order = np.argsort(dists)[:k]
    return [(pano_ids[idx], float(dists[idx])) for idx in order]


def find_first_clear(
    building_xy: np.ndarray, building_polygon_proj, building_id: str,
    pano_xy: np.ndarray, pano_ids: list, coverage_by_id, buildings, tree, epsg: int,
    max_search_m: float, skip_pano_ids: set, fov_deg: float,
):
    """Guarantee-at-least-one-clear-view search: walks candidates nearest-first
    (skipping ones already tried) out to max_search_m, checking the polygon-
    based occlusion for each, and returns the FIRST unoccluded one found as
    (pano_id, distance_m) -- or (None, None) if nothing clear turns up that
    close. Only called when none of a building's initial nearest-k views were
    clear; existing occluded candidates aren't re-saved, just skipped past."""
    dists = np.linalg.norm(pano_xy - building_xy[None, :], axis=1)
    order = np.argsort(dists)
    for idx in order:
        d = dists[idx]
        if d > max_search_m:
            break
        pano_id = pano_ids[idx]
        if pano_id in skip_pano_ids:
            continue
        cam_row = coverage_by_id.loc[pano_id]
        cam_xy = project(cam_row["lat"], cam_row["lon"], epsg)
        if not is_occluded(cam_xy, building_polygon_proj, building_id, buildings, tree,
                            id_col="building_id", fov_deg=fov_deg):
            return pano_id, float(d)
    return None, None


async def fetch_panoramas(pano_ids: set, images_dir: str, concurrency: int = 4) -> dict:
    """Downloads each unique panorama once (cached to disk by download_panorama_image),
    returns {pano_id: local_image_path or None}."""
    import aiohttp
    sem = asyncio.Semaphore(concurrency)
    paths = {}

    async def _one(session, pano_id):
        cached = Path(images_dir) / f"{pano_id}.jpg"
        if cached.exists():
            paths[pano_id] = str(cached)
            return
        async with sem:
            path = await download_panorama_image(pano_id, session, images_dir)
        paths[pano_id] = path

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[_one(session, pid) for pid in pano_ids])
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--buildings-geojson", default="../output/buildings_aoi.geojson")
    p.add_argument("--coverage-csv", default="outputs/coverage.csv")
    p.add_argument("--images-dir", default="outputs/panoramas")
    p.add_argument("--crops-dir", default="outputs/crops",
                   help="Each building gets its own subfolder here, e.g. outputs/crops/<building_id>/")
    p.add_argument("--out-csv", default="outputs/crops_manifest.csv")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N buildings (testing)")
    p.add_argument("--sample-n", type=int, default=None,
                   help="Randomly sample N buildings to process instead of taking the first --limit "
                        "(sequential rows tend to be geographically clustered, biasing tests toward one "
                        "area/building-type -- e.g. the first 100 rows here are mostly CBD/commercial)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for --sample-n, for reproducibility")
    p.add_argument("--num-views", type=int, default=3, help="How many nearest panoramas to mark per building")
    p.add_argument("--fov-deg", type=float, default=120.0,
                   help="Fixed crop width in degrees, centered on the aim heading (60 either side by default) "
                        "-- not computed per-building; see PLAN.md for why a dynamic FOV was cutting buildings off")
    p.add_argument("--guarantee-clear-max-m", type=float, default=50.0,
                   help="If none of the --num-views nearest views are clear (unoccluded), keep searching "
                        "farther out (nearest-first, skipping already-tried panoramas) up to this distance "
                        "for one that is, and add it as an extra view -- so every building with ANY clear "
                        "view within this radius gets at least one to actually judge the framing by")
    p.add_argument("--image-concurrency", type=int, default=4)
    p.add_argument("--skip-maps", action="store_true",
                   help="Don't generate the per-view occlusion map PNGs (map_path left blank in the "
                        "manifest) -- much faster for a first pass; run crops/add_maps.py afterward "
                        "on the resulting manifest to fill them in without redoing any matching/cropping")
    args = p.parse_args()

    print(f"Loading buildings from {args.buildings_geojson} ...")
    buildings_all = load_buildings(args.buildings_geojson)
    print(f"  -> {len(buildings_all)} buildings total in AOI")

    if args.sample_n:
        buildings = buildings_all.sample(n=args.sample_n, random_state=args.seed).copy()
        print(f"  -> randomly sampled {len(buildings)} buildings (seed={args.seed})")
    elif args.limit:
        buildings = buildings_all.iloc[:args.limit].copy()
        print(f"  -> using the first {len(buildings)} buildings")
    else:
        buildings = buildings_all

    print(f"Loading coverage from {args.coverage_csv} ...")
    coverage = load_coverage(args.coverage_csv)
    print(f"  -> {len(coverage)} panoramas")

    center = buildings.geometry.iloc[0].centroid
    epsg = utm_epsg_for(center.y, center.x)
    print(f"Using UTM EPSG:{epsg} for distance/occlusion/bearing math")

    # Occlusion index is built from ALL buildings in the AOI, not just the
    # (possibly small, possibly randomly-scattered) subset being processed --
    # a real neighbor that could actually block a view must be considered
    # even if it wasn't itself picked by --limit/--sample-n. Sequential
    # --limit rows happened to mostly dodge this (file order tends to cluster
    # geographically); a random sample would have masked almost all real
    # occlusion otherwise.
    buildings_proj = buildings_all.to_crs(epsg=epsg)
    tree = build_index(buildings_proj)
    proj_centroid_by_id = {
        bid: (geom.centroid.x, geom.centroid.y)
        for bid, geom in zip(buildings_proj["building_id"], buildings_proj.geometry)
    }
    buildings_all_by_id = buildings_all.set_index("building_id")  # WGS84 geometry lookup, for the map plots

    pano_xy = np.array([project(lat, lon, epsg) for lat, lon in zip(coverage["lat"], coverage["lon"])])
    pano_ids_list = coverage["pano_id"].tolist()
    coverage_by_id = coverage.set_index("pano_id")

    Path(args.crops_dir).mkdir(parents=True, exist_ok=True)
    Path(args.images_dir).mkdir(parents=True, exist_ok=True)

    print(f"Finding the {args.num_views} nearest panoramas per building ...")
    matches = []  # one entry per (building, rank)
    has_clear_by_building = {}  # building_id -> True if ANY view ended up clear (initial or forced-search)
    n_total_buildings = len(buildings)
    t_match_start = time.monotonic()
    PROGRESS_EVERY = 200
    for match_i, (i, row) in enumerate(buildings.iterrows(), start=1):
        if match_i % PROGRESS_EVERY == 0 or match_i == n_total_buildings:
            elapsed = time.monotonic() - t_match_start
            rate = match_i / elapsed if elapsed > 0 else 0
            remaining = (n_total_buildings - match_i) / rate if rate > 0 else float("inf")
            print(f"  matching {match_i}/{n_total_buildings} buildings "
                  f"({rate:.1f}/s, {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining) ...", flush=True)
        proj_row = buildings_proj.loc[i]
        building_xy = np.array(proj_row.geometry.centroid.coords[0])
        candidates = nearest_k_panoramas(building_xy, pano_xy, pano_ids_list, args.num_views)

        if not candidates:
            matches.append({
                "building_id": row["building_id"], "rank": 1, "status": "no_coverage",
                "pano_id": "", "distance_m": "", "occluded": "",
                "height_m": row.get("height_m", ""), "height_quality": row.get("height_quality", ""),
                "confidence": row.get("confidence", ""), "_polygon_wgs84": row.geometry,
            })
            continue

        seen_pano_ids = set()
        any_clear = False
        for rank, (pano_id, dist_m) in enumerate(candidates, start=1):
            seen_pano_ids.add(pano_id)
            cam_row = coverage_by_id.loc[pano_id]
            cam_xy = project(cam_row["lat"], cam_row["lon"], epsg)
            occluders = find_occluders(cam_xy, proj_row.geometry, row["building_id"], buildings_proj, tree,
                                        id_col="building_id", fov_deg=args.fov_deg)
            occluded = len(occluders) > 0
            any_clear = any_clear or not occluded
            primary_occluder = occluders[0] if occluders else None
            matches.append({
                "building_id": row["building_id"], "rank": rank, "status": "ok",
                "pano_id": pano_id, "distance_m": round(dist_m, 1), "occluded": occluded,
                "occluder_building_id": primary_occluder["building_id"] if primary_occluder else "",
                "occluder_distance_m": round(primary_occluder["distance_m"], 1) if primary_occluder else "",
                "occluder_fraction": round(primary_occluder["occlusion_fraction"], 2) if primary_occluder else "",
                "forced_clear": False,
                "height_m": row.get("height_m", ""), "height_quality": row.get("height_quality", ""),
                "confidence": row.get("confidence", ""), "_polygon_wgs84": row.geometry,
            })

        if not any_clear:
            # None of the nearest --num-views were clear -- keep looking, nearest-first,
            # out to --guarantee-clear-max-m, so this building isn't left with only
            # ambiguous/occluded views to judge framing by.
            extra_pano_id, extra_dist_m = find_first_clear(
                building_xy, proj_row.geometry, row["building_id"], pano_xy, pano_ids_list,
                coverage_by_id, buildings_proj, tree, epsg, args.guarantee_clear_max_m, seen_pano_ids,
                args.fov_deg,
            )
            if extra_pano_id is not None:
                any_clear = True
                matches.append({
                    "building_id": row["building_id"], "rank": len(candidates) + 1, "status": "ok",
                    "pano_id": extra_pano_id, "distance_m": round(extra_dist_m, 1), "occluded": False,
                    "forced_clear": True,
                    "height_m": row.get("height_m", ""), "height_quality": row.get("height_quality", ""),
                    "confidence": row.get("confidence", ""), "_polygon_wgs84": row.geometry,
                })

        has_clear_by_building[row["building_id"]] = any_clear

    n_buildings = buildings["building_id"].nunique()
    n_no_coverage = sum(1 for m in matches if m["status"] == "no_coverage")
    n_views = sum(1 for m in matches if m["status"] == "ok")
    n_occluded_views = sum(1 for m in matches if m.get("occluded"))
    n_forced_clear = sum(1 for m in matches if m.get("forced_clear"))
    n_still_all_occluded = sum(1 for v in has_clear_by_building.values() if not v)
    print(f"  {n_buildings} buildings -> {n_views} views total "
          f"({n_no_coverage} buildings with zero coverage, {n_occluded_views} views flagged occluded, "
          f"{n_forced_clear} extra views added to guarantee a clear one, "
          f"{n_still_all_occluded} buildings still with NO clear view within {args.guarantee_clear_max_m}m)")

    needed_panos = {m["pano_id"] for m in matches if m["status"] == "ok"}
    print(f"Downloading {len(needed_panos)} unique panorama images ...")
    pano_paths = asyncio.run(fetch_panoramas(needed_panos, args.images_dir, args.image_concurrency))

    print(f"Computing bearing and cropping each view ({args.fov_deg:g} deg fixed FOV) ...")
    rows = []
    n_total_matches = len(matches)
    t_crop_start = time.monotonic()
    for crop_i, m in enumerate(matches, start=1):
        if crop_i % PROGRESS_EVERY == 0 or crop_i == n_total_matches:
            elapsed = time.monotonic() - t_crop_start
            rate = crop_i / elapsed if elapsed > 0 else 0
            remaining = (n_total_matches - crop_i) / rate if rate > 0 else float("inf")
            print(f"  cropping {crop_i}/{n_total_matches} views "
                  f"({rate:.1f}/s, {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining) ...", flush=True)
        if m["status"] != "ok":
            m["image_path"] = ""
            rows.append(m)
            continue

        pano_path = pano_paths.get(m["pano_id"])
        if not pano_path:
            m["status"] = "download_failed"
            m["image_path"] = ""
            rows.append(m)
            continue

        cam_row = coverage_by_id.loc[m["pano_id"]]
        centroid = m["_polygon_wgs84"].centroid
        heading_deg = bearing_deg(cam_row["lat"], cam_row["lon"], centroid.y, centroid.x, epsg)
        m["bearing_deg"] = round(heading_deg, 2)

        pano_img = Image.open(pano_path)
        crop = slice_direction(
            pano_img, heading_deg=heading_deg,
            pano_heading_deg=float(cam_row["heading_deg"]) if pd.notna(cam_row["heading_deg"]) else 0.0,
            fov_h_deg=args.fov_deg,
        )
        marked = mark_center(crop)

        # If occluded, show WHICH building is in the way: its own bearing from
        # this same camera point, mapped into this crop's pixel space, plus a
        # label with its building_id + distance -- so it can be looked up
        # exactly rather than just trusting the occluded flag on faith.
        if m["occluded"] and m.get("occluder_building_id"):
            occ_xy = proj_centroid_by_id.get(m["occluder_building_id"])
            if occ_xy is not None:
                cam_xy = project(cam_row["lat"], cam_row["lon"], epsg)
                occ_heading = bearing_xy(occ_xy[0] - cam_xy[0], occ_xy[1] - cam_xy[1])
                occ_x = heading_offset_to_crop_x(occ_heading, heading_deg, crop.width, args.fov_deg)
                occ_tag = (f"blocked by {m['occluder_building_id']} ({m['occluder_distance_m']}m, "
                           f"{m['occluder_fraction']*100:.0f}% of facade)")
                marked = mark_occluder(marked, occ_x, occ_tag)

        status_text = "OCCLUDED" if m["occluded"] else "CLEAR"
        if m.get("forced_clear"):
            status_text += " (forced search)"
        status_color = (255, 70, 70) if m["occluded"] else (60, 230, 60)
        marked = label_image(marked, status_text, color=status_color)

        # Top-level split: buildings with >=1 clear view vs. buildings where
        # every view came back occluded -- so browsing the folder alone tells
        # you what you're looking at, no need to cross-reference the manifest.
        category = "has_clear_view" if has_clear_by_building.get(m["building_id"]) else "all_occluded"
        building_dir = Path(args.crops_dir) / category / m["building_id"]
        building_dir.mkdir(parents=True, exist_ok=True)
        image_path = str(building_dir / f"{m['building_id']}_{m['rank']}.jpg")
        marked.save(image_path, "JPEG", quality=90)
        m["image_path"] = image_path

        # Top-down map: target + occluder (if any) + nearby buildings for
        # context + the camera point and sightline -- see this occluded/clear
        # call geometrically instead of just trusting the street photo.
        # Skippable (--skip-maps) for a fast first pass; crops/add_maps.py
        # fills these in afterward from the manifest alone, no re-matching.
        if args.skip_maps:
            m["map_path"] = ""
        else:
            occluder_polys = []
            if m["occluded"] and m.get("occluder_building_id") and m["occluder_building_id"] in buildings_all_by_id.index:
                occluder_polys = [buildings_all_by_id.loc[m["occluder_building_id"]].geometry]

            minx, maxx = cam_row["lon"] - NEARBY_CONTEXT_DEG, cam_row["lon"] + NEARBY_CONTEXT_DEG
            miny, maxy = cam_row["lat"] - NEARBY_CONTEXT_DEG, cam_row["lat"] + NEARBY_CONTEXT_DEG
            nearby = buildings_all.cx[minx:maxx, miny:maxy]
            exclude_ids = {m["building_id"]}
            if m.get("occluder_building_id"):
                exclude_ids.add(m["occluder_building_id"])
            nearby_polys = [geom for bid, geom in zip(nearby["building_id"], nearby.geometry) if bid not in exclude_ids]

            map_path = str(building_dir / f"{m['building_id']}_{m['rank']}_map.png")
            plot_occlusion(
                target_poly=m["_polygon_wgs84"], occluder_polys=occluder_polys, nearby_polys=nearby_polys,
                cam_lat=cam_row["lat"], cam_lon=cam_row["lon"], out_path=map_path,
                title=f"{m['building_id']}  rank {m['rank']}  dist={m['distance_m']}m  "
                      f"{'OCCLUDED' if m['occluded'] else 'CLEAR'}",
            )
            m["map_path"] = map_path
        rows.append(m)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for m in rows:
            writer.writerow({k: m.get(k, "") for k in MANIFEST_FIELDS})

    print(f"\nWrote {len(rows)} rows ({n_buildings} buildings x up to {args.num_views} views) to {args.out_csv}")
    print(f"Marked panoramas saved under {args.crops_dir}/{{has_clear_view,all_occluded}}/<building_id>/")

    summary_path = str(Path(args.out_csv).parent / "building_summary.csv")
    summary = build_summary(rows, summary_path)
    print(f"Per-building summary ({len(summary)} buildings, "
          f"{summary['has_clear_view'].sum()} with >=1 clear view) written to {summary_path}")


def build_summary(rows: list, out_path: str) -> pd.DataFrame:
    """One row per building_id: has_clear_view (True if ANY of its views came
    back unoccluded), num_views actually produced, best_clear_distance_m
    among the clear ones (blank if none clear) -- the quick lookup table for
    "which buildings actually have a usable view" without re-scanning the
    full multi-row-per-building manifest each time."""
    df = pd.DataFrame(rows)
    out_rows = []
    for bid, g in df.groupby("building_id"):
        ok = g[g["status"] == "ok"]
        clear = ok[ok["occluded"] == False]  # noqa: E712 (explicit False, not falsy-check, matches stored bool)
        out_rows.append({
            "building_id": bid,
            "num_views": len(ok),
            "num_clear_views": len(clear),
            "has_clear_view": len(clear) > 0,
            "best_clear_distance_m": clear["distance_m"].min() if len(clear) else "",
            "height_m": g["height_m"].iloc[0],
            "height_quality": g["height_quality"].iloc[0],
            "confidence": g["confidence"].iloc[0],
        })
    summary = pd.DataFrame(out_rows)
    summary.to_csv(out_path, index=False)
    return summary


if __name__ == "__main__":
    main()
