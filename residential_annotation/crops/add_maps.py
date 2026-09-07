"""
Fills in the per-view occlusion map PNGs for an existing crops_manifest.csv
produced by build_crops.py --skip-maps -- reads the manifest + the buildings/
coverage data already on disk, no re-matching, no re-cropping, no re-download.
Safe to interrupt and re-run: skips any row that already has a map_path
filled in (or whose map file already exists on disk).
"""
import argparse
import csv
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # residential_annotation/
from geometry.occlusion_plot import plot_occlusion

NEARBY_CONTEXT_DEG = 0.0012  # ~130m -- same context radius build_crops.py uses


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="outputs/crops_manifest.csv")
    p.add_argument("--buildings-geojson", default="../output/buildings_aoi.geojson")
    p.add_argument("--coverage-csv", default="outputs/coverage.csv")
    args = p.parse_args()

    print(f"Loading buildings from {args.buildings_geojson} ...")
    buildings_all = gpd.read_file(args.buildings_geojson)
    if buildings_all.crs is None:
        buildings_all = buildings_all.set_crs(epsg=4326)
    buildings_all_by_id = buildings_all.set_index("building_id")
    print(f"  -> {len(buildings_all)} buildings")

    print(f"Loading coverage from {args.coverage_csv} ...")
    coverage = pd.read_csv(args.coverage_csv).set_index("pano_id")

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys()) if rows else []
    if "map_path" not in fieldnames:
        sys.exit(f"'{args.manifest}' has no map_path column -- was it produced by a build_crops.py "
                  f"new enough to have --skip-maps? Re-run build_crops.py first.")

    todo = [r for r in rows if r["status"] == "ok" and not r.get("map_path")]
    print(f"{len(rows)} manifest rows, {len(todo)} need a map")

    done = skipped_missing_building = 0
    for r in todo:
        bid = r["building_id"]

        image_path = Path(r["image_path"])
        map_path = str(image_path.with_name(image_path.stem + "_map.png"))
        if Path(map_path).exists():
            r["map_path"] = map_path  # already rendered in a prior partial run -- just record it
            done += 1
            continue

        if bid not in buildings_all_by_id.index or r["pano_id"] not in coverage.index:
            skipped_missing_building += 1
            continue

        target_poly = buildings_all_by_id.loc[bid].geometry
        cam = coverage.loc[r["pano_id"]]

        occluder_polys = []
        occ_id = r.get("occluder_building_id")
        if r["occluded"] == "True" and occ_id and occ_id in buildings_all_by_id.index:
            occluder_polys = [buildings_all_by_id.loc[occ_id].geometry]

        minx, maxx = cam["lon"] - NEARBY_CONTEXT_DEG, cam["lon"] + NEARBY_CONTEXT_DEG
        miny, maxy = cam["lat"] - NEARBY_CONTEXT_DEG, cam["lat"] + NEARBY_CONTEXT_DEG
        nearby = buildings_all.cx[minx:maxx, miny:maxy]
        exclude_ids = {bid}
        if occ_id:
            exclude_ids.add(occ_id)
        nearby_polys = [g for b, g in zip(nearby["building_id"], nearby.geometry) if b not in exclude_ids]

        plot_occlusion(
            target_poly=target_poly, occluder_polys=occluder_polys, nearby_polys=nearby_polys,
            cam_lat=cam["lat"], cam_lon=cam["lon"], out_path=map_path,
            title=f"{bid}  rank {r['rank']}  dist={r['distance_m']}m  "
                  f"{'OCCLUDED' if r['occluded'] == 'True' else 'CLEAR'}",
        )
        r["map_path"] = map_path
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(todo)} maps done", flush=True)

    with open(args.manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone: {done} maps generated/recorded, {skipped_missing_building} skipped "
          f"(building or pano no longer found), manifest updated at {args.manifest}")


if __name__ == "__main__":
    main()
