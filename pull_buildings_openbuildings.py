#!/usr/bin/env python3
"""
Pull building footprints intersecting our 5 km^2 AOI circle from the local
Google Open Buildings (2.5D Temporal) extract for Jakarta --
jakarta_buildings_2023_heights.geojson -- instead of Overture Maps.

Why the switch (from pull_buildings_5km.py's Overture source):
  - Height coverage: ~96% of buildings here have a height_m estimate (with a
    height_quality flag), vs. Overture's ~4% in the same AOI. This directly
    unlocks things Overture's sparse heights couldn't (e.g. any future
    vertical framing work).
  - This IS fundamentally Google Open Buildings data (same source Overture
    itself partially ingests, per our earlier research) -- just the raw,
    unfiltered version with Google's own per-building height model attached,
    rather than Overture's fused/re-tagged version.
  - Trade-off: no `class`/`subtype`/`name` tags at all (Overture had these,
    sparsely, via its OSM fusion) -- just geometry + area + confidence +
    height. Fine for this pipeline, which doesn't rely on those tags anyway
    (see PLAN.md: Overture's class/subtype was already too sparse to use as
    more than a soft signal).

Extraction approach: the source file is 1.58GB, a single unindexed
FeatureCollection covering all of Jakarta. A DuckDB spatial ST_Read() bbox
query against it estimated ~6.3 HOURS (GDAL's GeoJSON driver has no spatial
index to exploit, so it was scanning naively). geopandas.read_file(...,
bbox=..., engine="pyogrio") is dramatically faster in practice (~49s for a
404,739-row wider-bbox pull) -- use that, not DuckDB, for this file.

Usage:
  python3 pull_buildings_openbuildings.py
  python3 pull_buildings_openbuildings.py --lat -6.18 --lon 106.82 --area-km2 5
"""
import argparse
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from pull_buildings_5km import (
    DEFAULT_LAT, DEFAULT_LON, DEFAULT_AREA_KM2,
    radius_km_from_area_km2, utm_epsg_for, degree_bbox_for_radius,
)

SOURCE_GEOJSON = Path(__file__).parent / "jakarta_buildings_2023_heights.geojson"


def fetch_openbuildings(bbox) -> gpd.GeoDataFrame:
    print(f"Reading {SOURCE_GEOJSON.name} (1.58GB, bbox-filtered via pyogrio) for bbox={bbox} ...")
    t0 = time.monotonic()
    gdf = gpd.read_file(SOURCE_GEOJSON, bbox=bbox, engine="pyogrio")
    print(f"  -> {len(gdf):,} buildings in bounding box ({time.monotonic() - t0:.1f}s)")

    # Source is built from overlapping tile-processing blocks (see partition_id,
    # e.g. JKT_r12_c08_b00 vs ...b01) -- a building near a block boundary gets
    # emitted once per overlapping block that processed it. Confirmed these are
    # byte-for-byte identical re-emissions (same height/confidence/geometry),
    # not genuinely different candidates, so dropping duplicates is safe and
    # necessary -- otherwise ~1/3 of "buildings" here are the same physical
    # building double-counted, corrupting every downstream count/crop/stat.
    before = len(gdf)
    gdf = gdf.drop_duplicates(subset="building_id", keep="first").reset_index(drop=True)
    if before != len(gdf):
        print(f"  -> dropped {before - len(gdf):,} duplicate re-emissions "
              f"(overlapping tile-block artifact) -> {len(gdf):,} unique buildings")
    return gdf


def filter_to_circle(gdf: gpd.GeoDataFrame, lat: float, lon: float, radius_km: float):
    epsg = utm_epsg_for(lat, lon)
    print(f"Projecting to UTM EPSG:{epsg} for exact distance/intersection math ...")

    gdf_proj = gdf.to_crs(epsg=epsg)
    center_proj = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=epsg).iloc[0]
    circle = center_proj.buffer(radius_km * 1000.0)

    mask = gdf_proj.geometry.intersects(circle)
    kept = gdf_proj[mask].copy()
    kept["distance_from_center_m"] = kept.geometry.distance(center_proj).round(1)
    kept["area_m2"] = kept.geometry.area.round(1)
    kept["intersects_boundary"] = kept.geometry.crosses(circle.boundary) | kept.geometry.touches(circle.boundary)

    kept_wgs84 = kept.to_crs(epsg=4326).sort_values("distance_from_center_m")
    return kept_wgs84, circle, epsg


def summarize(gdf: gpd.GeoDataFrame, radius_km: float, area_km2: float):
    n = len(gdf)
    print("\n=== Summary ===")
    print(f"Buildings intersecting {area_km2:g} km^2 AOI (radius={radius_km:.4f} km): {n:,}")
    if n == 0:
        return
    print(f"Total footprint area: {gdf['area_m2'].sum() / 1e6:,.3f} km^2")
    print(f"Nearest building distance: {gdf['distance_from_center_m'].min():.1f} m")
    print(f"Farthest (still intersecting) distance: {gdf['distance_from_center_m'].max():.1f} m")
    heights = pd.to_numeric(gdf["height_m"], errors="coerce").dropna()
    print(f"Height available for {len(heights):,}/{n:,} buildings ({len(heights)/n*100:.1f}%) "
          f"(min={heights.min():.1f}m, mean={heights.mean():.1f}m, max={heights.max():.1f}m)")
    if "height_quality" in gdf.columns:
        print("Height quality breakdown:", gdf["height_quality"].value_counts().to_dict())
    print(f"Confidence: mean={gdf['confidence'].mean():.3f}, min={gdf['confidence'].min():.3f}")
    crossing = int(gdf["intersects_boundary"].sum())
    print(f"Buildings straddling the AOI boundary itself: {crossing:,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    parser.add_argument("--area-km2", type=float, default=DEFAULT_AREA_KM2)
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "output"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    radius_km = radius_km_from_area_km2(args.area_km2)
    print(f"AOI area = {args.area_km2:g} km^2  ->  derived radius = {radius_km:.4f} km "
          f"({radius_km * 1000:.1f} m)")

    bbox = degree_bbox_for_radius(args.lat, args.lon, radius_km)
    raw = fetch_openbuildings(bbox)
    buildings, circle_proj, epsg = filter_to_circle(raw, args.lat, args.lon, radius_km)

    geojson_path = out_dir / "buildings_aoi.geojson"
    csv_path = out_dir / "buildings_aoi.csv"
    center_path = out_dir / "aoi_circle.geojson"

    buildings.to_file(geojson_path, driver="GeoJSON")

    csv_cols = [c for c in ["building_id", "height_m", "height_quality", "height_valid", "confidence",
                             "full_plus_code", "distance_from_center_m", "area_m2", "intersects_boundary"]
                if c in buildings.columns]
    buildings[csv_cols].to_csv(csv_path, index=False)

    circle_wgs84 = gpd.GeoSeries([circle_proj], crs=epsg).to_crs(epsg=4326)
    circle_gdf = gpd.GeoDataFrame(
        {"lat": [args.lat], "lon": [args.lon], "area_km2": [args.area_km2], "radius_km": [radius_km]},
        geometry=circle_wgs84, crs="EPSG:4326",
    )
    circle_gdf.to_file(center_path, driver="GeoJSON")

    print(f"\nWrote {len(buildings):,} buildings to:")
    print(f"  {geojson_path}")
    print(f"  {csv_path}")
    print(f"AOI circle written to:")
    print(f"  {center_path}")

    summarize(buildings, radius_km, args.area_km2)


if __name__ == "__main__":
    main()
