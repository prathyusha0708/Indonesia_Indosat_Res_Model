#!/usr/bin/env python3
"""
Pull building footprints intersecting a circular area-of-interest around a
point, where the AOI is specified by its *area* (default 5 km^2) rather than
a radius. The equivalent radius is derived as r = sqrt(area / pi).

Data source: Overture Maps Foundation "buildings" theme (open, global building
footprint dataset, published as hive-partitioned GeoParquet on S3). Overture is
used instead of raw OSM/Overpass because Overpass has query-size/timeout limits
that are easily hit in dense urban areas (this point sits in central Jakarta,
which alone returns 100k+ buildings even for a small bounding box).

What it does:
  1. Takes a center point (lat, lon) and an AOI area in km^2 (default 5 km^2),
     and converts it to a circle radius (r = sqrt(area / pi)).
  2. Downloads all Overture building footprints in the bounding box around the
     point (with a safety margin), using bbox pushdown so we don't scan the
     whole planet.
  3. Reprojects into a local UTM CRS (auto-picked from the point) so distance
     math is in meters, not degrees.
  4. Builds the true AOI circle (buffer around the point, using the derived
     radius) and keeps every building whose polygon *intersects* that circle
     boundary/interior -- i.e. buildings straddling the AOI edge are included
     too, not just centroids inside it.
  5. Writes the filtered buildings out as GeoJSON + a CSV summary, and prints
     stats (count, area, height where available).

Usage:
  python3 pull_buildings_5km.py
  python3 pull_buildings_5km.py --lat -6.1799849294994065 --lon 106.82189361088099 --area-km2 5
"""

import argparse
import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

try:
    from overturemaps.core import geodataframe as overture_geodataframe
except ImportError:
    sys.exit(
        "Missing dependency 'overturemaps'. Install with:\n"
        "  pip3 install overturemaps geopandas shapely pyproj"
    )

DEFAULT_LAT = -6.1799849294994065
DEFAULT_LON = 106.82189361088099
DEFAULT_AREA_KM2 = 5.0
EARTH_RADIUS_M = 6_371_000.0


def radius_km_from_area_km2(area_km2: float) -> float:
    """Radius (km) of a circle with the given area (km^2): r = sqrt(area / pi)."""
    return math.sqrt(area_km2 / math.pi)


def utm_epsg_for(lat: float, lon: float) -> int:
    """Pick the UTM EPSG code covering (lat, lon)."""
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    return (32600 if lat >= 0 else 32700) + zone


def degree_bbox_for_radius(lat: float, lon: float, radius_km: float, margin_ratio: float = 0.05):
    """
    Rough (but safely oversized) lat/lon bounding box that fully contains a
    `radius_km` circle around (lat, lon), padded by `margin_ratio` extra so we
    never clip real edge cases before the precise UTM-based filtering step.
    """
    radius_m = radius_km * 1000.0 * (1.0 + margin_ratio)
    dlat = math.degrees(radius_m / EARTH_RADIUS_M)
    dlon = math.degrees(radius_m / (EARTH_RADIUS_M * math.cos(math.radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)  # (xmin, ymin, xmax, ymax)


def fetch_overture_buildings(bbox) -> gpd.GeoDataFrame:
    print(f"Downloading Overture 'building' footprints for bbox={bbox} ...")
    gdf = overture_geodataframe("building", bbox=bbox)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    print(f"  -> {len(gdf):,} buildings in bounding box")
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
    kept["intersects_boundary"] = kept.geometry.crosses(circle.boundary) | kept.geometry.touches(
        circle.boundary
    )

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
    if "height" in gdf.columns:
        heights = pd.to_numeric(gdf["height"], errors="coerce").dropna()
        if len(heights):
            print(
                f"Height available for {len(heights):,}/{n:,} buildings "
                f"(min={heights.min():.1f}m, mean={heights.mean():.1f}m, max={heights.max():.1f}m)"
            )
    crossing = int(gdf["intersects_boundary"].sum())
    print(f"Buildings straddling the AOI boundary itself: {crossing:,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    parser.add_argument(
        "--area-km2",
        type=float,
        default=DEFAULT_AREA_KM2,
        help="AOI area in km^2 (default 5). Radius is derived as sqrt(area/pi).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).parent / "output"),
        help="Directory to write buildings_aoi.geojson / .csv into",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    radius_km = radius_km_from_area_km2(args.area_km2)
    print(f"AOI area = {args.area_km2:g} km^2  ->  derived radius = {radius_km:.4f} km "
          f"({radius_km * 1000:.1f} m)")

    bbox = degree_bbox_for_radius(args.lat, args.lon, radius_km)
    raw = fetch_overture_buildings(bbox)
    buildings, circle_proj, epsg = filter_to_circle(raw, args.lat, args.lon, radius_km)

    geojson_path = out_dir / "buildings_aoi.geojson"
    csv_path = out_dir / "buildings_aoi.csv"
    center_path = out_dir / "aoi_circle.geojson"

    keep_cols = [c for c in buildings.columns if c != "sources"]  # 'sources' is nested/complex
    buildings[keep_cols].to_file(geojson_path, driver="GeoJSON")

    csv_cols = [c for c in ["id", "height", "num_floors", "class", "subtype",
                             "distance_from_center_m", "area_m2", "intersects_boundary"]
                if c in buildings.columns]
    buildings[csv_cols].to_csv(csv_path, index=False)

    circle_wgs84 = gpd.GeoSeries([circle_proj], crs=epsg).to_crs(epsg=4326)
    circle_gdf = gpd.GeoDataFrame(
        {"lat": [args.lat], "lon": [args.lon], "area_km2": [args.area_km2], "radius_km": [radius_km]},
        geometry=circle_wgs84,
        crs="EPSG:4326",
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
