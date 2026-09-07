# Tower Identification — Buildings Pull

Pulls every building footprint that **intersects a circular AOI of a given
area** (default **5 km²**) around a given lat/lon.

**Current canonical source: Google Open Buildings** (2.5D Temporal, 2023),
via a local Jakarta-wide extract — `jakarta_buildings_2023_heights.geojson`
— pulled with `pull_buildings_openbuildings.py`. Switched from the original
Overture-based pull (`pull_buildings_5km.py`, still present, output backed up
in `output_overture_backup/`) because ~96% of buildings here have a
`height_m` estimate (with a `height_quality` flag), vs. Overture's ~4% in the
same AOI. Trade-off: no `class`/`subtype`/`name` tags (Overture had these,
sparsely) — not something this pipeline relied on beyond soft stratification
signal anyway.

The AOI is specified by **area**, not radius: the equivalent radius is derived
as `r = sqrt(area / pi)`. For the default 5 km², that's a radius of
**~1.2616 km (1261.6 m)**.

## Point used

```
lat: -6.1799849294994065
lon: 106.82189361088099
```

This sits in central Jakarta, right at **PT Indosat HQ** (the nearest building,
0.0 m away — this point falls inside its footprint) — likely the tower/site
of interest.

## Run

```bash
pip3 install -r requirements.txt
python3 pull_buildings_openbuildings.py                     # current canonical source
python3 pull_buildings_openbuildings.py --lat <lat> --lon <lon> --area-km2 <area>

python3 pull_buildings_5km.py                                # legacy Overture-based pull, kept for reference
```

## How it works (`pull_buildings_openbuildings.py`)

1. Converts the requested AOI area (km²) to an equivalent circle radius:
   `r = sqrt(area / pi)` (reused from `pull_buildings_5km.py`).
2. Reads `jakarta_buildings_2023_heights.geojson` (a 1.58GB, unindexed,
   Jakarta-wide FeatureCollection) with `geopandas.read_file(..., bbox=...,
   engine="pyogrio")` — a bbox-filtered read that takes ~45-50s. **Do not**
   use DuckDB's spatial `ST_Read()` on this file — it has no spatial index to
   exploit and was estimated at ~6.3 *hours* for the same query; pyogrio's
   GDAL bindings are dramatically faster here.
3. Reprojects to the local UTM zone (auto-picked from the point) so distance
   math is in meters.
4. Builds the exact AOI circle around the point (using the derived radius) and
   keeps every building polygon that **intersects** it (not just
   centroid-inside) — buildings straddling the boundary are included.
5. Writes results to `output/`.

## Output (`output/`)

- `buildings_aoi.geojson` — full building polygons + Open Buildings attributes
  (`building_id`, `height_m`, `height_quality`, `height_valid`, `confidence`,
  `full_plus_code`) + computed `distance_from_center_m`, `area_m2`,
  `intersects_boundary`.
- `buildings_aoi.csv` — flat table of the key attributes for quick review.
- `aoi_circle.geojson` — the AOI circle polygon itself (with `area_km2` and
  `radius_km` properties), for sanity-checking in a GIS viewer (QGIS,
  geojson.io, kepler.gl, etc.).

**Note:** the source file re-emits a building once per overlapping tile-processing
block near block boundaries (see `partition_id`, e.g. `..._b00` vs `..._b01` for
the same `building_id`) — confirmed byte-for-byte identical duplicates, not
genuinely different candidates. `pull_buildings_openbuildings.py` drops these
(`drop_duplicates(subset="building_id")`) right after the bbox read. Without
this, ~1/3 of "buildings" in the AOI were the same physical building
double-counted.

Last run (5 km² AOI, radius 1.2616 km, post-dedup): **7,280 unique buildings**
intersecting the circle, ~1.413 km² of footprint area, 251 straddling the AOI
boundary. Height available for 7,010/7,280 (96.3%) — quality breakdown: HIGH
5,972, MEDIUM 529, LOW 432, LOW_RELAXED 77, none 270. Mean confidence 0.779.

(Prior Overture-based run, for reference: 7,916 buildings, ~1.59 km² footprint
area, 228 straddling the boundary, height available for only 326/7,916.)
