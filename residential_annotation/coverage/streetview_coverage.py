# Vendored as-is from Tower_Identification_Pipeline/coverage/streetview_coverage.py
# (same repo family, see /workspace/Prathyusha/Tower_Identification_Pipeline).
# No changes: this is pure Street View panorama discovery, no tower-specific
# logic anywhere in it -- exactly what this pipeline needs too, to know every
# candidate vantage point in the AOI before matching buildings to the nearest
# one in crops/build_crops.py. See PLAN.md's reuse table.
"""
Given a boundary polygon (a GeoPackage of one or more polygons), find every
real Google Street View panorama Google has captured inside it — via a graph
walk of the panorama connectivity graph (the same `links` that power the
forward/backward arrows in the Street View viewer itself), not by guessing at
sample spacing.

Free / no API key: uses the same unofficial GeoPhotoService endpoint (via the
`streetlevel` package) that tower_detection_app's own capture pipeline
already relies on for individual lookups — this just walks it broadly instead
of looking up one point at a time.

Usage (this pipeline uses --circle-center/--circle-radius-m, not a .gpkg,
matching the AOI already defined in pull_buildings_5km.py):
    python3 coverage/streetview_coverage.py \\
        --circle-center -6.1799849294994065 106.82189361088099 \\
        --circle-radius-m 1261.6 --out outputs/coverage.csv

Requires: streetlevel, shapely, aiohttp.
"""
import argparse
import asyncio
import csv
import io
import math
import os
import random
import sqlite3
import sys

import aiohttp
from aiohttp import ClientSession
from PIL import Image
from shapely import affinity as shapely_affinity
from shapely import wkb as shapely_wkb
from shapely.geometry import Point
from shapely.ops import unary_union

from streetlevel.streetview import find_panorama_async, find_panorama_by_id_async

# Tile grid dimensions per zoom level (width x height in tiles, 512px each).
# Matches tower_detection_app/backend/vendor/tower_scanner_modular/capture_v2.py's
# _download_tiles — same approach, copied rather than imported so this project
# stays independent of tower_detection_app's internals.
_ZOOM_GRID = {0: (1, 1), 1: (2, 1), 2: (4, 2), 3: (8, 4), 4: (16, 8), 5: (26, 13)}
_TILE_URL = (
    "https://streetviewpixels-pa.googleapis.com/v1/tile"
    "?cb_client=maps_sv.tactile&panoid={panoid}&x={x}&y={y}&zoom={zoom}&nbt=1&fover=2"
)
# The tile endpoint 403s without headers that look like a real browser request —
# no cookies needed (verified), just these. Metadata lookups (find_panorama_async
# etc.) don't need this; only the raw tile/image fetch does.
_CHROME_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/maps/",
    "sec-fetch-dest": "image",
    "sec-fetch-mode": "no-cors",
    "sec-fetch-site": "cross-site",
}


async def _download_tiles(pano_id: str, session: ClientSession, zoom: int = 3) -> Image.Image | None:
    """Downloads and stitches one panorama's tiles from the same tile endpoint
    the Maps browser client itself uses (no API key). Returns None if too many
    tiles fail, so the caller can retry at a lower zoom."""
    cols, rows = _ZOOM_GRID[zoom]
    tile_size = 512
    canvas = Image.new("RGB", (cols * tile_size, rows * tile_size))
    sem = asyncio.Semaphore(8)

    async def fetch_tile(x, y):
        url = _TILE_URL.format(panoid=pano_id, x=x, y=y, zoom=zoom)
        async with sem:
            for attempt in range(3):
                try:
                    async with session.get(url, headers=_CHROME_HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        if resp.status == 200:
                            return x, y, await resp.read()
                        if resp.status == 429:
                            await asyncio.sleep(2.0 ** attempt + random.uniform(0.3, 0.8))
                            continue
                        return x, y, None
                except (asyncio.TimeoutError, aiohttp.ClientError):
                    await asyncio.sleep(1.0)
            return x, y, None

    tiles = await asyncio.gather(*[fetch_tile(x, y) for y in range(rows) for x in range(cols)])
    ok = sum(1 for _, _, data in tiles if data is not None)
    if ok < cols * rows * 0.8:
        return None
    for x, y, data in tiles:
        if data is None:
            continue
        try:
            canvas.paste(Image.open(io.BytesIO(data)).convert("RGB"), (x * tile_size, y * tile_size))
        except Exception:
            pass
    return canvas


async def download_panorama_image(pano_id: str, session: ClientSession, out_dir: str) -> str | None:
    """Tries zoom 3 (4096x2048) first, falls back to zoom 2 if too many tiles
    are missing. Returns the saved file path, or None on total failure."""
    for zoom in (3, 2):
        img = await _download_tiles(pano_id, session, zoom=zoom)
        if img is not None:
            path = os.path.join(out_dir, f"{pano_id}.jpg")
            img.save(path, "JPEG", quality=90)
            return path
    return None

# GPKG binary geometry blob = an 8+N byte header (magic "GP", version, flags,
# srs_id) followed by standard WKB. N depends on the envelope flag bits.
_ENVELOPE_SIZES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def _parse_gpkg_geom(blob: bytes):
    assert blob[0:2] == b"GP", "not a GeoPackage geometry blob"
    flags = blob[3]
    envelope_code = (flags >> 1) & 0x07
    header_len = 8 + _ENVELOPE_SIZES[envelope_code]
    return shapely_wkb.loads(blob[header_len:])


def load_boundary(gpkg_path: str):
    """Reads every polygon in the GeoPackage's single feature table and
    unions them into one boundary geometry. Works via raw sqlite3 + shapely
    (GeoPackage is just SQLite) since fiona/geopandas aren't installed —
    avoids adding a dependency just to read one file."""
    con = sqlite3.connect(gpkg_path)
    cur = con.cursor()
    (table_name,) = cur.execute(
        "SELECT table_name FROM gpkg_contents WHERE data_type='features'"
    ).fetchone()
    (geom_col,) = cur.execute(
        "SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?", (table_name,)
    ).fetchone()
    geoms = [
        _parse_gpkg_geom(blob)
        for (blob,) in cur.execute(f"SELECT {geom_col} FROM {table_name}")
    ]
    con.close()
    return unary_union(geoms)


async def _throttled(sem, coro):
    async with sem:
        return await coro


async def discover_panoramas(boundary, seed_lat: float, seed_lon: float,
                              max_panoramas: int | None = None, concurrency: int = 4,
                              seed_radius: int = 100, results: dict | None = None,
                              visited: set | None = None, session: ClientSession | None = None):
    """BFS over the Street View link graph, starting near (seed_lat, seed_lon),
    expanding only to neighbors whose real capture point falls inside
    `boundary`. Returns (results, hit_cap, failed_lookups).

    `max_panoramas=None` (the default) means genuinely unlimited -- the walk
    only ever stops because it ran out of real neighbors inside `boundary`,
    i.e. it reached the polygon's edges. The boundary is what determines
    coverage, not this number. Pass an actual int only if you deliberately
    want a partial/quick sample instead of full coverage.

    `results`/`visited` can be passed in (and are mutated in place) so multiple
    calls from different seeds accumulate into one shared set — needed because
    a single BFS only ever reaches panoramas connected to its seed by real
    links; a boundary with a genuinely disconnected street cluster (cut off by
    a river, a gated area, a highway with no pedestrian-level connectivity,
    etc.) would otherwise be silently missed with no error, just a suspiciously
    low count. See find_extra_seeds() for the multi-seed probe that catches this."""
    sem = asyncio.Semaphore(concurrency)
    if results is None:
        results = {}
    if visited is None:
        visited = set()
    failed = 0
    hit_cap = False

    async def _run(session):
        nonlocal failed, hit_cap
        seed = await find_panorama_async(seed_lat, seed_lon, session, radius=seed_radius)
        if seed is None:
            raise RuntimeError(
                f"No Street View panorama found within {seed_radius}m of the seed point "
                f"({seed_lat}, {seed_lon}) — pick a different seed inside the boundary."
            )
        if not boundary.contains(Point(seed.lon, seed.lat)):
            print(
                f"  warning: seed panorama {seed.id} landed at ({seed.lat}, {seed.lon}), "
                f"just outside the boundary — starting the walk from there anyway.",
                file=sys.stderr,
            )

        if seed.id in visited:
            return  # this seed is already part of a previously-discovered cluster — nothing new here
        queue = [seed.id]
        visited.add(seed.id)
        # Seed metadata itself doesn't need re-fetching by ID — record it directly.
        results[seed.id] = {
            "pano_id": seed.id, "lat": seed.lat, "lon": seed.lon,
            "heading_deg": _deg(seed.heading), "date": str(seed.date) if seed.date else "",
            "street_names": _street_names(seed),
        }

        while queue and (max_panoramas is None or len(results) < max_panoramas):
            batch, queue = queue[: concurrency * 4], queue[concurrency * 4:]
            fetched = await asyncio.gather(
                *[_throttled(sem, find_panorama_by_id_async(pid, session)) for pid in batch],
                return_exceptions=True,
            )
            for pano in fetched:
                if pano is None or isinstance(pano, Exception):
                    failed += 1
                    continue
                for link in pano.links:
                    lp = link.pano
                    if lp is None or lp.id in visited:
                        continue
                    visited.add(lp.id)
                    if lp.lat is None or lp.lon is None:
                        continue
                    if not boundary.contains(Point(lp.lon, lp.lat)):
                        continue  # real neighbor, but outside the requested area
                    if max_panoramas is not None and len(results) >= max_panoramas:
                        hit_cap = True
                        break
                    results[lp.id] = {
                        "pano_id": lp.id, "lat": lp.lat, "lon": lp.lon,
                        "heading_deg": _deg(getattr(lp, "heading", None)),
                        "date": str(lp.date) if getattr(lp, "date", None) else "",
                        "street_names": _street_names(lp),
                    }
                    queue.append(lp.id)
                if hit_cap:
                    break

    if session is not None:
        await _run(session)
    else:
        async with ClientSession() as new_session:
            await _run(new_session)

    return list(results.values()), hit_cap, failed


def sample_grid_candidates(boundary, grid_size: int = 10) -> list:
    """Evenly-spaced (lat, lon) points inside the polygon (not just its
    bbox) — used to probe for Street View coverage disconnected from the
    primary seed's BFS walk."""
    import numpy as np
    minx, miny, maxx, maxy = boundary.bounds
    xs = np.linspace(minx, maxx, grid_size)
    ys = np.linspace(miny, maxy, grid_size)
    return [(y, x) for x in xs for y in ys if boundary.contains(Point(x, y))]


async def find_working_seed(boundary, session: ClientSession, seed_lat: float, seed_lon: float,
                             seed_radius: int = 100, grid_size: int = 10) -> tuple:
    """Tries the requested (seed_lat, seed_lon) first; if there's no Street
    View coverage within seed_radius of it (e.g. the polygon's representative
    point happened to land in a field, a lake, a gated area with no imagery),
    falls back to probing the boundary's candidate grid in random order until
    one has coverage, instead of crashing and making the caller guess a
    manual seed by hand. Returns the actual panorama's (lat, lon) to seed the
    real walk from. Raises only if truly nothing in the boundary has coverage."""
    seed = await find_panorama_async(seed_lat, seed_lon, session, radius=seed_radius)
    if seed is not None:
        return seed.lat, seed.lon

    print(f"  no Street View within {seed_radius}m of ({seed_lat}, {seed_lon}) — "
          f"trying random points across the boundary instead ...", file=sys.stderr)
    candidates = sample_grid_candidates(boundary, grid_size)
    random.shuffle(candidates)
    for lat, lon in candidates:
        pano = await find_panorama_async(lat, lon, session, radius=seed_radius)
        if pano is not None:
            print(f"  found coverage near ({lat:.5f}, {lon:.5f}) -> panorama at "
                  f"({pano.lat}, {pano.lon})", file=sys.stderr)
            return pano.lat, pano.lon

    raise RuntimeError(
        f"No Street View panorama found anywhere in the boundary (tried the requested seed "
        f"plus {len(candidates)} random points across it) — this area may genuinely have no coverage."
    )


async def find_extra_seeds(boundary, session: ClientSession, visited: set,
                            grid_size: int = 10, probe_radius: int = 150) -> list:
    """Probes a grid of candidate points for Street View coverage not already
    reachable from the panoramas we've found so far. Returns a list of
    (lat, lon) seeds for genuinely new, disconnected clusters."""
    candidates = sample_grid_candidates(boundary, grid_size)
    new_seeds = []
    for lat, lon in candidates:
        pano = await find_panorama_async(lat, lon, session, radius=probe_radius)
        if pano is not None and pano.id not in visited:
            new_seeds.append((pano.lat, pano.lon))
    return new_seeds


def _deg(radians):
    return round(radians * 57.29577951308232, 1) if radians is not None else None


def _street_names(pano):
    labels = getattr(pano, "street_names", None)
    if not labels:
        return ""
    return "; ".join(sorted({label.name.value for label in labels}))


async def main_async(args):
    if args.circle_center is not None:
        clat, clon = args.circle_center
        # meters -> degrees: 111,320 m/deg latitude everywhere, longitude scaled by cos(lat)
        # since a degree of longitude covers less real ground the further from the equator.
        radius_deg_lat = args.circle_radius_m / 111320.0
        radius_deg_lon = args.circle_radius_m / (111320.0 * math.cos(math.radians(clat)))
        # buffer() takes one radius, not independent lat/lon ones -- build a circle in
        # lat-scaled units then stretch it back, rather than accepting the (usually tiny,
        # but non-zero away from the equator) distortion of a plain single-radius buffer.
        boundary = shapely_affinity.scale(
            Point(clon, clat).buffer(radius_deg_lat), xfact=radius_deg_lon / radius_deg_lat, yfact=1.0,
            origin=(clon, clat))
        print(f"Circular boundary: center=({clat}, {clon}), radius={args.circle_radius_m}m "
              f"(no .gpkg file involved)", file=sys.stderr)
    else:
        print(f"Reading boundary from {args.gpkg} ...", file=sys.stderr)
        boundary = load_boundary(args.gpkg)
    minx, miny, maxx, maxy = boundary.bounds
    print(f"  boundary bbox: lon [{minx:.5f}, {maxx:.5f}], lat [{miny:.5f}, {maxy:.5f}]", file=sys.stderr)

    if args.seed_lat is not None and args.seed_lon is not None:
        seed_lat, seed_lon = args.seed_lat, args.seed_lon
    elif args.circle_center is not None:
        seed_lat, seed_lon = args.circle_center
    else:
        rep = boundary.representative_point()
        seed_lat, seed_lon = rep.y, rep.x
    print(f"  seeding walk from ({seed_lat}, {seed_lon})", file=sys.stderr)

    async with ClientSession() as seed_session:
        seed_lat, seed_lon = await find_working_seed(
            boundary, seed_session, seed_lat, seed_lon,
            seed_radius=args.seed_radius, grid_size=args.seed_grid_size,
        )

    shared_results: dict = {}
    shared_visited: set = set()
    _, hit_cap, failed = await discover_panoramas(
        boundary, seed_lat, seed_lon, max_panoramas=args.max_panoramas,
        concurrency=args.concurrency, seed_radius=args.seed_radius,
        results=shared_results, visited=shared_visited,
    )
    print(f"  primary walk found {len(shared_results)} panoramas ({failed} lookups failed/skipped)",
          file=sys.stderr)

    n_extra_seeds = 0
    if not hit_cap and not args.no_disconnected_check:
        print(f"  probing a {args.seed_grid_size}x{args.seed_grid_size} grid for street clusters "
              f"disconnected from the primary walk ...", file=sys.stderr)
        async with ClientSession() as session:
            extra_seeds = await find_extra_seeds(boundary, session, shared_visited,
                                                  grid_size=args.seed_grid_size)
            for lat, lon in extra_seeds:
                if args.max_panoramas is not None and len(shared_results) >= args.max_panoramas:
                    hit_cap = True
                    break
                before = len(shared_results)
                _, cap_hit, extra_failed = await discover_panoramas(
                    boundary, lat, lon, max_panoramas=args.max_panoramas,
                    concurrency=args.concurrency, seed_radius=args.seed_radius,
                    results=shared_results, visited=shared_visited, session=session,
                )
                hit_cap = hit_cap or cap_hit
                failed += extra_failed
                gained = len(shared_results) - before
                if gained > 0:
                    n_extra_seeds += 1
                    print(f"    extra seed ({lat:.5f}, {lon:.5f}) found a disconnected cluster "
                          f"of {gained} more panoramas", file=sys.stderr)

    results = list(shared_results.values())
    print(f"\nFound {len(results)} panoramas inside the boundary "
          f"({failed} lookups failed/skipped, {n_extra_seeds} disconnected cluster(s) recovered).",
          file=sys.stderr)
    if hit_cap:
        print(f"  NOTE: hit --max-panoramas cap of {args.max_panoramas} — "
              f"this is a PARTIAL result, not full coverage. Re-run with a "
              f"higher cap for the complete set.", file=sys.stderr)

    if args.images_dir:
        os.makedirs(args.images_dir, exist_ok=True)
        print(f"\nDownloading {len(results)} panorama images to {args.images_dir}/ "
              f"(concurrency={args.image_concurrency}) ...", file=sys.stderr)
        sem = asyncio.Semaphore(args.image_concurrency)
        saved = 0

        async def _fetch_one(session, row):
            nonlocal saved
            async with sem:
                path = await download_panorama_image(row["pano_id"], session, args.images_dir)
            row["image_path"] = path or ""
            if path:
                saved += 1
            else:
                print(f"  warning: could not fetch tiles for {row['pano_id']}", file=sys.stderr)

        async with ClientSession() as session:
            done = 0
            for i in range(0, len(results), 20):
                chunk = results[i:i + 20]
                await asyncio.gather(*[_fetch_one(session, row) for row in chunk])
                done += len(chunk)
                print(f"  {done}/{len(results)} processed ({saved} saved)", file=sys.stderr)
        print(f"Saved {saved}/{len(results)} panorama images "
              f"({len(results) - saved} failed — likely no imagery at that zoom).", file=sys.stderr)
    else:
        for row in results:
            row["image_path"] = ""

    fieldnames = ["pano_id", "lat", "lon", "heading_deg", "date", "street_names", "image_path"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {args.out}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gpkg", nargs="?", default=None,
                   help="Path to the GeoPackage boundary file. Omit if using --circle-center instead.")
    p.add_argument("--circle-center", type=float, nargs=2, default=None, metavar=("LAT", "LON"),
                   help="Alternative to a .gpkg file: a simple circular boundary around one point. "
                        "No GeoPackage-writing tools needed (fiona/geopandas aren't installed) -- "
                        "built directly as a shapely buffer, fine for a quick single-point scan. "
                        "Requires --circle-radius-m too.")
    p.add_argument("--circle-radius-m", type=float, default=None,
                   help="Radius in meters for --circle-center's boundary")
    p.add_argument("--out", default="outputs/coverage/out_all.csv", help="Output CSV path")
    p.add_argument("--seed-lat", type=float, default=None, help="Optional explicit seed latitude")
    p.add_argument("--seed-lon", type=float, default=None, help="Optional explicit seed longitude")
    p.add_argument("--seed-radius", type=int, default=100,
                   help="Meters to search for a Street View panorama near the seed point — raise this "
                        "if the auto-picked seed (polygon representative point) lands off-road with no "
                        "coverage nearby")
    p.add_argument("--seed-grid-size", type=int, default=10,
                   help="After the primary walk, probe an NxN grid of points across the boundary "
                        "for Street View coverage disconnected from it (a separate street cluster "
                        "cut off from the seed by a river, gated area, etc.) and merge in any found")
    p.add_argument("--no-disconnected-check", action="store_true",
                   help="Skip the disconnected-cluster grid probe (faster, but may silently miss "
                        "isolated street clusters not reachable from the primary seed)")
    p.add_argument("--max-panoramas", type=int, default=None,
                   help="Off by default -- genuinely unlimited. The boundary polygon alone determines "
                        "coverage: the walk only follows neighbors whose real coordinates fall inside it, "
                        "so it already stops on its own once it reaches the polygon's edges. Pass an int "
                        "here only if you deliberately want a partial/quick sample instead of full "
                        "coverage (e.g. for a fast smoke test on a huge region).")
    p.add_argument("--concurrency", type=int, default=4, help="Concurrent in-flight lookups")
    p.add_argument("--images-dir", default=None,
                   help="If set, also download each panorama's stitched equirectangular "
                        "image (JPEG) into this directory. Off by default — discovery is "
                        "cheap, downloading full panoramas for thousands of points is not.")
    p.add_argument("--image-concurrency", type=int, default=3,
                   help="Concurrent panorama image downloads (each one fetches up to 32 "
                        "tiles internally, so keep this lower than --concurrency)")
    args = p.parse_args()
    if args.gpkg is None and args.circle_center is None:
        p.error("either a gpkg path or --circle-center LAT LON (with --circle-radius-m) is required")
    if args.circle_center is not None and args.circle_radius_m is None:
        p.error("--circle-center requires --circle-radius-m")
    if args.gpkg is not None and args.circle_center is not None:
        p.error("pass either gpkg or --circle-center, not both")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
