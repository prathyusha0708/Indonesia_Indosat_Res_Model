"""
NEW module -- nothing like this exists in Tower_Identification_Pipeline (it
never needed one: a tower's location was the thing being *found*, so there
was nothing to check line-of-sight against ahead of time). We already know
every building's exact location, so we can cheaply ask, ahead of spending any
SAM3/YOLO time: "standing at this Street View point, is some OTHER building
physically in the way, blocking the view of this one?"

v2: occlusion checked against the target's full FOOTPRINT POLYGON, not a
single centroid point.

v3: only counts as occluding if it covers MORE than MIN_OCCLUSION_FRACTION
(30%) of the relevant angular window (a trivial corner-graze intersection doesn't
disqualify a view where the target is still mostly visible).

v4 (per direct feedback after a real mismatch: a building flagged OCCLUDED
turned out, in the actual photo, to be completely unobstructed -- the
"occluding" building wasn't even in frame): v3 measured the occlusion
fraction against the TARGET's own polygon-derived angular span. That's
fragile when the camera sits close to one corner of an irregular (non-
convex) target footprint -- a nearby vertex's bearing swings wildly for a
tiny position change, inflating the computed span far beyond the building's
real visible width. Worse, crops/build_crops.py stopped using a polygon-
derived FOV for the actual photo crop back in v4 of THAT module (fixed
120-degree window now, not a per-building computed one) -- so the occlusion
check and the actual photo were measuring two different windows entirely.

Fixed: occlusion is now measured against the SAME fixed FOV window the photo
is actually cropped to (see build_view_window()), not the target polygon's
own (unreliable) angular extent. What counts as "in the way" now means "in
the way of what the camera actually photographs" -- which is the only thing
that was ever supposed to matter.

Why this is cheap: we already have every building's full footprint polygon
in one GeoDataFrame (buildings_aoi.geojson) for the whole AOI. No new data,
no new imagery -- pure geometry against data we already downloaded. Must be
called with everything in the same *projected* (UTM, meters) CRS as the rest
of this pipeline's geometry -- see geometry/bearing.py's module docstring for
why planar math is used throughout.
"""
import math

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.strtree import STRtree

try:
    from geometry.bearing import bearing_xy
except ImportError:
    # Running this file directly (python3 geometry/line_of_sight.py, e.g. for
    # the self-test below) -- the parent dir isn't on sys.path by default in
    # that case, unlike when imported normally from crops/build_crops.py
    # (which already adds it). Add it here too so both ways of running work.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from geometry.bearing import bearing_xy

MIN_OCCLUSION_FRACTION = 0.30  # candidate must cover MORE than this much of the visible window to count
                                # (exactly at the threshold still counts as clear -- see the ">" below)
DEFAULT_FOV_DEG = 120.0        # must match crops/build_crops.py's --fov-deg default
DEFAULT_MAX_SIGHT_M = 200.0    # generous cap for the window's far extent (well beyond match distances used)


def build_index(buildings: gpd.GeoDataFrame) -> STRtree:
    """Spatial index over every building polygon (projected CRS), built once
    and reused across all is_occluded() calls -- an STRtree query is a fast
    bounding-box lookup, not a linear scan over every building each time."""
    return STRtree(buildings.geometry.values)


def build_view_window(cam_xy: tuple[float, float], center_bearing: float, fov_deg: float,
                       max_distance_m: float = DEFAULT_MAX_SIGHT_M, n_arc_points: int = 9) -> Polygon:
    """Sector-shaped polygon for the actual FIXED fov_deg window the camera is
    aimed at (the same window slice_direction() crops the photo to), fanning
    out from the camera to max_distance_m. This -- not the target's own
    footprint -- is the region occlusion is measured against, so what's
    flagged as occluding matches what the photo itself actually shows."""
    half = fov_deg / 2.0
    step = fov_deg / (n_arc_points - 1)
    pts = [cam_xy]
    for i in range(n_arc_points):
        a = math.radians(center_bearing - half + i * step)
        pts.append((cam_xy[0] + math.sin(a) * max_distance_m, cam_xy[1] + math.cos(a) * max_distance_m))
    return Polygon(pts)


def _angular_span(cam_xy: tuple[float, float], polygon: Polygon, ref_bearing: float) -> tuple[float, float]:
    """(min_offset, max_offset) in degrees, relative to ref_bearing, spanning
    every vertex of polygon as seen from cam_xy -- wraparound-safe (offsets
    wrapped to (-180, 180] before min/max)."""
    offsets = []
    for x, y in polygon.exterior.coords:
        b = bearing_xy(x - cam_xy[0], y - cam_xy[1])
        offsets.append(((b - ref_bearing + 180.0) % 360.0) - 180.0)
    return min(offsets), max(offsets)


def occlusion_fraction_in_window(cam_xy: tuple[float, float], center_bearing: float, fov_deg: float,
                                  candidate_polygon: Polygon) -> float:
    """What fraction of the FIXED fov_deg window (centered on center_bearing
    -- the same window the photo is actually cropped to) is covered by
    candidate_polygon's angular span, as seen from cam_xy. This is what
    decides whether a candidate counts as a real occluder -- NOT how much of
    the target's own (potentially unreliable, if the target is irregular and
    the camera sits close to a corner) polygon shape it covers."""
    half = fov_deg / 2.0
    c_min, c_max = _angular_span(cam_xy, candidate_polygon, center_bearing)
    overlap = max(0.0, min(half, c_max) - max(-half, c_min))
    return overlap / fov_deg


def find_occluders(
    cam_xy: tuple[float, float],
    target_polygon: Polygon,
    target_building_id: str,
    buildings: gpd.GeoDataFrame,
    tree: STRtree,
    id_col: str = "building_id",
    min_fraction: float = MIN_OCCLUSION_FRACTION,
    fov_deg: float = DEFAULT_FOV_DEG,
) -> list:
    """
    Returns every OTHER building that (a) sits BETWEEN the camera and the
    target (nearer than the target itself -- something behind it can't block
    it), (b) falls within the fixed fov_deg window the photo is actually
    cropped to, AND (c) covers at least `min_fraction` of that window --
    i.e. a real, meaningful block of what's actually photographed, not a
    trivial graze and not a polygon-shape artifact. List of dicts
    {building_id, distance_m, occlusion_fraction}, sorted by
    occlusion_fraction descending (most-blocking first).

    `cam_xy` is (x, y), `target_polygon` is the target's own footprint (used
    only to get its centroid bearing -- the window's center -- and its own
    distance from the camera, for the "must be nearer than the target" gate)
    -- both in the same projected CRS as `buildings`. `buildings`/`tree`
    should be the *same* GeoDataFrame/index for every call in a batch (build
    once with build_index(), reuse per building).
    """
    from shapely.geometry import Point

    cam_pt = Point(cam_xy)
    center_bearing = bearing_xy(target_polygon.centroid.x - cam_xy[0], target_polygon.centroid.y - cam_xy[1])
    target_dist_m = cam_pt.distance(target_polygon)

    window = build_view_window(cam_xy, center_bearing, fov_deg)
    hits = tree.query(window)
    occluders = []
    for idx in hits:
        row = buildings.iloc[idx]
        if row[id_col] == target_building_id:
            continue
        cand_dist_m = cam_pt.distance(row.geometry)
        if cand_dist_m >= target_dist_m:
            continue  # at or beyond the target -- can't be in front of it
        if not row.geometry.intersects(window):
            continue
        frac = occlusion_fraction_in_window(cam_xy, center_bearing, fov_deg, row.geometry)
        if frac > min_fraction:  # strict: exactly at min_fraction still counts as clear, not occluded
            occluders.append({"building_id": row[id_col], "distance_m": cand_dist_m, "occlusion_fraction": frac})
    occluders.sort(key=lambda o: o["occlusion_fraction"], reverse=True)
    return occluders


def is_occluded(
    cam_xy: tuple[float, float],
    target_polygon: Polygon,
    target_building_id: str,
    buildings: gpd.GeoDataFrame,
    tree: STRtree,
    id_col: str = "building_id",
    min_fraction: float = MIN_OCCLUSION_FRACTION,
    fov_deg: float = DEFAULT_FOV_DEG,
) -> bool:
    """True if find_occluders() found anything -- see that function for which
    building(s) specifically are in the way and by how much."""
    return len(find_occluders(cam_xy, target_polygon, target_building_id, buildings, tree,
                               id_col, min_fraction, fov_deg)) > 0


if __name__ == "__main__":
    # Quick sanity check -- run directly: python3 geometry/line_of_sight.py
    from shapely.geometry import Point

    cam = (0.0, 0.0)
    # Target 45m north (its centroid at (0,50), so bearing 0 deg / due north).
    target_poly = Polygon([(-15, 45), (15, 45), (15, 55), (-15, 55)])

    # Case 1: a wide blocker (24m wide) close in front -- should clearly occlude.
    buildings_wide_blocker = gpd.GeoDataFrame(
        {"building_id": ["target", "blocker"]},
        geometry=[target_poly, Polygon([(-12, 20), (12, 20), (12, 30), (-12, 30)])],
    )
    tree1 = build_index(buildings_wide_blocker)
    occ1 = find_occluders(cam, target_poly, "target", buildings_wide_blocker, tree1)
    frac_str = f"{occ1[0]['occlusion_fraction']:.2f}" if occ1 else "n/a"
    print(f"Wide blocker in front: occluded = {len(occ1) > 0} (expect True), fraction = {frac_str}")

    # Case 2: a tiny blocker (2m) -- should NOT count (well under 25% of the
    # 120 deg window even close up).
    buildings_tiny = gpd.GeoDataFrame(
        {"building_id": ["target", "tiny"]},
        geometry=[target_poly, Polygon([(13.5, 20), (15.5, 20), (15.5, 22), (13.5, 22)])],
    )
    tree2 = build_index(buildings_tiny)
    occ2 = find_occluders(cam, target_poly, "target", buildings_tiny, tree2)
    print(f"Tiny blocker: occluded = {len(occ2) > 0} (expect False)")

    # Case 3: a blocker well outside the 120 deg window (e.g. due east, 90 deg off).
    buildings_bystander = gpd.GeoDataFrame(
        {"building_id": ["target", "bystander"]},
        geometry=[target_poly, Point(60.0, 0.0).buffer(10.0)],
    )
    tree3 = build_index(buildings_bystander)
    r3 = is_occluded(cam, target_poly, "target", buildings_bystander, tree3)
    print(f"Bystander outside the 120 deg window: occluded = {r3} (expect False)")

    # Case 4: a big building dead ahead but BEHIND the target (80m, target is
    # only ~45m away) -- must never count, even though it's a wide match for
    # bearing and would easily clear the 25% threshold if distance weren't checked.
    buildings_behind = gpd.GeoDataFrame(
        {"building_id": ["target", "behind"]},
        geometry=[target_poly, Polygon([(-20, 75), (20, 75), (20, 85), (-20, 85)])],
    )
    tree4 = build_index(buildings_behind)
    r4 = is_occluded(cam, target_poly, "target", buildings_behind, tree4)
    print(f"Wide building behind the target: occluded = {r4} (expect False)")

    # Case 5: the real bug case this fix targets -- camera very close to one
    # corner of an L-shaped (non-convex) target, with a neighbor immediately
    # beside that corner. Old (v3) logic inflated the target's own angular
    # span here and wrongly called this occluded; new logic should not,
    # since the neighbor sits outside the fixed 120 deg window centered on
    # the target's CENTROID bearing.
    l_shaped_target = Polygon([(0, 1), (10, 1), (10, 10), (5, 10), (5, 5), (0, 5)])  # camera sits near (0,1) corner
    neighbor = Polygon([(10.5, 0), (14, 0), (14, 3), (10.5, 3)])  # immediately east of the L's near corner
    cam_near_corner = (0.2, 1.2)
    buildings_l = gpd.GeoDataFrame({"building_id": ["target", "neighbor"]}, geometry=[l_shaped_target, neighbor])
    tree5 = build_index(buildings_l)
    r5 = is_occluded(cam_near_corner, l_shaped_target, "target", buildings_l, tree5)
    print(f"Neighbor beside a near corner of an L-shaped target: occluded = {r5} (expect False)")
