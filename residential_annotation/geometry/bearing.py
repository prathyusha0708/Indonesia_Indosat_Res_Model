"""
NEW module (nothing like this exists in Tower_Identification_Pipeline -- every
"bearing" there goes pixel-position-in-a-crop -> compass angle, never
lat/lon-pair -> compass angle; see PLAN.md's reuse table).

Two things, both driven off the same planar-projection idea:

1. bearing_deg(lat1, lon1, lat2, lon2) -- the compass heading FROM point 1
   TO point 2. This is the `heading_deg` that direction_view.slice_direction()
   needs to aim a crop at a specific building.

2. heading_and_fov_for_building(cam_lat, cam_lon, building_polygon) -- given
   a camera point and a building's full footprint polygon (which we already
   have from Overture), returns (heading_deg, fov_deg): the bearing to aim at
   the building's centroid, AND how wide a field of view is needed to frame
   the whole footprint as seen from that camera point. This is what lets
   every building get an auto-sized crop instead of one fixed FOV for all.

Why planar (UTM-projected) math instead of the standard spherical
great-circle bearing formula: pano-to-building distances in this pipeline are
capped at ~80m (see PLAN.md defaults). At that scale flat-earth trig is exact
to well under a millimeter, and we're already reprojecting buildings into a
local UTM CRS elsewhere in this project (pull_buildings_5km.py's
utm_epsg_for() + to_crs(epsg=...)) for the same reason. Reusing that same
projected coordinate space here means one consistent geometry system across
the whole pipeline, instead of separate lat/lon trig living only in this file.
"""
import math

from pyproj import Transformer
from shapely.geometry import Polygon


def _transformer(epsg: int) -> Transformer:
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def project(lat: float, lon: float, epsg: int) -> tuple[float, float]:
    """lat/lon (WGS84) -> (x, y) meters in the given UTM EPSG."""
    x, y = _transformer(epsg).transform(lon, lat)
    return x, y


def bearing_xy(dx: float, dy: float) -> float:
    """Compass bearing (0-360, 0=N, clockwise) of the vector (dx=east, dy=north)."""
    return math.degrees(math.atan2(dx, dy)) % 360.0


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float, epsg: int) -> float:
    """Compass bearing FROM (lat1, lon1) TO (lat2, lon2), 0-360, 0=N, clockwise."""
    x1, y1 = project(lat1, lon1, epsg)
    x2, y2 = project(lat2, lon2, epsg)
    return bearing_xy(x2 - x1, y2 - y1)


def _angular_diff(a: float, b: float) -> float:
    """Signed difference a-b wrapped to (-180, 180], so spans across the 0/360
    seam (e.g. bearings 350 and 10) don't come out as 340 instead of 20."""
    return (a - b + 180.0) % 360.0 - 180.0


def heading_and_fov_for_building(
    cam_lat: float,
    cam_lon: float,
    building_polygon: Polygon,
    epsg: int,
    padding_ratio: float = 0.25,
    min_fov_deg: float = 40.0,
    max_fov_deg: float = 100.0,
) -> tuple[float, float]:
    """
    Returns (heading_deg, fov_deg):
      - heading_deg: bearing from (cam_lat, cam_lon) to the building's centroid
        -- what to pass as slice_direction()'s heading_deg.
      - fov_deg: wide enough to cover every exterior vertex of the building's
        footprint as seen from the camera point, padded by `padding_ratio` and
        clamped to [min_fov_deg, max_fov_deg] -- what to pass as fov_h_deg.

    Method: bearing from the camera to the centroid is the reference (0
    offset). Every polygon vertex's bearing is expressed as a signed offset
    from that reference (via _angular_diff, so it's wraparound-safe even if
    the building straddles due-north). The offsets' min/max give the true
    angular span without ever computing a raw max-minus-min on raw compass
    bearings (which breaks near the 0/360 seam).
    """
    centroid = building_polygon.centroid
    ref_heading = bearing_deg(cam_lat, cam_lon, centroid.y, centroid.x, epsg)

    coords = list(building_polygon.exterior.coords)
    offsets = []
    for lon, lat in coords:
        h = bearing_deg(cam_lat, cam_lon, lat, lon, epsg)
        offsets.append(_angular_diff(h, ref_heading))

    span = max(offsets) - min(offsets)
    # Re-center heading on the true midpoint of the vertex spread, not just the
    # centroid's own bearing -- for an irregular/rotated footprint these can
    # differ slightly, and we want the crop centered on the visible extent.
    mid_offset = (max(offsets) + min(offsets)) / 2.0
    heading_deg = (ref_heading + mid_offset) % 360.0

    fov_deg = span * (1.0 + padding_ratio)
    fov_deg = max(min_fov_deg, min(max_fov_deg, fov_deg))

    return heading_deg, fov_deg


if __name__ == "__main__":
    # Quick sanity checks -- run directly: python3 geometry/bearing.py
    epsg = 32748  # same zone as pull_buildings_5km.py for our Jakarta AOI

    lat0, lon0 = -6.1799849294994065, 106.82189361088099

    # A point ~30m due north should read ~0 deg; ~30m due east should read ~90.
    x0, y0 = project(lat0, lon0, epsg)
    import pyproj
    inv = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    def latlon_at_offset(dx, dy):
        lon, lat = inv.transform(x0 + dx, y0 + dy)
        return lat, lon

    tests = {
        "north (expect ~0)": latlon_at_offset(0, 30),
        "east (expect ~90)": latlon_at_offset(30, 0),
        "south (expect ~180)": latlon_at_offset(0, -30),
        "west (expect ~270)": latlon_at_offset(-30, 0),
    }
    print("bearing_deg sanity checks from the Indosat point:")
    for label, (lat, lon) in tests.items():
        b = bearing_deg(lat0, lon0, lat, lon, epsg)
        print(f"  {label:22s} -> {b:6.2f} deg")

    # A small square footprint straddling due north of the camera, to check
    # the wraparound-safe angular span math.
    square = Polygon([
        latlon_at_offset(-5, 40)[::-1],
        latlon_at_offset(5, 40)[::-1],
        latlon_at_offset(5, 50)[::-1],
        latlon_at_offset(-5, 50)[::-1],
    ])
    heading, fov = heading_and_fov_for_building(lat0, lon0, square, epsg)
    print(f"\nSquare building ~45m north, ~10m wide:")
    print(f"  heading_deg = {heading:.2f} (expect ~0)")
    print(f"  fov_deg     = {fov:.2f} (raw span ~{2*math.degrees(math.atan2(5, 40)):.2f}, padded)")
