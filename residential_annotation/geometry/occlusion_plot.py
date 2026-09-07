"""
NEW module. A top-down (lat/lon) debug plot per view: target building,
occluder building (if any), nearby buildings for context, the camera point,
and the sightline between them -- so occlusion can be checked geometrically,
not just guessed at from the street-level photo. Saved alongside each
marked panorama crop in crops/build_crops.py.

Plain matplotlib, no basemap tiles (no network dependency, no extra
service) -- just the building footprints themselves, which is all that
matters for judging why something was flagged occluded/clear. If you want
this rendered against a real map instead, the same lat/lon data can be
dropped into kepler.gl or QGIS directly (see PLAN.md's earlier kepler.gl
discussion) -- this is the "always works, zero setup" version.
"""
import matplotlib
matplotlib.use("Agg")  # no display in this environment -- render straight to file
import matplotlib.pyplot as plt


def plot_occlusion(
    target_poly, occluder_polys: list, nearby_polys: list,
    cam_lat: float, cam_lon: float, out_path: str, title: str = "",
):
    """
    All polygons/points in WGS84 (lon, lat) -- matches how shapely stores
    coordinates for geometries read from our GeoJSON files, so no
    reprojection needed just to draw a picture.

    target_poly: the building this view was aimed at (shapely Polygon).
    occluder_polys: 0+ polygons of buildings flagged as blocking this view.
    nearby_polys: other buildings in the area, for context only (light gray).
    cam_lat/cam_lon: the Street View panorama's own position.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    for poly in nearby_polys:
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, color="0.85", edgecolor="0.6", linewidth=0.6, zorder=1)

    for poly in occluder_polys:
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, color="orange", edgecolor="darkorange", linewidth=1.5, zorder=3, label="occluder")

    xs, ys = target_poly.exterior.xy
    ax.fill(xs, ys, color="dodgerblue", edgecolor="blue", linewidth=1.5, zorder=4, label="target")

    tc = target_poly.centroid
    ax.plot(cam_lon, cam_lat, marker="*", color="red", markersize=16, zorder=5, label="camera")
    ax.plot([cam_lon, tc.x], [cam_lat, tc.y], color="red", linewidth=1.3, linestyle="--", zorder=4)

    # De-dupe legend entries (multiple occluder polygons would otherwise repeat the label).
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=7)

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
