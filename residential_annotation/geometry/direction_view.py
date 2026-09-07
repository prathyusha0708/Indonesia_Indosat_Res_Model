# Vendored (trimmed) from Tower_Identification_Pipeline/detection/direction_view.py
# (same repo family, see /workspace/Prathyusha/Tower_Identification_Pipeline).
# Only slice_direction() is kept -- it already takes an arbitrary heading_deg
# + fov_h_deg, not just a cardinal direction, so it needs zero changes for our
# use. The source file also has slice_all_directions(), a wrapper that calls
# this 4 times at fixed 0/90/180/270 headings (the tower pipeline's "N/E/S/W
# crop" behavior) -- we don't use that wrapper at all: every building gets its
# own single heading_deg/fov_h_deg computed in geometry/bearing.py, so that
# wrapper would just be dead code here. Dropped to keep this file to only what
# this pipeline actually calls.
"""
Slices a full equirectangular Street View panorama into one tall "direction
view" — the FULL vertical extent of the panorama (nadir to zenith) for a
horizontal wedge centered on an arbitrary compass heading — one image showing
everything from street level up through rooftops to sky for that direction.

This is a plain equirectangular column-slice: no reprojection math needed,
because equirect rows already map linearly to elevation (row 0 = straight up,
middle row = horizon, bottom row = straight down) and columns map linearly to
azimuth — true regardless of which row you're looking at. Trade-off: content
far from the row-center (near-zenith rooftops, near-nadir ground) is
horizontally stretched/compressed same as the full panorama would be, just
confined to one wedge instead of the whole 360°.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def heading_to_column(heading_deg: float, pano_heading_deg: float, width: int) -> float:
    """Pixel column in a `width`-wide equirectangular panorama corresponding to
    `heading_deg` (compass heading), correcting for the panorama's own capture
    orientation. +180: pano.heading (from streetlevel) is the capture
    vehicle's heading, but the raw equirect's front-center column faces the
    OPPOSITE way from that -- same correction already applied on the
    frontend's Street View deep-link (ResultCard.jsx), baked in here instead
    so every use of this is right at the source."""
    return ((heading_deg - pano_heading_deg + 180.0) / 360.0) % 1.0 * width


def slice_direction(pano: Image.Image, heading_deg: float, pano_heading_deg: float = 0.0,
                     fov_h_deg: float = 100.0) -> Image.Image:
    """Returns a tall crop covering `fov_h_deg` degrees of azimuth centered on
    `heading_deg` (compass heading, corrected for the pano's own capture
    orientation), keeping the full source height (full vertical extent).

    Un-parked: back in use by crops/build_crops.py with a FIXED fov_h_deg
    (120 deg -- 60 either side of the aim line), not the earlier per-building
    dynamic FOV computed from the footprint polygon. That dynamic version was
    the actual problem (it clamped down and cut wide/close buildings off); a
    fixed, generous width avoids that without needing to compute anything
    building-specific."""
    arr = np.array(pano.convert("RGB"))
    H, W = arr.shape[:2]

    center_u = heading_to_column(heading_deg, pano_heading_deg, W)
    shift = int(round(W / 2.0 - center_u))
    arr = np.roll(arr, shift, axis=1)

    half_w = int(round((fov_h_deg / 360.0) * W / 2.0))
    x0, x1 = W // 2 - half_w, W // 2 + half_w
    return Image.fromarray(arr[:, x0:x1])


def mark_heading(pano: Image.Image, heading_deg: float, pano_heading_deg: float = 0.0,
                  color: tuple = (255, 0, 0), width: int = 6) -> Image.Image:
    """Draws a single vertical line at `heading_deg` on the FULL, UNCROPPED
    panorama (full resolution, not a thumbnail) and returns the marked copy.
    Replaces slice_direction() for now, per the "don't crop, just mark the
    centroid bearing" decision -- see this module's docstring."""
    out = pano.convert("RGB").copy()
    W, H = out.size
    x = heading_to_column(heading_deg, pano_heading_deg, W)
    draw = ImageDraw.Draw(out)
    draw.line([(x, 0), (x, H)], fill=color, width=width)
    return out


def mark_center(img: Image.Image, color: tuple = (255, 0, 0), width: int = 6) -> Image.Image:
    """Draws a vertical line straight down the exact horizontal center of
    `img`. Use this (not mark_heading()) on an image slice_direction() already
    cropped centered on the aim heading -- the center IS where that heading
    landed, no need to recompute a column position on a differently-sized/
    already-rolled image."""
    out = img.convert("RGB").copy()
    W, H = out.size
    draw = ImageDraw.Draw(out)
    draw.line([(W / 2, 0), (W / 2, H)], fill=color, width=width)
    return out


def heading_offset_to_crop_x(heading_deg: float, center_heading_deg: float,
                              crop_width: int, fov_h_deg: float) -> float:
    """Pixel x-position within a slice_direction()-produced crop (centered on
    center_heading_deg, fov_h_deg wide) corresponding to an arbitrary
    heading_deg -- e.g. to mark where a DIFFERENT building (an occluder) sits
    within a crop that was aimed at the target building's own heading. Can
    return a value outside [0, crop_width] if heading_deg isn't within this
    particular crop's field of view at all -- caller's job to check that."""
    diff = ((heading_deg - center_heading_deg + 180.0) % 360.0) - 180.0  # wrap to (-180, 180]
    return crop_width / 2.0 + (diff / fov_h_deg) * crop_width


def mark_occluder(img: Image.Image, x: float, text: str, color: tuple = (255, 140, 0),
                   line_width: int = 5, font_size: int = 32) -> Image.Image:
    """Marks where an occluding building sits within the crop: a vertical
    line at pixel column x, plus a small text tag near the bottom (so it
    doesn't clash with label_image()'s top-left status bar), in a distinct
    color from mark_center()'s line. Does nothing if x falls outside the
    image (the occluder's own bearing isn't within this crop's FOV at all --
    can happen since the crop is aimed at the TARGET's heading, not the
    occluder's)."""
    out = img.convert("RGB").copy()
    W, H = out.size
    if not (0 <= x <= W):
        return out
    draw = ImageDraw.Draw(out)
    draw.line([(x, 0), (x, H)], fill=color, width=line_width)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    ty = H - font_size - 20
    tx = max(4, min(W - 4, x + 8))
    bbox = draw.textbbox((tx, ty), text, font=font)
    draw.rectangle([bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4], fill=(0, 0, 0))
    draw.text((tx, ty), text, fill=color, font=font)
    return out


def label_image(img: Image.Image, text: str, color: tuple = (255, 255, 0),
                 bg: tuple = (0, 0, 0), font_size: int = 56) -> Image.Image:
    """Draws a filled label bar with `text` in the top-left corner of a copy
    of `img` -- full resolution, so it's readable even when the source image
    itself is large (e.g. a 4096x2048 panorama). Used to bake the
    occluded/clear status directly onto the saved image, not just into the
    separate inspect_crops.py thumbnails."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0] + 24, bbox[3] - bbox[1] + 24
    draw.rectangle([0, 0, w, h], fill=bg)
    draw.text((12, 8), text, fill=color, font=font)
    return out
