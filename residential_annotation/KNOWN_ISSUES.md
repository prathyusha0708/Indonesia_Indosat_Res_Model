# Known Issues — to fix later, not blocking the current full run

Both found 2026-09-07 while eyeballing real crops from the in-progress full
7,280-building `build_crops.py` run. Neither has been fixed in the running
process (left untouched, per explicit decision) -- issue #1 IS fixed in the
source file already (safe, cosmetic-only, doesn't affect any currently-running
process); issue #2 is NOT yet fixed anywhere.

## 1. [FIXED in source, not yet in the running process] Occluder label text
   clipped off-canvas near the crop's right edge

**File:** `geometry/direction_view.py`, `mark_occluder()`

The on-image debug label ("blocked by <building_id> (<dist>m, <frac>% of
facade)") is drawn starting at a clamped x-position, but the OLD code never
checked whether the text, once drawn, fit before the image's right edge --
so a label starting near the edge got silently clipped by the canvas
boundary. This truncated both the building_id (making it unrecoverable /
not matching any real building when read back off the image) and the
percentage (e.g. showing "30%" when the real value was "30.34%").

Confirmed via building `06b500000000000a4c9f`, view rank 3: label showed
occluder id `06b500000000037b799` (19 chars, not found in the buildings
dataset) -- the real id, recomputed directly from geometry, is
`06b5000000000037b799` (21 chars). The real fraction was 30.34% (marginally
above the 30% MIN_OCCLUSION_FRACTION threshold -- so the occluded=True call
itself was correct, just the label was unreadable).

**Fix applied:** `mark_occluder()` now also clamps `tx` by the text's own
rendered width (`draw.textbbox` at origin), so the whole string is guaranteed
to stay on-canvas. Sanity-tested with a synthetic worst-case call (marker at
the extreme right edge with a long string) -- no clipping.

**Status:** fixed in `geometry/direction_view.py`, but the full run started
before this fix was applied and has the old module already loaded in its
running process -- everything produced by that run (in progress as of this
writing) still has the old clipping bug. Purely cosmetic (never affected the
actual occluded/clear decision or any CSV data) -- left as-is per explicit
decision, only relevant to future runs.

## 2. [NOT YET FIXED] Occluder angular-span wraparound bug for very-close occluders

**File:** `geometry/line_of_sight.py`, `_angular_span()` /
`occlusion_fraction_in_window()`

Same root-cause class as the v3->v4 bug already fixed for the TARGET
polygon (camera close to a corner -> per-vertex bearing wraparound math
becomes unstable) -- but that fix was only ever applied to the target's own
window computation, never to how a CANDIDATE (occluder) polygon's own
angular span is computed against the window.

`_angular_span()` computes, for every exterior vertex of a polygon, its
bearing from the camera, wraps the offset from a reference bearing into
`(-180, 180]` via `((b - ref_bearing + 180) % 360) - 180`, then takes a
naive `min()`/`max()` over all vertex offsets as "the polygon's angular
span." This silently breaks when the camera sits very close to the
candidate polygon (a handful of meters) AND at least one vertex's offset
lands almost exactly on the +-180 deg boundary -- floating-point/exact
wraparound behavior can flip that one vertex from (correctly) ~+179.5 deg to
(incorrectly) ~-179.5 deg. The naive min/max then reports a span like
`(-179.5, +173.2)`, i.e. "wraps almost the entire circle," when the
polygon's true visible silhouette from the camera is nowhere near that wide
and nowhere near the direction actually being checked.

Confirmed via building `06b500000000000ae9d1`, view rank 2 (pano
`gcnDj73w6SEd2qR1MjlMEQ`, 27m from the target): occluder
`06b500000000003e5895` is a REAL 65m-tall, ~60x30m building (HIGH-quality
height data, not a data artifact) sitting only ~2.4-3m from the camera at
one corner. Its true vertex bearings from that camera all fall between
~0.5 deg and ~173 deg -- nowhere near the crop's actual window (centered at
307.5 deg, i.e. the completely different direction the crop was aimed at, at
the target building). But the single nearest vertex (3.01m away, bearing
128.0 deg absolute) has offset-from-307.5 deg landing right at the wraparound
boundary and flips to -179.5 deg instead of +179.5 deg, making the computed
span `(-179.5, 173.2)` -- which fully contains the crop's [-60,+60] window,
so `occlusion_fraction_in_window()` returns exactly 1.0 (maximum-confidence
"fully occluded"), even though this building is not physically in the
direction being photographed at all.

**Effect on data (confirmed, see conversation):** this bug can only ever
INFLATE a computed occlusion fraction -- it never shrinks one. So it can
only produce **false negatives for "clear"** (a genuinely clear view wrongly
marked occluded=True), and it can **never** produce a false positive for
"clear" (a genuinely occluded view will never get wrongly marked clear by
this bug). Practical effect:
- `has_clear_view/` stays trustworthy -- nothing fake was ever let in by
  this bug.
- `all_occluded/` may contain some buildings that actually do have a clear
  view, wrongly excluded by this bug.
- The matching-phase summary counts ("views flagged occluded", "buildings
  still with NO clear view within 50m") are likely somewhat inflated versus
  the true numbers, for the same reason.

**Proposed fix (not yet implemented):** replace the naive per-vertex
min/max wraparound approach with the standard "angular size of a convex
polygon as seen from an external point" algorithm -- take the candidate
polygon's convex hull, then find the two tangent lines from the camera
point to that hull (the two hull vertices where the cross-product sign of
consecutive edge vectors, relative to the camera, changes sign) -- those
two tangent bearings ARE the true visible angular span, correctly handling
the "camera very close to / almost inside the hull" case that breaks the
current per-vertex min/max. Needs a corresponding edge case for the (rare)
situation where the camera point is literally inside the polygon.

**Status:** NOT fixed. Explicitly deferred per user request ("keep this in
mind, will fix later") -- current full run left completely untouched.
