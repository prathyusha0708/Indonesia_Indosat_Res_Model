"""
Bulk pre-fetch ALL panorama images listed in a coverage CSV into the local
disk cache, so later crops/build_crops.py runs (on any subset -- --limit,
--sample-n, or eventually the full 7,280-building set) never need to
download on the fly and just always hit a warm cache. Idempotent -- reuses
download_panorama_image's existing disk-cache check (skips anything already
saved), so re-running only fetches what's still missing; safe to re-run
after an interruption.
"""
import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # residential_annotation/
from crops.build_crops import fetch_panoramas  # reuse the exact cached/concurrent fetch logic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coverage-csv", default="outputs/coverage.csv")
    p.add_argument("--images-dir", default="outputs/panoramas")
    p.add_argument("--concurrency", type=int, default=6)
    args = p.parse_args()

    coverage = pd.read_csv(args.coverage_csv)
    pano_ids = set(coverage["pano_id"])
    print(f"{len(pano_ids)} unique panoramas in {args.coverage_csv}", flush=True)

    Path(args.images_dir).mkdir(parents=True, exist_ok=True)
    already = sum(1 for pid in pano_ids if (Path(args.images_dir) / f"{pid}.jpg").exists())
    print(f"{already} already cached, {len(pano_ids) - already} to download", flush=True)

    paths = asyncio.run(fetch_panoramas(pano_ids, args.images_dir, args.concurrency))
    ok = sum(1 for v in paths.values() if v)
    failed = len(pano_ids) - ok
    print(f"\nDone: {ok}/{len(pano_ids)} available locally in {args.images_dir}/ ({failed} failed)", flush=True)


if __name__ == "__main__":
    main()
