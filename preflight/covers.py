"""Download each labelled Reel's cover image while its signed url still works.

    uv run python -m preflight.covers [--every 30 --hours 10]

Saves `data/covers/<media id>.jpg`, at most 512 px on the long side: the
models downscale further anyway, and full-size covers take several times the
disk.

Only Reels inside the label window (see `preflight.dataset`) are fetched.
Younger ones have no settled view count yet, and their urls expire before
they age in. The urls are Instagram CDN links that Business Discovery hands
out and they expire within days, so this runs alongside the crawl.
"""
import argparse
import io
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from PIL import Image

from preflight.dataset import WINDOW_DAYS, parse
from preflight.store import DEFAULT_PATH

COVERS = DEFAULT_PATH.parent / "covers"
LONG_SIDE = 512
WORKERS = 4


def pending(skip: set[str]) -> list[tuple[str, str]]:
    db = sqlite3.connect(f"file:{DEFAULT_PATH}?mode=ro", uri=True)  # the crawler may be writing
    found = []
    for media_id, url, posted, fetched in db.execute(
            "SELECT id, thumbnail_url, timestamp, fetched_at FROM media "
            "WHERE media_product_type = 'REELS' AND thumbnail_url IS NOT NULL"):
        age = (parse(fetched) - parse(posted)).total_seconds() / 86400
        if (WINDOW_DAYS[0] <= age <= WINDOW_DAYS[1] and media_id not in skip
                and not (COVERS / f"{media_id}.jpg").exists()):
            found.append((media_id, url))
    return found


def download(item: tuple[str, str]) -> str:
    media_id, url = item
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "PreFlight/0.1"})
        with urllib.request.urlopen(request, timeout=30) as reply:
            image = Image.open(io.BytesIO(reply.read())).convert("RGB")
    except urllib.error.HTTPError as failure:
        return "expired" if failure.code in (403, 410) else f"http {failure.code}"
    except OSError:  # network errors, and images Pillow cannot read
        return "failed"
    image.thumbnail((LONG_SIDE, LONG_SIDE))
    partial = COVERS / f"{media_id}.part"
    image.save(partial, "JPEG", quality=85)
    partial.rename(COVERS / f"{media_id}.jpg")  # never a half-written .jpg
    return "saved"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--every", type=float, default=0, help="minutes between passes; 0 = one pass")
    parser.add_argument("--hours", type=float, default=10)
    args = parser.parse_args()

    COVERS.mkdir(parents=True, exist_ok=True)
    deadline, expired = time.time() + args.hours * 3600, set()
    while True:
        items = pending(expired)
        with ThreadPoolExecutor(WORKERS) as pool:
            results = list(pool.map(download, items))
        expired |= {media_id for (media_id, _), result in zip(items, results) if result == "expired"}
        print(f"{datetime.now():%H:%M:%S} {dict(Counter(results)) or 'nothing new'}; "
              f"{sum(1 for _ in COVERS.glob('*.jpg'))} covers on disk", flush=True)
        if not args.every or time.time() + args.every * 60 > deadline:
            return 0
        time.sleep(args.every * 60)


if __name__ == "__main__":
    sys.exit(main())
