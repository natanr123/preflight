"""Download each labelled Reel's video while its signed url still works.

    uv run python -m preflight.videos [--every 30 --hours 10]

Saves `data/videos/<media id>.mp4`, shrunk to 360 px on the short side, 15
frames a second and mono sound: about a sixth of Instagram's 720p file
(measured on six of zachking's Reels: 32 MB down to 5.7 MB), and still more
than the models look at.

Only Reels of labelled accounts (see `preflight.dataset`) are fetched: a video
means something to the model only next to its account's other Reels. The urls
are Instagram CDN links that Business Discovery hands out, and they expire
about a day and a half after the crawl read them (their `oe` parameter), so
this runs alongside the crawl, and `preflight.crawl refresh` reads accounts
again for fresh ones.
"""
import argparse
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from preflight.dataset import MIN_REELS, window_reels
from preflight.store import DEFAULT_PATH

VIDEOS = DEFAULT_PATH.parent / "videos"
SHORT_SIDE, FPS = 360, 15
#: Two videos at a time, two ffmpeg threads each: the laptop is shared (see
#: `preflight.baseline.THREADS`).
WORKERS, FFMPEG_THREADS = 2, 2


def video_path(media_id: str) -> Path:
    return VIDEOS / f"{media_id}.mp4"


def live(url: str | None, now: datetime) -> bool:
    """Whether a signed CDN url still works, by its `oe` parameter (hex seconds)."""
    if not url:
        return False
    expires = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get("oe")
    return not expires or datetime.fromtimestamp(int(expires[0], 16), timezone.utc) > now


def labelled_reels() -> dict[str, list[dict]]:
    db = sqlite3.connect(f"file:{DEFAULT_PATH}?mode=ro", uri=True)  # the crawler may be writing
    db.row_factory = sqlite3.Row
    return window_reels(db, MIN_REELS)


def missing() -> Counter:
    """How many labelled Reels each account has with neither a video on disk
    nor a link that still works: the accounts `crawl refresh` reads again."""
    now, counts = datetime.now(timezone.utc), Counter()
    for username, reels in labelled_reels().items():
        for reel in reels:
            if not video_path(reel["id"]).exists() and not live(reel["media_url"], now):
                counts[username] += 1
    return counts


def pending(skip: set[str]) -> list[tuple[str, str]]:
    now = datetime.now(timezone.utc)
    return [(reel["id"], reel["media_url"]) for reels in labelled_reels().values() for reel in reels
            if live(reel["media_url"], now) and reel["id"] not in skip
            and not video_path(reel["id"]).exists()]


def download(item: tuple[str, str]) -> str:
    media_id, url = item
    original, partial = VIDEOS / f"{media_id}.download", VIDEOS / f"{media_id}.part"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "PreFlight/0.1"})
        with urllib.request.urlopen(request, timeout=60) as reply, original.open("wb") as handle:
            shutil.copyfileobj(reply, handle)
        shrunk = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-threads", str(FFMPEG_THREADS), "-i", str(original),
             "-vf", f"scale={SHORT_SIDE}:{SHORT_SIDE}:force_original_aspect_ratio=increase"
                    f":force_divisible_by=2,fps={FPS}",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
             "-c:a", "aac", "-b:a", "48k", "-ac", "1", "-movflags", "+faststart", "-f", "mp4", str(partial)],
            capture_output=True, timeout=600)
    except urllib.error.HTTPError as failure:
        return "expired" if failure.code in (403, 410) else f"http {failure.code}"
    except (OSError, subprocess.TimeoutExpired):  # network errors, and an ffmpeg that hangs
        partial.unlink(missing_ok=True)
        return "failed"
    finally:
        original.unlink(missing_ok=True)
    if shrunk.returncode:
        partial.unlink(missing_ok=True)
        return "unreadable"
    partial.rename(video_path(media_id))  # never a half-written .mp4
    return "saved"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--every", type=float, default=0, help="minutes between passes; 0 = one pass")
    parser.add_argument("--hours", type=float, default=10)
    args = parser.parse_args()

    VIDEOS.mkdir(parents=True, exist_ok=True)
    deadline, expired = time.time() + args.hours * 3600, set()
    while True:
        items = pending(expired)
        with ThreadPoolExecutor(WORKERS) as pool:
            results = list(pool.map(download, items))
        expired |= {media_id for (media_id, _), result in zip(items, results) if result == "expired"}
        print(f"{datetime.now():%H:%M:%S} {dict(Counter(results)) or 'nothing new'}; "
              f"{sum(1 for _ in VIDEOS.glob('*.mp4'))} videos on disk", flush=True)
        if not args.every or time.time() + args.every * 60 > deadline:
            return 0
        time.sleep(args.every * 60)


if __name__ == "__main__":
    sys.exit(main())
