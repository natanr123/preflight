"""Label every Reel by how it did against its own account's usual.

    uv run python -m preflight.dataset [--min-reels 8]

Writes `data/dataset.jsonl`, one Reel per line: the caption and what else is
known before posting, then the outcome.

**Why relative to the account.** Raw views mostly measure the account, and the
account is fixed before anyone posts. What a creator can still change is the
Reel, so the label is its views over its account's median: 2.0 is twice that
account's usual. Follower count and posting history drop out of the label,
which is the equal starting ground the comparison needs.

**Why 14 to 120 days old.** Younger Reels are still collecting views, and the
crawl pages back no further than 120 days. Age still counts a little inside
the window (measured 2026-10-27: about 7% more views per doubling of age), so
views are age-adjusted with the within-account slope before the median.

**Bands.** Under half the median is a flop and over twice it outperforms (17%
and 24% of Reels on 2026-10-27); the rest is normal.
"""
import argparse
import json
import math
import re
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from langdetect import DetectorFactory, LangDetectException, detect

from preflight.store import DEFAULT_PATH

OUT = DEFAULT_PATH.parent / "dataset.jsonl"
WINDOW_DAYS = (14, 120)
MIN_REELS = 8
FLOP, OUTPERFORM = 0.5, 2.0
#: Recent Reels averaged for the account's streak, a reference feature.
STREAK = 5

HASHTAG = re.compile(r"#\w+")
URL = re.compile(r"https?://\S+|www\.\S+")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
LETTER = re.compile(r"[^\W\d_]")
CTA = re.compile(r"\b(link in (?:the |our |my )?bio|comment|follow|tag (?:a|your|someone)|share|"
                 r"save (?:this|it)|subscribe|shop now|dm)\b", re.IGNORECASE)

DetectorFactory.seed = 0  # langdetect is random otherwise


def parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("+0000", "+00:00"))


def language(caption: str | None) -> str:
    """'und' when there are too few letters to tell, as with emoji and hashtags."""
    text = HASHTAG.sub(" ", URL.sub(" ", caption or ""))
    if len(LETTER.findall(text)) < 15:
        return "und"
    try:
        return detect(text)
    except LangDetectException:
        return "und"


def caption_features(caption: str | None) -> dict:
    text = caption or ""
    return {"caption_chars": len(text), "words": len(text.split()),
            "hashtags": len(HASHTAG.findall(text)), "lines": text.count("\n") + 1 if text else 0,
            "emojis": len(EMOJI.findall(text)), "question": "?" in text, "cta": bool(CTA.search(text))}


def follower_band(followers: int) -> str:
    for limit, name in ((5_000, "<5K"), (50_000, "5K-50K"), (500_000, "50K-500K"), (2_000_000, "500K-2M")):
        if followers < limit:
            return name
    return ">2M"


def window_reels(db: sqlite3.Connection, min_reels: int) -> dict[str, list[dict]]:
    """Each account's Reels inside the age window, oldest first, with the hours
    since the account's previous post of any kind."""
    posts = defaultdict(list)
    for row in db.execute("SELECT m.*, a.followers_count, a.media_count FROM media m "
                          "JOIN accounts a USING (username) WHERE m.timestamp IS NOT NULL"):
        posts[row["username"]].append(dict(row))
    accounts = {}
    for username, items in posts.items():
        items.sort(key=lambda item: item["timestamp"])
        reels, previous = [], None
        for item in items:
            posted = parse(item["timestamp"])
            gap = (posted - previous).total_seconds() / 3600 if previous else None
            previous = posted
            if item["media_product_type"] != "REELS" or item["view_count"] is None:
                continue
            age = (parse(item["fetched_at"]) - posted).total_seconds() / 86400
            if WINDOW_DAYS[0] <= age <= WINDOW_DAYS[1]:
                reels.append(item | {"posted": posted, "age_days": age, "gap_hours": gap})
        if len(reels) >= min_reels:
            accounts[username] = reels
    return accounts


def age_slope(accounts: dict[str, list[dict]]) -> float:
    """Slope of log views on log age within accounts, pooled over all of them."""
    covariance = variance = 0.0
    for reels in accounts.values():
        ages = [math.log(reel["age_days"]) for reel in reels]
        views = [math.log(max(reel["view_count"], 1)) for reel in reels]
        mean_age, mean_views = statistics.fmean(ages), statistics.fmean(views)
        covariance += sum((a - mean_age) * (v - mean_views) for a, v in zip(ages, views))
        variance += sum((a - mean_age) ** 2 for a in ages)
    return covariance / variance if variance else 0.0


def label(accounts: dict[str, list[dict]], slope: float) -> list[dict]:
    rows = []
    for username, reels in accounts.items():
        adjusted = [math.log(max(reel["view_count"], 1)) - slope * math.log(reel["age_days"])
                    for reel in reels]
        usual, streak = statistics.median(adjusted), []
        for reel, value in zip(reels, adjusted):
            log_ratio = value - usual
            ratio = math.exp(log_ratio)
            rows.append({
                "id": reel["id"], "username": username, "permalink": reel["permalink"],
                "timestamp": reel["timestamp"], "caption": reel["caption"] or "",
                "language": language(reel["caption"]),
                "followers": reel["followers_count"], "follower_band": follower_band(reel["followers_count"]),
                "media_count": reel["media_count"],
                "hour_utc": reel["posted"].hour, "weekday": reel["posted"].weekday(),
                "gap_hours": reel["gap_hours"], **caption_features(reel["caption"]),
                "views": reel["view_count"], "age_days": round(reel["age_days"], 2),
                "log_ratio": log_ratio, "ratio": ratio, "beats_median": ratio > 1,
                "band": "flop" if ratio < FLOP else "outperform" if ratio > OUTPERFORM else "normal",
                "streak": statistics.fmean(streak[-STREAK:]) if streak else None,
            })
            streak.append(log_ratio)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-reels", type=int, default=MIN_REELS,
                        help="accounts with fewer Reels in the window have no stable median")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    db = sqlite3.connect(f"file:{DEFAULT_PATH}?mode=ro", uri=True)  # the crawler may be writing
    db.row_factory = sqlite3.Row
    accounts = window_reels(db, args.min_reels)
    slope = age_slope(accounts)
    rows = label(accounts, slope)
    with args.out.open("w") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

    share = lambda key: ", ".join(f"{name} {count / len(rows):.0%}"  # noqa: E731
                                  for name, count in Counter(row[key] for row in rows).most_common(6))
    print(f"wrote {len(rows)} Reels from {len(accounts)} accounts to {args.out}")
    print(f"age slope {slope:.3f} (log views per log day)")
    print(f"bands: {share('band')}\nlanguages: {share('language')}\nfollowers: {share('follower_band')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
