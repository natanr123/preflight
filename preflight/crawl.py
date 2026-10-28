"""Crawl Instagram professional accounts for Reels and their view counts.

    uv run python -m preflight.crawl seed seeds/instagram.txt
    uv run python -m preflight.crawl run [--max-calls N] [--hours H]
    uv run python -m preflight.crawl refresh [--max-calls N] [--hours H]
    uv run python -m preflight.crawl stats

**Where accounts come from.** Business Discovery needs a username; searching
for creators needs App Review we will not get this week. So the queue starts
from seed lists (`seeds/`, including Wikidata's) and snowballs through @mentions
in BIOS, one hop deeper each time. Not captions: Business Discovery strips the
"@" from captions (measured: natgeo, 0 of 50 photo credits kept it), while
bios keep theirs.

**How deep per account.** Every account gets one page (100 posts). Accounts in
the follower band get more pages, until the posts are older than
`HISTORY_DAYS`, because labelling a Reel against its own account's median
needs that account's other Reels from the same months.

**Reading again.** `refresh` reads the labelled accounts again, those with
the most Reels lacking a video first: video links expire about a day and a
half after they are read, and accounts crawled before the crawl asked for
them have none.

**Rate.** The token belongs to bookpopup's live app, and Graph rate limits are
shared with its real publishing. The crawl pauses at `PAUSE_AT` percent of
any usage counter, leaving the other half for bookpopup.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from preflight.ig import BusinessDiscovery, GraphError, NotFound, RateLimited, TokenInvalid
from preflight.store import Store
from preflight.videos import missing

#: Accounts in this follower band get their history paginated.
BAND = (5_000, 2_000_000)
HISTORY_DAYS = 120
MAX_PAGES = 2
MAX_DEPTH = 3

#: Usage percentages (of the app's rolling-hour budget) that slow or pause us.
SLOW_AT, PAUSE_AT = 25, 50
BASE_DELAY, SLOW_DELAY, PAUSE_DELAY = 2, 60, 15 * 60

MENTION = re.compile(r"(?<![\w.@])@([A-Za-z0-9._]{1,30})")


def mentions(text: str, own: str) -> set[str]:
    found = {name.rstrip(".").lower() for name in MENTION.findall(text or "")}
    return {name for name in found if name and name != own}


def oldest(media: list[dict]) -> datetime | None:
    stamps = [item["timestamp"] for item in media if item.get("timestamp")]
    return datetime.strptime(min(stamps), "%Y-%m-%dT%H:%M:%S%z") if stamps else None


class Crawler:
    def __init__(self, store: Store, client: BusinessDiscovery, max_calls: int, deadline: float):
        self.store, self.client = store, client
        self.max_calls, self.deadline = max_calls, deadline
        self.calls = 0

    def budget_left(self) -> bool:
        return self.calls < self.max_calls and time.time() < self.deadline

    def fetch(self, username: str, after: str | None = None):
        """One call, retried through throttling and transient failures."""
        for attempt in range(4):
            self.calls += 1
            try:
                page = self.client.page(username, after)
            except RateLimited as limited:
                log(f"rate limited ({limited}); pausing {PAUSE_DELAY // 60} min")
                time.sleep(PAUSE_DELAY)
                continue
            except GraphError as failure:
                if failure.error.get("is_transient") and attempt < 3:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
            except OSError as failure:  # network: timeouts, resets, DNS
                if attempt < 3:
                    log(f"network error ({failure}); retrying")
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
            self.pace(page.usage)
            return page
        raise RuntimeError("gave up after repeated throttling")

    def pace(self, usage: int) -> None:
        if usage >= PAUSE_AT:
            log(f"usage {usage}% >= {PAUSE_AT}%; pausing {PAUSE_DELAY // 60} min to leave bookpopup room")
            time.sleep(PAUSE_DELAY)
        else:
            time.sleep(SLOW_DELAY if usage >= SLOW_AT else BASE_DELAY)

    def read(self, username: str) -> tuple[dict, list[dict], int, int]:
        """Save the account's first page, and its history if it is in the band.
        Returns the profile, the posts, the pages read and the last usage."""
        page = self.fetch(username)
        media, profile = list(page.media), page.profile
        self.store.save_page(username, profile, page.media)
        followers = profile.get("followers_count") or 0
        horizon = datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)
        pages, after = 1, page.after
        while (BAND[0] <= followers <= BAND[1] and after and pages < MAX_PAGES
               and (oldest(page.media) or horizon) > horizon and self.budget_left()):
            try:
                page = self.fetch(username, after)
            except (NotFound, GraphError):
                break  # keep what the first pages gave
            self.store.save_page(username, page.profile, page.media)
            media += page.media
            pages, after = pages + 1, page.after
        return profile, media, pages, page.usage

    def crawl_one(self, account) -> None:
        username = account["username"]
        try:
            profile, media, pages, usage = self.read(username)
        except NotFound:
            self.store.mark(username, "not_found")
            return
        except GraphError as failure:
            self.store.mark(username, "error", json.dumps(failure.error)[:500])
            return

        if account["depth"] < MAX_DEPTH:
            added = self.store.enqueue(mentions(profile.get("biography"), username),
                                       account["depth"] + 1, username)
        else:
            added = 0
        self.store.mark(username, "done")
        reels = [item for item in media if item.get("media_product_type") == "REELS"]
        log(f"{username:<28} followers={profile.get('followers_count') or 0:>11,} posts={len(media):>3} "
            f"reels={len(reels):>3} pages={pages} +queued={added:<3} usage={usage}% calls={self.calls}")

    def run(self) -> None:
        while self.budget_left():
            account = self.store.next_queued()
            if account is None:
                log("queue empty")
                return
            self.crawl_one(account)

    def refresh(self, usernames: list[str]) -> None:
        """Read crawled accounts again, for links that expired or were not yet
        asked for. Their posts' counts are read afresh too."""
        for username in usernames:
            if not self.budget_left():
                return
            try:
                _, media, pages, usage = self.read(username)
            except (NotFound, GraphError) as failure:
                log(f"{username:<28} not read again: {failure}")
                continue
            reels = [item for item in media if item.get("media_product_type") == "REELS"]
            videos = sum(1 for item in reels if item.get("media_url"))
            log(f"{username:<28} reels={len(reels):>3} with video={videos:>3} pages={pages} "
                f"usage={usage}% calls={self.calls}")


def log(message: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    seed = commands.add_parser("seed")
    seed.add_argument("file", type=Path)
    for name in ("run", "refresh"):
        command = commands.add_parser(name)
        command.add_argument("--max-calls", type=int, default=500)
        command.add_argument("--hours", type=float, default=12)
    commands.add_parser("stats")
    args = parser.parse_args()

    store = Store()
    if args.command == "seed":
        # Ordered and deduplicated: the file's order is the crawl order.
        names = list(dict.fromkeys(line.split("#")[0].strip().lstrip("@").lower()
                                   for line in args.file.read_text().splitlines()))
        names = [name for name in names if name]
        print(f"queued {store.enqueue(names, 0, 'seed')} new seeds of {len(names)}")
    elif args.command in ("run", "refresh"):
        crawler = Crawler(store, BusinessDiscovery(), args.max_calls, time.time() + args.hours * 3600)
        try:
            if args.command == "run":
                crawler.run()
            else:
                # Most videos missing first.
                crawler.refresh([username for username, _ in missing().most_common()])
        except TokenInvalid as invalid:
            log(f"token rejected ({invalid}); reconnect Facebook in bookpopup, then rerun")
            return 1
        finally:
            log(f"stopped after {crawler.calls} calls: {json.dumps(store.stats())}")
    else:
        print(json.dumps(store.stats(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
