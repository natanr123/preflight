"""Instagram Business Discovery: other accounts' Reels, with their view counts.

Business Discovery is the official way to read another professional account's
media, and since June 2025 the only place `view_count` exists for Reels
(combined Instagram + Facebook views for crossposts since April 2026). Queries
run AS our own professional account, naming the target by username. Personal
and age-gated accounts answer "Cannot find User" (code 110), like a typo does.

The token is read where its owner keeps it (`IG_TOKEN_FILE` in
`secrets/ig.env`) and never copied, printed or logged.
"""
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "secrets" / "ig.env"

PROFILE_FIELDS = "id,username,name,biography,followers_count,follows_count,media_count,website"
#: media_url (the video) is fetched with everything else: the model works on
#: the video, and Business Discovery cannot read a single post afterwards, so
#: fetching it later means fetching every page again. The signed url expires
#: about a day and a half later (its `oe` parameter), so `preflight.videos`
#: downloads soon after. Instagram leaves it out of Reels it holds rights
#: claims on: measured, khaby00 97 of 98 and zachking 100 of 100 came with
#: one, verge 0 of 312 and nickelodeon 0 of 67.
#: thumbnail_url (the cover) is no longer asked for: covers are not a model
#: input for now, and the budget that runs out is the app's `total_time`
#: (measured: 19% vs 5% call_count after ~15 calls with both urls).
MEDIA_FIELDS = ("id,media_product_type,media_type,timestamp,caption,like_count,comments_count,"
                "view_count,permalink,media_url")
#: Measured 2026-10-27: the nested media edge accepts 100.
PAGE_SIZE = 100

#: Throttling codes: app (4), user (17), page (32), API-specific (613).
RATE_LIMIT_CODES = {4, 17, 32, 613}


class NotFound(Exception):
    """Personal, age-gated, renamed or nonexistent: nothing to read."""


class RateLimited(Exception):
    pass


class TokenInvalid(Exception):
    pass


class GraphError(Exception):
    def __init__(self, error: dict):
        super().__init__(error.get("message", "unknown Graph error"))
        self.error = error


@dataclass
class Page:
    profile: dict
    media: list[dict]
    after: str | None
    usage: int  # the highest of the app-usage percentages after this call


def load_config(path: Path = CONFIG) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _usage(headers) -> int:
    """Highest percentage across the app and business-use-case counters."""
    highest = 0
    app = headers.get("x-app-usage")
    if app:
        highest = max(highest, *json.loads(app).values())
    buc = headers.get("x-business-use-case-usage")
    if buc:
        for entries in json.loads(buc).values():
            for entry in entries:
                highest = max(highest, entry.get("call_count", 0),
                              entry.get("total_cputime", 0), entry.get("total_time", 0))
    return int(highest)


class BusinessDiscovery:
    def __init__(self, config: dict | None = None):
        config = config or load_config()
        self._token = json.loads(Path(config["IG_TOKEN_FILE"]).read_text())["access_token"]
        self._caller = config["IG_USER_ID"]
        self._version = config.get("GRAPH_VERSION", "v26.0")

    def page(self, username: str, after: str | None = None) -> Page:
        edge = f"media.after({after})" if after else "media"
        fields = (f"business_discovery.username({username})"
                  f"{{{PROFILE_FIELDS},{edge}.limit({PAGE_SIZE}){{{MEDIA_FIELDS}}}}}")
        query = urllib.parse.urlencode({"fields": fields, "access_token": self._token})
        url = f"https://graph.facebook.com/{self._version}/{self._caller}?{query}"
        try:
            with urllib.request.urlopen(url, timeout=90) as reply:
                body, usage = json.load(reply), _usage(reply.headers)
        except urllib.error.HTTPError as failure:
            raw = failure.read().decode(errors="replace")
            error = (json.loads(raw) if raw.startswith("{") else {}).get("error", {"message": raw[:200]})
            code = error.get("code")
            if code == 110:
                raise NotFound(error.get("error_user_msg") or error.get("message")) from None
            if code in RATE_LIMIT_CODES or failure.code == 429:
                raise RateLimited(error.get("message")) from None
            if code == 190:
                raise TokenInvalid(error.get("message")) from None
            raise GraphError(error) from None

        found = body["business_discovery"]
        media = found.pop("media", {})
        data = media.get("data", [])
        # Business Discovery pages carry a cursor but never a `next` link
        # (measured), so a full page is the only sign that more may follow.
        cursor = media.get("paging", {}).get("cursors", {}).get("after")
        return Page(profile=found, media=data,
                    after=cursor if len(data) == PAGE_SIZE else None, usage=usage)
