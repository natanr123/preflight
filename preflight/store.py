"""SQLite store for the crawl: a resumable account queue and every post seen.

Stopping and restarting the crawler loses nothing: the queue is the table.
Media rows keep `fetched_at`, because a view count only means something next to
how old the post was when it was read.
"""
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "preflight.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    username        TEXT PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'queued',  -- queued | done | not_found | error
    depth           INTEGER NOT NULL,                -- 0 = seed, n = mentioned n hops away
    source          TEXT,                            -- 'seed' or the account that mentioned it
    mentions        INTEGER NOT NULL DEFAULT 1,      -- how many crawled accounts mentioned it
    ig_id           TEXT,
    name            TEXT,
    biography       TEXT,
    website         TEXT,
    followers_count INTEGER,
    follows_count   INTEGER,
    media_count     INTEGER,
    pages           INTEGER NOT NULL DEFAULT 0,
    fetched_at      TEXT,
    error           TEXT
);
CREATE TABLE IF NOT EXISTS media (
    id                 TEXT PRIMARY KEY,
    username           TEXT NOT NULL,
    media_product_type TEXT,   -- REELS | FEED | ...
    media_type         TEXT,   -- VIDEO | IMAGE | CAROUSEL_ALBUM
    timestamp          TEXT,
    caption            TEXT,
    like_count         INTEGER,
    comments_count     INTEGER,
    view_count         INTEGER,
    permalink          TEXT,
    thumbnail_url      TEXT,   -- signed CDN urls: they expire within days
    media_url          TEXT,
    followers_at_fetch INTEGER,
    fetched_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS media_by_account ON media(username);
"""

MEDIA_COLUMNS = ("id", "media_product_type", "media_type", "timestamp", "caption", "like_count",
                 "comments_count", "view_count", "permalink", "thumbnail_url", "media_url")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path = DEFAULT_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def enqueue(self, usernames: Iterable[str], depth: int, source: str) -> int:
        """Queue the new names, in the order given; count one more mention for
        the known ones."""
        added = 0
        with self.db:
            for name in usernames:
                cursor = self.db.execute(
                    "INSERT OR IGNORE INTO accounts (username, depth, source) VALUES (?, ?, ?)",
                    (name, depth, source))
                if cursor.rowcount:
                    added += 1
                else:
                    self.db.execute("UPDATE accounts SET mentions = mentions + 1 WHERE username = ?",
                                    (name,))
        return added

    def next_queued(self) -> sqlite3.Row | None:
        """Breadth first; within a depth, the most-mentioned account first."""
        return self.db.execute(
            "SELECT * FROM accounts WHERE status = 'queued' "
            "ORDER BY depth, mentions DESC, rowid LIMIT 1").fetchone()

    def save_page(self, username: str, profile: dict, media: list[dict]) -> None:
        fetched = now()
        with self.db:
            self.db.execute(
                "UPDATE accounts SET ig_id = ?, name = ?, biography = ?, website = ?, "
                "followers_count = ?, follows_count = ?, media_count = ?, pages = pages + 1, "
                "fetched_at = ? WHERE username = ?",
                (profile.get("id"), profile.get("name"), profile.get("biography"),
                 profile.get("website"), profile.get("followers_count"),
                 profile.get("follows_count"), profile.get("media_count"), fetched, username))
            self.db.executemany(
                f"INSERT OR REPLACE INTO media ({', '.join(MEDIA_COLUMNS)}, username, "
                f"followers_at_fetch, fetched_at) VALUES ({', '.join('?' * (len(MEDIA_COLUMNS) + 3))})",
                [tuple(item.get(column) for column in MEDIA_COLUMNS)
                 + (username, profile.get("followers_count"), fetched) for item in media])

    def mark(self, username: str, status: str, error: str | None = None) -> None:
        with self.db:
            self.db.execute("UPDATE accounts SET status = ?, error = ? WHERE username = ?",
                            (status, error, username))

    def stats(self) -> dict:
        one = lambda sql: self.db.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "accounts": dict(self.db.execute(
                "SELECT status, COUNT(*) FROM accounts GROUP BY status").fetchall()),
            "media": one("SELECT COUNT(*) FROM media"),
            "reels": one("SELECT COUNT(*) FROM media WHERE media_product_type = 'REELS'"),
            "reels_with_views": one("SELECT COUNT(*) FROM media WHERE view_count IS NOT NULL"),
            "accounts_with_reels": one(
                "SELECT COUNT(DISTINCT username) FROM media WHERE media_product_type = 'REELS'"),
        }
