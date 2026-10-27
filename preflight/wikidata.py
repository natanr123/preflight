"""Seed accounts from Wikidata's Instagram usernames (property P2003).

    uv run python -m preflight.wikidata [--max 5000] [--max-sitelinks 10]

Writes `seeds/wikidata.tsv` (username, kind, category, sitelinks, country) and
`seeds/wikidata.txt` (usernames, in crawl order).

**Why Wikidata.** It is public domain (CC0), so the seed list carries no rights
question of its own, and it spans every niche: chefs, illustrators, bands,
museums, bakeries, sports clubs.

**Why few sitelinks.** An entity with one or two Wikipedia articles is notable
but not famous, which is where mid-size accounts live; household names have
fifty. **Why a random sample with a per-category cap.** Random order keeps
categories in proportion (musicians and athletes outnumber radio stations, as
on Instagram itself) while the cap stops one category from swamping the rest.
Round-robin was tried first and gave 3,339 categories one account each.
**Why some categories are left out.** Political, legal and government
accounts: the hackathon bars "anything political", and none of them are what a
creator would compare their Reels against.
"""
import argparse
import csv
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: QLever's Wikidata mirror (Uni Freiburg), not query.wikidata.org: WDQS was
#: rate-limiting to 1 request/minute during an outage on 2026-10-27, and QLever
#: answered the same counts in seconds.
ENDPOINT = "https://qlever.cs.uni-freiburg.de/api/wikidata"
USER_AGENT = "PreFlight/0.1 (hackathon research: short-video popularity)"
USERNAME = re.compile(r"^[a-z0-9._]{1,30}$")

#: QLever predefines nothing; WDQS predefines all of these.
PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

COUNTRIES = """VALUES (?c ?cc) { (wd:Q30 "US") (wd:Q145 "GB") (wd:Q16 "CA")
                                (wd:Q408 "AU") (wd:Q27 "IE") (wd:Q664 "NZ") }"""

PEOPLE = PREFIXES + """
SELECT ?item ?ig (SAMPLE(?label) AS ?category) (SAMPLE(?sl) AS ?sitelinks) (SAMPLE(?cc) AS ?country)
WHERE {
  ?item wdt:P2003 ?ig ; wdt:P31 wd:Q5 ; wdt:P27 ?c ; wikibase:sitelinks ?sl .
  %s
  FILTER(?sl >= 1 && ?sl <= %d)
  OPTIONAL { ?item wdt:P106 ?occ . ?occ rdfs:label ?label . FILTER(LANG(?label) = "en") }
} GROUP BY ?item ?ig
"""

ORGS = PREFIXES + """
SELECT ?item ?ig (SAMPLE(?label) AS ?category) (SAMPLE(?sl) AS ?sitelinks) (SAMPLE(?cc) AS ?country)
WHERE {
  ?item wdt:P2003 ?ig ; wdt:P17 ?c ; wikibase:sitelinks ?sl .
  %s
  FILTER NOT EXISTS { ?item wdt:P31 wd:Q5 }
  FILTER(?sl >= 1 && ?sl <= %d)
  OPTIONAL { ?item wdt:P31 ?type . ?type rdfs:label ?label . FILTER(LANG(?label) = "en") }
} GROUP BY ?item ?ig
"""


def query(sparql: str) -> list[dict]:
    request = urllib.request.Request(
        f"{ENDPOINT}?{urllib.parse.urlencode({'query': sparql, 'format': 'json'})}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as reply:
                return json.load(reply)["results"]["bindings"]
        except urllib.error.HTTPError as failure:
            # WDQS answers 429 with "1 req / min" during outages (seen 2026-10-27).
            if failure.code != 429 or attempt == 3:
                raise
            wait = int(failure.headers.get("Retry-After") or 65)
            print(f"wikidata rate limit; waiting {wait}s")
            time.sleep(wait)


def rows(kind: str, sparql: str) -> list[dict]:
    found = []
    for binding in query(sparql):
        name = binding["ig"]["value"].strip().lstrip("@").lower()
        if USERNAME.match(name):
            found.append({"username": name, "kind": kind,
                          "category": binding.get("category", {}).get("value", "unknown"),
                          "sitelinks": int(binding["sitelinks"]["value"]),
                          "country": binding["country"]["value"]})
    return found


EXCLUDED = re.compile(r"politic|government|parliament|senat|legislat|congress|governor|mayor|"
                      r"diplomat|ambassador|embassy|ministry|judge|court|lawyer|law firm|attorney|"
                      r"election|military|police|municipal|council|party", re.IGNORECASE)


def sample(entries: list[dict], limit: int, cap_share: float = 0.04, seed: int = 7) -> list[dict]:
    pool = [entry for entry in entries if not EXCLUDED.search(entry["category"])]
    random.Random(seed).shuffle(pool)
    cap = max(5, int(limit * cap_share))
    taken, ordered = defaultdict(int), []
    for entry in pool:
        if taken[entry["category"]] < cap:
            taken[entry["category"]] += 1
            ordered.append(entry)
            if len(ordered) == limit:
                break
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max", type=int, default=5000)
    parser.add_argument("--max-sitelinks", type=int, default=10)
    args = parser.parse_args()

    entries = {}
    for kind, template in (("person", PEOPLE), ("org", ORGS)):
        found = rows(kind, template % (COUNTRIES, args.max_sitelinks))
        print(f"{kind}: {len(found)} usernames")
        for entry in found:
            entries.setdefault(entry["username"], entry)

    ordered = sample(list(entries.values()), args.max)
    out = ROOT / "seeds"
    with (out / "wikidata.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)
    (out / "wikidata.txt").write_text("".join(f"{entry['username']}\n" for entry in ordered))
    categories = defaultdict(int)
    for entry in ordered:
        categories[entry["category"]] += 1
    print(f"wrote {len(ordered)} of {len(entries)} unique; {len(categories)} categories; top: "
          + ", ".join(f"{name} {count}" for name, count in sorted(categories.items(), key=lambda kv: -kv[1])[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
