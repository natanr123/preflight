"""Which of two Reels from one account did better? Asked of a judge that only
reads their written-down openings.

    uv run python -m preflight.pairs run --judge codex:gpt-6.1-sol:medium --describer codex:gpt-6.1-sol [--caption]
    uv run python -m preflight.pairs report --judge codex:gpt-6.1-sol:medium --describer codex:gpt-6.1-sol

**Why.** It is the product's question: of two openings, which will do better
for this account? Comparing within one account leaves out what differs between
accounts (size, niche, how often Instagram shows them), so only the openings
are left to tell the two apart. The same pairs score every judge: a frontier
model through Codex now, Muse Spark as it comes (the baseline), then the
fine-tuned Glimmer (the number the pitch leads with).

**Pairs.** One Reel that beat its account's usual and one that did not, at
least `GAP` times apart, at most `PER_ACCOUNT` from one account. They are drawn
once, from the Reels the first `--describer` wrote down, and kept in
`data/pairs/pairs.json`, so every later judge gets exactly the same pairs, each
with the same Reel shown first.

**Score.** How often the judge picks the Reel that did better, with a 95%
interval (pairs from one account are not quite independent, so read it as
approximate); 50% is a coin flip. How often it picks whichever came first shows
any bias for position.
"""
import argparse
import json
import math
import random
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from preflight.dataset import OUT as DATASET
from preflight.describe import DescriberFailed, OutOfAllowance, as_text, ask, load_descriptions, log

PAIRS = DATASET.parent / "pairs"
#: The better Reel got at least this many times its partner's views, each
#: against its account's usual: closer pairs are mostly luck.
GAP = 2
PER_ACCOUNT = 3
SEED = 0

VERDICT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["better", "why"],
    "properties": {
        "better": {"type": "string", "enum": ["A", "B"]},
        "why": {"type": "string", "description": "One sentence."},
    },
}

PROMPT = """\
Two Reels were posted on the same Instagram account. Below is what the first five seconds of each show and say,
written down by someone who watched them{captions}.

Which of the two got more views, measured against how this account's Reels usually do? These openings are all you
have to go on; there is always an answer.

Reel A
{a}

Reel B
{b}

Answer with the form only.
"""


def build(rows: list[dict]) -> list[dict]:
    """Same-account pairs; whether the better Reel comes first is a coin toss."""
    rng, by_account = random.Random(SEED), defaultdict(list)
    for row in rows:
        by_account[row["username"]].append(row)
    pairs = []
    for account in sorted(by_account):
        reels = by_account[account]
        candidates = [(hit, miss) for hit in reels if hit["beats_median"] for miss in reels
                      if not miss["beats_median"] and hit["ratio"] >= GAP * miss["ratio"]]
        rng.shuffle(candidates)
        for hit, miss in candidates[:PER_ACCOUNT]:
            first, second = (hit, miss) if rng.random() < 0.5 else (miss, hit)
            pairs.append({"account": account, "a": first["id"], "b": second["id"],
                          "better": "A" if first is hit else "B"})
    return pairs


def load_pairs(rows: dict[str, dict], describer: str) -> list[dict]:
    path = PAIRS / "pairs.json"
    if not path.exists():
        forms = load_descriptions(describer)
        pairs = build([row for row in rows.values() if row["id"] in forms])
        PAIRS.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"drawn_from": describer, "gap": GAP, "per_account": PER_ACCOUNT,
                                    "pairs": pairs}, indent=1))
    return json.loads(path.read_text())["pairs"]


def verdict_path(judge: str, describer: str, captions: bool, pair: dict) -> Path:
    setup = describer + ("+captions" if captions else "")
    return PAIRS / setup.replace(":", "-") / judge.replace(":", "-") / f"{pair['a']}-{pair['b']}.json"


def judge_pair(judge: str, describer: str, captions: bool, pair: dict, forms: dict, rows: dict) -> dict:
    def shown(media_id: str) -> str:
        caption = f"\nCaption: {rows[media_id]['caption'] or '(none)'}" if captions else ""
        return as_text(forms[media_id]) + caption

    prompt = PROMPT.format(captions=", and each one's caption" if captions else "",
                           a=shown(pair["a"]), b=shown(pair["b"]))
    started = time.time()
    answer, usage = ask(judge, prompt, VERDICT)
    record = {**pair, "judge": judge, "describer": describer, "captions": captions, "said": answer["better"],
              "right": answer["better"] == pair["better"], "why": answer["why"],
              "seconds": round(time.time() - started, 1), "usage": usage}
    out = verdict_path(judge, describer, captions, pair)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    return record


def wilson(right: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 1.0
    share = right / n
    centre = (share + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(share * (1 - share) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def report(judge: str, describer: str, captions: bool) -> None:
    folder = verdict_path(judge, describer, captions, {"a": "", "b": ""}).parent
    records = [json.loads(path.read_text()) for path in folder.glob("*.json")]
    if not records:
        log(f"no verdicts in {folder}")
        return
    n, right = len(records), sum(record["right"] for record in records)
    low, high = wilson(right, n)
    first = sum(record["said"] == "A" for record in records)
    log(f"{judge} reading {describer}{' + captions' if captions else ''}: picked the better Reel in {right} of {n} "
        f"pairs = {right / n:.0%} (95%: {low:.0%}-{high:.0%}); said A {first / n:.0%} of the time, "
        f"from {len({record['account'] for record in records})} accounts")


def run(judge: str, describer: str, captions: bool, workers: int, limit: int | None) -> None:
    rows = {row["id"]: row for row in map(json.loads, DATASET.open())}
    forms = load_descriptions(describer)
    pairs = [pair for pair in load_pairs(rows, describer)[:limit] if pair["a"] in forms and pair["b"] in forms]
    todo = [pair for pair in pairs if not verdict_path(judge, describer, captions, pair).exists()]
    log(f"{len(todo)} of {len(pairs)} pairs to judge with {judge}")
    stop, counts = threading.Event(), Counter()

    def one(pair: dict) -> str:
        if stop.is_set():
            return "left for later"
        try:
            return "right" if judge_pair(judge, describer, captions, pair, forms, rows)["right"] else "wrong"
        except OutOfAllowance as limit_reached:
            stop.set()
            log(f"out of allowance, stopping: {limit_reached}")
            return "out of allowance"
        except (DescriberFailed, subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
            log(f"{pair['a']}-{pair['b']}: {type(error).__name__}: {error}")
            return "failed"

    with ThreadPoolExecutor(workers) as pool:
        for count, result in enumerate(pool.map(one, todo), 1):
            counts[result] += 1
            if count % 25 == 0:
                log(f"{count}/{len(todo)} {dict(counts)}")
    log(f"done: {dict(counts)}")
    report(judge, describer, captions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("--judge", default="codex:gpt-6.1-sol:medium")
    parser.add_argument("--describer", default="codex:gpt-6.1-sol", help="whose descriptions the judge reads")
    parser.add_argument("--caption", dest="captions", action="store_true", help="show each Reel's caption too")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, help="only the first N pairs")
    args = parser.parse_args()
    if args.command == "run":
        run(args.judge, args.describer, args.captions, args.workers, args.limit)
    else:
        report(args.judge, args.describer, args.captions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
