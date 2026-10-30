"""Which of two Reels from one account did better? Asked of a judge that only
reads their written-down openings.

    uv run python -m preflight.pairs run --judge codex:gpt-6.1-sol:medium --describer codex:gpt-6-luna [--caption]
                                         [--first-frames] [--sound]
    uv run python -m preflight.pairs report --judge codex:gpt-6.1-sol:medium --describer codex:gpt-6-luna
    uv run python -m preflight.pairs compare --judge glimmer:pairs-spark --against meta:muse-spark-1.3:medium
                                             --describer meta:muse-spark-1.3:low

**Why.** It is the product's question: of two openings, which will do better
for this account? Comparing within one account leaves out what differs between
accounts (size, niche, how often Instagram shows them), so only the openings
are left to tell the two apart. The same pairs score every judge: a frontier
model through Codex, Muse Spark zero-shot (the baseline), then the fine-tuned
Glimmer (the number the pitch leads with).

**Pairs.** One Reel that beat its account's usual and one that did not, at
least `GAP` times apart, at most `PER_ACCOUNT` from one account, and only from
the held-out accounts (`held_out`), which no fine-tune ever trains on. They are
drawn once from every such Reel with a video and kept in
`data/pairs/<set>/pairs.json`, so every judge gets exactly the same pairs, each
with the same Reel shown first; a judge takes those both sides of which its
describer wrote down. (`--set sol-sample` is the first, exploratory set: 160
pairs from the 497 Reels Codex Sol described, held-out accounts or not.)

**First frames.** `--first-frames` also shows the judge both Reels' first
frames, side by side, for what the words may miss of how they look. The
written-down openings already describe that frame, so it asks whether the
pixels add anything.

**Sound.** `--sound` adds a line under each opening with what audio models
measured of its sound (`preflight.sound`): the judges cannot hear, and the
written-down opening has only the words and how loud each second is.

**Score.** How often the judge picks the Reel that did better, with a 95%
interval (pairs from one account are not quite independent, so read it as
approximate); 50% is a coin flip. How often it picks whichever came first shows
any bias for position.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from preflight.dataset import OUT as DATASET
from preflight.describe import FAILURES, STOPS, as_text, ask, load_descriptions, log
from preflight.sound import load_sound, sound_line
from preflight.videos import first_frame, video_path

PAIRS = DATASET.parent / "pairs"
#: One account in this many, picked by a hash of its name, is held out of
#: every fine-tune for testing: 44 accounts and 24% of the described Reels on
#: 2026-10-29. (Holding out the sol-sample's accounts instead would have left
#: 28% of the Reels to train on: they were the big accounts.)
HOLD_OUT = 4
#: The better Reel got at least this many times its partner's views, each
#: against its account's usual: closer pairs are mostly luck.
GAP = 2
PER_ACCOUNT = 6
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
written down by someone who watched them{captions}.{frames}{sound}

Which of the two got more views, measured against how this account's Reels usually do? These openings are all you
have to go on; there is always an answer.

Reel A
{a}

Reel B
{b}

Answer with the form only.
"""
FRAMES = " The attached picture shows each one's first frame: Reel A's on the left, Reel B's on the right."
SOUND = " Under each one, a line says what audio models measured of its sound."


def held_out(username: str) -> bool:
    return int(hashlib.sha256(username.encode()).hexdigest(), 16) % HOLD_OUT == 0


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


def load_pairs(name: str, rows: dict[str, dict]) -> list[dict]:
    path = PAIRS / name / "pairs.json"
    if not path.exists():
        pairs = build([row for row in rows.values()
                       if held_out(row["username"]) and video_path(row["id"]).exists()])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"hold_out": HOLD_OUT, "gap": GAP, "per_account": PER_ACCOUNT,
                                    "pairs": pairs}, indent=1))
    return json.loads(path.read_text())["pairs"]


def verdict_path(name: str, judge: str, describer: str, captions: bool, pair: dict, frames: bool = False,
                 sound: bool = False) -> Path:
    setup = (describer + ("+captions" if captions else "") + ("+first-frames" if frames else "")
             + ("+sound" if sound else ""))
    return PAIRS / name / setup.replace(":", "-") / judge.replace(":", "-") / f"{pair['a']}-{pair['b']}.json"


def first_frames(name: str, pair: dict) -> Path:
    """Both Reels' first frames side by side, as the prompt says: A's on the left."""
    out = PAIRS / name / "first-frames" / f"{pair['a']}-{pair['b']}.jpg"
    if not out.exists():
        from PIL import Image, ImageDraw, ImageFont

        frames = [Image.open(first_frame(pair[side])).convert("RGB") for side in ("a", "b")]
        height, label, gutter = min(frame.height for frame in frames), 44, 16
        frames = [frame.resize((round(frame.width * height / frame.height), height)) for frame in frames]
        sheet = Image.new("RGB", (frames[0].width + gutter + frames[1].width, label + height), "white")
        draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=30)
        x = 0
        for side, frame in zip("AB", frames):
            sheet.paste(frame, (x, label))
            draw.text((x + 8, 6), f"Reel {side}", fill="black", font=font)
            x += frame.width + gutter
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out, quality=90)
    return out


def judge_pair(name: str, judge: str, describer: str, captions: bool, pair: dict, forms: dict, rows: dict,
               frames: bool = False, sound: bool = False) -> dict:
    def shown(media_id: str) -> str:
        caption = f"\nCaption: {rows[media_id]['caption'] or '(none)'}" if captions else ""
        return as_text(forms[media_id]) + caption + (f"\n{sound_line(load_sound(media_id))}" if sound else "")

    prompt = PROMPT.format(captions=", and each one's caption" if captions else "", frames=FRAMES if frames else "",
                           sound=SOUND if sound else "", a=shown(pair["a"]), b=shown(pair["b"]))
    started = time.time()
    answer, usage = ask(judge, prompt, VERDICT, first_frames(name, pair) if frames else None)
    record = {**pair, "judge": judge, "describer": describer, "captions": captions, "first_frames": frames,
              "sound": sound, "said": answer["better"], "right": answer["better"] == pair["better"],
              "why": answer["why"], "seconds": round(time.time() - started, 1), "usage": usage}
    out = verdict_path(name, judge, describer, captions, pair, frames, sound)
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


def report(name: str, judge: str, describer: str, captions: bool, frames: bool = False, sound: bool = False) -> None:
    folder = verdict_path(name, judge, describer, captions, {"a": "", "b": ""}, frames, sound).parent
    records = [json.loads(path.read_text()) for path in folder.glob("*.json")]
    if not records:
        log(f"no verdicts in {folder}")
        return
    n, right = len(records), sum(record["right"] for record in records)
    low, high = wilson(right, n)
    first = sum(record["said"] == "A" for record in records)
    extras = (" + captions" if captions else "") + (" + first frames" if frames else "") + (" + sound" if sound else "")
    log(f"{name}: {judge} reading {describer}{extras}: "
        f"picked the better Reel in {right} of {n} pairs = {right / n:.0%} (95%: {low:.0%}-{high:.0%}); "
        f"said A {first / n:.0%} of the time, from {len({record['account'] for record in records})} accounts")


def tuned_chances(name: str, judge: str) -> dict[tuple[str, str], float]:
    """A fine-tuned judge's (`glimmer:<run>`) chance that Reel A did better,
    pair by pair, from its run, whose chances follow the pairs its describer
    wrote both Reels of."""
    result = json.loads((DATASET.parent / "finetune" / f"{judge.split(':', 1)[1]}.json").read_text())
    described = load_descriptions(result["describer"])
    pairs = [pair for pair in load_pairs(name, {}) if pair["a"] in described and pair["b"] in described]
    tuned = result["tuned"]
    if [pair["better"] for pair in pairs] != tuned.get("better", [pair["better"] for pair in pairs]):
        raise ValueError(f"{judge} was scored on other pairs")
    return {(pair["a"], pair["b"]): chance for pair, chance in zip(pairs, tuned["chances"], strict=True)}


def rights(name: str, judge: str, describer: str) -> dict[tuple[str, str], bool]:
    """Whether the judge picked the better Reel, pair by pair: a zero-shot
    judge's from its verdicts, a fine-tuned one's from its chances."""
    if judge.startswith("glimmer:"):
        better = {(pair["a"], pair["b"]): pair["better"] for pair in load_pairs(name, {})}
        return {pair: (chance > 0.5) == (better[pair] == "A") for pair, chance in tuned_chances(name, judge).items()}
    folder = verdict_path(name, judge, describer, False, {"a": "", "b": ""}).parent
    return {(record["a"], record["b"]): record["right"]
            for record in (json.loads(path.read_text()) for path in folder.glob("*.json"))}


def compare(name: str, judge: str, other: str, describer: str) -> None:
    """Two judges on the pairs both answered: how many only one of them got
    right, and how likely a split that uneven is if they are equally good
    (McNemar's exact test). Where they agree, how often they are right."""
    mine, theirs = rights(name, judge, describer), rights(name, other, describer)
    pairs = mine.keys() & theirs.keys()
    if not pairs:
        log(f"{name}: no pairs that {judge} and {other} (reading {describer}) both answered")
        return
    alone, others_alone = (sum(first[pair] and not second[pair] for pair in pairs)
                           for first, second in ((mine, theirs), (theirs, mine)))
    split = alone + others_alone
    p = min(1.0, 2 * sum(math.comb(split, k) for k in range(min(alone, others_alone) + 1)) / 2 ** split)
    agree = len(pairs) - split
    right = sum(mine[pair] and theirs[pair] for pair in pairs)
    log(f"{name}: {judge} against {other} on {len(pairs)} pairs: right on {sum(mine[pair] for pair in pairs)} "
        f"against {sum(theirs[pair] for pair in pairs)}; only {judge} right on {alone}, only {other} on "
        f"{others_alone} (McNemar p = {p:.2f}); they pick the same Reel in {agree} pairs and are right on "
        f"{right} of them ({right / max(agree, 1):.0%})")


def run(name: str, judge: str, describer: str, captions: bool, workers: int, limit: int | None,
        frames: bool = False, sound: bool = False) -> None:
    rows = {row["id"]: row for row in map(json.loads, DATASET.open())}
    forms = load_descriptions(describer)
    pairs = [pair for pair in load_pairs(name, rows)[:limit] if pair["a"] in forms and pair["b"] in forms
             and (not sound or (load_sound(pair["a"]) and load_sound(pair["b"])))]
    todo = [pair for pair in pairs if not verdict_path(name, judge, describer, captions, pair, frames, sound).exists()]
    log(f"{len(todo)} of {len(pairs)} pairs to judge with {judge}")
    stop, counts = threading.Event(), Counter()

    def one(pair: dict) -> str:
        if stop.is_set():
            return "left for later"
        try:
            record = judge_pair(name, judge, describer, captions, pair, forms, rows, frames, sound)
            return "right" if record["right"] else "wrong"
        except STOPS as limit:
            stop.set()
            log(f"stopping: {type(limit).__name__}: {limit}")
            return "out of allowance"
        except FAILURES as error:
            log(f"{pair['a']}-{pair['b']}: {type(error).__name__}: {error}")
            return "failed"

    with ThreadPoolExecutor(workers) as pool:
        for count, result in enumerate(pool.map(one, todo), 1):
            counts[result] += 1
            if count % 25 == 0:
                log(f"{count}/{len(todo)} {dict(counts)}")
    log(f"done: {dict(counts)}")
    report(name, judge, describer, captions, frames, sound)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "report", "compare"])
    parser.add_argument("--judge", default="codex:gpt-6.1-sol:medium")
    parser.add_argument("--against", help="compare: the judge to set against --judge, on the same describer")
    parser.add_argument("--set", dest="name", default="held-out", help="which pairs (held-out, sol-sample)")
    parser.add_argument("--describer", default="codex:gpt-6-luna", help="whose descriptions the judge reads")
    parser.add_argument("--caption", dest="captions", action="store_true", help="show each Reel's caption too")
    parser.add_argument("--first-frames", dest="frames", action="store_true", help="show both first frames too")
    parser.add_argument("--sound", action="store_true", help="add what audio models measured of each sound")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, help="only the first N pairs")
    args = parser.parse_args()
    if args.command == "run":
        run(args.name, args.judge, args.describer, args.captions, args.workers, args.limit, args.frames, args.sound)
    elif args.command == "compare":
        compare(args.name, args.judge, args.against, args.describer)
    else:
        report(args.name, args.judge, args.describer, args.captions, args.frames, args.sound)
    return 0


if __name__ == "__main__":
    sys.exit(main())
