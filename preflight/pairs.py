"""Which of two Reels from one account did better? Asked of a judge that only
reads their written-down openings.

    uv run python -m preflight.pairs run --judge codex:gpt-6.1-sol:medium --describer codex:gpt-6-luna [--caption]
                                         [--first-frames] [--sound]
    uv run python -m preflight.pairs report --judge codex:gpt-6.1-sol:medium --describer codex:gpt-6-luna
    uv run python -m preflight.pairs compare --judge glimmer:pairs-spark --against meta:muse-spark-1.3:medium
                                             --describer meta:muse-spark-1.3:low
    uv run python -m preflight.pairs together --judge glimmer:pairs-spark --against meta:muse-spark-1.3:medium
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
`--set fresh` is a second test, drawn on 2026-10-31 from the held-out Reels
with a video that the first set did not use: 419 pairs from 75 accounts, 53 of
them in no earlier pair. Its calls are read with the first set's line for sure
(`levels`), as the demo reads them.

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

**Together.** `together` reads a fine-tuned judge and another side by side, as
the demo does (`LEVELS`): how often each is right alone, and where they
disagree, agree, and agree with the fine-tuned one sure. For each, what the
picked Reel got against its account's usual next to the one passed over, and
95% ranges that draw whole accounts again, since pairs from one account are not
independent.
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
from statistics import median

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
#: With a second judge, the kinds of call, each with its own record: the two
#: pick different Reels; the same, with the fine-tuned judge less sure than on
#: half the pairs; the same, and it surer.
LEVELS = ("disagree", "agree", "agree, sure")

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
        # A later set (`fresh`) draws only from the Reels the held-out pairs left unused.
        used = set() if name == "held-out" else {pair[side] for pair in load_pairs("held-out", rows)
                                                 for side in ("a", "b")}
        pairs = build([row for row in rows.values() if held_out(row["username"])
                       and video_path(row["id"]).exists() and row["id"] not in used])
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
    wrote both Reels of; another set's are scored apart (`--test-set`)."""
    scored = judge.split(":", 1)[1] + ("" if name == "held-out" else f"-{name}")
    result = json.loads((DATASET.parent / "finetune" / f"{scored}.json").read_text())
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


def levels(name: str, judge: str, other: str, describer: str) -> tuple[float, dict[tuple[str, str], int]]:
    """Which of `LEVELS` a fine-tuned judge's call and another judge's make,
    pair by pair (two answers agree when both are right or both wrong), and the
    sureness from which the fine-tuned one's call is sure: its median on the
    held-out pairs, the line the demo draws, whichever set is read."""
    chances, mine, theirs = tuned_chances(name, judge), rights(name, judge, describer), rights(name, other, describer)
    ruler = sorted(abs(chance - 0.5) for chance in tuned_chances("held-out", judge).values())
    sure_from = ruler[len(ruler) // 2]
    return sure_from, {pair: 2 if mine[pair] == theirs[pair] and abs(chances[pair] - 0.5) >= sure_from
                       else int(mine[pair] == theirs[pair]) for pair in mine.keys() & theirs.keys()}


def by_account(calls: list[tuple], stat, draws: int = 10_000) -> tuple[float, float]:
    """A 95% range for `stat` of `calls`, each led by its account, from drawing
    whole accounts again with replacement."""
    accounts = defaultdict(list)
    for call in calls:
        accounts[call[0]].append(call)
    rng, names = random.Random(SEED), sorted(accounts)
    values = sorted(stat([call for account in rng.choices(names, k=len(names)) for call in accounts[account]])
                    for _ in range(draws))
    return values[int(0.025 * draws)], values[int(0.975 * draws)]


def together(name: str, judge: str, other: str, describer: str) -> None:
    """A fine-tuned judge and another, alone and read together (`levels`): how
    often the pick did better, what it got against its account's usual next to
    the Reel passed over (medians), and typically how many times the other's
    views (the geometric mean). The pairs are at least `GAP` times apart, so
    between two closer openings a right call gains less."""
    rows = {row["id"]: row for row in map(json.loads, DATASET.open())}
    pairs = {(pair["a"], pair["b"]): pair for pair in load_pairs(name, rows)}
    sure_from, level = levels(name, judge, other, describer)
    mine, theirs, ratios = rights(name, judge, describer), rights(name, other, describer), {}
    for key in level:
        pair = pairs[key]
        hit, miss = (pair["a"], pair["b"]) if pair["better"] == "A" else (pair["b"], pair["a"])
        ratios[key] = pair["account"], rows[hit]["ratio"], rows[miss]["ratio"]

    def calls(right: dict, keys: list) -> list[tuple]:
        return [(account, right[key], *((up, down) if right[key] else (down, up)))
                for key in keys for account, up, down in [ratios[key]]]

    def share(some: list[tuple]) -> float:
        return sum(call[1] for call in some) / len(some)

    def gain(some: list[tuple]) -> float:
        return math.exp(sum(math.log(call[2] / call[3]) for call in some) / len(some))

    log(f"{name}: {judge} with {other}, reading {describer}; sure from {sure_from:.4f}")
    for label, right, keys in [(f"{judge} alone", mine, list(level)), (f"{other} alone", theirs, list(level)),
                               *((kind, mine, [key for key in level if level[key] == n])
                                 for n, kind in enumerate(LEVELS)),
                               ("agree, sure or not", mine, [key for key in level if level[key]])]:
        some = calls(right, keys)
        (low, high), (least, most) = by_account(some, share), by_account(some, gain)
        log(f"  {label}: right on {sum(call[1] for call in some)} of {len(some)} = {share(some):.0%} "
            f"(95% by account: {low:.0%}-{high:.0%}); the pick got {median(call[2] for call in some):.2f}x its "
            f"usual, the other {median(call[3] for call in some):.2f}x; typically {gain(some):.2f}x the other's "
            f"views ({least:.2f}x-{most:.2f}x)")


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
    parser.add_argument("command", choices=["run", "report", "compare", "together"])
    parser.add_argument("--judge", default="codex:gpt-6.1-sol:medium")
    parser.add_argument("--against", help="compare, together: the judge to set against --judge, on the same describer")
    parser.add_argument("--set", dest="name", default="held-out", help="which pairs (held-out, fresh, sol-sample)")
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
    elif args.command == "together":
        together(args.name, args.judge, args.against, args.describer)
    else:
        report(args.name, args.judge, args.describer, args.captions, args.frames, args.sound)
    return 0


if __name__ == "__main__":
    sys.exit(main())
