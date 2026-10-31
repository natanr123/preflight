# PreFlight

**Two openings, one account. See which will do better before you post.**

A creator often has two takes of a Reel, or two ideas for one, and only finds
out which works after posting. PreFlight takes the two videos and says which
opening is likely to do better *against that account's usual*, how often calls
like this one were right in testing, and why. It reads the first five
seconds, where viewers decide whether to stay: what is seen, what is written
on screen, what is said and heard.

- **Muse Spark** watches each opening and writes it down, second by second.
- **Muse Glimmer 30B, fine-tuned** with QLoRA on Modal, reads the two
  write-ups and picks the one that will do better, with how often calls as
  sure as this one were right in testing.
- **Muse Spark** then says why, and what could lift the other one.

On 626 test pairs from 90 accounts no model trained on, drawn in two rounds,
the fine-tuned Glimmer picks the Reel that did better **67% of the time**,
against 57% for Muse Spark asked zero-shot (and 50% for Glimmer before
fine-tuning). **In its surest quarter of calls it is right 82% of the time,
and the Reel it picks typically got three times the views of the other.**

## How it was built

Everything below was written during the hackathon (Oct 26 to Nov 4, 2026);
the git history shows it.

```
Instagram Graph API ──► labels ──► first 5 s ──► Muse Spark ──► pairs ──► QLoRA on Modal ──► held-out tests
(Business Discovery)   vs the      10 frames,    writes it      same         Muse Glimmer 30B     626 pairs,
                       account's   words, loud-  down, second   account,     answers "A" or "B"   90 unseen
                       usual       ness, cuts    by second      ≥2x apart                         accounts
```

1. **Data, through the official API only** (`preflight/crawl.py`,
   `preflight/ig.py`). Instagram's Business Discovery returns another
   professional account's Reels with their view counts and a link to the
   video. The crawl starts from seed lists and follows @mentions in bios:
   1,787 accounts read, 62,482 Reels. No scraping.
2. **Labels against the account's usual** (`preflight/dataset.py`). Raw views
   mostly measure the account, which is fixed before anyone posts. So each
   Reel's views are divided by its account's median (Reels 14 to 120 days
   old, age-adjusted): 23,040 labelled Reels from 679 accounts. Instagram
   withholds the video of rights-claimed Reels, which leaves 8,974 videos.
3. **The opening** (`preflight/describe.py`). The first five seconds become
   ten frames on one contact sheet, one every half second, plus what is
   heard: the words (faster-whisper), how loud each second is and where the
   shots change (ffmpeg).
4. **Muse Spark writes it down.** `muse-spark-1.3` fills the same form for
   every opening: for each second, what is seen, what text is on screen and
   what is heard, then the hook. About $0.012 an opening; 4,042 openings for
   training and the first test, 605 for the second.
5. **Pairs within one account** (`preflight/pairs.py`). One Reel that beat
   its account's usual and one that did not, at least 2x apart; closer pairs
   are mostly luck. A quarter of the accounts, picked by a hash of the name,
   never train: 207 test pairs (311 Reels, 37 accounts). Training gets 3,414
   pairs from 289 other accounts, each shown in both orders. Once all else was
   fixed, a second test was drawn from the held-out Reels the first left
   unused: 419 pairs (605 Reels) from 75 accounts, 53 of them new.
6. **Fine-tune Muse Glimmer 30B** (`preflight/finetune.py`). QLoRA after
   Meta's post-training guide for Glimmer (4-bit NF4, r 16, alpha 32, the
   language model's attention projections) with TRL, on one Modal GPU; the
   answer is "A" or "B", without reasoning. Two epochs (854 steps) took 4.7
   hours and about $10.
7. **Score** on the held-out pairs: the probability of the answer token, read
   with the pair both ways round and averaged, so a lean towards A or B
   cancels out.
8. **Demo** (`preflight/demo.py`). A page that takes two videos and runs the
   same path: cut, listen, Muse Spark writes it down, the fine-tuned Glimmer
   on Modal makes the call, Muse Spark explains. About half a minute a
   comparison once the judge's GPU is up.

## Results

Every judge takes the same test pairs, from accounts no fine-tune saw: first
207 pairs from 37 accounts, then 419 more, drawn once everything else was
fixed, from 75 accounts (53 of them new). 50% is a coin toss. Pairs from one
account are not independent, so the 95% ranges come from drawing whole
accounts again (`preflight.pairs together`).

| Judge | First test (207) | Second test (419) | Both (626) |
|---|---|---|---|
| Muse Glimmer 30B, as it comes | 104 (50%) | | |
| Muse Glimmer 30B, fine-tuned, after 1 epoch | 131 (63%) | | |
| **Muse Glimmer 30B, fine-tuned (2 epochs)** | **133 (64%)** | **285 (68%)** | **418 (67%, 61–72%)** |
| Muse Spark 1.3, zero-shot | 135 (65%) | 222 (53%) | 357 (57%, 52–62%) |

On the first test the two looked level, and right on different pairs: each
alone got about 40 that the other missed (38 and 40, McNemar p = 0.91). The
second test, twice the size, settled it. The fine-tuned Glimmer alone was
right on 117 pairs Muse Spark got wrong, Spark alone on 54 (p < 0.0001), and
Glimmer held up on the 53 accounts new to the second test (197 of 293, 67%).
Across both tests, its pick got a median 1.52x its account's usual and the
Reel passed over 0.76x: typically (the geometric mean) 1.69x the other's
views.

### How sure it is

Glimmer's own probability is not worth showing: trained to near-zero loss, it
is 99% sure or more of nearly every pair. How sure it is still ranks its
calls, though. Split into quarters by it, over both tests, it was right 60%,
61%, 65% and 82% of the time:

| Glimmer's calls | Right (95% range) | The pick's views, typically |
|---|---|---|
| Its surest quarter | 128 of 157 (82%, 73–89%) | 3.03x the other's (1.83–5.02x) |
| The rest | 290 of 469 (62%, 56–67%) | 1.39x (1.03–1.87x) |

So the demo quotes each call the record of its band, never a made-up
percentage. This held up out of sample: the line for a sure call was drawn on
the first test (its median sureness), and on the second, the calls past it
were right on 156 of 209 (75%), those below it on 129 of 210 (61%). Views are
each Reel's against its account's usual, and the test pairs are at least 2x
apart, so between two closer openings a right call gains less.

### Two judges, tried and dropped

On the first test, asking Glimmer and Muse Spark together looked like the
signal: where they agreed they were right on 95 of 129 pairs (74%), where they
did not on 38 of 78 (49%), so the demo called those pairs too close. The
second test did not bear it out: where they disagreed, Glimmer was right on
117 of 171 (68%). Over both, agreement adds little to Glimmer's own sureness
(Glimmer is right on 70% of the pairs where they agree, 62% where they do
not), so the demo now lets Glimmer make every call.

### What did not help

Measured, then left out:

- **One Reel at a time.** Asked "did this Reel beat its usual?", Glimmer
  learned nothing: every held-out Reel got exactly 0.5. Comparing two Reels
  from one account takes the account out of the question.
- **The first frame as an image.** Fine-tuned with each Reel's first frame
  above its write-up, Glimmer got 129 of 207 right against 127 without it (a
  one-epoch practice run), at 1.7 times the GPU time; Muse Spark shown both
  first frames got 135, as without them. The write-up already says what is in
  the frame.
- **More training pairs.** 3,414 pairs (133 right) did no better than
  practice runs on 1,282 pairs written down by another model (127 to 139):
  the limit seems to be what five seconds written down can tell, not how much
  the model has seen.
- **Sound beyond words and loudness.** AudioSet labels, CLAP embeddings, beat
  and brightness: on their own, at chance on the held-out pairs. Reels using
  a sound another account used first (a "trending" sound) did no better within
  their account.
- **What is known before posting.** Caption words, length, hashtags and
  posting time, and CLIP embeddings of the cover, score an AUC of 0.49 to 0.53
  on accounts the model never saw (`preflight/baseline.py`): a coin toss.

## Meta AI used

- **Muse Spark 1.3** through the Meta Model API (`preflight/meta.py`): writes
  down every opening (low effort), is the zero-shot judge every test is set
  against (medium effort), and explains the demo's call. Every call goes
  through a ledger that refuses anything past the credit.
- **Muse Glimmer 30B** (open weights, `meta-models/Muse-Glimmer-30B`):
  fine-tuned with QLoRA and served from Modal (`preflight.finetune.Judge`),
  where it makes the demo's call.
- **Instagram Graph API** (Business Discovery) for the data.

## Try it

**https://natanr123--preflight-web-web-site.us-east.modal.run**

Drop in two videos you might post on one account. The first comparison after
a quiet spell waits about a minute while the judge's GPU starts; then each
takes about half a minute. The live demo (`preflight/web.py`) runs on Modal:
one CPU container for the page, ffmpeg and speech, the fine-tuned Glimmer on
its GPU, and Muse Spark through the Meta Model API. It spends the hackathon's
credit, so it keeps its own ledger with a hard limit, and allows 20
comparisons a day to the public (judges get a link without the daily limit).
Videos are deleted within 15 minutes.

## Run it

Needs Python 3.13, [uv](https://docs.astral.sh/uv/), ffmpeg, a Meta Model API
key and a Modal account.

```bash
uv sync
echo "META_API_KEY=..." > secrets/meta.env
uv run python -m preflight.meta arm --limit 100     # the most the ledger will let it spend
uvx modal token new
uv run modal deploy -m preflight.finetune           # the fine-tuned judge, on Modal
uv run python -m preflight.demo                     # Muse Spark + glimmer:pairs-spark
```

Then open http://127.0.0.1:8008 and drop in two videos. The first comparison
waits about a minute while Modal starts the judge's GPU.

To host it as above, put the key in a Modal secret and deploy the web app:

```bash
uv run --with python-dotenv modal secret create preflight-meta --from-dotenv secrets/meta.env
echo "PREFLIGHT_CODE=$(openssl rand -hex 6)" > secrets/web.env  # judges' code
uv run --with python-dotenv modal secret create preflight-web --from-dotenv secrets/web.env
uv run modal deploy -m preflight.web
```

The whole pipeline, from the crawl to the fine-tune (needs an Instagram
professional account's token in `secrets/ig.env`):

```bash
uv run python -m preflight.crawl seed seeds/instagram.txt
uv run python -m preflight.crawl run --hours 10
uv run python -m preflight.dataset
uv run python -m preflight.videos --every 30 --hours 10
uv run python -m preflight.describe prepare
uv run python -m preflight.describe run --with meta:muse-spark-1.3:low --pairs 15
uv run python -m preflight.pairs run --judge meta:muse-spark-1.3:medium --describer meta:muse-spark-1.3:low
uv run modal run -m preflight.finetune --run pairs-spark --describer meta:muse-spark-1.3:low --model glimmer --epochs 2
# the second test, drawn once all else is fixed: draw it, write it down, judge it
uv run python -m preflight.pairs run --set fresh --judge meta:muse-spark-1.3:medium --describer meta:muse-spark-1.3:low
uv run python -m preflight.describe run --with meta:muse-spark-1.3:low --pairs 0
uv run python -m preflight.pairs run --set fresh --judge meta:muse-spark-1.3:medium --describer meta:muse-spark-1.3:low
uv run modal run -m preflight.finetune --run pairs-spark --describer meta:muse-spark-1.3:low --model glimmer --no-retrain --no-baseline --test-set fresh
```

## Data and rights

- Official APIs only: Business Discovery for Reels and view counts, the Meta
  Model API and Modal for the models.
- The Reels belong to their creators. The dataset, videos, frames,
  write-ups and fine-tuned weights stay private and are not published, and
  the raw crawl is deleted after the hackathon. The demo video uses the
  author's own Reels.
- The Meta Model API is used in standard mode (not contributor mode), on the
  hackathon's credit.
- The code is under the MIT license (`LICENSE`).

## Limitations

- 626 test pairs from 90 accounts: the ranges, drawn by account, are still
  about ±5 points (61–72% for the fine-tuned Glimmer).
- The pairs are at least 2x apart. Between two openings that would do about
  equally well, PreFlight is guessing; it still makes a call, quoting the
  record of calls as sure.
- Only the first five seconds, and only what can be written down: no pacing
  beyond the half-second frames, no music beyond loudness and words, and not
  the frame's shape (a landscape and a vertical cut of one opening are written
  down alike).
- Professional accounts large enough to be found through mentions; small
  creators may behave differently. On the author's own channel (animated
  children's stories, 57 followers), Glimmer got 13 of 29 same-story pairs
  right and Muse Spark 19, and Glimmer was never in its surest quarter there:
  three stories say little, but it is not the kind of account it learned
  from.
