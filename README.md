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
  write-ups and picks the one that will do better. **Muse Spark** judges the
  same pair alongside it.
- When the two agree, PreFlight says which to post; when they do not, it says
  the pair is too close to call. **Muse Spark** then says why, and what could
  lift the other one.

On 207 pairs from 37 accounts no model trained on, fine-tuning takes Glimmer
from a coin toss (104 right, 50%) to **133 (64%)**, level with Muse Spark
asked zero-shot (135, 65%). The two are right on different pairs, though:
**where they agree, they are right 74% of the time, and the Reel they pick
typically got twice the views of the other**; 81% when Glimmer is also sure.
Where they disagree, it is a coin toss, and PreFlight says so.

## How it was built

Everything below was written during the hackathon (Oct 26 to Nov 4, 2026);
the git history shows it.

```
Instagram Graph API ──► labels ──► first 5 s ──► Muse Spark ──► pairs ──► QLoRA on Modal ──► held-out test
(Business Discovery)   vs the      10 frames,    writes it      same         Muse Glimmer 30B     207 pairs,
                       account's   words, loud-  down, second   account,     answers "A" or "B"   37 unseen
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
   what is heard, then the hook. About $0.012 an opening; 4,042 openings.
5. **Pairs within one account** (`preflight/pairs.py`). One Reel that beat
   its account's usual and one that did not, at least 2x apart; closer pairs
   are mostly luck. A quarter of the accounts, picked by a hash of the name,
   never train: 207 test pairs (311 Reels, 37 accounts). Training gets 3,414
   pairs from 289 other accounts, each shown in both orders.
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
   on Modal and Muse Spark judge side by side, Muse Spark explains. About 26
   seconds a comparison once the judge's GPU is up.

## Results

Every judge takes the same 207 held-out pairs, from accounts no fine-tune saw.
50% is a coin toss. Pairs from one account are not independent, so the 95%
ranges here come from drawing whole accounts again (`preflight.pairs
together`): about ±10 points.

| Judge | Right of 207 |
|---|---|
| Muse Glimmer 30B, as it comes | 104 (50%) |
| Muse Glimmer 30B, fine-tuned, after 1 epoch | 131 (63%) |
| **Muse Glimmer 30B, fine-tuned (2 epochs)** | **133 (64%)** |
| Muse Spark 1.3, zero-shot | 135 (65%) |

The fine-tuned Glimmer's range is 54–74%, Spark's 58–72%. Set pair by pair
(`preflight.pairs compare`), the fine-tuned Glimmer alone is right on 38 pairs
Spark gets wrong, and Spark alone on 40 (McNemar p = 0.91): as good as each
other, but not the same.

### Knowing when it knows

So PreFlight asks both and reads their answers together:

| The two judges | Pairs | Right (95% range) | The pick's views, typically |
|---|---|---|---|
| Disagree | 78 | 38 (49%, 35–64%) | 1.06x the other's |
| Agree, Glimmer less sure than on half the pairs | 61 | 40 (66%, 56–76%) | 1.13x |
| Agree, Glimmer surer | 68 | 55 (81%, 68–91%) | 3.29x |
| Agree, either way | 129 | 95 (74%, 63–83%) | 1.99x |

Views are each Reel's against its account's usual, and "typically" is the
geometric mean of the pick's over the other's. Where the two agree, the pick
got a median 1.59x its account's usual and the Reel passed over 0.72x. Where
they disagree PreFlight makes no call; that row is what following Glimmer
anyway would have done. The test pairs are at least 2x apart, so between two
closer openings a right call gains less.

Asked twice, Spark agrees with itself far more often (175 pairs, with and
without the first frames), but is then right only 68% of the time: it is two
different models agreeing that carries the signal.

Glimmer's own probability is not worth showing: trained to near-zero loss, it
is 99% sure or more of 162 of the 207 pairs and right on 68% of those. How sure
it is still ranks its calls, though: its least sure quarter is right on 25 of
51 pairs (49%) and its surest on 38 of 52 (73%). So the demo quotes each call
the record of the held-out calls like it, never a made-up percentage.

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
  down every opening (low effort), judges each pair zero-shot alongside the
  fine-tuned Glimmer (medium effort), and explains the verdict. Every call
  goes through a ledger that refuses anything past the credit.
- **Muse Glimmer 30B** (open weights, `meta-models/Muse-Glimmer-30B`):
  fine-tuned with QLoRA and served from Modal (`preflight.finetune.Judge`).
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

- 207 test pairs from 37 accounts: the ranges, drawn by account, are wide
  (54–74% for the fine-tuned Glimmer alone).
- The pairs are at least 2x apart. Between two openings that would do about
  equally well, PreFlight is guessing, and says so when its judges disagree.
- Only the first five seconds, and only what can be written down: no pacing
  beyond the half-second frames, and no music beyond loudness and words.
- Professional accounts large enough to be found through mentions; small
  creators may behave differently.
