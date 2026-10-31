"""PreFlight's demo: two openings in; which will do better on the account, and why.

    uv run python -m preflight.demo [--describer meta:muse-spark-1.3:low] [--judge glimmer:pairs-spark]
                                    [--second meta:muse-spark-1.3:medium] [--explainer meta:muse-spark-1.3:medium]
                                    [--port 8008]

**What it does.** You drop in two videos you might post on one account: two
takes of a Reel, or two different Reels. Each one's start is shrunk the way
the training videos were (`preflight.videos.shrink`), its first five seconds
become a contact sheet and what is heard in them
(`preflight.describe.opening`), and the describer writes it down second by
second, in the form the models were trained and tested on. The judge then
says which will do better against the account's usual, and the explainer says
why and what could lift the other one.

**Judges.** `glimmer:<run>` is the fine-tuned Muse Glimmer on Modal
(`preflight.finetune.Judge`, deployed first), reading the describer its
adapter was trained on. It reads the pair both ways round and averages the
two, as in its test. Its own probability is not shown: trained to near-zero
loss, it is all but certain of nearly every pair (162 of 207 held-out pairs at
99% or more for `pairs-spark`, right on 68% of them). How sure it is still
ranks its calls, though: split by it into quarters, `pairs-spark` got 25 of 51
held-out pairs right in its least sure quarter and 38 of 52 in its surest. So
the page quotes a call the record of the held-out calls about as sure as it
(`BANDS`). Any other judge is a model asked zero-shot, the way the pair test
asks it (`preflight.pairs.PROMPT`).

**A second judge.** Glimmer fine-tuned and Muse Spark zero-shot are about as
good (133 and 135 of the 207 held-out pairs) but right on different pairs, so
`--second` asks Muse Spark too, alongside. Where the two picked the same Reel
they were right on 95 of 129 held-out pairs (74%), and on 55 of 68 when
Glimmer was also surer than on half of them; where they differed, on 38 of 78,
a coin toss. So the page calls those too close, and quotes each call the
record of its kind (`preflight.pairs.LEVELS`).

**Cached.** An opening is kept by its file's hash (`data/demo/<hash>/`),
with each describer's form, so showing the same videos again costs nothing.

**Warm.** The fine-tuned judge's GPU goes cold after four idle minutes and
takes about a minute to come back, so the page asks for it as soon as a video
is chosen (`/api/warm`), and it loads while the openings are read.

**On Modal.** `preflight.web` serves the same page and API (`make_app`) to
anyone with the link.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from preflight.describe import SECONDS, as_text, ask, log, opening, write_down
from preflight.pairs import LEVELS, PROMPT, VERDICT, levels, rights, tuned_chances, verdict_path
from preflight.store import DEFAULT_PATH
from preflight.videos import shrink

DEMO = DEFAULT_PATH.parent / "demo"
PAGE = Path(__file__).with_name("demo.html")
SIDES = ("a", "b")
#: A fine-tuned judge's chance this near a half means its two readings of the
#: pair contradict each other (each all but certain, the other way round). On
#: the held-out pairs: 11 of 22 right for `pairs-luna-3ep`, 9 of 13 for `pairs-spark`.
CLOSE = 0.1
#: The held-out pairs in this many equal bands, from the fine-tuned judge's
#: least sure calls to its surest, each band with its own record.
BANDS = 4
#: Ask for the fine-tuned judge's GPU at most this often, in seconds: a little
#: less than it stays up idle (`preflight.finetune.Judge`).
WARM_EVERY = 3 * 60

EXPLAIN = """\
Two openings for Reels on the same Instagram account are written down below, second by second. PreFlight {expects}
Reel {better} to get more views than Reel {other}, measured against how the account's Reels usually do.

In `why`, say in one or two sentences what in the openings most likely makes that difference, pointing to what is
seen, shown on screen or heard. In `lift`, say in one sentence what could make Reel {other}'s opening stronger.

Reel A
{a}

Reel B
{b}

Answer with the form only.
"""

EXPLANATION = {
    "type": "object",
    "additionalProperties": False,
    "required": ["why", "lift"],
    "properties": {"why": {"type": "string"}, "lift": {"type": "string"}},
}

#: Every comparison's progress and results, by job, for the page to poll.
JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def update(target: dict, **changes) -> None:
    with _LOCK:  # the page reads these while they are filled in
        target.update(changes)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()[:16]


def ready(upload: Path, describer: str, progress: dict) -> dict:
    """An uploaded video's opening, written down by `describer`: its form."""
    key = digest(upload)
    folder = DEMO / key
    facts_path, form_path = folder / "facts.json", folder / f"{describer.replace(':', '-')}.json"
    if not facts_path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        update(progress, stage="cutting")
        if shrink(upload, folder / "opening.mp4", SECONDS + 1).returncode:
            raise ValueError(f"{progress['name']} could not be read as a video")
        update(progress, stage="listening")
        facts = opening(folder / "opening.mp4", folder / "sheet.jpg")
        facts_path.write_text(json.dumps(facts, ensure_ascii=False))
    facts = json.loads(facts_path.read_text())
    update(progress, key=key, words=" ".join(word["word"] for word in facts["words"]))
    if not form_path.exists():
        update(progress, stage="describing")
        form, _ = write_down(describer, facts, folder / "sheet.jpg")
        form_path.write_text(json.dumps(form, ensure_ascii=False))
    form = json.loads(form_path.read_text())
    update(progress, stage="described", form=form)
    return form


def judge(judge: str, forms: dict[str, dict]) -> dict:
    """Which opening the judge expects to do better, with how sure it is when
    it can say."""
    texts = {side: as_text(form) for side, form in forms.items()}
    if judge.startswith("glimmer:"):
        import modal

        from preflight.finetune import pair_text

        rate = modal.Cls.from_name("preflight", "Judge")(run=judge.split(":", 1)[1]).rate
        a_first, b_first = rate.remote([pair_text(texts, "a", "b"), pair_text(texts, "b", "a")])
        chance = (a_first + 1 - b_first) / 2  # that A does better, read both ways round
        return {"better": "A" if chance > 0.5 else "B", "close": abs(chance - 0.5) < CLOSE,
                "sureness": abs(chance - 0.5)}
    answer, _ = ask(judge, PROMPT.format(captions="", frames="", sound="", a=texts["a"], b=texts["b"]), VERDICT)
    return {"better": answer["better"], "close": False}


def decide(config: dict, forms: dict[str, dict]) -> dict:
    """The judge's call; with a second judge, asked at the same time, which of
    `LEVELS` the two calls make, and too close to call when they differ."""
    judges = [config["judge"]] + ([config["second"]] if config["second"] else [])
    with ThreadPoolExecutor(len(judges)) as pool:
        call, *second = pool.map(lambda name: judge(name, forms), judges)
    if not second:
        return call
    agree, sure_from = second[0]["better"] == call["better"], (config["record"] or {}).get("sure_from")
    sure = sure_from is not None and call.get("sureness", 0) >= sure_from
    return call | {"second": second[0]["better"], "close": not agree, "level": 2 if agree and sure else int(agree)}


def explain(explainer: str, forms: dict[str, dict], verdict: dict) -> dict:
    better = verdict["better"]
    prompt = EXPLAIN.format(expects="finds the two close, but leans to" if verdict["close"] else "expects",
                            better=better, other="B" if better == "A" else "A",
                            a=as_text(forms["a"]), b=as_text(forms["b"]))
    return ask(explainer, prompt, EXPLANATION)[0]


def compare(job: str, uploads: dict[str, Path], config: dict) -> None:
    state = JOBS[job]
    try:
        with ThreadPoolExecutor(len(SIDES)) as pool:
            forms = dict(zip(SIDES, pool.map(
                lambda side: ready(uploads[side], config["describer"], state["openings"][side]), SIDES)))
        update(state, stage="judging")
        verdict = decide(config, forms)
        update(state, stage="explaining", verdict=verdict)
        update(state, stage="done", verdict=verdict | explain(config["explainer"], forms, verdict))
    except Exception as error:  # shown on the page, and the server carries on
        log(f"{job}: {type(error).__name__}: {error}")
        update(state, stage="failed", error=f"{type(error).__name__}: {error}")
    finally:
        for upload in uploads.values():
            upload.unlink(missing_ok=True)


def track_record(judge: str, describer: str, second: str | None) -> dict | None:
    """How the judge did on the held-out pairs, for the page to quote: in all,
    and for a fine-tuned judge, in `BANDS` bands by how sure it was, least sure
    first, each with the least sureness in it; with a second judge too, by
    `LEVELS`, with the sureness that makes a call sure."""
    if not judge.startswith("glimmer:"):
        folder = verdict_path("held-out", judge, describer, False, {"a": "", "b": ""}).parent
        records = [json.loads(path.read_text()) for path in folder.glob("*.json")]
        return {"right": sum(record["right"] for record in records), "pairs": len(records)} if records else None
    if not (DEMO.parent / "finetune" / f"{judge.split(':', 1)[1]}.json").exists():
        return None
    chances, right = tuned_chances("held-out", judge), rights("held-out", judge, describer)
    record = {"right": sum(right.values()), "pairs": len(right)}
    calls = sorted(((abs(chances[pair] - 0.5), right[pair]) for pair in right), key=lambda call: call[0])
    bands = [calls[len(calls) * band // BANDS:len(calls) * (band + 1) // BANDS] for band in range(BANDS)]
    record["bands"] = [{"from": band[0][0], "right": sum(hit for _, hit in band), "pairs": len(band)}
                       for band in bands]
    if second:
        sure_from, level = levels("held-out", judge, second, describer)
        record |= {"sure_from": sure_from,
                   "levels": [{"right": sum(right[pair] for pair in level if level[pair] == n),
                               "pairs": sum(level[pair] == n for pair in level)} for n in range(len(LEVELS))]}
    return record


def make_app(config: dict):
    """The page and its API, for `config` with its "record" already in."""
    from fastapi import FastAPI, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse

    app, warmed = FastAPI(title="PreFlight"), [float("-inf")]

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE.read_text()

    @app.get("/api/config")
    def settings() -> dict:
        return config

    @app.post("/api/warm")
    def warm() -> dict:
        if config["judge"].startswith("glimmer:") and time.monotonic() - warmed[0] > WARM_EVERY:
            import modal

            warmed[0] = time.monotonic()
            modal.Cls.from_name("preflight", "Judge")(run=config["judge"].split(":", 1)[1]).ping.spawn()
        return {}

    @app.post("/api/compare")
    def start(a: UploadFile, b: UploadFile) -> dict:
        job, folder = uuid.uuid4().hex[:12], DEMO / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        uploads = {side: folder / f"{job}-{side}" for side in SIDES}
        for side, upload in zip(SIDES, (a, b)):
            with uploads[side].open("wb") as out:
                shutil.copyfileobj(upload.file, out)
        JOBS[job] = {"stage": "openings",
                     "openings": {side: {"name": upload.filename, "stage": "waiting"}
                                  for side, upload in zip(SIDES, (a, b))}}
        threading.Thread(target=compare, args=(job, uploads, config), daemon=True).start()
        return {"job": job}

    @app.get("/api/jobs/{job}")
    def progress(job: str) -> dict:
        if job not in JOBS:
            raise HTTPException(404)
        with _LOCK:
            return json.loads(json.dumps(JOBS[job]))

    @app.get("/api/openings/{key}/sheet.jpg")
    def sheet(key: str) -> FileResponse:
        if not re.fullmatch(r"[0-9a-f]{16}", key) or not (DEMO / key / "sheet.jpg").exists():
            raise HTTPException(404)
        return FileResponse(DEMO / key / "sheet.jpg")

    return app


def serve(config: dict, port: int) -> None:
    import uvicorn

    config["record"] = track_record(config["judge"], config["describer"], config["second"])
    log(f"PreFlight on http://127.0.0.1:{port} (describer {config['describer']}, judge {config['judge']}, "
        f"second judge {config['second']}, explainer {config['explainer']})")
    uvicorn.run(make_app(config), host="127.0.0.1", port=port, log_level="warning")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--describer", default="meta:muse-spark-1.3:low")
    parser.add_argument("--judge", default="glimmer:pairs-spark")
    parser.add_argument("--second", default="meta:muse-spark-1.3:medium", help="a zero-shot judge asked alongside; "
                        "'none' for the judge alone")
    parser.add_argument("--explainer", default="meta:muse-spark-1.3:medium")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()
    serve({"describer": args.describer, "judge": args.judge, "second": None if args.second == "none" else args.second,
           "explainer": args.explainer}, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
