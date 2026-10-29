"""PreFlight's demo: two openings in; which will do better on the account, and why.

    uv run python -m preflight.demo [--describer codex:gpt-6-luna] [--judge codex:gpt-6.1-sol:medium]
                                    [--explainer codex:gpt-6.1-sol:medium] [--port 8008]

**What it does.** You drop in two videos you might post on one account: two
takes of a Reel, or two different Reels. Each one's start is shrunk the way
the training videos were (`preflight.videos.shrink`), its first five seconds
become a contact sheet and what is heard in them
(`preflight.describe.opening`), and the describer writes it down second by
second, in the form the models were trained and tested on. The judge then
says which will do better against the account's usual, and the explainer says
why and what could lift the other one.

**Judges.** `glimmer:<run>` is the fine-tuned Muse Glimmer on Modal
(`preflight.finetune.Judge`, deployed first). It reads the pair both ways
round and averages the two, as in its test. Its own probability is not shown:
trained to near-zero loss, it is all but certain of nearly every pair (172 of
207 held-out pairs at 99% or more for `pairs-luna-3ep`, right on 69% of
them). The page says instead how often calls like this one were right on the
held-out pairs, and when the two readings contradict each other, that it is
too close to call. Any other judge is a model asked zero-shot, the way the
pair test asks it (`preflight.pairs.PROMPT`).

**Stand-ins.** Until the Meta credit is in, the defaults run on the user's own
subscriptions. The real thing is `--describer meta:muse-spark-1.3 --judge
glimmer:<run> --explainer meta:muse-spark-1.3`, with the describer the
judge's adapter was trained on.

**Cached.** An opening is kept by its file's hash (`data/demo/<hash>/`),
with each describer's form, so showing the same videos again costs nothing.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from preflight.describe import SECONDS, as_text, ask, log, opening, write_down
from preflight.pairs import PROMPT, VERDICT, verdict_path
from preflight.store import DEFAULT_PATH
from preflight.videos import shrink

DEMO = DEFAULT_PATH.parent / "demo"
PAGE = Path(__file__).with_name("demo.html")
SIDES = ("a", "b")
#: A fine-tuned judge's chance this near a half means its two readings of the
#: pair contradict each other (each all but certain, the other way round). On
#: the held-out pairs these were coin tosses: 11 of 22 right for `pairs-luna-3ep`.
CLOSE = 0.1

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
        return {"better": "A" if chance > 0.5 else "B", "close": abs(chance - 0.5) < CLOSE}
    answer, _ = ask(judge, PROMPT.format(captions="", frames="", a=texts["a"], b=texts["b"]), VERDICT)
    return {"better": answer["better"], "close": False}


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
        verdict = judge(config["judge"], forms)
        update(state, stage="explaining", verdict=verdict)
        update(state, stage="done", verdict=verdict | explain(config["explainer"], forms, verdict))
    except Exception as error:  # shown on the page, and the server carries on
        log(f"{job}: {type(error).__name__}: {error}")
        update(state, stage="failed", error=f"{type(error).__name__}: {error}")
    finally:
        for upload in uploads.values():
            upload.unlink(missing_ok=True)


def track_record(judge: str, describer: str) -> dict | None:
    """How the judge did on the held-out pairs, for the page to quote: in all,
    and for a fine-tuned judge, apart on its firm and its close calls."""
    if judge.startswith("glimmer:"):
        path = DEMO.parent / "finetune" / f"{judge.split(':', 1)[1]}.json"
        if not path.exists():
            return None
        tuned = json.loads(path.read_text())["tuned"]
        record = {"right": tuned["pairs_right"], "pairs": tuned["pairs"]}
        if "better" not in tuned:  # scored before the answers were kept with the chances
            return record
        for kind, close in (("firm", False), ("close", True)):
            calls = [(chance > 0.5) == (better == "A") for chance, better in zip(tuned["chances"], tuned["better"])
                     if (abs(chance - 0.5) < CLOSE) == close]
            record[kind] = {"right": sum(calls), "pairs": len(calls)}
        return record
    folder = verdict_path("held-out", judge, describer, False, {"a": "", "b": ""}).parent
    records = [json.loads(path.read_text()) for path in folder.glob("*.json")]
    return {"right": sum(record["right"] for record in records), "pairs": len(records)} if records else None


def serve(config: dict, port: int) -> None:
    import uvicorn
    from fastapi import FastAPI, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse

    app = FastAPI(title="PreFlight")

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE.read_text()

    @app.get("/api/config")
    def settings() -> dict:
        return config | {"record": track_record(config["judge"], config["describer"])}

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

    log(f"PreFlight on http://127.0.0.1:{port} (describer {config['describer']}, judge {config['judge']}, "
        f"explainer {config['explainer']})")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--describer", default="codex:gpt-6-luna")
    parser.add_argument("--judge", default="codex:gpt-6.1-sol:medium")
    parser.add_argument("--explainer", default="codex:gpt-6.1-sol:medium")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()
    serve({"describer": args.describer, "judge": args.judge, "explainer": args.explainer}, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
