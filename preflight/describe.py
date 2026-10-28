"""Write down what each Reel's opening shows and says, as text a model can learn from.

    uv run python -m preflight.describe pilot [--n 12] [--with codex:gpt-6-luna claude:sonnet]
    uv run python -m preflight.describe run [--with codex:gpt-6-luna[:effort]] [--workers 3] [--hours 10]

**What the describer gets.** The Reel's first five seconds as ten frames, one
every half second, on one contact sheet stamped with their times
(`data/openings/<id>.jpg`), and what is heard in them. Neither Codex nor
Claude can hear, so the sound is worked out here and handed over as text: the
words (faster-whisper, on the CPU), how loud each second is, and where the
shots change (ffmpeg's scene score). The caption is left out on purpose: it is
an input of its own, and the free test compares the two.

**What it writes.** The same form every time (`FORM`), second by second, so
openings of one, three and five seconds can be compared from one description:
`data/descriptions/<describer>/<id>.json`.

**Who describes.** A CLI agent on the user's own subscriptions, `codex exec`
or `claude -p`, so a description costs allowance rather than money; Codex is
the cheaper of the two (the user, 2026-10-28). Muse Spark can take over once
the Meta API key is in, and it hears for itself.
"""
import argparse
import base64
import functools
import json
import math
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from preflight.dataset import age_slope, label
from preflight.store import DEFAULT_PATH
from preflight.videos import labelled_reels, video_path

OPENINGS = DEFAULT_PATH.parent / "openings"
DESCRIPTIONS = DEFAULT_PATH.parent / "descriptions"
SECONDS, FRAMES_PER_SECOND, COLUMNS = 5, 2, 5
#: A band above each frame for its time, so the stamp hides nothing on screen.
STAMP = 40
#: Speech to text on the CPU: "small" is multilingual and takes about a second
#: an opening (measured). The laptop is shared; see `preflight.baseline.THREADS`.
WHISPER, THREADS, RATE = "small", 4, 16_000
#: Whisper's own threshold for a segment that is probably not speech.
NO_SPEECH = 0.6
#: ffmpeg's scene score (0 to 1) above which a frame starts a new shot.
CUT = 0.3
#: Raised whenever FORM or PROMPT changes, so older descriptions can be told apart.
VERSION = 3
TIMEOUT = 300

FORM = {
    "type": "object",
    "additionalProperties": False,
    "required": ["first_frame", "seconds", "people", "camera", "setting", "kind", "production", "sound", "hook"],
    "properties": {
        "first_frame": {"type": "string", "description": "What is on screen at 0.0 s, as it shows in the feed."},
        "seconds": {
            "type": "array",
            "description": "One entry per second, in order.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["second", "seen", "text_on_screen", "heard"],
                "properties": {
                    "second": {"type": "string", "description": '"0-1 s", "1-2 s", ...'},
                    "seen": {"type": "string", "description": "What happens on screen in that second."},
                    "text_on_screen": {"type": "string", "description": 'Verbatim, or "none".'},
                    "heard": {"type": "string",
                              "description": "The words said in that second, verbatim from the transcript; "
                                             "otherwise the sound, from the loudness."},
                },
            },
        },
        "people": {"type": "string",
                   "description": "Who is on screen, without names: whether a face shows and looks at the camera."},
        "camera": {"type": "string", "description": "Framing, camera and subject movement, and the pace of the cuts."},
        "setting": {"type": "string"},
        # Not "format": that is a JSON Schema keyword, and the pilot's Codex answered "json" to it.
        "kind": {"type": "string",
                 "description": "What kind of video: talking to camera, tutorial, skit, vlog, montage, product "
                                "demo, animation, screen recording, ..."},
        "production": {"type": "string", "description": "Phone-shot, studio, animated, edited stock, ... and how "
                                                        "polished."},
        "sound": {"type": "string", "description": "Speech, other sound or near silence across the opening."},
        "hook": {"type": "string", "description": "What the opening makes a viewer expect or want to know, in one "
                                                  "sentence."},
    },
}

PROMPT = """\
Fill in a form about the first {seconds} seconds of an Instagram Reel, for a dataset. Describe only what is there.

The attached image is a contact sheet of {frames} frames from those seconds, one every half second, left to right
and then top to bottom. The black band above each frame, with its time in yellow, was added for you: it is not part
of the Reel, so leave it out of the form.

You cannot hear the Reel, so software listened and measured for you:
{measured}

Rules:
- Plain English. On-screen text and speech stay verbatim, in their own language.
- Do not name anyone or guess who they are. Do not judge quality or guess how popular the Reel is.
- Write "none" where there is nothing to report.
- Short and specific: what a viewer would notice.
- `seconds` has one entry per second, "0-1 s" up to "{last}".

Answer with the form only.
"""

#: What the CLIs say when the subscription's allowance runs out: the run stops
#: there instead of failing every Reel after it.
LIMIT = re.compile(r"usage limit|limit reached|rate.?limit|too many requests|\b429\b|quota", re.IGNORECASE)


class DescriberFailed(Exception):
    pass


class OutOfAllowance(Exception):
    pass


def log(message: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {message}", flush=True)


def description_path(describer: str, media_id: str) -> Path:
    return DESCRIPTIONS / describer.replace(":", "-") / f"{media_id}.json"


# The opening ----------------------------------------------------------------------------------------------------------
def contact_sheet(frames: list[Path], out: Path) -> None:
    tiles = [Image.open(frame).convert("RGB") for frame in frames]
    width, height = tiles[0].size
    sheet = Image.new("RGB", (width * min(COLUMNS, len(tiles)), (height + STAMP) * math.ceil(len(tiles) / COLUMNS)),
                      "black")
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=STAMP - 10)
    for index, tile in enumerate(tiles):
        x, y = (index % COLUMNS) * width, (index // COLUMNS) * (height + STAMP)
        draw.text((x + 8, y + 4), f"{index / FRAMES_PER_SECOND:.1f} s", fill="yellow", font=font)
        sheet.paste(tile.resize((width, height)), (x, y + STAMP))
    sheet.save(out, quality=85)


@functools.cache
def speech_model():
    from faster_whisper import WhisperModel  # slow to import; only `prepare` needs it

    return WhisperModel(WHISPER, device="cpu", compute_type="int8", cpu_threads=THREADS)


_LISTENING = threading.Lock()


def listen(video: Path) -> dict:
    """The words, with when each starts, and each second's loudness in dBFS.
    Decoded by ffmpeg: faster-whisper's own decoder breaks on PyAV 19."""
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-t", str(SECONDS), "-i", str(video), "-vn",
                          "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"],
                         capture_output=True, timeout=120).stdout
    audio = np.frombuffer(raw, np.float32)
    if not len(audio):
        return {"sound_track": False, "language": None, "words": [], "loudness": []}
    loudness = [round(20 * math.log10(float(np.sqrt(np.mean(audio[start:start + RATE] ** 2))) + 1e-9), 1)
                for start in range(0, len(audio), RATE)]
    with _LISTENING:  # one model, one transcription at a time
        segments, info = speech_model().transcribe(audio, word_timestamps=True, vad_filter=True)
        # Whisper hears words in music: a lone "You" at no-speech 0.8 (measured).
        words = [{"start": round(word.start, 2), "word": word.word.strip()}
                 for segment in segments if segment.no_speech_prob <= NO_SPEECH for word in segment.words]
    return {"sound_track": True, "language": info.language if words else None, "words": words, "loudness": loudness}


def cuts(video: Path) -> list[float]:
    found = subprocess.run(["ffmpeg", "-nostdin", "-t", str(SECONDS), "-i", str(video), "-an",
                            "-vf", f"select='gt(scene,{CUT})',showinfo", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=120).stderr
    return [round(float(stamp), 1) for stamp in re.findall(r"pts_time:([\d.]+)", found)]


def prepare(media_id: str) -> dict:
    """The opening's contact sheet and measurements, made once and kept."""
    facts_path = OPENINGS / f"{media_id}.json"
    if facts_path.exists():
        return json.loads(facts_path.read_text())
    OPENINGS.mkdir(parents=True, exist_ok=True)
    video = video_path(media_id)
    with tempfile.TemporaryDirectory(prefix="preflight-frames-") as work:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-t", str(SECONDS), "-i", str(video),
                        "-vf", f"fps={FRAMES_PER_SECOND}", "-q:v", "2", f"{work}/%02d.jpg"],
                       check=True, capture_output=True, timeout=120)
        frames = sorted(Path(work).glob("*.jpg"))
        contact_sheet(frames, OPENINGS / f"{media_id}.jpg")
    facts = {"frames": len(frames), **listen(video), "cuts": cuts(video)}
    partial = facts_path.with_suffix(".part")
    partial.write_text(json.dumps(facts, ensure_ascii=False))
    partial.rename(facts_path)  # its existence means the sheet is complete too
    return facts


def loudness_word(dbfs: float) -> str:
    return "silent" if dbfs < -50 else "quiet" if dbfs < -35 else "medium" if dbfs < -20 else "loud"


def measured(facts: dict) -> str:
    lines = []
    if not facts["sound_track"]:
        lines.append("- The Reel has no sound track.")
    else:
        if facts["words"]:
            lines.append(f"- Language of the speech: {facts['language']}")
        for second, level in enumerate(facts["loudness"]):
            said = " ".join(word["word"] for word in facts["words"] if second <= word["start"] < second + 1)
            lines.append(f"- {second}-{second + 1} s: " + (f'speech "{said}"' if said else "no speech")
                         + f", {loudness_word(level)}")
    shots = ", ".join(f"{stamp} s" for stamp in facts["cuts"]) or "none"
    lines.append(f"- Shot changes at: {shots}")
    return "\n".join(lines)


# The describers ---------------------------------------------------------------------------------------------------------
def events(output: str) -> list[dict]:
    found = []
    for line in output.splitlines():
        try:
            found.append(json.loads(line))
        except ValueError:
            continue
    return [event for event in found if isinstance(event, dict)]


def failure(output: str) -> Exception:
    said = output.strip()[-400:]
    return OutOfAllowance(said) if LIMIT.search(said) else DescriberFailed(said)


def codex(model: str, effort: str, prompt: str, sheet: Path) -> tuple[dict, dict]:
    with tempfile.TemporaryDirectory(prefix="preflight-codex-") as work:
        schema, answer = Path(work) / "form.schema.json", Path(work) / "form.json"
        schema.write_text(json.dumps(FORM))
        # Without the user's config: it would start their MCP servers on every call.
        argv = ["codex", "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
                "--sandbox", "read-only", "-C", work, "-m", model, "-c", f'model_reasoning_effort="{effort}"',
                "--output-schema", str(schema), "-o", str(answer), "--json", "--color", "never", "-i", str(sheet)]
        done = subprocess.run(argv, input=prompt, capture_output=True, text=True, timeout=TIMEOUT)
        usage = next((event["usage"] for event in reversed(events(done.stdout)) if "usage" in event), {})
        if not answer.exists() or not answer.read_text().strip():
            raise failure(done.stdout + done.stderr)
        return json.loads(answer.read_text()), usage


def claude(model: str, effort: str, prompt: str, sheet: Path) -> tuple[dict, dict]:
    image = base64.b64encode(sheet.read_bytes()).decode()
    message = {"type": "user", "message": {"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image}},
        {"type": "text", "text": prompt}]}}
    # The image comes inline, so no tools are needed and one turn answers.
    argv = ["claude", "-p", "--model", model, "--effort", effort, "--restricted", "--strict-mcp-config",
            "--tools", "", "--no-session-persistence", "--input-format", "stream-json",
            "--output-format", "stream-json", "--verbose", "--json-schema", json.dumps(FORM)]
    with tempfile.TemporaryDirectory(prefix="preflight-claude-") as work:
        done = subprocess.run(argv, input=json.dumps(message) + "\n", capture_output=True, text=True,
                              timeout=TIMEOUT, cwd=work)
    result = next((event for event in reversed(events(done.stdout)) if event.get("type") == "result"), None)
    if not result or result.get("is_error"):
        raise failure(json.dumps(result) if result else done.stdout + done.stderr)
    form = result.get("structured_output") or json.loads(result["result"])
    return form, {**(result.get("usage") or {}), "cost_usd": result.get("total_cost_usd")}


DESCRIBERS = {"codex": codex, "claude": claude}


def describe(describer: str, media_id: str) -> dict:
    facts = prepare(media_id)
    seconds = math.ceil(facts["frames"] / FRAMES_PER_SECOND)
    prompt = PROMPT.format(seconds=seconds, frames=facts["frames"], measured=measured(facts),
                           last=f"{seconds - 1}-{seconds} s")
    kind, model, *effort = describer.split(":")  # codex:gpt-6-luna, or codex:gpt-6-luna:medium
    started = time.time()
    form, usage = DESCRIBERS[kind](model, effort[0] if effort else "low", prompt, OPENINGS / f"{media_id}.jpg")
    record = {"id": media_id, "describer": describer, "version": VERSION,
              "seconds": round(time.time() - started, 1), "usage": usage, "form": form}
    out = description_path(describer, media_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    return record


OVERVIEW = ("people", "camera", "setting", "kind", "production", "sound", "hook")


def as_text(form: dict, seconds: int | None = None) -> str:
    """A description as the model reads it. With `seconds`, only the first
    frame and that many seconds, without the overview of the whole opening,
    which is how openings of different lengths are compared."""
    lines = [f"First frame: {form['first_frame']}"]
    lines += [f"{second['second']}: {second['seen']} On screen: {second['text_on_screen'].rstrip('.')}. "
              f"Heard: {second['heard'].rstrip('.')}." for second in form["seconds"][:seconds]]
    if seconds is None:
        lines += [f"{name.capitalize()}: {form[name]}" for name in OVERVIEW]
    return "\n".join(lines)


def load_descriptions(describer: str) -> dict[str, dict]:
    """Each described Reel's form, by media id."""
    folder = DESCRIPTIONS / describer.replace(":", "-")
    return {path.stem: json.loads(path.read_text())["form"] for path in folder.glob("*.json")}


# Commands -------------------------------------------------------------------------------------------------------------
def labelled() -> list[dict]:
    accounts = labelled_reels()
    return label(accounts, age_slope(accounts))


def pilot(n: int, describers: list[str]) -> None:
    """A few Reels from each band, one per account, through every describer,
    side by side in `data/descriptions/pilot.md` for judging by eye."""
    rows = [row for row in labelled() if video_path(row["id"]).exists()]
    random.Random(0).shuffle(rows)
    picked, accounts = [], set()
    for band in ("outperform", "normal", "flop"):
        for row in rows:
            if (row["band"] == band and row["username"] not in accounts
                    and sum(1 for chosen in picked if chosen["band"] == band) < n // 3):
                picked.append(row)
                accounts.add(row["username"])
    for row in picked:
        prepare(row["id"])
    log(f"{len(picked)} openings ready")

    results = {}
    for describer in describers:
        def one(row, describer=describer):
            try:
                return describe(describer, row["id"])
            except (DescriberFailed, OutOfAllowance, subprocess.SubprocessError, ValueError) as error:
                log(f"{describer} {row['id']}: {type(error).__name__}: {error}")
                return None
        with ThreadPoolExecutor(3) as pool:
            results[describer] = list(pool.map(one, picked))
        done = [record for record in results[describer] if record]
        log(f"{describer}: {len(done)}/{len(picked)} described, "
            f"{sum(record['seconds'] for record in done) / max(len(done), 1):.0f} s each")

    report = [f"# Pilot: {len(picked)} openings\n"]
    for index, row in enumerate(picked):
        facts = prepare(row["id"])
        report.append(f"## {row['id']} ({row['band']}, {row['ratio']:.2f}x its account's usual)\n\n"
                      f"Sheet: `{OPENINGS / (row['id'] + '.jpg')}`\n\nMeasured:\n{measured(facts)}\n")
        for describer in describers:
            record = results[describer][index]
            report.append(f"### {describer}" + (f" ({record['seconds']} s, {json.dumps(record['usage'])})\n\n"
                                                f"{as_text(record['form'])}\n" if record else "\n\nfailed\n"))
    (DESCRIPTIONS / "pilot.md").write_text("\n".join(report))
    log(f"side by side: {DESCRIPTIONS / 'pilot.md'}")


def run(describer: str, workers: int, deadline: float, sample: int | None) -> None:
    """Every labelled Reel with a video, in one fixed shuffled order: a run cut
    short by the allowance still covers every account, and `--sample N` gives
    each describer the same first N Reels to compare on."""
    todo = sorted(reel["id"] for reels in labelled_reels().values() for reel in reels
                  if video_path(reel["id"]).exists())
    random.Random(0).shuffle(todo)
    todo = [media_id for media_id in todo[:sample] if not description_path(describer, media_id).exists()]
    log(f"{len(todo)} openings to describe with {describer}")
    stop, counts = threading.Event(), Counter()

    def one(media_id: str) -> str:
        if stop.is_set() or time.time() > deadline:
            return "left for later"
        try:
            describe(describer, media_id)
            return "described"
        except OutOfAllowance as limit:
            stop.set()
            log(f"out of allowance, stopping: {limit}")
            return "out of allowance"
        except (DescriberFailed, subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
            log(f"{media_id}: {type(error).__name__}: {error}")
            return "failed"

    with ThreadPoolExecutor(workers) as pool:
        for count, result in enumerate(pool.map(one, todo), 1):
            counts[result] += 1
            if count % 25 == 0:
                log(f"{count}/{len(todo)} {dict(counts)}")
    log(f"done: {dict(counts)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    trial = commands.add_parser("pilot")
    trial.add_argument("--n", type=int, default=12)
    trial.add_argument("--with", dest="describers", nargs="+", default=["codex:gpt-6-luna", "claude:sonnet"])
    full = commands.add_parser("run")
    full.add_argument("--with", dest="describer", default="codex:gpt-6-luna")
    full.add_argument("--workers", type=int, default=3)
    full.add_argument("--hours", type=float, default=10)
    full.add_argument("--sample", type=int, help="only the first N Reels of the fixed order")
    args = parser.parse_args()

    if args.command == "pilot":
        pilot(args.n, args.describers)
    else:
        run(args.describer, args.workers, time.time() + args.hours * 3600, args.sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())
