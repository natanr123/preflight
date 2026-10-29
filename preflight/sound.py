"""What each Reel's opening sounds like, measured: the sound's counterpart of
the first frame, for judges that cannot hear.

    python -m preflight.sound [--workers 2] [--limit N]

**Why.** The describer and the judges only read, and what they read of the
sound is the words (Whisper) and how loud each second is. A third of the
openings have sound but no speech, mostly music, and of those a judge learns
only "no speech, loud". Glimmer cannot hear; Muse Spark can, but only on the
Meta credit. So open models listen here, on the CPU, the way CLIP looked at
the first frame:

- **What it is:** AudioSet's labels, from AST: music, speech, singing,
  genres, moods, laughter, waves, ...
- **How it sounds:** the tempo and how steady the beat is, how busy (sound
  events a second), how bright (the spectral centroid), when the sound
  starts, and where someone talks, the voice's pitch, how far it moves and
  how fast the words come (Whisper's word times, from `preflight.describe`).
- **A vector:** CLAP's embedding, the audio counterpart of CLIP, so a probe
  can ask whether the sound alone tells two Reels apart.

Each prepared opening gets `data/sound/<id>.json` (labels and measures) and
`<id>.npz` (the vectors); `sound_line` writes the measures as one line for a
judge to read.

**Environment.** torch, transformers and librosa are not in the project's
environment, so this runs in one that has them as well as the project's own
packages, with the repo on PYTHONPATH. `sound_line` needs none of them.
"""
import argparse
import json
import math
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from preflight.describe import OPENINGS, SECONDS, log
from preflight.store import DEFAULT_PATH
from preflight.videos import video_path

SOUND = DEFAULT_PATH.parent / "sound"
TAGGER = "MIT/ast-finetuned-audioset-10-10-0.4593"
EMBEDDER = "laion/larger_clap_music_and_speech"
RATE, CLAP_RATE, HOP = 16_000, 48_000, 512
#: Raised whenever what is measured changes, so older openings are measured again.
MEASURED = 1
#: torch threads per worker: the laptop is shared (see `preflight.baseline.THREADS`).
THREADS = 4
#: A stretch at or above this is sound rather than near silence (`preflight.describe.loudness_word`'s "quiet").
AUDIBLE = -35
#: Labels are kept from this probability up.
KEPT = 0.05
#: AudioSet labels said without whose voice it is.
FOLD = {"Male speech, man speaking": "Speech", "Female speech, woman speaking": "Speech",
        "Child speech, kid speaking": "Speech", "Male singing": "Singing", "Female singing": "Singing",
        "Child singing": "Singing"}

_models = {}


def _load() -> None:
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification, ClapModel, ClapProcessor

    torch.set_num_threads(THREADS)
    _models["tag"] = (ASTFeatureExtractor.from_pretrained(TAGGER),
                      ASTForAudioClassification.from_pretrained(TAGGER).eval())
    _models["embed"] = ClapProcessor.from_pretrained(EMBEDDER), ClapModel.from_pretrained(EMBEDDER).eval()


def decode(media_id: str, rate: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-t", str(SECONDS), "-i", str(video_path(media_id)),
                          "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"],
                         capture_output=True, timeout=120).stdout
    return np.frombuffer(raw, np.float32)


def labels(wave: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    import torch

    extractor, model = _models["tag"]
    with torch.inference_mode():
        chances = torch.sigmoid(model(**extractor(wave, sampling_rate=RATE, return_tensors="pt")).logits[0]).numpy()
    kept = {}
    for index in np.flatnonzero(chances >= KEPT):
        name = FOLD.get(model.config.id2label[int(index)], model.config.id2label[int(index)])
        kept[name] = max(kept.get(name, 0.0), round(float(chances[index]), 3))
    return dict(sorted(kept.items(), key=lambda item: -item[1])), chances


def embedding(wave: np.ndarray) -> np.ndarray:
    import torch

    processor, model = _models["embed"]
    with torch.inference_mode():
        vector = model.get_audio_features(**processor(audio=wave, sampling_rate=CLAP_RATE, return_tensors="pt"))
    return getattr(vector, "pooler_output", vector)[0].numpy()


def measure(wave: np.ndarray, words: list[dict]) -> dict:
    """How the opening sounds, from its samples at RATE and Whisper's word times."""
    import librosa

    seconds = len(wave) / RATE
    tenth = RATE // 10
    levels = np.array([20 * math.log10(float(np.sqrt(np.mean(wave[start:start + tenth] ** 2))) + 1e-9)
                       for start in range(0, len(wave), tenth)])
    audible = np.flatnonzero(levels >= AUDIBLE)
    onset = librosa.onset.onset_strength(y=wave, sr=RATE, hop_length=HOP)
    # How steady the beat is: the onset envelope's autocorrelation at its best lag for 60-200 BPM, over lag 0.
    correlation = librosa.autocorrelate(onset - onset.mean())
    frames_per_second = RATE / HOP
    lags = correlation[int(frames_per_second * 60 / 200):int(frames_per_second) + 1]
    loud_frames = librosa.amplitude_to_db(librosa.feature.rms(y=wave, hop_length=HOP)[0]) >= AUDIBLE
    centroid = librosa.feature.spectral_centroid(y=wave, sr=RATE, hop_length=HOP)[0]
    found = {
        "seconds": round(seconds, 2),
        "sound_from": round(audible[0] / 10, 1) if len(audible) else None,
        "audible_share": round(len(audible) / len(levels), 2),
        "loudest": round(float(levels.max()), 1),
        "tempo": round(float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset, sr=RATE, hop_length=HOP))[0])),
        "beat": round(float(lags.max() / correlation[0]), 2) if correlation[0] > 0 and len(lags) else 0.0,
        "events_per_second": round(len(librosa.onset.onset_detect(onset_envelope=onset, sr=RATE,
                                                                  hop_length=HOP)) / seconds, 1),
        "brightness_hz": round(float(np.median(centroid[loud_frames]))) if loud_frames.any() else None,
    }
    if words:
        start, end = words[0]["start"], min(seconds, words[-1]["start"] + 0.5)
        found["words_per_second"] = round(len(words) / max(end - start, 1.0), 1)
        span = wave[int(start * RATE):int(end * RATE)]
        pitch, voiced = (librosa.pyin(span, fmin=65, fmax=600, sr=RATE, frame_length=1024)[:2]
                         if len(span) >= RATE // 4 else (np.array([]), np.array([], bool)))
        pitch = pitch[voiced & ~np.isnan(pitch)]
        if len(pitch) >= 10:
            semitones = 12 * np.log2(pitch)
            found["pitch_hz"] = round(float(np.median(pitch)))
            # From the 10th to the 90th percentile: octave slips at the ends do not count.
            found["pitch_range"] = round(float(np.percentile(semitones, 90) - np.percentile(semitones, 10)), 1)
    return found


def hear(media_id: str) -> str:
    import soxr

    wave = decode(media_id, CLAP_RATE)
    record = {"measured": MEASURED, "sound_track": bool(len(wave))}
    if len(wave):
        heard = json.loads((OPENINGS / f"{media_id}.json").read_text())
        low = soxr.resample(wave, CLAP_RATE, RATE)
        record["labels"], chances = labels(low)
        record |= measure(low, heard.get("words", []))
        np.savez(SOUND / f"{media_id}.npz", clap=embedding(wave).astype(np.float32),
                 audioset=chances.astype(np.float16))
    partial = SOUND / f"{media_id}.part"
    partial.write_text(json.dumps(record))
    partial.rename(SOUND / f"{media_id}.json")  # its existence means the vectors are complete too
    return "heard" if len(wave) else "no sound track"


def attempt(media_id: str) -> str:
    try:
        return hear(media_id)
    except Exception as error:  # one unreadable video should not stop the rest
        log(f"{media_id}: {type(error).__name__}: {error}")
        return "failed"


def load_sound(media_id: str) -> dict | None:
    path = SOUND / f"{media_id}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return record if record.get("measured") == MEASURED else None


def sound_line(record: dict) -> str:
    """The measures as one line for a judge. The words' cut-offs are about the
    10th and 90th percentiles of the first openings heard (2026-10-30). The
    voice's own pitch is left out: it mostly says whether a man or a woman talks."""
    if not record["sound_track"]:
        return "Sound, as audio models measured it: no sound track."
    labels = record["labels"]
    said = [f"{name.lower()} {labels.get(name, 0.0):.2f}" for name in ("Music", "Speech")]
    others = [f"{name} {chance:.2f}" for name, chance in labels.items()
              if chance >= 0.1 and name not in ("Music", "Speech", "Musical instrument", "Television", "Radio")][:4]
    if others:
        said.append("also " + ", ".join(others))
    if record["beat"] >= 0.35:
        said.append("a steady beat" + (f" at {record['tempo']} BPM" if labels.get("Music", 0.0) >= 0.3 else ""))
    elif record["beat"] >= 0.2:
        said.append("a loose beat")
    else:
        said.append("no steady beat")
    events = record["events_per_second"]
    said.append(f"{'sparse, ' if events < 4 else 'busy, ' if events > 7.5 else ''}{events} sound events a second")
    if record["brightness_hz"] is not None and not 1000 <= record["brightness_hz"] <= 1900:
        said.append("dark, bass-heavy" if record["brightness_hz"] < 1000 else "bright, treble-heavy")
    start = record["sound_from"]
    said.append("silent throughout" if start is None else f"silent until {start} s" if start >= 0.5
                else "sound from the first moment")
    if "words_per_second" in record:
        pace = record["words_per_second"]
        voice = f"{'slow, ' if pace < 2.3 else 'fast, ' if pace > 4 else ''}{pace} words a second"
        if "pitch_range" in record:
            moves = record["pitch_range"]
            voice += f", pitch moving {moves} semitones"
            voice += " (flat)" if moves < 4.5 else " (lively)" if moves > 10 else ""
        said.append("voice: " + voice)
    return "Sound, as audio models measured it: " + "; ".join(said) + "."


def pending(limit: int | None) -> list[str]:
    """Prepared openings not yet heard, the held-out pairs' first so they can be probed early."""
    from preflight.pairs import PAIRS  # it reads `sound_line`

    held_out = json.loads((PAIRS / "held-out" / "pairs.json").read_text())["pairs"]
    first = {pair[side] for pair in held_out for side in ("a", "b")}
    ready = [path.stem for path in OPENINGS.glob("*.json") if video_path(path.stem).exists()]
    return sorted((media_id for media_id in ready if load_sound(media_id) is None),
                  key=lambda media_id: (media_id not in first, media_id))[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, help="only the first N openings")
    args = parser.parse_args()
    todo = pending(args.limit)
    SOUND.mkdir(parents=True, exist_ok=True)
    log(f"{len(todo)} openings to hear, {args.workers} workers")
    started, counts = time.time(), Counter()
    with ProcessPoolExecutor(args.workers, initializer=_load) as pool:
        for count, result in enumerate(pool.map(attempt, todo), 1):
            counts[result] += 1
            if count % 200 == 0 or count == len(todo):
                log(f"{count}/{len(todo)} {dict(counts)}, {(time.time() - started) / count:.2f} s each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
