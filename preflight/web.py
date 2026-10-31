"""PreFlight's demo on Modal, for anyone with the link.

    uv run modal deploy -m preflight.finetune    # the fine-tuned judge, first
    uv run modal deploy -m preflight.web         # then this, which prints the link

**What runs where.** One CPU container (`Web`) serves the demo's page and API
(`preflight.demo.make_app`): ffmpeg cuts each opening, faster-whisper listens,
and Muse Spark writes it down and explains the call through the Meta Model API,
with the key from the Modal secret `preflight-meta`. The call is the fine-tuned
Glimmer's, on the GPU deployed with `preflight.finetune`. There is only ever one
container, so the jobs and the ledger live in one place, as on the laptop.

**Spending.** Strangers spend the hackathon's credit here. So the demo keeps a
Meta ledger of its own (`preflight.meta`) on the Volume `preflight-web`, armed
at `LIMIT` dollars apart from the laptop's $100, which together stay under the
$150 credit. And without the code in the Modal secret `preflight-web` (a link
with `?code=`, for the judges), at most `DAILY` comparisons start a day, and
at most `DAILY` early wake-ups of the GPU, since each keeps it up for minutes;
a comparison that finds the GPU cold starts it too. Past the comparisons'
limit the page says so; past the wake-ups', a comparison just waits longer.

**Nothing kept.** An upload is deleted once its comparison is over; its
opening (the first seconds, their frames and what was heard) `KEEP` minutes
later, so the page can still show them; the container's own disk when it stops.
"""
import hmac
import json
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("preflight-web")
volume = modal.Volume.from_name("preflight-web", create_if_missing=True)
KEPT = Path("/ledger")
USAGE = KEPT / "usage.json"
RECORD = Path(__file__).resolve().parents[1] / "data" / "demo" / "record.json"
CONFIG = {"describer": "meta:muse-spark-1.3:low", "judge": "glimmer:pairs-spark", "second": None,
          "explainer": "meta:muse-spark-1.3:medium"}
#: Dollars of the Meta credit this demo may spend in all: about 400 comparisons.
LIMIT = 20
#: Comparisons a day, and wakes of the GPU, without the code.
DAILY = 20
#: Minutes an opening is kept after it is read.
KEEP = 15
#: The most a video may weigh, in bytes; only its first seconds are read.
LARGEST = 150 << 20
NOTICE = (f"A free demo: {DAILY} comparisons a day for everyone. Videos are deleted within {KEEP} minutes; their "
          "frames and the words heard go to Meta's Muse Spark to be written down.")

if modal.is_local():  # deploying: the image is defined, and the judges' record made, here
    from preflight.demo import track_record
    from preflight.describe import WHISPER

    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(track_record(CONFIG["judge"], CONFIG["describer"], CONFIG["second"])))
else:
    WHISPER = None


def fetch_whisper(name: str) -> None:
    from faster_whisper import WhisperModel

    WhisperModel(name, device="cpu", compute_type="int8")


image = (modal.Image.debian_slim(python_version="3.13")
         .apt_install("ffmpeg")
         .pip_install("fastapi==0.162.0", "python-multipart==0.0.32", "httpx==0.28.1", "numpy==2.5.4",
                      "pillow==13.0.0", "langdetect==1.0.9", "faster-whisper==1.2.1")
         .env({"HF_HOME": "/root/hf", "PREFLIGHT_LEDGER": str(KEPT / "meta_spend.json")})
         .run_function(fetch_whisper, kwargs={"name": WHISPER})
         .add_local_file(RECORD, "/root/record.json")
         # demo.html too, which only-Python would leave out
         .add_local_python_source("preflight", ignore=["**/__pycache__/**"]))

_LOCK = threading.Lock()


def refuse(headers, path: str) -> tuple[int, str] | None:
    """Why a comparison, or a wake of the GPU for one, may not go ahead, if it
    may not. Each that goes ahead is counted against the day."""
    from preflight import meta

    if int(headers.get("content-length") or 0) > 2 * LARGEST + (1 << 20):
        return 413, f"Each video can be at most {LARGEST >> 20} MB."
    if meta.left() < 0.25:
        return 429, "The demo has spent its share of the hackathon credit. The video shows it at work."
    code = os.environ.get("PREFLIGHT_CODE", "")
    if code and hmac.compare_digest(headers.get("x-preflight-code", "").encode(), code.encode()):
        return None
    today = str(datetime.now(timezone.utc).date())
    kinds = [today] if path == "/api/compare" else [today, f"{today} warm"]  # no wake once the day's runs are done
    with _LOCK:
        usage = json.loads(USAGE.read_text()) if USAGE.exists() else {}
        if any(usage.get(kind, 0) >= DAILY for kind in kinds):
            return 429, f"Today's {DAILY} free comparisons are used up: come back tomorrow, or watch the video."
        usage[kinds[-1]] = usage.get(kinds[-1], 0) + 1
        USAGE.write_text(json.dumps(usage))
    return None


def tend() -> None:
    """Every minute: drop the openings read more than `KEEP` minutes ago, and
    keep the ledger and the day's count."""
    from preflight.demo import DEMO

    while True:
        time.sleep(60)
        for folder in DEMO.iterdir() if DEMO.exists() else []:
            if folder.is_dir() and folder.name != "uploads" and time.time() - folder.stat().st_mtime > KEEP * 60:
                shutil.rmtree(folder, ignore_errors=True)
        volume.commit()


@app.cls(image=image, secrets=[modal.Secret.from_name("preflight-meta"), modal.Secret.from_name("preflight-web")],
         volumes={str(KEPT): volume}, cpu=4, memory=4096, max_containers=1, scaledown_window=5 * 60,
         timeout=10 * 60)
@modal.concurrent(max_inputs=50)
class Web:
    @modal.enter()
    def start(self):
        from preflight import meta

        meta.arm(LIMIT)
        volume.commit()
        threading.Thread(target=tend, daemon=True).start()

    @modal.exit()
    def stop(self):
        volume.commit()

    @modal.asgi_app()
    def site(self):
        from fastapi import Request
        from fastapi.responses import JSONResponse

        from preflight.demo import make_app

        site = make_app(CONFIG | {"record": json.loads(Path("/root/record.json").read_text()), "notice": NOTICE})

        @site.middleware("http")
        async def gate(request: Request, call_next):
            if request.method == "POST" and request.url.path in ("/api/compare", "/api/warm"):
                if refusal := refuse(request.headers, request.url.path):
                    if request.url.path == "/api/warm":
                        return JSONResponse({})  # nothing to tell: the comparison itself will say
                    status, detail = refusal
                    return JSONResponse({"detail": detail}, status_code=status)
            return await call_next(request)

        return site
