"""Meta Model API, on the hackathon's $150 credit and never past it.

    uv run python -m preflight.meta arm --limit 100    # only once the credit shows on Billing
    uv run python -m preflight.meta spent

**Why a ledger.** The user will not spend real money ("i dont want to use real
money", 2026-10-29), but Meta's "spend limit" only sends an email, and anything
past the credit, or sent before the credit is applied, is charged to the card
on the account. So every call goes through `respond` or `transcribe` here:
each reserves its worst case before it is sent and settles at the published
prices from the usage Meta reports, in `data/meta_spend.json`, and a call that
could take the total past `limit_usd` is refused. Nothing is sent at all until
the ledger is armed, which is done by hand once the credit shows under "free
credits" on Billing.

**Not "contributor" mode.** It costs a tenth, but lets Meta train on what is
sent, and these are other creators' Reels.

**On Modal.** The hosted demo (`preflight.web`) takes the key from the Modal
secret `preflight-meta`, as `META_API_KEY`, and keeps a ledger of its own at
`PREFLIGHT_LEDGER`, with its own limit, apart from this laptop's.
"""
import argparse
import base64
import json
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import httpx

from preflight.store import DEFAULT_PATH

API = "https://api.meta.ai/v1"
KEY_FILE = Path(__file__).resolve().parents[1] / "secrets" / "meta.env"
LEDGER = Path(os.environ.get("PREFLIGHT_LEDGER") or DEFAULT_PATH.parent / "meta_spend.json")
#: USD per million tokens, standard mode (dev.meta.ai/docs/pricing-rate-limits, 2026-10-29).
PRICES = {"muse-spark-1.3": {"input": 1.25, "cached": 0.15, "output": 4.25}}
TRANSCRIBE_PER_HOUR = 0.18
#: Caps each answer (reasoning included, which is billed as output), so a
#: call's worst case is known before it is sent.
MAX_OUTPUT = 4000
#: Asks for one answer: now and then the reasoning runs to `MAX_OUTPUT` and the
#: answer stops short (status "incomplete", or JSON cut off: 5 in about 4,500
#: calls by 2026-10-31), and asking again has always done it.
TRIES = 3
TIMEOUT = 300


class NotArmed(Exception):
    pass


class OverBudget(Exception):
    pass


class MetaFailed(Exception):
    pass


_LOCK = threading.Lock()


def _key() -> str:
    if key := os.environ.get("META_API_KEY", "").strip():
        return key
    for line in KEY_FILE.read_text().splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "META_API_KEY" and value.strip():
            return value.strip()
    raise NotArmed(f"no META_API_KEY in {KEY_FILE}")


def _read() -> dict:
    if not LEDGER.exists():
        raise NotArmed("the Meta ledger is not armed: run `python -m preflight.meta arm` once the credit shows "
                       "on Billing")
    return json.loads(LEDGER.read_text())


def _write(ledger: dict) -> None:
    partial = LEDGER.with_suffix(".part")
    partial.write_text(json.dumps(ledger, indent=1))
    partial.rename(LEDGER)


@contextmanager
def _budget(worst: float):
    """Hold `worst` dollars while a call is out; the caller reports the
    actual cost to the yielded function, and only that stays spent."""
    with _LOCK:
        ledger = _read()
        if ledger["spent_usd"] + ledger["reserved_usd"] + worst > ledger["limit_usd"]:
            raise OverBudget(f"${ledger['spent_usd']:.2f} spent and ${ledger['reserved_usd']:.2f} out, "
                             f"limit ${ledger['limit_usd']:.2f}")
        ledger["reserved_usd"] += worst
        _write(ledger)
    actual = [worst]  # if the call fails without usage, assume the worst

    def settle(cost: float) -> None:
        actual[0] = cost

    try:
        yield settle
    finally:
        with _LOCK:
            ledger = _read()
            ledger["reserved_usd"] = max(0.0, ledger["reserved_usd"] - worst)
            ledger["spent_usd"] += actual[0]
            ledger["calls"] += 1
            _write(ledger)


def _post(path: str, **kwargs) -> dict:
    reply = httpx.post(f"{API}/{path}", headers={"Authorization": f"Bearer {_key()}"}, timeout=TIMEOUT, **kwargs)
    if reply.status_code != 200:
        raise MetaFailed(f"HTTP {reply.status_code}: {reply.text[:400]}")
    return reply.json()


def token_cost(model: str, usage: dict) -> float:
    price = PRICES[model]
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
    return ((usage.get("input_tokens", 0) - cached) * price["input"] + cached * price["cached"]
            + usage.get("output_tokens", 0) * price["output"]) / 1e6


def respond(model: str, effort: str, prompt: str, schema: dict, image: Path | None = None) -> tuple[dict, dict]:
    """One answer in `schema` from a Muse model, through the Responses API."""
    content = [{"type": "input_text", "text": prompt}]
    if image:
        content.append({"type": "input_image",
                        "image_url": "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()})
    body = {"model": model, "input": [{"type": "message", "role": "user", "content": content}],
            "reasoning": {"effort": effort}, "max_output_tokens": MAX_OUTPUT,
            "text": {"format": {"type": "json_schema", "name": "answer", "schema": schema, "strict": True}}}
    # The worst case: every input token uncached (an image is under 3,000) and every output token used.
    worst = ((len(prompt) // 2 + (3000 if image else 0)) * PRICES[model]["input"]
             + MAX_OUTPUT * PRICES[model]["output"]) / 1e6
    for _ in range(TRIES):
        with _budget(worst) as settle:
            reply = _post("responses", json=body)
            usage = reply.get("usage") or {}
            settle(token_cost(model, usage))
        texts = [part["text"] for item in reply.get("output", []) if item.get("type") == "message"
                 for part in item.get("content", []) if part.get("type") == "output_text"]
        try:
            if texts:
                return json.loads(texts[-1]), {**usage, "cost_usd": round(token_cost(model, usage), 6)}
        except json.JSONDecodeError:
            pass
    raise MetaFailed(f"no whole answer in {TRIES} tries: status {reply.get('status')}, {json.dumps(reply)[:400]}")


def transcribe(wav: bytes, seconds: float) -> dict:
    """What is said in a mono 16-bit WAV, as turns with start and end times
    (Muse Voice Transcribe gives no word times)."""
    request = {"mode": "DIARIZATION", "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"}
    with _budget(seconds / 3600 * TRANSCRIBE_PER_HOUR) as settle:
        reply = _post("asr/transcribe", files={"request": (None, json.dumps(request), "application/json"),
                                               "audio": ("opening.wav", wav, "audio/wav")})
        settle(int(seconds) / 3600 * TRANSCRIBE_PER_HOUR)  # billed in whole seconds
    return reply


def arm(limit: float) -> dict:
    """Let the ledger spend up to `limit` dollars in all, keeping what it has spent."""
    with _LOCK:
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"spent_usd": 0.0, "reserved_usd": 0.0,
                                                                           "calls": 0}
        ledger["limit_usd"] = limit
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        _write(ledger)
    return ledger


def left() -> float:
    """Dollars the ledger can still spend; nothing until it is armed."""
    with _LOCK:
        ledger = _read() if LEDGER.exists() else {"limit_usd": 0.0, "spent_usd": 0.0, "reserved_usd": 0.0}
    return ledger["limit_usd"] - ledger["spent_usd"] - ledger["reserved_usd"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("arm").add_argument("--limit", type=float, default=100)
    commands.add_parser("spent")
    args = parser.parse_args()
    if args.command == "arm":
        arm(args.limit)
    print(json.dumps(_read(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
