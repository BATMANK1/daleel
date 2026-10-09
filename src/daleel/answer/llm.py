"""A client for language models behind an API, built for evaluation runs.

Every reply is stored on disk under the hash of its request, so a run that is
repeated, or stopped and started again, asks the API only what it has not
asked before, and every number is computed from the same saved answers. The
client keeps under each model's free-tier limits: it spaces requests to the
model's requests per minute and tokens per minute, waits as long as the API
says when it is refused all the same, retries the API's own errors, and
stops with QuotaExhaustedError when the day's quota is spent. Each request
that reaches the API is logged with its tokens, for the cost of a run at
paid prices.

The API key is read from the environment or from `.env`, and is never
written anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LLM = Path("data/interim/llm")
API = "https://generativelanguage.googleapis.com/v1beta/models"
KEY_VARIABLE = "GEMINI_API_KEY"
# Words per token is not known before a request is answered, so a request's
# tokens are estimated from its characters, generously for Arabic.
CHARS_PER_TOKEN = 2.5


@dataclass(frozen=True)
class Model:
    """A model and the free tier's limits on it, with paid prices for costing runs."""

    name: str
    requests_per_minute: int
    tokens_per_minute: int
    requests_per_day: int
    # US dollars per million tokens at paid rates, None where unpublished.
    price_in: float | None = None
    price_out: float | None = None


MODELS = {
    model.name: model
    for model in (
        Model("gemini-3.5-flash-lite", 15, 250_000, 500, 0.30, 2.50),
        Model("gemini-3.8-flash", 5, 250_000, 20, 0.75, 3.75),
        Model("gemini-3.7-flash", 5, 250_000, 20, 0.75, 3.75),
        Model("gemma-4-31b-it", 30, 16_000, 14_400),
        Model("gemma-4-26b-a4b-it", 30, 16_000, 14_400),
    )
}
GENERATOR = "gemini-3.5-flash-lite"
JUDGE = "gemma-4-31b-it"


@dataclass(frozen=True)
class Request:
    """What is asked of a model. Two equal requests get the same stored reply."""

    model: str
    prompt: str
    system: str = ""
    max_tokens: int = 2048
    # None leaves the model's default, which Gemini 3 models are tuned for.
    temperature: float | None = None
    # A label kept with the reply, such as "arm C" or "judge"; not part of the key.
    purpose: str = field(default="", compare=False)

    def key(self) -> str:
        body = {k: v for k, v in asdict(self).items() if k != "purpose"}
        return hashlib.sha256(
            json.dumps(body, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    def body(self) -> dict[str, Any]:
        config: dict[str, Any] = {"maxOutputTokens": self.max_tokens}
        if self.temperature is not None:
            config["temperature"] = self.temperature
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": self.prompt}]}],
            "generationConfig": config,
        }
        if self.system:
            body["systemInstruction"] = {"parts": [{"text": self.system}]}
        return body

    def estimated_tokens(self) -> int:
        """The prompt's tokens, estimated before the API counts them."""
        return int((len(self.prompt) + len(self.system)) / CHARS_PER_TOKEN)


@dataclass(frozen=True)
class Reply:
    """A model's answer, with what it cost and where it came from."""

    text: str
    finish: str
    model_version: str
    tokens_in: int
    tokens_out: int
    tokens_thinking: int
    seconds: float
    cached: bool = False

    def cost(self, model: Model) -> float | None:
        """US dollars at paid rates, thinking billed as output, or None if unpriced."""
        if model.price_in is None or model.price_out is None:
            return None
        out = self.tokens_out + self.tokens_thinking
        return (self.tokens_in * model.price_in + out * model.price_out) / 1e6


class LlmError(RuntimeError):
    """The API refused a request for a reason a retry will not mend."""


class QuotaExhaustedError(LlmError):
    """The day's quota for a model is spent. Run again once it resets."""


# --- talking to the API ------------------------------------------------------------

Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, dict[str, Any]]]


def urllib_transport(
    url: str, headers: Mapping[str, str], data: bytes, timeout: float
) -> tuple[int, dict[str, Any]]:
    """POST data and return the status and the JSON body, errors included."""
    request = urllib.request.Request(url, data=data, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.loads(error.read() or b"{}")
        except json.JSONDecodeError:
            return error.code, {}


def api_key(env: Mapping[str, str] = os.environ, dotenv: Path = Path(".env")) -> str:
    """The key from the environment, or from a KEY=value line of `.env`."""
    if env.get(KEY_VARIABLE):
        return env[KEY_VARIABLE]
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            name, _, value = line.strip().partition("=")
            if name.strip().removeprefix("export ").strip() == KEY_VARIABLE and value.strip():
                return value.strip().strip("'\"")
    raise LlmError(f"no {KEY_VARIABLE} in the environment or in {dotenv}")


def parse_reply(body: Mapping[str, Any], seconds: float) -> Reply:
    """A Reply from a generateContent response, leaving out the model's thoughts."""
    usage = body.get("usageMetadata", {})
    candidates = body.get("candidates") or []
    blocked = body.get("promptFeedback", {}).get("blockReason")
    if not candidates:
        finish, text = f"BLOCKED: {blocked}" if blocked else "NO_CANDIDATE", ""
    else:
        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
        finish = candidate.get("finishReason", "")
    return Reply(
        text=text.strip(),
        finish=finish,
        model_version=body.get("modelVersion", ""),
        tokens_in=usage.get("promptTokenCount", 0),
        tokens_out=usage.get("candidatesTokenCount", 0),
        tokens_thinking=usage.get("thoughtsTokenCount", 0),
        seconds=round(seconds, 3),
    )


def retry_delay(error: Mapping[str, Any]) -> float | None:
    """The wait the API asks for, in seconds, from its RetryInfo, if it gives one."""
    for detail in error.get("details", []):
        delay = detail.get("retryDelay")
        if isinstance(delay, str) and delay.endswith("s"):
            return float(delay[:-1])
    found = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", error.get("message", ""))
    if found:
        hours, minutes, seconds = found.groups()
        return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds)
    return None


def daily_quota(error: Mapping[str, Any]) -> bool:
    """Whether a refusal is for the day's quota rather than the minute's."""
    text = json.dumps(error)
    return "PerDay" in text or "per_day" in text.lower()


# --- the client ---------------------------------------------------------------------


class Pacer:
    """Spaces requests to one model's limits per minute, in requests and tokens."""

    def __init__(self, model: Model, clock: Callable[[], float], sleep: Callable[[float], None]):
        self.model = model
        self.clock = clock
        self.sleep = sleep
        self.sent: deque[tuple[float, int]] = deque()

    def wait(self, tokens: int) -> None:
        while True:
            now = self.clock()
            while self.sent and now - self.sent[0][0] >= 60:
                self.sent.popleft()
            spent = sum(count for _, count in self.sent)
            requests_left = len(self.sent) < self.model.requests_per_minute
            tokens_left = spent + tokens <= self.model.tokens_per_minute or not self.sent
            if requests_left and tokens_left:
                self.sent.append((now, tokens))
                return
            self.sleep(60 - (now - self.sent[0][0]) + 0.5)


class Client:
    """Asks models, with stored replies, pacing, retries and a log of tokens."""

    def __init__(
        self,
        root: Path = LLM,
        key: str | None = None,
        transport: Transport = urllib_transport,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = 300,
        retries: int = 5,
    ):
        self.root = root
        self._key = key
        self.transport = transport
        self.clock = clock
        self.sleep = sleep
        self.timeout = timeout
        self.retries = retries
        self.pacers: dict[str, Pacer] = {}
        self.asked = 0

    def stored(self, request: Request) -> Path:
        key = request.key()
        return self.root / "replies" / request.model / key[:2] / f"{key}.json"

    def ask(self, request: Request) -> Reply:
        """The model's reply, from the store if this request was asked before."""
        path = self.stored(request)
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            return Reply(**{**saved["reply"], "cached": True})
        reply = self._call(request)
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = {"request": asdict(request), "reply": asdict(reply)}
        path.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
        self._log(request, reply)
        return reply

    def _call(self, request: Request) -> Reply:
        model = MODELS.get(request.model)
        if model is None:
            raise LlmError(f"unknown model {request.model}; add it to MODELS with its limits")
        if self._key is None:
            self._key = api_key()
        pacer = self.pacers.setdefault(model.name, Pacer(model, self.clock, self.sleep))
        url = f"{API}/{model.name}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._key}
        data = json.dumps(request.body(), ensure_ascii=False).encode()
        for attempt in range(self.retries + 1):
            pacer.wait(request.estimated_tokens())
            started = self.clock()
            status, body = self.transport(url, headers, data, self.timeout)
            seconds = self.clock() - started
            if status == 200:
                self.asked += 1
                return parse_reply(body, seconds)
            error = body.get("error", {})
            if status == 429 and daily_quota(error):
                raise QuotaExhaustedError(f"{model.name}: the day's quota is spent")
            if attempt == self.retries or status not in (429, 500, 502, 503, 504):
                raise LlmError(f"{model.name}: HTTP {status}: {error.get('message', '')[:300]}")
            delay = retry_delay(error) if status == 429 else None
            self.sleep(delay + 1 if delay is not None else min(60, 2 ** (attempt + 1)))
        raise AssertionError("unreachable")

    def _log(self, request: Request, reply: Reply) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        model = MODELS[request.model]
        line = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": request.model,
            "version": reply.model_version,
            "purpose": request.purpose,
            "request": request.key()[:16],
            "tokens_in": reply.tokens_in,
            "tokens_out": reply.tokens_out,
            "tokens_thinking": reply.tokens_thinking,
            "seconds": reply.seconds,
            "usd_at_paid_rates": reply.cost(model),
            "finish": reply.finish,
        }
        with (self.root / "ledger.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(line) + "\n")
