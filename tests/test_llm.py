"""Tests for the language model client. A fake connection stands in for the API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daleel.answer.llm import (
    MODELS,
    Client,
    LlmError,
    Model,
    Pacer,
    QuotaExhaustedError,
    Request,
    api_key,
    parse_reply,
    retry_delay,
)

KEY = "test-key-never-stored"


def answer(text: str, *, thought: str = "", version: str = "gemini-3.5-flash-lite-001") -> dict:
    parts = [{"text": thought, "thought": True}] if thought else []
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [*parts, {"text": text}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 120,
            "candidatesTokenCount": 30,
            "thoughtsTokenCount": 15,
            "totalTokenCount": 165,
        },
        "modelVersion": version,
    }


def refusal(code: int, message: str, details: list | None = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details or []}}


class FakeApi:
    """Answers with the next of its responses, and keeps what it was sent."""

    def __init__(self, *responses: tuple[int, dict]):
        self.responses = list(responses)
        self.sent: list[dict[str, Any]] = []

    def __call__(self, url: str, headers, data: bytes, timeout: float) -> tuple[int, dict]:
        self.sent.append({"url": url, "headers": dict(headers), "body": json.loads(data)})
        return self.responses.pop(0)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def client(tmp_path: Path, api: FakeApi, clock: FakeClock | None = None) -> Client:
    clock = clock or FakeClock()
    return Client(tmp_path / "llm", key=KEY, transport=api, clock=clock, sleep=clock.sleep)


QUESTION = Request("gemini-3.5-flash-lite", "ما عاصمة السعودية؟", system="أجب بإيجاز.")


def test_a_reply_leaves_out_the_model_s_thoughts_and_counts_its_tokens() -> None:
    reply = parse_reply(answer("الرياض", thought="The capital is..."), seconds=1.25)
    assert reply.text == "الرياض"
    assert (reply.tokens_in, reply.tokens_out, reply.tokens_thinking) == (120, 30, 15)
    assert reply.finish == "STOP" and reply.model_version == "gemini-3.5-flash-lite-001"
    # Thinking is billed as output: 120 * 0.30 + 45 * 2.50 per million tokens.
    assert reply.cost(MODELS["gemini-3.5-flash-lite"]) == pytest.approx(148.5e-6)
    assert reply.cost(MODELS["gemma-4-31b-it"]) is None


def test_a_blocked_prompt_gives_an_empty_reply_that_says_why() -> None:
    reply = parse_reply({"promptFeedback": {"blockReason": "SAFETY"}}, seconds=0.1)
    assert reply.text == "" and reply.finish == "BLOCKED: SAFETY"


def test_the_request_carries_the_system_text_and_leaves_the_default_temperature() -> None:
    body = QUESTION.body()
    assert body["systemInstruction"] == {"parts": [{"text": "أجب بإيجاز."}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "ما عاصمة السعودية؟"}]}]
    assert "temperature" not in body["generationConfig"]
    assert Request("m", "q", temperature=0.5).body()["generationConfig"]["temperature"] == 0.5


def test_a_request_is_stored_under_what_was_asked_not_why() -> None:
    assert QUESTION.key() == Request(**{**QUESTION.__dict__, "purpose": "judge"}).key()
    assert QUESTION.key() != Request(QUESTION.model, QUESTION.prompt).key()


def test_a_question_asked_twice_reaches_the_api_once(tmp_path: Path) -> None:
    api = FakeApi((200, answer("الرياض")))
    first = client(tmp_path, api).ask(QUESTION)
    again = client(tmp_path, api).ask(QUESTION)
    assert len(api.sent) == 1
    assert first.text == again.text == "الرياض"
    assert (first.cached, again.cached) == (False, True)
    assert api.sent[0]["url"].endswith("/gemini-3.5-flash-lite:generateContent")
    assert api.sent[0]["headers"]["x-goog-api-key"] == KEY


def test_the_key_is_never_written_to_disk(tmp_path: Path) -> None:
    client(tmp_path, FakeApi((200, answer("الرياض")))).ask(QUESTION)
    written = [path.read_text(encoding="utf-8") for path in (tmp_path / "llm").rglob("*.json*")]
    assert written and not any(KEY in text for text in written)


def test_each_request_that_reaches_the_api_is_logged_with_its_tokens(tmp_path: Path) -> None:
    api = FakeApi((200, answer("الرياض")))
    client(tmp_path, api).ask(Request(**{**QUESTION.__dict__, "purpose": "arm C"}))
    client(tmp_path, api).ask(QUESTION)
    lines = (tmp_path / "llm" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["purpose"] == "arm C" and logged["tokens_in"] == 120
    assert logged["usd_at_paid_rates"] == pytest.approx(148.5e-6)


def test_a_refusal_for_the_minute_waits_as_long_as_the_api_asks(tmp_path: Path) -> None:
    wait = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"}
    api = FakeApi((429, refusal(429, "Resource exhausted", [wait])), (200, answer("الرياض")))
    clock = FakeClock()
    assert client(tmp_path, api, clock).ask(QUESTION).text == "الرياض"
    assert 35 in clock.slept and len(api.sent) == 2


def test_a_refusal_for_the_day_stops_the_run(tmp_path: Path) -> None:
    quota = {
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
    }
    api = FakeApi((429, refusal(429, "You exceeded your current quota", [quota])))
    with pytest.raises(QuotaExhaustedError):
        client(tmp_path, api).ask(QUESTION)
    assert not list((tmp_path / "llm").rglob("*.json"))


def test_the_api_s_own_errors_are_retried_and_others_are_not(tmp_path: Path) -> None:
    api = FakeApi((500, refusal(500, "Internal error encountered.")), (200, answer("الرياض")))
    assert client(tmp_path, api).ask(QUESTION).text == "الرياض"
    with pytest.raises(LlmError, match="HTTP 400"):
        client(tmp_path, FakeApi((400, refusal(400, "Invalid argument")))).ask(
            Request(QUESTION.model, "سؤال آخر")
        )


def test_a_model_without_known_limits_is_refused_before_any_request(tmp_path: Path) -> None:
    api = FakeApi()
    with pytest.raises(LlmError, match="unknown model"):
        client(tmp_path, api).ask(Request("gemini-9-ultra", "q"))
    assert not api.sent


def test_requests_are_spaced_to_the_model_s_limit_per_minute() -> None:
    clock = FakeClock()
    pacer = Pacer(Model("m", 2, 1_000, 100), clock, clock.sleep)
    for _ in range(3):
        pacer.wait(10)
    assert clock.slept and clock.now >= 60
    clock = FakeClock()
    pacer = Pacer(Model("m", 100, 1_000, 100), clock, clock.sleep)
    pacer.wait(800)
    pacer.wait(800)
    assert clock.now >= 60


def test_the_wait_is_read_from_the_message_when_the_details_lack_it() -> None:
    assert retry_delay({"message": "Please retry in 9h34m50.5s."}) == 9 * 3600 + 34 * 60 + 50.5
    assert retry_delay({"details": [{"retryDelay": "12.5s"}]}) == 12.5
    assert retry_delay({"message": "no hint"}) is None


def test_the_key_comes_from_the_environment_or_from_dotenv(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("# comment\nOTHER=1\nexport GEMINI_API_KEY='abc123'\n", encoding="utf-8")
    assert api_key({}, dotenv) == "abc123"
    assert api_key({"GEMINI_API_KEY": "from-env"}, dotenv) == "from-env"
    with pytest.raises(LlmError, match="no GEMINI_API_KEY"):
        api_key({}, tmp_path / "missing")


def test_the_real_connection_returns_the_api_s_errors_as_json() -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from daleel.answer.llm import urllib_transport

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            sent = json.loads(self.rfile.read(length))
            status = 200 if sent.get("ok") else 429
            body = json.dumps(answer("الرياض") if status == 200 else refusal(429, "slow down"))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/"
    try:
        status, body = urllib_transport(url, {}, json.dumps({"ok": True}).encode(), 5)
        assert status == 200 and parse_reply(body, 0).text == "الرياض"
        status, body = urllib_transport(url, {}, b"{}", 5)
        assert status == 429 and body["error"]["message"] == "slow down"
    finally:
        server.shutdown()
