"""Tests for the dots.mocr engine.

No model runs for these tests, CI included: a stand-in answers the engine the
way vLLM 0.30 does when it serves dots.mocr, and a small local HTTP server
checks the client that talks to the real one.
"""

from __future__ import annotations

import base64
import io
import json
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from daleel.ocr.dots import (
    DEFAULT,
    IMAGE_TOKENS,
    LAYOUT_PROMPT,
    DotsEngine,
    DotsError,
    Settings,
    chat_request,
    error_message,
    fetch_json,
    hub_cache,
    layout_text,
    markdown_text,
    revision,
    table_text,
)

SERVER = "http://localhost:8000"
MODEL = "rednote-hilab/dots.mocr"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
CARD = {"id": MODEL, "object": "model", "root": MODEL, "max_model_len": 16384}
HUB_VARIABLES = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME")


def png_of(mode: str, colour: Any = "white") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (40, 20), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def sent_image(body: dict[str, Any]) -> Image.Image:
    url = body["messages"][0]["content"][0]["image_url"]["url"]
    prefix = "data:image/png;base64,"
    assert url.startswith(prefix)
    return Image.open(io.BytesIO(base64.b64decode(url.removeprefix(prefix))))


def block(category: str, text: str | None = None, box: int = 0) -> dict[str, Any]:
    """One block of a layout, its box told apart from others by a number."""
    found: dict[str, Any] = {"bbox": [box, box, box + 100, box + 20], "category": category}
    if text is not None:
        found["text"] = text
    return found


def test_the_request_is_the_one_dots_mocr_s_parser_sends() -> None:
    png = png_of("RGB")
    body = chat_request(MODEL, png)
    (message,) = body["messages"]
    image, prompt = message["content"]
    assert message["role"] == "user"
    assert image["type"] == "image_url"
    assert image["image_url"] == {"url": "data:image/png;base64," + base64.b64encode(png).decode()}
    assert prompt == {"type": "text", "text": IMAGE_TOKENS + LAYOUT_PROMPT}
    assert body["model"] == MODEL
    assert (body["temperature"], body["top_p"], body["seed"]) == (0.1, 1.0, 0)
    # No token limit: the answer may run to the end of the server's context.
    assert "max_completion_tokens" not in body
    # The model's own pixel limit, which the server applies unless told otherwise.
    assert "mm_processor_kwargs" not in body


def test_a_token_limit_is_sent_when_set() -> None:
    body = chat_request(MODEL, png_of("RGB"), Settings(max_tokens=4096))
    assert body["max_completion_tokens"] == 4096


def test_a_pixel_limit_goes_with_every_page() -> None:
    body = chat_request(MODEL, png_of("RGB"), Settings(max_pixels=4_000_000))
    assert body["mm_processor_kwargs"] == {"max_pixels": 4_000_000}


def test_a_pixel_limit_below_the_model_s_smallest_image_is_refused() -> None:
    with pytest.raises(ValueError, match="max_pixels is at least 3136"):
        Settings(max_pixels=784)


def test_the_same_page_sent_twice_is_new_to_the_server_each_time() -> None:
    png = png_of("RGB")
    first, second = chat_request(MODEL, png), chat_request(MODEL, png)
    assert first["messages"][0]["content"][0]["uuid"] != second["messages"][0]["content"][0]["uuid"]


@pytest.mark.parametrize("mode", ["L", "P", "RGBA"])
def test_a_page_is_sent_in_rgb(mode: str) -> None:
    assert sent_image(chat_request(MODEL, png_of(mode))).mode == "RGB"


def test_transparency_is_sent_as_white_as_dots_mocr_does() -> None:
    image = sent_image(chat_request(MODEL, png_of("RGBA", (0, 0, 0, 0))))
    assert image.getpixel((0, 0)) == (255, 255, 255)


def test_a_quantization_must_be_a_name() -> None:
    with pytest.raises(ValueError, match="quantization"):
        Settings(quantization="fp8 please")


def test_blocks_are_read_in_order_and_pictures_carry_no_text() -> None:
    answer = [
        block("Page-header", "الكلية", 0),
        block("Picture", None, 1),
        block("Section-header", "## المادة الأولى", 2),
        block("Text", "يلتزم الطالب", 3),
        block("Page-footer", "5", 4),
    ]
    expected = "\n\n".join(["الكلية", "المادة الأولى", "يلتزم الطالب", "5"])
    assert layout_text(json.dumps(answer)) == (expected, [])


@pytest.mark.parametrize(
    ("markdown", "text"),
    [
        ("# عنوان", "عنوان"),
        ("### عنوان فرعي", "عنوان فرعي"),
        ("**تعريفات:**", "تعريفات:"),
        ("*مائل*", "مائل"),
        ("- يلتزم الطالب", "يلتزم الطالب"),
        ("1. يلتزم الطالب", "1. يلتزم الطالب"),
        ("1\\. يلتزم الطالب", "1. يلتزم الطالب"),
        ("\\*", "*"),
        ("نسبة $70\\%$ من الدرجة", "نسبة 70% من الدرجة"),
        ("سطر<br>سطر", "سطر\nسطر"),
        ("<u>نص</u>", "نص"),
        ("5 < 6", "5 < 6"),
        ("A &amp; B", "A & B"),
    ],
)
def test_markdown_is_read_as_the_text_it_marks_up(markdown: str, text: str) -> None:
    assert markdown_text(markdown) == text


def test_a_table_becomes_a_line_per_row() -> None:
    table = (
        "<table><thead><tr><th>المادة</th><th>الدرجة</th></tr></thead>"
        "<tbody><tr><td>عربي<br>101</td><td>70%</td></tr>"
        "<tr><td>A &amp; B</td><td></td></tr></tbody></table>"
    )
    assert table_text(table) == "المادة الدرجة\nعربي 101 70%\nA & B"


def test_a_cell_left_open_ends_at_the_next() -> None:
    assert table_text("<table><tr><td>a<td>b<tr><td>c</table>") == "a b\nc"


def test_a_table_not_in_html_is_read_as_markdown() -> None:
    assert layout_text(json.dumps([block("Table", "**جدول**")])) == ("جدول", [])


def test_blocks_missing_the_comma_between_them_are_recovered() -> None:
    answer = f"[{json.dumps(block('Text', 'one', 0))}{json.dumps(block('Text', 'two', 1))}]"
    assert layout_text(answer) == (
        "one\n\ntwo",
        ["the layout is not valid JSON, so its blocks were read one by one: 2 read, 0 lost"],
    )


def test_a_block_cut_off_at_the_token_limit_is_lost() -> None:
    whole = json.dumps([block("Text", "one", 0), block("Text", "two words", 1)])
    assert layout_text(whole[: whole.index("two") + 3]) == (
        "one",
        ["the layout is not valid JSON, so its blocks were read one by one: 1 read, 1 lost"],
    )


def test_repeats_are_removed_from_a_recovered_layout() -> None:
    # The model looped: one text in five boxes, then a new text in a box it had used.
    loop = [block("Text", "again", box) for box in range(5)] + [block("Text", "new", 0)]
    text, _ = layout_text(json.dumps(loop).removesuffix("]"))
    assert text == "again"


def test_a_repeat_seen_fewer_than_five_times_is_kept() -> None:
    blocks = [block("Text", "again", box) for box in range(4)]
    text, _ = layout_text(json.dumps(blocks).removesuffix("]"))
    assert text == "\n\n".join(["again"] * 4)


def test_a_valid_layout_keeps_its_repeats_as_dots_mocr_does() -> None:
    blocks = [block("Text", "again", 0)] * 6
    assert layout_text(json.dumps(blocks)) == ("\n\n".join(["again"] * 6), [])


def test_an_answer_in_text_rather_than_json_is_read_as_markdown() -> None:
    assert layout_text("## عنوان\n\nنص") == ("عنوان\n\nنص", ["the answer is text, not a layout"])


def test_a_layout_without_text_says_so() -> None:
    assert layout_text(json.dumps([block("Picture")])) == (
        "",
        ["no block of the layout holds text"],
    )


class FakeServer:
    """Answers the engine like vLLM 0.30 serving dots.mocr, recording each request."""

    def __init__(
        self,
        content: str = "[]",
        finish_reason: str = "stop",
        models: list[dict[str, Any]] | None = None,
        version: Any = None,
    ) -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.models = [CARD] if models is None else models
        self.version = {"version": "0.30.0"} if version is None else version
        self.choices: list[dict[str, Any]] | None = None
        self.requests: list[tuple[str, Any, float]] = []

    def __call__(self, url: str, body: Any = None, timeout: float = 30.0) -> Any:
        self.requests.append((url, body, timeout))
        path = url.removeprefix(SERVER)
        if path == "/version":
            return self.version
        if path == "/v1/models":
            return {"object": "list", "data": self.models}
        if path == "/v1/chat/completions":
            message = {"role": "assistant", "content": self.content}
            choices = [{"index": 0, "message": message, "finish_reason": self.finish_reason}]
            return {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": body["model"],
                "choices": choices if self.choices is None else self.choices,
                "usage": {"prompt_tokens": 5217, "completion_tokens": 812, "total_tokens": 6029},
            }
        raise AssertionError(f"unexpected request to {url}")


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A Hugging Face cache holding dots.mocr at one commit."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    ref = tmp_path / "models--rednote-hilab--dots.mocr" / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text(COMMIT + "\n")
    return tmp_path


def test_the_engine_reads_the_server_once_and_names_what_it_serves(hub: Path) -> None:
    server = FakeServer()
    engine = DotsEngine(Settings(quantization="fp8", max_pixels=4_000_000), fetch=server)
    assert [url for url, _, _ in server.requests] == [f"{SERVER}/version", f"{SERVER}/v1/models"]
    assert engine.tag() == "dots.mocr-01234567-vllm0.30.0-fp8-max4000000px"
    assert engine.describe() == (
        "rednote-hilab/dots.mocr revision 0123456789ab; vLLM 0.30.0 at http://localhost:8000, "
        "quantization fp8, as stated, context 16384 tokens; layout prompt, pages of up to "
        "4,000,000 pixels, answers of up to the rest of the context, temperature 0.1, "
        "top_p 1, seed 0"
    )


def test_the_model_s_own_limits_are_named_as_such(hub: Path) -> None:
    engine = DotsEngine(fetch=FakeServer())
    assert engine.tag() == "dots.mocr-01234567-vllm0.30.0"
    assert ", unquantized, " in engine.describe()
    assert " pages of up to 11,289,600 pixels, " in engine.describe()


def test_an_openai_base_url_names_the_same_server(hub: Path) -> None:
    server = FakeServer()
    DotsEngine(Settings(server=f"{SERVER}/v1/"), fetch=server)
    assert server.requests[0][0] == f"{SERVER}/version"


def test_a_served_model_name_is_traced_to_the_hub_name(hub: Path) -> None:
    server = FakeServer(models=[{**CARD, "id": "model"}])
    engine = DotsEngine(fetch=server)
    engine.recognize(png_of("RGB"))
    assert server.requests[-1][1]["model"] == "model"
    assert engine.tag().startswith("dots.mocr-01234567-")


def test_a_page_is_read_and_the_whole_answer_kept(hub: Path) -> None:
    server = FakeServer(json.dumps([block("Title", "**لائحة**", 0), block("Text", "نص", 1)]))
    result = DotsEngine(fetch=server).recognize(png_of("RGB"))
    url, body, timeout = server.requests[-1]
    assert url == f"{SERVER}/v1/chat/completions"
    assert body["model"] == MODEL
    assert timeout == DEFAULT.timeout
    assert result.text == "لائحة\n\nنص"
    assert result.warnings == ""
    assert result.seconds >= 0
    assert json.loads(result.response)["usage"]["prompt_tokens"] == 5217


def test_an_answer_cut_off_at_the_token_limit_is_flagged(hub: Path) -> None:
    whole = json.dumps([block("Text", "one", 0), block("Text", "two", 1)])
    server = FakeServer(whole[:-10], finish_reason="length")
    result = DotsEngine(fetch=server).recognize(png_of("RGB"))
    assert result.text == "one"
    assert result.warnings.startswith("the answer was cut off at the token limit; ")


def test_an_answer_without_a_single_choice_is_an_error(hub: Path) -> None:
    server = FakeServer()
    server.choices = []
    with pytest.raises(DotsError, match="no single answer"):
        DotsEngine(fetch=server).recognize(png_of("RGB"))


def test_a_model_missing_from_the_cache_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(DotsError, match="not in the Hugging Face cache"):
        DotsEngine(fetch=FakeServer())


@pytest.mark.parametrize("models", [[], [CARD, {**CARD, "id": "other"}]])
def test_a_server_must_serve_one_model(hub: Path, models: list[dict[str, Any]]) -> None:
    with pytest.raises(DotsError, match=f"serves {len(models)} models"):
        DotsEngine(fetch=FakeServer(models=models))


def test_a_server_other_than_vllm_is_refused(hub: Path) -> None:
    with pytest.raises(DotsError, match="not a vLLM server"):
        DotsEngine(fetch=FakeServer(version={"detail": "Not Found"}))


def test_the_revision_is_the_commit_the_cache_holds(hub: Path) -> None:
    assert revision(MODEL) == COMMIT


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ({"HF_HUB_CACHE": "/a", "HUGGINGFACE_HUB_CACHE": "/b", "HF_HOME": "/c"}, "/a"),
        ({"HUGGINGFACE_HUB_CACHE": "/b", "HF_HOME": "/c"}, "/b"),
        ({"HF_HOME": "/c", "XDG_CACHE_HOME": "/d"}, "/c/hub"),
        ({"XDG_CACHE_HOME": "/d"}, "/d/huggingface/hub"),
    ],
)
def test_the_cache_is_found_as_the_hub_client_finds_it(
    monkeypatch: pytest.MonkeyPatch, variables: dict[str, str], expected: str
) -> None:
    for name in HUB_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    assert hub_cache() == Path(expected)


def test_the_cache_is_in_the_home_folder_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in HUB_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert hub_cache() == tmp_path / ".cache" / "huggingface" / "hub"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b'{"error": {"message": "bad", "type": "BadRequestError", "code": 400}}', "bad"),
        (b'{"object": "error", "message": "old", "type": "BadRequestError", "code": 400}', "old"),
        (b"Internal Server Error", "Internal Server Error"),
    ],
)
def test_the_message_is_taken_from_an_error(body: bytes, message: str) -> None:
    assert error_message(body) == message


class Handler(BaseHTTPRequestHandler):
    """A few answers a vLLM server can give, and one it should never."""

    def do_GET(self) -> None:
        self.answer(200, {"version": "0.30.0"})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/echo":
            self.answer(200, {"type": self.headers["Content-Type"], "body": body})
        elif self.path == "/too-long":
            message = "This model's maximum context length is 16384 tokens."
            error = {"message": message, "type": "BadRequestError", "param": None, "code": 400}
            self.answer(400, {"error": error})
        elif self.path == "/slow":
            time.sleep(0.5)
            self.answer(200, {})
        # Anything else is left unanswered: the connection closes without a response.

    def answer(self, status: int, content: Any) -> None:
        data = json.dumps(content).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def local_server() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_the_client_gets_and_posts_json(local_server: str) -> None:
    assert fetch_json(f"{local_server}/version") == {"version": "0.30.0"}
    echoed = fetch_json(f"{local_server}/echo", {"text": "نص"})
    assert echoed == {"type": "application/json", "body": {"text": "نص"}}


def test_a_refusal_carries_the_server_s_message(local_server: str) -> None:
    with pytest.raises(DotsError, match="answered 400: This model's maximum context length"):
        fetch_json(f"{local_server}/too-long", {})


def test_a_server_that_hangs_up_is_reported(local_server: str) -> None:
    with pytest.raises(DotsError, match="broke off its answer"):
        fetch_json(f"{local_server}/hang-up", {})


def test_a_server_that_takes_too_long_is_reported(local_server: str) -> None:
    with pytest.raises(DotsError, match=r"no answer in 0\.1 seconds"):
        fetch_json(f"{local_server}/slow", {}, timeout=0.1)


def test_no_server_is_reported() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    # The port is free again once the probe closes, so nothing listens on it.
    with pytest.raises(DotsError, match="is vllm serve running"):
        fetch_json(f"http://127.0.0.1:{port}/version", timeout=5)
