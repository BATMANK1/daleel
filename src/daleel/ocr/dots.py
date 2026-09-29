"""Read pages with dots.mocr, a vision language model, through a vLLM server.

dots.mocr (github.com/rednote-hilab/dots.mocr) reads a page image and answers
with its layout: a JSON list of the page's blocks in reading order, each with a
box, a category and the text inside it. vLLM serves the model over the OpenAI
chat API, so this engine is an HTTP client that needs only the standard library
and Pillow; the model, the GPU and their packages stay with the server.

Each request is the one dots.mocr's own parser sends (dots_mocr/parser.py): the
page as a PNG data URL, then the layout prompt behind the image tokens, with
temperature 0.1 and top_p 1. A fixed seed makes that sampling repeatable. Each
image is sent with a new id, which vLLM keys its caches on in place of the
image, so it cannot reuse its work on a page it has seen: otherwise the first
page, read once untimed before the timed run, would be timed without being read.

A block's text is Markdown, or HTML for a table, so it is turned back into the
plain text the page prints. Pictures carry no text. Page headers and footers
are kept, as the ground truth keeps them. When the answer is not valid JSON, its
complete blocks are recovered and repeats removed, as dots.mocr's own cleaner
does (dots_mocr/utils/output_cleaner.py).

What produced each result is recorded: the vLLM version and the model the
server reports, and the model's revision, read from the Hugging Face cache, so
the client runs on the machine that serves. The server does not report whether
it quantized the model, so that is stated when the engine is built.

A page larger than the model's pixel limit is scaled down to fit. vLLM sets
aside memory for the largest image a request may hold, so a small GPU needs a
lower limit, given to the server when it starts (--mm-processor-kwargs). The
engine sends its own limit with every page, so the one recorded is the one used.
"""

from __future__ import annotations

import base64
import html
import http.client
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from PIL import Image

from daleel.ocr.engine import Result

# prompt_layout_all_en in dots_mocr/utils/prompts.py, verbatim.
LAYOUT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""  # noqa: E501

# Written before the prompt, the image tokens keep vLLM from adding a newline
# between the image and the prompt (dots_mocr/model/inference.py).
IMAGE_TOKENS = "<|img|><|imgpad|><|endofimg|>"

_NAME = re.compile(r"[A-Za-z0-9_.-]+")

# dots_mocr/utils/consts.py: the smallest image the model's processor takes,
# and the largest it takes without scaling down.
MIN_PIXELS = 3136
MAX_PIXELS = 11289600


class DotsError(RuntimeError):
    """The server could not be reached, refused a request, or gave an answer that cannot be read."""


@dataclass(frozen=True)
class Settings:
    """Where the server is, how it serves the model, and everything sent that can change a page."""

    server: str = "http://localhost:8000"
    # vLLM's --quantization, as the server was started. vLLM does not report it.
    quantization: str = "none"
    # The most pixels of a page the model sees: the model's image processor
    # scales a larger page down to fit. None keeps the model's own limit.
    max_pixels: int | None = None
    # The defaults of dots.mocr's parser, stated so that a change to them shows.
    temperature: float = 0.1
    top_p: float = 1.0
    max_tokens: int = 32768
    # Not the parser's: a fixed seed makes the sampling repeatable.
    seed: int = 0
    # Seconds to wait for a page. A long page on a small GPU takes minutes.
    timeout: float = 900.0

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.quantization):
            raise ValueError(f"a quantization is a name such as fp8, not {self.quantization!r}")
        if self.max_pixels is not None and self.max_pixels < MIN_PIXELS:
            raise ValueError(f"max_pixels is at least {MIN_PIXELS}, not {self.max_pixels}")


DEFAULT = Settings()


def rgb_png(png: bytes) -> bytes:
    """The page as an RGB PNG, as dots.mocr's parser gives every image (image_utils.to_rgb)."""
    with Image.open(io.BytesIO(png)) as image:
        if image.mode == "RGB":
            return png
        if image.mode == "RGBA":
            rgb = Image.new("RGB", image.size, "white")
            rgb.paste(image, mask=image.getchannel("A"))
        else:
            rgb = image.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, format="PNG")
    return buffer.getvalue()


def chat_request(model: str, png: bytes, settings: Settings = DEFAULT) -> dict[str, Any]:
    """The chat completion request that asks the model for one page's layout."""
    image = base64.b64encode(rgb_png(png)).decode("ascii")
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image}"},
            "uuid": uuid.uuid4().hex,
        },
        {"type": "text", "text": IMAGE_TOKENS + LAYOUT_PROMPT},
    ]
    request: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_completion_tokens": settings.max_tokens,
        "seed": settings.seed,
    }
    if settings.max_pixels is not None:
        # Sent with every page, so the limit a page was read at is the one recorded.
        request["mm_processor_kwargs"] = {"max_pixels": settings.max_pixels}
    return request


def error_message(body: bytes) -> str:
    """The message in an error body from vLLM, or the start of the body when it holds none."""
    try:
        answer = json.loads(body)
    except ValueError:
        return body.decode("utf-8", errors="replace").strip()[:500]
    if isinstance(answer, dict):
        error = answer.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        # Older releases of vLLM put the message at the top level.
        if answer.get("message"):
            return str(answer["message"])
    return str(answer)[:500]


def fetch_json(url: str, body: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
    """GET a URL, or POST a body to it as JSON, and return the JSON it answers with."""
    if body is None:
        request = urllib.request.Request(url)
    else:
        data = json.dumps(body).encode()
        request = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            answer = response.read()
    except urllib.error.HTTPError as exc:
        raise DotsError(f"{url} answered {exc.code}: {error_message(exc.read())}") from exc
    except urllib.error.URLError as exc:
        raise DotsError(f"nothing answers at {url} ({exc.reason}); is vllm serve running?") from exc
    except TimeoutError as exc:
        raise DotsError(f"{url} gave no answer in {timeout:g} seconds") from exc
    except (OSError, http.client.HTTPException) as exc:
        raise DotsError(f"{url} broke off its answer: {exc!r}") from exc
    try:
        return json.loads(answer)
    except ValueError as exc:
        raise DotsError(f"{url} answered with something other than JSON: {answer[:200]!r}") from exc


def _expanded(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def hub_cache() -> Path:
    """Where the Hugging Face Hub client keeps its downloads, found the way it finds it."""
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if folder := os.environ.get(variable):
            return _expanded(folder)
    if home := os.environ.get("HF_HOME"):
        return _expanded(home) / "hub"
    cache = os.environ.get("XDG_CACHE_HOME")
    return (_expanded(cache) if cache else Path.home() / ".cache") / "huggingface" / "hub"


def revision(model: str) -> str:
    """The commit of a model downloaded from the Hugging Face Hub, read from the local cache."""
    ref = hub_cache() / f"models--{model.replace('/', '--')}" / "refs" / "main"
    try:
        return ref.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise DotsError(
            f"{model} is not in the Hugging Face cache at {hub_cache()}, so its revision "
            "cannot be recorded: serve the model by its Hub name, downloaded with hf download"
        ) from exc


_ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")  # CommonMark: any ASCII punctuation
_SET_ASIDE = 0xE000  # the Private Use Area, where the model writes nothing
_SET_ASIDE_CHARS = re.compile("[\ue021-\ue07e]")
_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>")
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]+", re.MULTILINE)
_BULLET = re.compile(r"^[ \t]*[-*+][ \t]+", re.MULTILINE)
_STRONG = re.compile(r"\*\*|__")
_EMPHASIS = re.compile(r"\*(?=\S)(.+?)(?<=\S)\*")
_MATH = re.compile(r"\$([^$\n]+)\$")


def markdown_text(markdown: str) -> str:
    """The text of a Markdown block with its markup gone: what the page prints.

    Headings, bold, italics, inline maths and HTML tags lose their markup. List
    bullets go, as the ground truth leaves out decorative bullets
    (ANNOTATION.md 4.7), while a list's numbers stay.
    """
    # Escaped characters are text, not markup, so they are set aside meanwhile.
    text = _ESCAPE.sub(lambda match: chr(_SET_ASIDE + ord(match[1])), markdown)
    text = _BREAK.sub("\n", text)
    text = _TAG.sub("", text)
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _STRONG.sub("", text)
    text = _EMPHASIS.sub(r"\1", text)
    text = _MATH.sub(r"\1", text)
    text = _SET_ASIDE_CHARS.sub(lambda match: chr(ord(match[0]) - _SET_ASIDE), text)
    return html.unescape(text).strip()


class _Table(HTMLParser):
    """Collects an HTML table's cells, row by row."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.end_cell()
            self.rows.append([])
        elif tag in ("td", "th"):
            self.end_cell()
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self.end_cell()

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def end_cell(self) -> None:
        if self.cell is None:
            return
        if not self.rows:
            self.rows.append([])
        self.rows[-1].append(" ".join("".join(self.cell).split()))
        self.cell = None


def table_text(table: str) -> str:
    """The text of an HTML table: a line for each row, its cells in the order given."""
    parser = _Table()
    parser.feed(table)
    parser.close()
    parser.end_cell()
    lines = (" ".join(cell for cell in row if cell) for row in parser.rows)
    text = "\n".join(line for line in lines if line)
    # A table the model wrote as something other than HTML is read as Markdown.
    return text or markdown_text(table)


def block_text(block: dict[str, Any]) -> str:
    """The plain text of one layout block: none for a picture, a line per row for a table."""
    text = block.get("text")
    if block.get("category") == "Picture" or not isinstance(text, str):
        return ""
    if block.get("category") == "Table":
        return table_text(text)
    return markdown_text(text)


_BLOCK_START = re.compile(r'\{\s*"')


def _box(block: dict[str, Any]) -> str | None:
    box = block.get("bbox")
    return json.dumps(box) if isinstance(box, list) and box else None


def _pair(block: dict[str, Any]) -> str | None:
    if "category" in block and "text" in block:
        return json.dumps([block["category"], block["text"]], ensure_ascii=False)
    return None


def deduplicated(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The blocks less the repeats dots.mocr's cleaner removes.

    A box that appeared before, or a category and text that appear five times
    or more, keeps only its first block.
    """
    counts = Counter(_pair(block) for block in blocks)
    kept: list[dict[str, Any]] = []
    boxes: set[str | None] = set()
    pairs: set[str | None] = set()
    for block in blocks:
        box, pair = _box(block), _pair(block)
        repeat = (box is not None and box in boxes) or (
            pair is not None and counts[pair] >= 5 and pair in pairs
        )
        boxes.add(box)
        pairs.add(pair)
        if not repeat:
            kept.append(block)
    return kept


def recovered(answer: str) -> tuple[list[dict[str, Any]], int]:
    """The complete blocks of a layout that is not valid JSON, and how many were lost.

    A long answer can drop the comma between two blocks, repeat blocks in a
    loop, or stop in the middle of one at the token limit. Each block is read on
    its own, a block that cannot be read is lost, and repeats are removed.
    """
    decoder = json.JSONDecoder()
    blocks: list[dict[str, Any]] = []
    lost = end = 0
    for start in _BLOCK_START.finditer(answer):
        if start.start() < end:
            continue
        try:
            block, end = decoder.raw_decode(answer, start.start())
        except ValueError:
            lost += 1
            continue
        if isinstance(block, dict):
            blocks.append(block)
    return deduplicated(blocks), lost


def layout_text(answer: str) -> tuple[str, list[str]]:
    """The plain text of a layout answer, block by block, and notes on any repair it needed."""
    notes: list[str] = []
    try:
        layout = json.loads(answer)
    except ValueError:
        if not answer.lstrip().startswith(("[", "{")):
            return markdown_text(answer), ["the answer is text, not a layout"]
        layout, lost = recovered(answer)
        notes.append(
            "the layout is not valid JSON, so its blocks were read one by one: "
            f"{len(layout)} read, {lost} lost"
        )
    blocks = layout if isinstance(layout, list) else [layout]
    texts = [block_text(block) for block in blocks if isinstance(block, dict)]
    text = "\n\n".join(text for text in texts if text)
    if not text and answer.strip():
        notes.append("no block of the layout holds text")
    return text, notes


Fetch = Callable[..., Any]


class DotsEngine:
    """dots.mocr on a running vLLM server, whose version and model are read once when built.

    `fetch` stands in for the HTTP client in tests: it takes a URL, a body to
    post as JSON if any, and a timeout, and returns the JSON answer.
    """

    def __init__(self, settings: Settings = DEFAULT, fetch: Fetch = fetch_json) -> None:
        self.settings = settings
        self._fetch = fetch
        # An OpenAI client's base URL, which ends in /v1, names the same server.
        self.server = settings.server.rstrip("/").removesuffix("/v1")
        version = fetch(f"{self.server}/version")
        if not isinstance(version, dict) or "version" not in version:
            raise DotsError(f"{self.server} is not a vLLM server: /version gave {version!r}")
        self.vllm = str(version["version"])
        listing = fetch(f"{self.server}/v1/models")
        models = listing.get("data") if isinstance(listing, dict) else None
        if not isinstance(models, list) or len(models) != 1:
            count = len(models) if isinstance(models, list) else "no"
            raise DotsError(f"the server at {self.server} serves {count} models, not one")
        (card,) = models
        # Requests name the model as served; its Hub name is what vllm serve
        # was given, which --served-model-name can hide.
        self.model = str(card["id"])
        self.name = str(card.get("root") or card["id"])
        self.context = card.get("max_model_len")
        self.revision = revision(self.name)

    def tag(self) -> str:
        tag = f"{self.name.rsplit('/', 1)[-1]}-{self.revision[:8]}-vllm{self.vllm}"
        if self.settings.quantization != "none":
            tag += f"-{self.settings.quantization}"
        if self.settings.max_pixels is not None:
            tag += f"-max{self.settings.max_pixels}px"
        return tag

    def describe(self) -> str:
        settings = self.settings
        weights = (
            "unquantized"
            if settings.quantization == "none"
            else f"quantization {settings.quantization}, as stated"
        )
        pixels = MAX_PIXELS if settings.max_pixels is None else settings.max_pixels
        return (
            f"{self.name} revision {self.revision[:12]}; vLLM {self.vllm} at {self.server}, "
            f"{weights}, context {self.context} tokens; layout prompt, pages of up to "
            f"{pixels:,} pixels, temperature {settings.temperature:g}, "
            f"top_p {settings.top_p:g}, seed {settings.seed}"
        )

    def recognize(self, png: bytes) -> Result:
        settings = self.settings
        start = time.perf_counter()
        body = chat_request(self.model, png, settings)
        answer = self._fetch(f"{self.server}/v1/chat/completions", body, settings.timeout)
        seconds = time.perf_counter() - start
        try:
            (choice,) = answer["choices"]
            content = choice["message"]["content"] or ""
        except (KeyError, TypeError, ValueError) as exc:
            raise DotsError(f"no single answer in the server's response: {answer!r:.300}") from exc
        text, notes = layout_text(content)
        if choice.get("finish_reason") == "length":
            notes.insert(0, "the answer was cut off at the token limit")
        return Result(
            text=text,
            seconds=seconds,
            warnings="; ".join(notes),
            response=json.dumps(answer, ensure_ascii=False),
        )
