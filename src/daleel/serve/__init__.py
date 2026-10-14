"""The web service: the answering flow and search behind HTTP, and the page that asks them."""

from pathlib import Path

# What the service keeps: the query log and the model's stored replies,
# apart from the evaluation's.
SERVE = Path("data/serve")
QUERIES = SERVE / "queries.jsonl"
REPLIES = SERVE / "llm"

# Set, the service answers only requests that send this code (daleel.serve.app).
ACCESS_CODE = "DALEEL_ACCESS_CODE"
# Long enough not to be guessed: secrets.token_urlsafe(12) gives 16 characters.
MIN_CODE = 12
