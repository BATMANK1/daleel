"""The web service: the answering flow and search behind HTTP, and the page that asks them."""

from pathlib import Path

# What the service keeps: the query log and the model's stored replies,
# apart from the evaluation's.
SERVE = Path("data/serve")
QUERIES = SERVE / "queries.jsonl"
REPLIES = SERVE / "llm"
