#!/usr/bin/env python3
"""Ask a model one question through the evaluation client, and show what came back.

    python3 scripts/ask_llm.py "ما عاصمة السعودية؟" [--model gemma-4-31b-it] [--system TEXT]

Prints the reply, then its model version, finish reason, tokens, seconds and
whether it came from the store. Asking the same question again comes from
the store and costs no quota. Reads the key from GEMINI_API_KEY or `.env`.
"""

from __future__ import annotations

import argparse
import sys

from daleel.answer.llm import GENERATOR, MODELS, Client, LlmError, Request


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask a model one question.")
    parser.add_argument("prompt")
    parser.add_argument("--model", choices=sorted(MODELS), default=GENERATOR)
    parser.add_argument("--system", default="")
    args = parser.parse_args(argv)
    request = Request(args.model, args.prompt, system=args.system, purpose="ask_llm")
    try:
        reply = Client().ask(request)
    except LlmError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(reply.text)
    cost = reply.cost(MODELS[args.model])
    print(
        f"\n{reply.model_version or args.model}, finish {reply.finish}, "
        f"{reply.tokens_in} tokens in, {reply.tokens_out} out, {reply.tokens_thinking} thinking, "
        f"{reply.seconds} s, {'from the store' if reply.cached else 'asked now'}"
        + (f", ${cost:.6f} at paid rates" if cost is not None else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
