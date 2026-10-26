"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from .llm import DEFAULT_HOST, DEFAULT_MODEL, LLMError, OllamaClient
from .router import Attempt, RouteResult, route


def _print_human(res: RouteResult, verbose: bool) -> None:
    if res.status == "ok":
        print(f"Reasoning : {res.reasoning}")
        print(f"Function  : {res.function}")
        print(f"Arguments : {json.dumps(res.arguments)}")
        if res.result is not None:
            print(f"Result    : {json.dumps(res.result)}")
    elif res.status == "out_of_scope":
        print(f"Reasoning : {res.reasoning}")
        print("Function  : (none) - request is out of scope")
        print(f"Reason    : {res.reason}")
    else:
        print(f"Error     : {res.error}", file=sys.stderr)
    if verbose or len(res.attempts) > 1:
        print(f"Attempts  : {len(res.attempts)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="router",
        description="Route a natural-language request to a mock backend function via a local Ollama model.",
    )
    p.add_argument("request", nargs="*", help="the request, e.g. \"what's the weather in Lagos?\" (omit for interactive mode)")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    p.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama host (default: {DEFAULT_HOST})")
    p.add_argument("-r", "--retries", type=int, default=2, help="correction retries on invalid output (default: 2)")
    p.add_argument("--timeout", type=float, default=300.0, help="seconds to wait for Ollama per call (default: 300)")
    p.add_argument("--json", action="store_true", help="print the result as a single JSON object")
    p.add_argument("--no-exec", action="store_true", help="only route; don't call the mock function")
    p.add_argument("-v", "--verbose", action="store_true", help="show every raw model response and validation error")
    args = p.parse_args(argv)

    client = OllamaClient(model=args.model, host=args.host, timeout=args.timeout)

    def on_attempt(n: int, a: Attempt) -> None:
        if args.verbose:
            print(f"--- attempt {n} raw output ---\n{a.raw}", file=sys.stderr)
            for e in a.errors:
                print(f"    invalid: {e}", file=sys.stderr)

    def run(text: str) -> int:
        try:
            res = route(text, client, max_retries=args.retries, execute=not args.no_exec, on_attempt=on_attempt)
        except LLMError as e:
            res = RouteResult(status="error", error=str(e))
        if args.json:
            print(json.dumps(res.to_dict(), indent=2))
        else:
            _print_human(res, args.verbose)
        return 1 if res.status == "error" else 0

    if args.request:
        return run(" ".join(args.request))

    print(f"router ({args.model}) - type a request, or 'quit' to exit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text.lower() in {"quit", "exit", "q"}:
            return 0
        if text:
            run(text)
            print()


if __name__ == "__main__":
    sys.exit(main())
