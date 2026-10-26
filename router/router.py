"""Core routing loop: system prompt -> model -> parse -> validate -> retry -> execute."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable

from .llm import ChatClient
from .tools import TOOLS
from .validator import ParseError, parse_json, validate

SYSTEM_PROMPT = """\
You are a function-calling router. You NEVER answer the user directly and you NEVER chat.
Your only job is to map the user's request to exactly one of the functions below, or to
declare the request out of scope.

Today's date is {today} ({weekday}). Resolve relative dates like "tomorrow" or "next Friday"
by looking them up in this calendar of the coming days (do not do date arithmetic yourself):
{calendar}

AVAILABLE FUNCTIONS (JSON Schema for each function's arguments):
{tools}

RESPONSE FORMAT — reply with ONE JSON object and nothing else (no prose, no markdown):

For an in-scope request:
{{"reasoning": "<one or two sentences: which function fits and why>",
  "function": "<one of: {names}>",
  "arguments": {{<arguments matching that function's schema exactly>}}}}

For a request no function can fulfil (general questions, chit-chat, unsupported actions,
or a request missing information you cannot reasonably infer):
{{"reasoning": "<why none of the functions apply>",
  "function": null,
  "arguments": {{}},
  "reason": "<short user-facing explanation of what is not supported>"}}

RULES:
1. Think first: fill "reasoning" before choosing "function".
2. Use only the function names and argument names listed above. Never invent new ones.
3. Include every required argument. Omit optional arguments you have no value for.
4. Dates are YYYY-MM-DD, times are 24-hour HH:MM, emails must be real addresses
   given by the user. Never fabricate a required value such as an email address.
5. If the request asks for something the functions cannot do, use the out-of-scope form.
"""

CORRECTION_PROMPT = """\
Your previous response failed validation:
{errors}

Reply again with ONE corrected JSON object that follows the response format and the
function schemas exactly. Output only the JSON."""


def build_system_prompt(today: date | None = None) -> str:
    today = today or date.today()
    tools = "\n".join(
        f"- {t.name}: {t.description}\n  parameters: {json.dumps(t.parameters)}" for t in TOOLS.values()
    )
    calendar = "\n".join(
        f"  {d.strftime('%A')} {d.isoformat()}" + (" (tomorrow)" if i == 1 else "")
        for i, d in ((i, today + timedelta(days=i)) for i in range(1, 15))
    )
    return SYSTEM_PROMPT.format(
        today=today.isoformat(),
        weekday=today.strftime("%A"),
        calendar=calendar,
        tools=tools,
        names=", ".join(TOOLS),
    )


@dataclass
class Attempt:
    raw: str
    errors: list[str]


@dataclass
class RouteResult:
    status: str  # "ok" | "out_of_scope" | "error"
    function: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    reasoning: str | None = None
    reason: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: list[Attempt] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": self.status,
            "function": self.function,
            "arguments": self.arguments,
            "reasoning": self.reasoning,
            "attempts": len(self.attempts),
        }
        if self.status == "out_of_scope":
            out["reason"] = self.reason
        if self.result is not None:
            out["result"] = self.result
        if self.error is not None:
            out["error"] = self.error
        return out


def route(
    request: str,
    client: ChatClient,
    max_retries: int = 2,
    execute: bool = True,
    today: date | None = None,
    on_attempt: Callable[[int, Attempt], None] | None = None,
) -> RouteResult:
    """Route one natural-language request. Retries up to `max_retries` times on invalid output."""
    messages = [
        {"role": "system", "content": build_system_prompt(today)},
        {"role": "user", "content": request},
    ]
    attempts: list[Attempt] = []

    for n in range(1, max_retries + 2):
        raw = client.chat(messages)
        try:
            obj = parse_json(raw)
            errors = validate(obj)
        except ParseError as e:
            obj, errors = None, [str(e)]

        attempt = Attempt(raw=raw, errors=errors)
        attempts.append(attempt)
        if on_attempt:
            on_attempt(n, attempt)

        if not errors:
            assert obj is not None
            if obj["function"] is None:
                return RouteResult(
                    status="out_of_scope", reasoning=obj["reasoning"], reason=obj["reason"], attempts=attempts
                )
            fn, args = obj["function"], obj.get("arguments", {})
            result = TOOLS[fn].func(**args) if execute else None
            return RouteResult(
                status="ok", function=fn, arguments=args, reasoning=obj["reasoning"], result=result, attempts=attempts
            )

        # Feed the bad output back and ask the model to fix it.
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {"role": "user", "content": CORRECTION_PROMPT.format(errors="\n".join(f"- {e}" for e in errors))}
        )

    return RouteResult(
        status="error",
        error=f"model failed to produce a valid function call after {len(attempts)} attempts: "
        + "; ".join(attempts[-1].errors),
        attempts=attempts,
    )
