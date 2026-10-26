"""Parse raw model text into a dict and validate it against the tool registry."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

from .tools import TOOLS

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
TOP_LEVEL_KEYS = {"reasoning", "function", "arguments", "reason"}


class ParseError(ValueError):
    """The model output could not be turned into a JSON object."""


def parse_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model text (tolerates code fences / chatter)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start == -1:
            raise ParseError("no JSON object found in the response")
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as e:
            raise ParseError(f"invalid JSON: {e.msg} at position {e.pos}") from e
    if not isinstance(obj, dict):
        raise ParseError(f"expected a JSON object, got {type(obj).__name__}")
    return obj


def _check_value(name: str, value: Any, spec: dict[str, Any]) -> list[str]:
    t = spec.get("type")
    if t == "string" and not isinstance(value, str):
        return [f"argument '{name}' must be a string"]
    if t == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        return [f"argument '{name}' must be an integer"]
    if t == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        return [f"argument '{name}' must be a number"]
    if t == "boolean" and not isinstance(value, bool):
        return [f"argument '{name}' must be a boolean"]

    if isinstance(value, str) and not value.strip():
        return [f"argument '{name}' must not be empty"]
    if "enum" in spec and value not in spec["enum"]:
        return [f"argument '{name}' must be one of {spec['enum']}, got {value!r}"]

    fmt = spec.get("format")
    if fmt == "email" and not EMAIL_RE.match(value):
        return [f"argument '{name}' must be a valid email address, got {value!r}"]
    if fmt == "date":
        try:
            date.fromisoformat(value)
            if len(value) != 10:
                raise ValueError
        except ValueError:
            return [f"argument '{name}' must be a date in YYYY-MM-DD format, got {value!r}"]
    if fmt == "time":
        try:
            datetime.strptime(value, "%H:%M")
        except ValueError:
            return [f"argument '{name}' must be a 24-hour time in HH:MM format, got {value!r}"]
    return []


def validate(obj: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problems; empty list means the call is valid."""
    errors: list[str] = []

    extra = set(obj) - TOP_LEVEL_KEYS
    if extra:
        errors.append(f"unexpected top-level keys: {sorted(extra)}")
    if not isinstance(obj.get("reasoning"), str) or not obj.get("reasoning", "").strip():
        errors.append("'reasoning' must be a non-empty string explaining the tool choice")
    if "function" not in obj:
        errors.append("missing 'function' key (use null for out-of-scope requests)")
        return errors

    fn = obj["function"]
    args = obj.get("arguments", {})
    if not isinstance(args, dict):
        return errors + ["'arguments' must be a JSON object"]

    if fn is None:
        if args:
            errors.append("'arguments' must be {} when 'function' is null")
        if not isinstance(obj.get("reason"), str) or not obj["reason"].strip():
            errors.append("'reason' must explain why the request is out of scope when 'function' is null")
        return errors

    if fn not in TOOLS:
        return errors + [f"unknown function {fn!r}; must be one of {sorted(TOOLS)} or null"]

    params = TOOLS[fn].parameters
    props = params["properties"]
    for req in params.get("required", []):
        if req not in args:
            errors.append(f"missing required argument '{req}' for {fn}")
    for key, value in args.items():
        if key not in props:
            errors.append(f"unknown argument '{key}' for {fn}; allowed: {sorted(props)}")
        else:
            errors.extend(_check_value(key, value, props[key]))
    return errors
