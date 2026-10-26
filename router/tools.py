"""Mock backend functions and the schemas that describe them to the model.

Each tool is registered with a JSON-Schema-style parameter spec. The same spec
is (1) rendered into the system prompt and (2) used by the validator, so the
model is constrained to exactly the interface the code can execute.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # {"type": "object", "properties": {...}, "required": [...]}
    func: Callable[..., dict[str, Any]]


def _fake_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8]
    return f"{prefix}_{digest}"


# --- mock backends -----------------------------------------------------------

def get_weather(location: str, unit: str = "celsius") -> dict[str, Any]:
    # Deterministic fake reading derived from the location name.
    seed = int(hashlib.sha1(location.lower().encode()).hexdigest(), 16)
    temp_c = 5 + seed % 28
    temp = temp_c if unit == "celsius" else round(temp_c * 9 / 5 + 32)
    conditions = ["sunny", "cloudy", "rainy", "windy", "partly cloudy"][seed % 5]
    return {"location": location, "temperature": temp, "unit": unit, "conditions": conditions}


def send_email(to: str, subject: str, body: str) -> dict[str, Any]:
    return {"status": "queued", "message_id": _fake_id("msg", to, subject, body), "to": to}


def book_appointment(service: str, date: str, time: str, name: str | None = None) -> dict[str, Any]:
    return {
        "status": "confirmed",
        "confirmation_id": _fake_id("apt", service, date, time, name),
        "service": service,
        "date": date,
        "time": time,
        "name": name,
    }


# --- registry ----------------------------------------------------------------

TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            name="get_weather",
            description="Get the current weather for a city or location.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or place name, e.g. 'Lagos' or 'Paris, France'."},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit. Default celsius."},
                },
                "required": ["location"],
            },
            func=get_weather,
        ),
        Tool(
            name="send_email",
            description="Send an email to a single recipient.",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string", "format": "email", "description": "Recipient email address."},
                    "subject": {"type": "string", "description": "Email subject line."},
                    "body": {"type": "string", "description": "Plain-text email body."},
                },
                "required": ["to", "subject", "body"],
            },
            func=send_email,
        ),
        Tool(
            name="book_appointment",
            description="Book an appointment for a service (doctor, dentist, haircut, etc.) at a date and time.",
            parameters={
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "What the appointment is for, e.g. 'dentist'."},
                    "date": {"type": "string", "format": "date", "description": "Date as YYYY-MM-DD."},
                    "time": {"type": "string", "format": "time", "description": "24-hour time as HH:MM."},
                    "name": {"type": "string", "description": "Name of the person the booking is for (optional)."},
                },
                "required": ["service", "date", "time"],
            },
            func=book_appointment,
        ),
    ]
}
