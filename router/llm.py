"""Minimal Ollama chat client (stdlib only)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("ROUTER_MODEL", "gemma4:12b")


class LLMError(RuntimeError):
    pass


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str: ...


class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, timeout: float = 300.0):
        self.model = model
        self.host = host if host.startswith("http") else f"http://{host}"
        self.timeout = timeout
        # Loose JSON mode nudges the model toward a JSON object while leaving schema
        # enforcement to our validator. Some models/servers reject it (HTTP 501);
        # we then fall back to prompt-only enforcement for the rest of the session.
        self.json_mode = True

    def chat(self, messages: list[dict[str, str]]) -> str:
        try:
            return self._chat(messages)
        except LLMError as e:
            if self.json_mode and "structured output is unavailable" in str(e):
                self.json_mode = False
                return self._chat(messages)
            raise

    def _chat(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        }
        if self.json_mode:
            payload["format"] = "json"
        req = urllib.request.Request(
            f"{self.host.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data: dict[str, Any] = json.load(resp)
        except urllib.error.HTTPError as e:
            raise LLMError(f"Ollama returned HTTP {e.code}: {e.read().decode(errors='replace')}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise LLMError(f"could not reach Ollama at {self.host} ({e}). Is `ollama serve` running?") from e
        return data.get("message", {}).get("content", "")
