# function-router

A Python CLI that turns a natural-language request into a **validated function call** using a local
Ollama model. The model is held to a fixed interface of three mock backend functions. Its JSON output
is parsed into a Python `dict`, checked against each function's schema, and corrected through
retries if it doesn't match. Requests the tools can't handle are refused with an explanation.

Standard library only, so nothing needs installing beyond Python 3.10+ and a running Ollama.

## Usage

```bash
cd function-router
python3 -m router "What's the weather in Lagos in fahrenheit?"
python3 -m router --json "Book a dentist appointment tomorrow at 2:30pm"
python3 -m router -v "Order me a pizza"          # -v shows raw model output and validation errors
python3 -m router                                 # interactive mode
```

| flag | default | meaning |
|---|---|---|
| `-m/--model` | `gemma4:12b` (or `$ROUTER_MODEL`) | Ollama model |
| `--host` | `http://localhost:11434` (or `$OLLAMA_HOST`) | Ollama server |
| `-r/--retries` | `2` | correction retries after an invalid response |
| `--json` | off | print one JSON object (exit code 1 on error) |
| `--no-exec` | off | route only; don't call the mock function |
| `--timeout` | `300` | seconds per Ollama call |

## Functions

| name | required args | optional |
|---|---|---|
| `get_weather` | `location` | `unit` (`celsius`/`fahrenheit`) |
| `send_email` | `to` (email), `subject`, `body` | |
| `book_appointment` | `service`, `date` (YYYY-MM-DD), `time` (HH:MM) | `name` |

Defined in `router/tools.py`. The same schema is rendered into the system prompt and used by the validator.

## How it works

1. **System prompt** (`router/router.py`): forbids chatting, lists the function schemas, sets the
   required response shape, and includes today's date plus a 14-day calendar for relative dates.
2. **Chain of thought**: every response must start with a `reasoning` field explaining the tool choice.
3. **Parse** (`validator.parse_json`): pulls a JSON object out of the reply, handling code fences or stray text.
4. **Validate** (`validator.validate`): checks the function name, that required args are present,
   that no unknown args are passed, and the types, enums, email/date/time formats and out-of-scope shape.
5. **Retry**: if validation fails, the bad output and the list of errors go back to the model with a
   request to fix them. This repeats up to `--retries` times, then ends with `status: "error"`.
6. **Execute**: the validated arguments are passed to the mock function and its result is printed.

Response shapes the model must produce:

```json
{"reasoning": "...", "function": "get_weather", "arguments": {"location": "Lagos"}}
{"reasoning": "...", "function": null, "arguments": {}, "reason": "Ordering food is not supported."}
```

CLI `--json` output: `status` (`ok` | `out_of_scope` | `error`), `function`, `arguments`, `reasoning`,
`attempts`, plus `result`, `reason` or `error` depending on the status.

Ollama's `format: "json"` mode is requested first. If the server rejects it (some models return
HTTP 501), the client falls back to prompt-only enforcement.

## Tests

```bash
python3 -m unittest -v
```

A scripted fake model covers parsing, every validation rule, the retry-then-success path,
giving up after retries, and out-of-scope handling, all without Ollama.
