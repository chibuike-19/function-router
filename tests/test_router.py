import json
import unittest
from datetime import date

from router.router import build_system_prompt, route
from router.validator import ParseError, parse_json, validate


class FakeClient:
    """Returns scripted responses in order and records what it was sent."""

    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages):
        self.calls.append([dict(m) for m in messages])
        return self.responses.pop(0)


def call(fn, args, reasoning="fits"):
    return json.dumps({"reasoning": reasoning, "function": fn, "arguments": args})


class ParseTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_json('{"a": 1}'), {"a": 1})

    def test_code_fence_and_chatter(self):
        self.assertEqual(parse_json('Sure!\n```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json('Here you go: {"a": 1} hope that helps'), {"a": 1})

    def test_garbage(self):
        with self.assertRaises(ParseError):
            parse_json("I cannot do that")
        with self.assertRaises(ParseError):
            parse_json('{"a": ')
        with self.assertRaises(ParseError):
            parse_json("[1, 2]")


class ValidateTests(unittest.TestCase):
    def test_valid_calls(self):
        self.assertEqual(validate(json.loads(call("get_weather", {"location": "Lagos"}))), [])
        self.assertEqual(
            validate(json.loads(call("send_email", {"to": "a@b.com", "subject": "Hi", "body": "Yo"}))), []
        )
        self.assertEqual(
            validate(json.loads(call("book_appointment", {"service": "dentist", "date": "2026-10-27", "time": "14:00"}))),
            [],
        )

    def test_out_of_scope_valid(self):
        obj = {"reasoning": "no tool", "function": None, "arguments": {}, "reason": "can't order pizza"}
        self.assertEqual(validate(obj), [])

    def test_errors(self):
        cases = {
            "unknown function": call("order_pizza", {}),
            "missing required argument 'location'": call("get_weather", {}),
            "unknown argument 'city'": call("get_weather", {"location": "X", "city": "X"}),
            "must be one of": call("get_weather", {"location": "X", "unit": "kelvin"}),
            "valid email": call("send_email", {"to": "bob", "subject": "s", "body": "b"}),
            "YYYY-MM-DD": call("book_appointment", {"service": "s", "date": "tomorrow", "time": "14:00"}),
            "HH:MM": call("book_appointment", {"service": "s", "date": "2026-10-27", "time": "2pm"}),
            "must be a string": call("get_weather", {"location": 42}),
            "'reasoning'": json.dumps({"function": "get_weather", "arguments": {"location": "X"}}),
            "'reason' must explain": json.dumps({"reasoning": "x", "function": None, "arguments": {}}),
        }
        for needle, raw in cases.items():
            with self.subTest(needle):
                errors = validate(json.loads(raw))
                self.assertTrue(any(needle in e for e in errors), errors)


class RouteTests(unittest.TestCase):
    def test_happy_path_executes_mock(self):
        client = FakeClient(call("get_weather", {"location": "Lagos", "unit": "celsius"}))
        res = route("weather in Lagos?", client)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.function, "get_weather")
        self.assertEqual(res.arguments, {"location": "Lagos", "unit": "celsius"})
        self.assertEqual(res.result["location"], "Lagos")
        self.assertEqual(len(res.attempts), 1)

    def test_system_prompt_is_sent(self):
        client = FakeClient(call("get_weather", {"location": "Lagos"}))
        route("weather?", client)
        system = client.calls[0][0]
        self.assertEqual(system["role"], "system")
        for name in ("get_weather", "send_email", "book_appointment"):
            self.assertIn(name, system["content"])

    def test_retry_then_success(self):
        client = FakeClient(
            "not json at all",
            call("book_appointment", {"service": "dentist", "date": "tomorrow", "time": "14:00"}),
            call("book_appointment", {"service": "dentist", "date": "2026-10-27", "time": "14:00"}),
        )
        res = route("dentist tomorrow 2pm", client)
        self.assertEqual(res.status, "ok")
        self.assertEqual(len(res.attempts), 3)
        # The validation errors were fed back to the model.
        correction = client.calls[2][-1]["content"]
        self.assertIn("YYYY-MM-DD", correction)

    def test_gives_up_after_retries(self):
        client = FakeClient("nope", "still nope", "nope again")
        res = route("x", client, max_retries=2)
        self.assertEqual(res.status, "error")
        self.assertEqual(len(res.attempts), 3)
        self.assertIn("after 3 attempts", res.error)
        json.dumps(res.to_dict())  # still serialisable

    def test_out_of_scope(self):
        client = FakeClient(
            json.dumps({"reasoning": "no tool orders food", "function": None, "arguments": {}, "reason": "Ordering food is not supported."})
        )
        res = route("order me a pizza", client)
        self.assertEqual(res.status, "out_of_scope")
        self.assertIsNone(res.function)
        self.assertEqual(res.to_dict()["reason"], "Ordering food is not supported.")

    def test_no_exec(self):
        client = FakeClient(call("send_email", {"to": "a@b.com", "subject": "s", "body": "b"}))
        res = route("email", client, execute=False)
        self.assertIsNone(res.result)

    def test_today_in_prompt(self):
        self.assertIn("2026-10-26 (Monday)", build_system_prompt(date(2026, 10, 26)))


if __name__ == "__main__":
    unittest.main()
