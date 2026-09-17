#!/usr/bin/env python3
"""
Smoke test: does native OpenAI-style tool calling work for Nemotron on Token Factory?

Sends ONE chat completion with a trivial JSON Schema tool and prints the raw HTTP
response body exactly as the API returned it, then the fields that decide the
integration method, read separately:

  - message.content
  - message.reasoning_content (Nemotron reasoning models may put text there)
  - message.tool_calls
  - choices[0].finish_reason

The decision it informs (see HACKATHON.md, "Tool calling on Token Factory"):
  Method A: tool_calls is populated -> web search plugs in as a standard tool.
  Method B: HTTP 400, or no tool_calls with the decision only in text / reasoning
            -> prompt the model to emit a JSON action block and parse it ourselves.

It deliberately bypasses apu.inference.nebius_client: call_main_model returns only
message.content, which is exactly the field that would hide the answer here.

Usage:
    uv run python scripts/smoke_test_tool_calling.py
    uv run python scripts/smoke_test_tool_calling.py --model nvidia/nemotron-3-nano-30b-a3b
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import APIStatusError, OpenAI  # noqa: E402

from apu import config  # noqa: E402

# A tool the model has every reason to call and no way to answer without: the
# question asks for a value only the tool can provide.
LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_school_fact",
        "description": "Look up a short factual definition for a school topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to look up, e.g. 'photosynthesis'.",
                },
            },
            "required": ["topic"],
        },
    },
}

MESSAGES = [
    {
        "role": "user",
        "content": "Use the lookup_school_fact tool to look up 'photosynthesis'. "
                   "Do not answer from memory.",
    },
]


def _print_section(title: str, value) -> None:
    print(f"\n=== {title} ===")
    if isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2, ensure_ascii=False))
    else:
        print(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config.MAIN_MODEL)
    args = parser.parse_args()

    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        print("NEBIUS_API_KEY is not set (.env).", file=sys.stderr)
        return 2

    client = OpenAI(base_url=config.NEBIUS_BASE_URL, api_key=api_key, timeout=180)

    request = {
        "model": args.model,
        "messages": MESSAGES,
        "tools": [LOOKUP_TOOL],
        "tool_choice": "auto",
    }
    _print_section("REQUEST (sent to " + config.NEBIUS_BASE_URL + ")", request)

    try:
        raw = client.chat.completions.with_raw_response.create(**request)
    except APIStatusError as error:
        # The rejection itself is the result for Method B, so show all of it.
        _print_section("HTTP STATUS", error.status_code)
        _print_section("RAW ERROR BODY", error.response.text)
        return 1

    _print_section("HTTP STATUS", raw.http_response.status_code)
    try:
        _print_section("RAW RESPONSE BODY", json.loads(raw.http_response.text))
    except json.JSONDecodeError:
        _print_section("RAW RESPONSE BODY (not JSON)", raw.http_response.text)

    completion = raw.parse()
    choice = completion.choices[0]
    message = choice.message
    # reasoning_content is not part of the OpenAI schema, so the SDK keeps it in
    # model_extra rather than as an attribute; check both.
    reasoning = getattr(message, "reasoning_content", None) or (message.model_extra or {}).get(
        "reasoning_content"
    )

    _print_section("finish_reason", choice.finish_reason)
    _print_section("message.content", repr(message.content))
    _print_section("message.reasoning_content", repr(reasoning))
    _print_section(
        "message.tool_calls",
        [call.model_dump() for call in message.tool_calls] if message.tool_calls else None,
    )

    verdict = (
        "tool_calls populated -> native tool calling works (Method A)"
        if message.tool_calls
        else "no tool_calls -> native tool calling not usable as-is (Method B)"
    )
    _print_section("VERDICT", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
