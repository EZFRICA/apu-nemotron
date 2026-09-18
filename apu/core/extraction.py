"""
Parsing the memory-extraction model's output.

Ported unchanged from Akili (app_local/core/extraction.py). Model-agnostic on
purpose: it is what keeps a Nemotron response wrapped in prose or code fences from
silently discarding a turn's memory, exactly as it did for Gemma.

Extracted from `_update_student_memory` so it is testable without an LLM, and
made tolerant of what models actually emit. The old logic was
`raw.strip().replace("```json","").replace("```","")` followed by `json.loads`:
it handled a clean object and a cleanly fenced object, and nothing else. Leading
prose, trailing prose, a fence embedded in prose and a truncated object all
raised, and the caller's bare `except` logged one line and returned a
normal-looking turn — so memory silently stopped updating with no sign to the
student.
"""

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def _candidates(raw: str) -> list[str]:
    """Progressively more forgiving readings of the payload, best first."""
    out = [raw]

    fenced = _FENCE.search(raw)
    if fenced:
        out.append(fenced.group(1))

    # First balanced {...}, which survives prose on either side.
    start = raw.find("{")
    if start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(raw)):
            c = raw[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    out.append(raw[start:i + 1])
                    break
        else:
            # Never closed: a truncated object. Close what is open and let the
            # complete key/value pairs through rather than losing the whole turn.
            repaired = raw[start:]
            if in_str:
                repaired += '"'
            repaired = repaired.rstrip().rstrip(",")
            repaired += "}" * max(depth, 0)
            out.append(repaired)

    return out


def parse_extraction(raw) -> tuple[dict[str, str], list[str]]:
    """
    Return (updates, problems).

    `updates` holds only the keys that are usable: a non-empty string value of
    more than 5 characters. `problems` describes everything rejected, so the
    caller can surface it instead of swallowing it.

    Never raises. A parse failure yields ({}, [reason]).
    """
    problems: list[str] = []

    if isinstance(raw, list):
        # Some providers return a list of content blocks; thinking blocks carry
        # no "text" and must not be concatenated into the JSON.
        raw = " ".join(
            b.get("text", "") for b in raw if isinstance(b, dict)
        )
    if not isinstance(raw, str):
        raw = str(raw)

    if not raw.strip():
        return {}, ["extractor returned an empty response"]

    parsed = None
    for candidate in _candidates(raw):
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            parsed = value
            break

    if parsed is None:
        return {}, [f"no JSON object found in extractor output: {raw[:120]!r}"]

    updates: dict[str, str] = {}
    for key, value in parsed.items():
        # Per key, so one bad value cannot discard the good ones. The whole loop
        # used to sit inside the caller's try, making extraction all-or-nothing.
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if not isinstance(value, str):
            problems.append(f"{key}: expected a string, got {type(value).__name__}")
            continue
        if len(value.strip()) <= 5:
            continue
        updates[key] = value

    return updates, problems
