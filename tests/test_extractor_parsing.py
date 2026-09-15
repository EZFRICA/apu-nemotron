"""
Extractor JSON parsing against realistic malformed model output.

Target: apu/runtime/agent.py (_update_student_memory), apu/core/extraction.py

Drives the real write-back with the fake Nebius client returning each payload as
the extraction model's raw output, and observes which updates land in the DLL.
"""

import json

from apu.mmu import dll as mmu

GOOD = {
    "student_profile": "The student is named Marc and plays basketball.",
    "learning_preferences": "",
    "current_session": "The student is studying fractions in 6eme math.",
}


async def _extract(payload, fake_nebius):
    """Run the real extraction path with `payload` as the model's raw output."""
    import apu.runtime.agent as agent
    fake_nebius.extraction_replies = [payload]

    dll = await mmu.init_dll()
    problems = await agent._update_student_memory("a question", "an answer", dll)

    fresh = await mmu.load_dll()
    landed = {
        nid: node.get("content")
        for nid, node in fresh["nodes"].items()
        if node.get("content")
    }
    return landed, problems


# ── the call itself ──────────────────────────────────────────────────────────

async def test_extraction_goes_to_the_nano_model_deterministically_in_json_mode(
    akili_paths, stub_embeddings, fake_nebius
):
    """
    Every Akili extractor branch was temperature 0 and asked for JSON. The port
    keeps both, and sends the call to the extraction model, never the main one.
    """
    from apu import config

    await _extract(json.dumps(GOOD), fake_nebius)

    assert fake_nebius.main_calls == []
    [call] = fake_nebius.extraction_calls
    assert call["model"] == config.EXTRACTION_MODEL
    assert call["kwargs"]["temperature"] == 0.0
    assert call["kwargs"]["response_format"] == {"type": "json_object"}
    assert call["messages"][0]["role"] == "user"
    assert "a question" in call["messages"][0]["content"]


async def test_an_unreachable_extractor_is_reported_not_raised(
    akili_paths, stub_embeddings, fake_nebius
):
    landed, problems = await _extract(ConnectionError("endpoint down"), fake_nebius)
    assert landed == {}
    assert any("could not be reached" in p for p in problems), problems


async def test_an_empty_completion_is_reported(akili_paths, stub_embeddings, fake_nebius):
    """message.content can be None; it must not be stored as the string 'None'."""
    landed, problems = await _extract(None, fake_nebius)
    assert landed == {}
    assert any("empty response" in p for p in problems), problems


# ── shapes that parse ────────────────────────────────────────────────────────

async def test_bare_json_object_parses(akili_paths, stub_embeddings, fake_nebius):
    landed, errors = await _extract(json.dumps(GOOD), fake_nebius)
    assert landed == {
        "student_profile": GOOD["student_profile"],
        "current_session": GOOD["current_session"],
    }
    assert errors == []


async def test_fenced_json_parses(akili_paths, stub_embeddings, fake_nebius):
    payload = "```json\n" + json.dumps(GOOD) + "\n```"
    landed, errors = await _extract(payload, fake_nebius)
    assert "student_profile" in landed
    assert errors == []


async def test_bare_fence_without_a_language_tag_parses(
    akili_paths, stub_embeddings, fake_nebius
):
    payload = "```\n" + json.dumps(GOOD) + "\n```"
    landed, errors = await _extract(payload, fake_nebius)
    assert "student_profile" in landed
    assert errors == []


async def test_short_values_are_dropped_by_the_length_filter(
    akili_paths, stub_embeddings, fake_nebius
):
    """Values of 5 characters or fewer are dropped."""
    payload = json.dumps({"student_profile": "Marc", "current_session": "Fractions!"})
    landed, errors = await _extract(payload, fake_nebius)
    assert "student_profile" not in landed        # "Marc" is 4 chars
    assert landed["current_session"] == "Fractions!"
    assert errors == []


# ── shapes that used to be swallowed ─────────────────────────────────────────

async def test_trailing_prose_after_the_object_is_tolerated(
    akili_paths, stub_embeddings, fake_nebius
):
    payload = json.dumps(GOOD) + "\n\nI hope that helps with the extraction!"
    landed, problems = await _extract(payload, fake_nebius)
    assert landed["current_session"] == GOOD["current_session"]
    assert problems == []


async def test_leading_prose_before_the_object_is_tolerated(
    akili_paths, stub_embeddings, fake_nebius
):
    payload = "Here is the JSON you asked for:\n" + json.dumps(GOOD)
    landed, problems = await _extract(payload, fake_nebius)
    assert landed["current_session"] == GOOD["current_session"]
    assert problems == []


async def test_a_truncated_object_keeps_its_complete_keys(
    akili_paths, stub_embeddings, fake_nebius
):
    """Hit whenever the extractor runs out of output tokens."""
    payload = '{"student_profile": "The student is named Marc and plays bask'
    landed, problems = await _extract(payload, fake_nebius)
    assert "Marc" in landed["student_profile"]


async def test_a_fence_inside_prose_is_tolerated(akili_paths, stub_embeddings, fake_nebius):
    payload = "Sure! ```json\n" + json.dumps(GOOD) + "\n``` Let me know if that works."
    landed, problems = await _extract(payload, fake_nebius)
    assert landed["current_session"] == GOOD["current_session"]
    assert problems == []


async def test_single_quoted_pseudo_json_is_reported_not_swallowed(
    akili_paths, stub_embeddings, fake_nebius
):
    payload = "{'student_profile': 'The student is named Marc', 'current_session': 'x'}"
    landed, problems = await _extract(payload, fake_nebius)
    assert landed == {}
    assert any("no JSON object" in p for p in problems), problems


async def test_a_json_array_is_searched_for_an_object(
    akili_paths, stub_embeddings, fake_nebius
):
    payload = json.dumps([GOOD])
    landed, problems = await _extract(payload, fake_nebius)
    assert landed["current_session"] == GOOD["current_session"]


async def test_an_unknown_key_is_silently_discarded(
    akili_paths, stub_embeddings, fake_nebius
):
    """update_node_content returns early for an id not in the DLL, with no log line."""
    payload = json.dumps({"favourite_colour": "The student likes blue a lot."})
    landed, errors = await _extract(payload, fake_nebius)
    assert landed == {}
    assert errors == []


async def test_a_non_string_value_is_reported_and_the_rest_survives(
    akili_paths, stub_embeddings, fake_nebius
):
    payload = json.dumps({"student_profile": {"name": "Marc"}, "current_session": "x"})
    landed, problems = await _extract(payload, fake_nebius)
    assert landed == {}
    assert any("student_profile" in p and "dict" in p for p in problems), problems


async def test_one_bad_key_does_not_discard_the_good_ones(
    akili_paths, stub_embeddings, fake_nebius
):
    payload = json.dumps({
        "student_profile": 12345,                                   # not a str
        "current_session": "The student is studying fractions.",    # valid
    })
    landed, problems = await _extract(payload, fake_nebius)
    assert landed["current_session"] == "The student is studying fractions."
    assert "student_profile" not in landed
    assert any("student_profile" in p for p in problems), problems


async def test_realistic_malformed_output_still_updates_memory(
    akili_paths, stub_embeddings, fake_nebius
):
    payload = "Here you go:\n```json\n" + json.dumps(GOOD) + "\n```\nHope that helps!"
    landed, problems = await _extract(payload, fake_nebius)
    assert landed.get("current_session") == GOOD["current_session"]
