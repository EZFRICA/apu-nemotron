"""
The prompts the cloud registry installs, treated as instructions and validated as such.

Target: apu/runtime/prompts.py
"""

import json
import pathlib

import pytest

from apu import config
from apu.runtime import prompts


def write_prompts(data) -> None:
    path = pathlib.Path(prompts.prompts_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")


def test_without_a_file_the_built_in_tutor_applies(akili_paths):
    assert prompts.load_registry_prompts("6eme") == (prompts.DEFAULT_TUTOR_PERSONA, "")


def test_a_registry_persona_and_class_guidelines_are_used(akili_paths):
    write_prompts({"system_tutor": "You are Akili, a patient tutor.", "6eme": "Keep it concrete.",
                   "5eme": "Other class."})
    assert prompts.load_registry_prompts("6eme") == ("You are Akili, a patient tutor.", "Keep it concrete.")
    assert prompts.load_registry_prompts("4eme")[1] == "", "no guidelines for this class"


@pytest.mark.parametrize("data", [
    "{ not json",
    ["system_tutor", "a list is not a prompt file"],
    {"system_tutor": {"nested": "object"}},
    {"system_tutor": 42},
])
def test_a_file_that_is_not_a_prompt_file_falls_back_to_the_built_in_tutor(akili_paths, data):
    write_prompts(data)
    persona, guidelines = prompts.load_registry_prompts("6eme")
    assert persona == prompts.DEFAULT_TUTOR_PERSONA and guidelines == ""


def test_an_oversized_prompt_is_refused(akili_paths, caplog):
    """A persona long enough to crowd out the course context and the question is not loaded."""
    write_prompts({"system_tutor": "x" * (config.REGISTRY_PROMPT_MAX_CHARS + 1),
                   "6eme": "y" * (config.REGISTRY_PROMPT_MAX_CHARS + 1)})

    persona, guidelines = prompts.load_registry_prompts("6eme")

    assert persona == prompts.DEFAULT_TUTOR_PERSONA and guidelines == ""
    assert "over the" in caplog.text


async def test_a_turn_uses_the_validated_persona(akili_paths, no_network, stub_embeddings, fake_nebius):
    from langchain_core.messages import HumanMessage

    import apu.runtime.agent as agent

    write_prompts({"system_tutor": "You are Akili, a patient tutor.", "6eme": "Keep it concrete."})
    fake_nebius.main_replies = ["ok"]
    fake_nebius.extraction_replies = ["{}"]

    await agent.planner_node({
        "messages": [HumanMessage(content="Explain fractions")], "session_id": "session-test",
        "agent_id": "agent-test", "class_level": "6eme", "subject": "math",
        "memory_only_mode": False, "needs_new_block": "False", "proposed_block_config": {},
    })

    system = fake_nebius.main_calls[0]["messages"][0]["content"]
    assert "You are Akili, a patient tutor." in system and "Keep it concrete." in system
