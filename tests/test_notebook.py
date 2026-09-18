"""
The student notebook: saving (button and chat), the three kinds of entry, revision sheets,
and the notebook staying out of the tutor's context.

Target: apu/notebook/, apu/tools/notebook.py,
        apu/runtime/agent.py (save_to_notebook tool round)
"""

from datetime import UTC, datetime, timedelta

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from apu import config
from apu.guardrails import guard
from apu.guardrails.guard import TopicalGuard
from apu.notebook import service
from apu.notebook.store import EntryKind, EntryOrigin, NotebookStore, new_entry
from tests.conftest import TEST_SESSION_ID, make_classifier_llm, tool_call_reply

T0 = datetime(2026, 9, 17, 8, 0, tzinfo=UTC)
PREVIOUS_ANSWER = (
    "Great question! To add 1/4 and 1/6, put both over 12: 3/12 + 2/12 = 5/12. "
    "Can you try 1/3 + 1/6?"
)
KEY_POINTS = "- Put the fractions over a common denominator.\n- Add the numerators: 1/4 + 1/6 = 5/12."


def entry(student="eleve-aya", kind="full", text="A fraction is a share.", minutes=0,
          class_level="6eme", subject="math"):
    return new_entry(student, class_level, subject, kind, text, text, "button",
                     created_at=T0 + timedelta(minutes=minutes))


@pytest.fixture
def store(akili_paths):
    return NotebookStore()


# ── the store ────────────────────────────────────────────────────────────────

def test_entries_are_per_student_oldest_first_and_filterable_by_course(store):
    store.add(entry(text="second", minutes=5))
    store.add(entry(text="first", minutes=0))
    store.add(entry(text="history note", subject="history", minutes=1))
    store.add(entry(student="eleve-koffi", text="not Aya's"))

    assert [e.text for e in store.entries("eleve-aya")] == ["first", "history note", "second"]
    assert [e.text for e in store.entries("eleve-aya", "6eme", "math")] == ["first", "second"]
    assert store.entries("eleve-aya")[0].created_at == T0


def test_a_student_can_only_delete_their_own_entries(store):
    mine = store.add(entry())
    assert store.delete("eleve-koffi", mine.entry_id) is False
    assert store.delete("eleve-aya", mine.entry_id) is True
    assert store.entries("eleve-aya") == []


@pytest.mark.parametrize("kind,text,message", [
    ("full", "   ", "needs some text"),
    ("full", "x" * (config.NOTEBOOK_MAX_ENTRY_CHARS + 1), "at most"),
    ("summary", "text", "not a valid EntryKind"),
])
def test_an_invalid_entry_is_refused(kind, text, message):
    with pytest.raises(ValueError, match=message):
        new_entry("eleve-aya", "6eme", "math", kind, text, text, "button")


def test_the_notebook_lives_apart_from_the_tutoring_memory(store):
    store.add(entry())
    assert config.NOTEBOOK_DB_PATH != config.ESCALATION_DB_PATH
    assert not config.NOTEBOOK_DB_PATH.startswith(config.LANCE_DB_PATH)
    assert not config.NOTEBOOK_DB_PATH.startswith(config.METADATA_LINKS_PATH)


# ── saving, the three kinds ──────────────────────────────────────────────────

async def test_a_full_answer_is_kept_as_written_without_a_model_call(store, fake_nebius):
    saved = await service.save_entry(student_id="eleve-aya", class_level="6eme", subject="math",
                                     kind="full", answer=PREVIOUS_ANSWER, origin="button", store=store)
    assert saved.text == PREVIOUS_ANSWER and saved.kind is EntryKind.FULL
    assert fake_nebius.calls == []


async def test_key_points_are_condensed_by_nemotron_nano_in_plain_text(store, fake_nebius, no_network):
    fake_nebius.extraction_replies = [f"  {KEY_POINTS}\n"]
    saved = await service.save_entry(student_id="eleve-aya", class_level="6eme", subject="math",
                                     kind="key_points", answer=PREVIOUS_ANSWER, origin="button", store=store)

    assert saved.text == KEY_POINTS and saved.source_answer == PREVIOUS_ANSWER
    [call] = fake_nebius.calls
    assert call["model"] == config.EXTRACTION_MODEL and call["kwargs"]["temperature"] == 0.0
    prompt = call["messages"][0]["content"]
    assert PREVIOUS_ANSWER in prompt and "no LaTeX" in prompt


async def test_an_excerpt_keeps_the_chosen_passage(store, fake_nebius):
    saved = await service.save_entry(student_id="eleve-aya", class_level="6eme", subject="math",
                                     kind="excerpt", answer=PREVIOUS_ANSWER, excerpt="3/12 + 2/12 = 5/12",
                                     origin="button", store=store)
    assert saved.text == "3/12 + 2/12 = 5/12" and fake_nebius.calls == []
    with pytest.raises(ValueError, match="passage"):
        await service.save_entry(student_id="eleve-aya", class_level="6eme", subject="math",
                                 kind="excerpt", answer=PREVIOUS_ANSWER, excerpt=" ", origin="button", store=store)


async def test_an_empty_condensation_is_retried_once_then_reported(store, fake_nebius, no_network):
    fake_nebius.extraction_replies = [None]
    with pytest.raises(service.NotebookUnavailable):
        await service.save_entry(student_id="eleve-aya", class_level="6eme", subject="math",
                                 kind="key_points", answer=PREVIOUS_ANSWER, origin="button", store=store)
    assert len(fake_nebius.calls) == 2 and store.entries("eleve-aya") == []


def test_a_notebook_has_a_ceiling(store, monkeypatch):
    """Each save is a model call and a row on disk, both driven by the student."""
    from apu.notebook.store import NotebookFull

    monkeypatch.setattr(config, "NOTEBOOK_MAX_ENTRIES_PER_STUDENT", 3)
    for index in range(3):
        store.add(entry(text=f"note {index}", minutes=index))

    with pytest.raises(NotebookFull, match="full"):
        store.add(entry(text="one too many", minutes=9))
    assert store.count("eleve-aya") == 3
    assert store.count("eleve-koffi") == 0, "the ceiling is per student"


# ── revision sheets ──────────────────────────────────────────────────────────

def test_selected_entries_are_printed_as_written_with_a_heading_each():
    text = service.entries_as_text([entry(text="**Rule**: same denominator"), entry(kind="excerpt", text="5/12")])
    assert text == "1. math (6eme), full answer\nRule: same denominator\n\n2. math (6eme), excerpt\n5/12"


async def test_the_summary_is_written_by_nemotron_super_from_the_selected_entries(fake_nebius, no_network):
    fake_nebius.main_replies = ["Fractions\nAdd over a common denominator."]
    summary = await service.summarize_entries([entry(text="note one"), entry(text="note two")])

    assert summary == "Fractions\nAdd over a common denominator."
    [call] = fake_nebius.calls
    assert call["model"] == config.MAIN_MODEL
    prompt = call["messages"][0]["content"]
    assert "note one" in prompt and "note two" in prompt and "revision sheet" in prompt
    with pytest.raises(ValueError, match="nothing"):
        await service.summarize_entries([])


async def test_a_sheet_cannot_send_the_whole_notebook_to_the_model(fake_nebius, monkeypatch, no_network):
    monkeypatch.setattr(config, "NOTEBOOK_MAX_SHEET_ENTRIES", 3)
    with pytest.raises(ValueError, match="at most 3 entries"):
        await service.summarize_entries([entry(text=f"note {index}") for index in range(4)])

    monkeypatch.setattr(config, "NOTEBOOK_MAX_SHEET_CHARS", 50)
    with pytest.raises(ValueError, match="too long"):
        await service.summarize_entries([entry(text="x" * 40), entry(text="y" * 40)])
    assert fake_nebius.calls == [], "nothing was sent"


# ── saving from the chat ─────────────────────────────────────────────────────

def _state(query, **extra):
    return {
        "messages": [HumanMessage(content=query)],
        "session_id": TEST_SESSION_ID,
        "agent_id": "agent-test", "class_level": "6eme", "subject": "math",
        "memory_only_mode": False, "needs_new_block": "False", "proposed_block_config": {},
        **extra,
    }


@pytest.fixture
def turn(akili_paths, no_network, stub_embeddings, fake_nebius):
    return fake_nebius


async def test_asking_the_tutor_saves_the_key_points_of_its_previous_answer(turn):
    import apu.runtime.agent as agent
    turn.main_replies = [tool_call_reply("save_to_notebook", {"what": "key_points"}), "Saved!"]
    turn.extraction_replies = [KEY_POINTS, "{}"]

    out = await agent.planner_node(_state("Save the key points of your answer",
                                          previous_answer=PREVIOUS_ANSWER))

    [saved] = NotebookStore().entries("eleve-test")
    assert (saved.kind, saved.text, saved.origin) == (EntryKind.KEY_POINTS, KEY_POINTS, EntryOrigin.CHAT)
    assert (saved.class_level, saved.subject, saved.source_answer) == ("6eme", "math", PREVIOUS_ANSWER)
    assert out["notebook_saves"] == [{"entry_id": saved.entry_id, "kind": "key_points"}]
    assert out["messages"][0].content == "Saved!"

    first, second = turn.main_calls
    assert [tool["function"]["name"] for tool in first["kwargs"]["tools"]] == ["web_search", "save_to_notebook"]
    assert "NOTEBOOK:" in first["messages"][0]["content"]
    assert KEY_POINTS in second["messages"][-1]["content"], "the tutor can confirm what was kept"


async def test_the_student_comes_from_the_guard_session_not_the_model(turn):
    import apu.runtime.agent as agent
    turn.main_replies = [tool_call_reply("save_to_notebook", {"what": "full", "student_id": "eleve-koffi"}), "ok"]
    turn.extraction_replies = ["{}"]

    await agent.planner_node(_state("Keep that", previous_answer=PREVIOUS_ANSWER))
    assert [e.student_id for e in NotebookStore().entries("eleve-test")] == ["eleve-test"]
    assert NotebookStore().entries("eleve-koffi") == []


async def test_without_a_previous_answer_in_state_the_last_ai_message_is_used(turn):
    import apu.runtime.agent as agent
    turn.main_replies = [tool_call_reply("save_to_notebook", {"what": "full"}), "ok"]
    turn.extraction_replies = ["{}"]
    state = _state("Save that")
    state["messages"] = [HumanMessage(content="How do I add 1/4 and 1/6?"), AIMessage(content=PREVIOUS_ANSWER),
                         HumanMessage(content="Save that")]

    await agent.planner_node(state)
    assert [e.text for e in NotebookStore().entries("eleve-test")] == [PREVIOUS_ANSWER]


@pytest.mark.parametrize("arguments,previous,expected", [
    ({"what": "everything"}, PREVIOUS_ANSWER, "'what' must be one of"),
    ({"what": "full"}, "", "no previous answer"),
    ({"what": "excerpt"}, PREVIOUS_ANSWER, "passage"),
])
async def test_unusable_save_requests_are_explained_to_the_model_and_nothing_is_saved(turn, arguments, previous, expected):
    import apu.runtime.agent as agent
    turn.main_replies = [tool_call_reply("save_to_notebook", arguments), "ok"]
    turn.extraction_replies = ["{}"]

    out = await agent.planner_node(_state("Save it", previous_answer=previous))

    assert expected in turn.main_calls[1]["messages"][-1]["content"]
    assert NotebookStore().entries("eleve-test") == [] and out["notebook_saves"] == []


async def test_a_failed_condensation_is_reported_and_the_tutor_says_so(turn):
    import apu.runtime.agent as agent
    turn.main_replies = [tool_call_reply("save_to_notebook", {"what": "key_points"}), "Sorry, not saved."]
    turn.extraction_replies = [ConnectionError("Token Factory down"), "{}"]

    out = await agent.planner_node(_state("Save the key points", previous_answer=PREVIOUS_ANSWER))

    assert "failed" in turn.main_calls[1]["messages"][-1]["content"]
    assert any("Token Factory down" in problem for problem in out["tool_problems"])
    assert NotebookStore().entries("eleve-test") == []


async def test_an_unvalidated_turn_cannot_save(turn, monkeypatch):
    import apu.runtime.agent as agent
    monkeypatch.setattr(guard, "_guard", TopicalGuard(llm=make_classifier_llm(verdict_for=lambda p: "?")))
    turn.main_replies = ["ok"]
    turn.extraction_replies = ["{}"]

    await agent.planner_node(_state("Save that", previous_answer=PREVIOUS_ANSWER))
    [only_call] = turn.main_calls
    assert "tools" not in only_call["kwargs"] and "NOTEBOOK:" not in only_call["messages"][0]["content"]


async def test_a_save_needs_the_sessions_current_validated_turn(akili_paths, fake_nebius):
    from apu.guardrails.session import sessions
    from apu.tools.notebook import NotebookGateError, save_from_chat

    with pytest.raises(NotebookGateError):
        await save_from_chat({"what": "full"}, validated_turn=None, previous_answer=PREVIOUS_ANSWER,
                             class_level="6eme", subject="math")
    decision = await guard.get_topical_guard().check(TEST_SESSION_ID, "Explain fractions")
    sessions.get(TEST_SESSION_ID).invalidate_current_turn()
    with pytest.raises(NotebookGateError, match="current validated turn"):
        await save_from_chat({"what": "full"}, validated_turn=decision.validated_turn,
                             previous_answer=PREVIOUS_ANSWER, class_level="6eme", subject="math")


async def test_the_tutor_never_reads_the_notebook(turn):
    """Saved text must not reach any model call of a later turn: not the prompt, not a tool result."""
    import apu.runtime.agent as agent
    NotebookStore().add(new_entry("eleve-test", "6eme", "math", "full", "NOTEBOOK-MARKER-7f3a",
                                  "NOTEBOOK-MARKER-7f3a", "button"))
    turn.main_replies = ["A fraction is a share of a whole."]
    turn.extraction_replies = ["{}"]

    await agent.planner_node(_state("What is a fraction?"))

    assert turn.calls and all("NOTEBOOK-MARKER-7f3a" not in str(call["messages"]) for call in turn.calls)


# ── demo data ────────────────────────────────────────────────────────────────

def test_the_demo_notebook_is_seeded_and_reset_with_the_rest(akili_paths):
    import os

    from apu.demo import seed

    assert seed.seed_notebook(progress=lambda message: None) == len(seed.DEMO_NOTEBOOK)
    entries = NotebookStore().entries(config.DEMO_STUDENT_ID, "6eme", "math")
    assert {e.kind for e in entries} == set(EntryKind)

    assert config.NOTEBOOK_DB_PATH in seed._state_paths()
    seed.reset_demo_data(progress=lambda message: None)
    assert not os.path.exists(config.NOTEBOOK_DB_PATH)
