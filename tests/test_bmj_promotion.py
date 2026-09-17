"""
Does the BMJ move-to-front actually survive a turn?

Target: apu/mmu/dll.py (search_memory / BMJ), apu/runtime/agent.py (the turn's
single DLL handle). The vector search is stubbed so the routing is exercised
independently of LanceDB; inference goes to the fake Nebius client.
"""

import json

import pytest
from langchain_core.messages import HumanMessage

from apu.mmu import dll as mmu
from apu.storage import lance_driver

EXTRACTION_REPLY = json.dumps({
    "student_profile": "",
    "learning_preferences": "",
    "current_session": "The student is studying fractions.",
})


def _state():
    return {
        "messages": [HumanMessage(content="hi")],
        "session_id": "session-test",  # opened by the autouse topical_guard fixture
        "agent_id": "agent-test", "class_level": "6eme", "subject": "math",
        "memory_only_mode": False, "needs_new_block": "False",
        "proposed_block_config": {},
    }


@pytest.fixture
def hit_on_student_profile(monkeypatch):
    """Pretend the vector search found the student_profile DLL node."""
    async def fake_search(query_vector, limit=12, class_level=None, subject=None):
        return [{
            "block_id": "student_profile",
            "chapter_id": "student_profile",
            "block_type": "fondamental",
            "certainty": 0.99,
            "content": "Marc plays basketball",
            "source_table": "user_memory",
        }]

    monkeypatch.setattr(lance_driver, "search_block_index", fake_search)


async def test_search_memory_alone_does_promote_to_head(
    akili_paths, no_network, hit_on_student_profile
):
    await mmu.init_dll()
    assert (await mmu.load_dll())["head_id"] == "current_session"

    await mmu.search_memory([0.1] * 8, "6eme", "math")

    assert (await mmu.load_dll())["head_id"] == "student_profile"


async def test_the_promoted_order_is_persisted_by_the_turn(
    akili_paths, no_network, hit_on_student_profile, stub_embeddings, fake_nebius
):
    """
    The memory write-back used to persist a pre-promotion copy of the DLL and
    revert every pointer, not only the head. Asserts the whole chain.
    """
    import apu.runtime.agent as agent

    fake_nebius.main_replies = ["an answer"]
    fake_nebius.extraction_replies = [EXTRACTION_REPLY]

    await mmu.init_dll()
    await agent.planner_node(_state())

    after = await mmu.load_dll()
    assert after["head_id"] == "student_profile", "promotion did not survive"
    assert mmu._head_to_tail_order(after) == [
        "student_profile", "current_session", "active_course", "learning_preferences",
    ]
    assert mmu._tail_to_head_order(after) == [
        "learning_preferences", "active_course", "current_session", "student_profile",
    ]
    assert after["nodes"][after["head_id"]]["prev"] is None
    assert after["nodes"][after["tail_id"]]["next"] is None


async def test_search_memory_promotes_in_the_callers_handle(
    akili_paths, no_network, hit_on_student_profile
):
    await mmu.init_dll()
    mine = await mmu.load_dll()
    assert mine["head_id"] == "current_session"

    await mmu.search_memory([0.1] * 8, "6eme", "math", dll=mine)

    assert mine["head_id"] == "student_profile"          # caller's object moved
    assert mine["nodes"]["student_profile"]["prev"] is None
    assert (await mmu.load_dll())["head_id"] == "student_profile"

    # persisting the caller's handle afterwards must not undo it
    mmu.save_dll(mine)
    assert (await mmu.load_dll())["head_id"] == "student_profile"


async def test_search_memory_without_a_handle_still_loads_its_own(
    akili_paths, no_network, hit_on_student_profile
):
    await mmu.init_dll()
    await mmu.search_memory([0.1] * 8, "6eme", "math")
    assert (await mmu.load_dll())["head_id"] == "student_profile"


async def test_bmj_promotes_at_most_one_node_per_search(akili_paths, no_network,
                                                        monkeypatch):
    """Only the first matching DLL node moves, not every match."""
    async def two_hits(query_vector, limit=12, class_level=None, subject=None):
        return [
            {"block_id": "learning_preferences", "chapter_id": "x",
             "block_type": "fondamental", "certainty": 0.99, "content": "a",
             "source_table": "user_memory"},
            {"block_id": "student_profile", "chapter_id": "y",
             "block_type": "fondamental", "certainty": 0.98, "content": "b",
             "source_table": "user_memory"},
        ]

    monkeypatch.setattr(lance_driver, "search_block_index", two_hits)
    await mmu.init_dll()
    await mmu.search_memory([0.1] * 8, "6eme", "math")

    dll = await mmu.load_dll()
    assert dll["head_id"] == "learning_preferences"
    assert dll["nodes"]["student_profile"]["prev"] is not None


async def test_course_blocks_below_threshold_are_filtered_out(akili_paths, no_network,
                                                              monkeypatch):
    async def mixed(query_vector, limit=12, class_level=None, subject=None):
        return [
            {"block_id": "ch1", "chapter_id": "ch1", "block_type": "manual_chapter",
             "certainty": 0.46, "content": "kept", "source_table": "edu_registry"},
            {"block_id": "ch2", "chapter_id": "ch2", "block_type": "manual_chapter",
             "certainty": 0.44, "content": "dropped", "source_table": "edu_registry"},
            {"block_id": "ch3", "chapter_id": "ch3", "block_type": "temp",
             "certainty": 0.49, "content": "dropped, temp needs 0.50",
             "source_table": "edu_registry"},
        ]

    monkeypatch.setattr(lance_driver, "search_block_index", mixed)
    await mmu.init_dll()
    out = await mmu.search_memory([0.1] * 8, "6eme", "math")

    assert [r["block_id"] for r in out] == ["ch1"]
