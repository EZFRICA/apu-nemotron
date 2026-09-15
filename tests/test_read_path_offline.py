"""
How far the read path gets with no network.

Target: apu/runtime/agent.py (planner_node), apu/mmu/dll.py (update_node_content)

Inference goes to the fake Nebius client; everything else (embedding, LanceDB, the
DLL, L1) runs for real with INET sockets blocked.
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from apu.mmu import dll as mmu
from tests.conftest import V_A


def _state(query="I don't understand fractions"):
    return {
        "messages": [HumanMessage(content=query)],
        "agent_id": "agent-test",
        "class_level": "6eme",
        "subject": "math",
        "memory_only_mode": False,
        "needs_new_block": "False",
        "proposed_block_config": {},
    }


# ── the load-bearing tests: where does it die offline ────────────────────────

async def test_the_read_path_reaches_generation_offline(
    akili_paths, no_network, real_local_embedder, fake_nebius, monkeypatch
):
    """
    Uses the REAL ONNX embedder, so it proves the model runs with no network
    rather than proving a fixture works.
    """
    import apu.runtime.agent as agent

    reached = {"search": False}
    real_search = mmu.search_memory

    async def spy_search(*a, **k):
        reached["search"] = True
        return await real_search(*a, **k)

    monkeypatch.setattr(mmu, "search_memory", spy_search)
    fake_nebius.main_replies = ["answer"]
    fake_nebius.extraction_replies = ["{}"]

    await agent.planner_node(_state())

    assert reached["search"] is True, "retrieval was reached with sockets blocked"
    assert len(fake_nebius.main_calls) == 1, "the turn got all the way to generation"


async def test_generation_is_the_only_remaining_network_dependency(
    akili_paths, no_network, real_local_embedder, fake_nebius, monkeypatch
):
    """
    Embeddings and retrieval are local, inference is Nebius. When the endpoint is
    unreachable the turn fails at generation, after retrieval, not before it.
    """
    import apu.runtime.agent as agent

    reached = {"search": False}
    real_search = mmu.search_memory

    async def spy_search(*a, **k):
        reached["search"] = True
        return await real_search(*a, **k)

    monkeypatch.setattr(mmu, "search_memory", spy_search)
    fake_nebius.main_replies = [OSError("network is unreachable")]

    with pytest.raises(OSError):
        await agent.planner_node(_state())

    assert reached["search"] is True, "retrieval ran before the failure"
    assert len(fake_nebius.main_calls) == 1, "the failure came from generation"
    assert fake_nebius.extraction_calls == [], "no write-back for a failed turn"


async def test_everything_after_the_embedding_works_offline(
    akili_paths, no_network, stub_embeddings, fake_nebius
):
    import apu.runtime.agent as agent

    fake_nebius.main_replies = ["Great question. What do you already know about halves?"]
    fake_nebius.extraction_replies = [json.dumps({
        "student_profile": "",
        "learning_preferences": "",
        "current_session": "The student is studying fractions in 6eme math.",
    })]

    out = await agent.planner_node(_state())

    assert len(fake_nebius.main_calls) == 1
    assert len(fake_nebius.extraction_calls) == 1
    assert isinstance(out["messages"][0], AIMessage)
    assert out["messages"][0].content.startswith("Great question")
    assert out["needs_new_block"] == "False"
    assert out["memory_problems"] == []

    dll = await mmu.load_dll()
    assert dll["nodes"]["current_session"]["content"] == (
        "The student is studying fractions in 6eme math."
    )


async def test_course_context_reaches_the_prompt(
    akili_paths, no_network, stub_embeddings, fake_nebius
):
    import apu.runtime.agent as agent
    from apu.storage import lance_driver

    db = lance_driver.get_db()
    db.create_table("edu_registry", data=[{
        "id": "ch_fractions", "chapter": "ch_fractions",
        "content": "A fraction represents a part of a whole.",
        "block_type": "manual_chapter", "class_level": "6eme", "subject": "math",
        "vector": list(V_A), "updated_at": "2026-01-01T00:00:00",
    }])

    fake_nebius.main_replies = ["ok"]
    fake_nebius.extraction_replies = ["{}"]

    await agent.planner_node(_state())

    system_text = fake_nebius.main_calls[0]["messages"][0]["content"]
    assert "COURSE CONTEXT (Search Results):" in system_text
    after = system_text.split("COURSE CONTEXT (Search Results):", 1)[1]
    course_context = after.split("STUDENT MEMORY", 1)[0]
    assert "A fraction represents a part of a whole." in course_context
    assert "--- ch_fractions ---" in course_context


async def test_the_system_prompt_is_sent_with_the_system_role(
    akili_paths, no_network, stub_embeddings, fake_nebius
):
    import apu.runtime.agent as agent

    fake_nebius.main_replies = ["ok"]
    fake_nebius.extraction_replies = ["{}"]

    await agent.planner_node(_state())

    first = fake_nebius.main_calls[0]["messages"][0]
    assert first["role"] == "system"
    assert "You are Akili" in first["content"]


# ── the conversation window ──────────────────────────────────────────────────

def _log(n):
    """The UI's own transcript: n exchanges, oldest first."""
    out = []
    for i in range(1, n + 1):
        out.append({"role": "user", "content": f"q{i}"})
        out.append({"role": "assistant", "content": f"a{i}"})
    return out


def test_zero_exchanges_is_memory_only():
    from apu.runtime.agent import build_message_window

    window = build_message_window(_log(3), "now", 0)
    assert [m.content for m in window] == ["now"]


def test_the_window_keeps_the_most_recent_exchanges():
    from apu.runtime.agent import build_message_window

    window = build_message_window(_log(5), "now", 2)
    assert [m.content for m in window] == ["q4", "a4", "q5", "a5", "now"]


def test_roles_survive_the_round_trip():
    from apu.runtime.agent import build_message_window

    window = build_message_window(_log(1), "now", 1)
    assert isinstance(window[0], HumanMessage)
    assert isinstance(window[1], AIMessage)
    assert isinstance(window[2], HumanMessage)


def test_asking_for_more_history_than_exists_is_safe():
    from apu.runtime.agent import build_message_window

    window = build_message_window(_log(2), "now", 10)
    assert [m.content for m in window] == ["q1", "a1", "q2", "a2", "now"]
    assert build_message_window([], "first question", 5)[0].content == "first question"


async def test_the_window_reaches_the_model_with_openai_roles(
    akili_paths, no_network, stub_embeddings, fake_nebius
):
    """
    What build_message_window returns is what the model is sent, after the
    system prompt, converted to the roles the chat completions API expects.
    """
    import apu.runtime.agent as agent
    from apu.runtime.agent import build_message_window

    fake_nebius.main_replies = ["ok"]
    fake_nebius.extraction_replies = ["{}"]

    state = _state()
    state["messages"] = build_message_window(_log(2), "and now?", 2)
    await agent.planner_node(state)

    sent = fake_nebius.main_calls[0]["messages"][1:]      # [0] is the system prompt
    assert [m["content"] for m in sent] == ["q1", "a1", "q2", "a2", "and now?"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user", "assistant", "user"]
