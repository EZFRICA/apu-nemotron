"""
The runtime wired to Nebius: model routing, graph shape, import-time behaviour.

Target: apu/runtime/agent.py, apu/inference/nebius_client.py (used, not modified)

Replaces Akili's test_import_time.py and test_tool_graph.py, which pinned
llm_provider and the TEU loop, neither of which exists in this port.
"""

import os
import pathlib
import re
import subprocess
import sys
import textwrap

from langchain_core.messages import HumanMessage

from apu import config
from apu.mmu import dll as mmu

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _state(query="Combien font 3/4 + 2/5 ?"):
    return {
        "messages": [HumanMessage(content=query)],
        "agent_id": "agent-test", "class_level": "6eme", "subject": "math",
        "memory_only_mode": False, "needs_new_block": "False",
        "proposed_block_config": {},
    }


def _run(code, env_overrides):
    env = dict(os.environ)
    env.update(env_overrides)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, cwd=REPO_ROOT, env=env, timeout=180,
    )


# ── import-time behaviour ────────────────────────────────────────────────────

def test_importing_the_agent_without_a_nebius_key_succeeds():
    """
    nebius_client raises at import without a key. The agent imports it lazily, so
    a missing key fails the first question, not the start of the app.
    """
    proc = _run(
        """
        try:
            import apu.runtime.agent
            print("IMPORT_OK")
        except Exception as e:
            print("IMPORT_FAILED:", type(e).__name__, e)
        """,
        {"NEBIUS_API_KEY": ""},
    )
    assert "IMPORT_OK" in proc.stdout, proc.stdout + proc.stderr


def test_the_nebius_client_itself_still_fails_loudly_without_a_key():
    """Pins why the lazy import above is needed."""
    proc = _run(
        """
        try:
            import apu.inference.nebius_client
            print("IMPORT_OK")
        except RuntimeError as e:
            print("RAISED:", e)
        """,
        {"NEBIUS_API_KEY": ""},
    )
    assert "RAISED: NEBIUS_API_KEY is not set" in proc.stdout, proc.stdout + proc.stderr


def test_no_module_imports_an_ollama_or_google_client():
    """Every inference call goes through nebius_client; nothing else may creep back."""
    forbidden = re.compile(
        r"^\s*(from|import)\s+(ollama|langchain_ollama|langchain_google_genai|"
        r"google\.generativeai|google\.genai)\b",
        re.M,
    )
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in (REPO_ROOT / "apu").rglob("*.py")
        if forbidden.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []


# ── model routing ────────────────────────────────────────────────────────────

async def test_a_turn_sends_the_answer_to_super_and_the_extraction_to_nano(
    akili_paths, no_network, stub_embeddings, fake_nebius
):
    import apu.runtime.agent as agent

    fake_nebius.main_replies = ["Cela fait 23."]
    fake_nebius.extraction_replies = ['{"current_session": "Fractions en 6eme."}']

    await agent.planner_node(_state())

    assert [c["model"] for c in fake_nebius.calls] == [
        config.MAIN_MODEL, config.EXTRACTION_MODEL,
    ], "answer first, then the write-back, each on its own model"
    assert fake_nebius.main_calls[0]["kwargs"] == {"temperature": 0.7}


async def test_the_write_back_is_still_inline(
    akili_paths, no_network, stub_embeddings, fake_nebius
):
    """
    Pins Akili's behaviour: the extraction runs before planner_node returns, on the
    critical path. Moving it to the background is an open decision (COLD_PATH.md in
    Akili); invert this test when it is taken.
    """
    import apu.runtime.agent as agent

    fake_nebius.main_replies = ["ok"]
    fake_nebius.extraction_replies = ['{"current_session": "Fractions en 6eme."}']

    await agent.planner_node(_state())

    assert len(fake_nebius.extraction_calls) == 1
    dll = await mmu.load_dll()
    assert dll["nodes"]["current_session"]["content"] == "Fractions en 6eme."


async def test_an_empty_answer_is_not_the_string_none(
    akili_paths, no_network, stub_embeddings, fake_nebius
):
    import apu.runtime.agent as agent

    fake_nebius.main_replies = [None]
    fake_nebius.extraction_replies = ["{}"]

    out = await agent.planner_node(_state())
    assert out["messages"][0].content == ""


# ── the graph ────────────────────────────────────────────────────────────────

def test_the_graph_is_a_single_planner_node():
    from apu.runtime import agent

    graph = agent.create_agent_graph().get_graph()
    assert "Planner" in set(graph.nodes)
    assert "Tools" not in set(graph.nodes)
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("Planner", "__end__") in edges


async def test_a_turn_runs_end_to_end_through_the_graph(
    akili_paths, no_network, stub_embeddings, fake_nebius
):
    from apu.runtime import agent

    fake_nebius.main_replies = ["Cela fait 23."]
    fake_nebius.extraction_replies = ['{"current_session": "Fractions en 6eme."}']

    await mmu.init_dll()
    result = await agent.create_agent_graph().ainvoke(_state())

    assert result["messages"][-1].content == "Cela fait 23."
    assert result["memory_problems"] == []
    assert "Combien font 3/4 + 2/5 ?" in fake_nebius.extraction_calls[0]["messages"][0]["content"]
