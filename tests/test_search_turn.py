"""
Web search inside a tutoring turn: native tool calls (Method A), guard-gated, cited.

Target: apu/runtime/agent.py (_answer, _run_tool_call, _generate),
        apu/inference/nebius_client.py (call_main_model_message)

Nemotron is the fake Nebius client scripted with tool_calls shaped like the smoke test's
real response; Tavily is a fake tool behind the real TavilySearch gate.
"""

import pytest
from langchain_core.messages import HumanMessage

from apu import config
from apu.guardrails import guard
from apu.guardrails.guard import TopicalGuard
from apu.tools import web_search
from tests.conftest import OFF_TOPIC_MARKER, make_classifier_llm, tool_call_reply

WIKIPEDIA = {"title": "Fraction — Wikipédia", "url": "https://fr.wikipedia.org/wiki/Fraction",
             "content": "Pour additionner deux fractions, on les met au même dénominateur."}
REDDIT = {"title": "Post", "url": "https://www.reddit.com/r/maths/1", "content": "excluded globally"}
ANSWER = "Pour additionner deux fractions, commence par trouver un dénominateur commun."


def _state(query="Comment additionner deux fractions ?", **extra):
    return {
        "messages": [HumanMessage(content=query)],
        "session_id": "session-test",
        "agent_id": "agent-test", "class_level": "6eme", "subject": "math",
        "memory_only_mode": False, "needs_new_block": "False", "proposed_block_config": {},
        **extra,
    }


@pytest.fixture
def fake_tavily(monkeypatch):
    built = []

    class Tool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.queries = []
            built.append(self)

        async def ainvoke(self, payload):
            self.queries.append(payload["query"])
            return {"results": [WIKIPEDIA, REDDIT]}

    monkeypatch.setattr(web_search, "_web_search", web_search.TavilySearch(tool_factory=Tool))
    return built


@pytest.fixture
def turn(akili_paths, no_network, stub_embeddings, fake_nebius, fake_tavily):
    return fake_nebius, fake_tavily


async def test_a_validated_turn_searches_and_cites_its_sources(turn):
    import apu.runtime.agent as agent
    nebius, tavily = turn
    nebius.main_replies = [tool_call_reply("web_search", {"query": "addition de fractions"}), ANSWER]
    nebius.extraction_replies = ["{}"]

    out = await agent.planner_node(_state())

    [tool] = tavily
    assert tool.queries == ["addition de fractions"]
    assert tool.kwargs["exclude_domains"] == [*config.GLOBAL_EXCLUDED_DOMAINS, "youtube.com"]

    first, second = nebius.main_calls
    assert "tools" in first["kwargs"]
    assistant, tool_message = second["messages"][-2:]
    assert assistant["role"] == "assistant" and assistant["content"] is None
    assert assistant["tool_calls"][0]["function"]["name"] == "web_search"
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "call-1"
    assert "https://fr.wikipedia.org/wiki/Fraction" in tool_message["content"]
    assert "reddit.com" not in tool_message["content"]

    assert out["messages"][0].content == (
        f"{ANSWER}\n\nSources :\n1. Fraction — Wikipédia — https://fr.wikipedia.org/wiki/Fraction"
    )
    assert out["sources"] == [{"title": WIKIPEDIA["title"], "url": WIKIPEDIA["url"]}]
    assert out["tool_problems"] == []

    extraction_prompt = nebius.extraction_calls[0]["messages"][0]["content"]
    assert ANSWER in extraction_prompt and "Sources :" not in extraction_prompt


async def test_without_a_search_the_answer_has_no_source_list(turn):
    import apu.runtime.agent as agent
    nebius, tavily = turn
    nebius.main_replies = [ANSWER]
    nebius.extraction_replies = ["{}"]

    out = await agent.planner_node(_state())
    assert out["messages"][0].content == ANSWER
    assert tavily == [] and out["sources"] == []


async def test_an_unvalidated_turn_is_not_offered_the_tool(turn, monkeypatch):
    """A classifier without a usable verdict lets the turn be answered, but never searched."""
    import apu.runtime.agent as agent
    nebius, tavily = turn
    monkeypatch.setattr(guard, "_guard", TopicalGuard(llm=make_classifier_llm(verdict_for=lambda p: "?")))
    nebius.main_replies = [ANSWER]
    nebius.extraction_replies = ["{}"]

    await agent.planner_node(_state())

    [only_call] = nebius.main_calls
    assert "tools" not in only_call["kwargs"]
    assert "WEB SEARCH" not in only_call["messages"][0]["content"]
    assert tavily == []


async def test_an_off_topic_turn_never_reaches_the_model_or_the_search(turn):
    import apu.runtime.agent as agent
    nebius, tavily = turn
    nebius.main_replies = [tool_call_reply("web_search", {"query": "score du match"})]

    out = await agent.planner_node(_state(f"{OFF_TOPIC_MARKER} score du match ?"))
    assert out["off_topic"] is True
    assert nebius.calls == [] and tavily == []


async def test_search_rounds_are_capped_and_the_last_round_forces_an_answer(turn):
    import apu.runtime.agent as agent
    nebius, tavily = turn
    nebius.main_replies = [tool_call_reply("web_search", {"query": "fractions"})]   # sticky: always asks
    nebius.extraction_replies = ["{}"]

    out = await agent.planner_node(_state())

    assert len(nebius.main_calls) == agent.MAX_SEARCH_ROUNDS + 1
    assert "tools" not in nebius.main_calls[-1]["kwargs"]
    assert sum(len(tool.queries) for tool in tavily) == agent.MAX_SEARCH_ROUNDS
    assert out["sources"] == [{"title": WIKIPEDIA["title"], "url": WIKIPEDIA["url"]}], "deduplicated"


async def test_a_spoken_answer_names_sources_without_urls(turn):
    import apu.runtime.agent as agent
    nebius, _ = turn
    nebius.main_replies = [tool_call_reply("web_search", {"query": "fractions"}), ANSWER]
    nebius.extraction_replies = ["{}"]

    out = await agent.planner_node(_state(interaction_mode={"input_channel": "voice", "output_channel": "voice"}))

    assert out["rendered_answer"] == {"spoken": f"D'après Wikipédia : {ANSWER}", "written": None}
    assert out["messages"][0].content == f"D'après Wikipédia : {ANSWER}"


async def test_an_unavailable_search_still_answers_and_reports_it(turn, monkeypatch):
    import apu.runtime.agent as agent
    nebius, _ = turn
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(web_search, "_web_search", web_search.TavilySearch())
    nebius.main_replies = [tool_call_reply("web_search", {"query": "fractions"}), ANSWER]
    nebius.extraction_replies = ["{}"]

    out = await agent.planner_node(_state())

    assert out["messages"][0].content == ANSWER
    assert any("TAVILY_API_KEY" in problem for problem in out["tool_problems"])
    assert "unavailable" in nebius.main_calls[1]["messages"][-1]["content"]


@pytest.mark.parametrize("reply,expected", [
    (tool_call_reply("delete_everything", {"query": "x"}), "Unknown tool"),
    ({"content": None, "tool_calls": [{"id": "call-1", "name": "web_search", "arguments": "not json"}]},
     "non-empty 'query'"),
    (tool_call_reply("web_search", {"query": "   "}), "non-empty 'query'"),
])
async def test_malformed_tool_calls_are_answered_without_searching(turn, reply, expected):
    import apu.runtime.agent as agent
    nebius, tavily = turn
    nebius.main_replies = [reply, ANSWER]
    nebius.extraction_replies = ["{}"]

    await agent.planner_node(_state())

    assert expected in nebius.main_calls[1]["messages"][-1]["content"]
    assert all(tool.queries == [] for tool in tavily)


def test_call_main_model_message_keeps_the_tool_calls(fake_nebius):
    from apu.inference import nebius_client

    fake_nebius.main_replies = [tool_call_reply("web_search", {"query": "fractions"})]
    message = nebius_client.call_main_model_message([{"role": "user", "content": "q"}],
                                                    tools=[web_search.WEB_SEARCH_TOOL])
    assert message.content is None
    assert message.tool_calls[0].function.name == "web_search"
    assert nebius_client.call_main_model([{"role": "user", "content": "q"}]) is None, (
        "call_main_model alone would have lost the request"
    )
