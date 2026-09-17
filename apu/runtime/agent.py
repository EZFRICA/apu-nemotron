"""LangGraph runtime for the tutor: one planner node per turn.

Ported from Akili (app_local/runtime/agent.py). What changed in the port, and why:

  - Every turn first goes through the topical guard (apu.guardrails). An off-topic turn
    gets the guard's reply and nothing else.
  - Inference goes through apu.inference.nebius_client: call_main_model_message for the
    answer, call_extraction_model for the memory write-back. Both are synchronous
    OpenAI-client calls, so they run in a worker thread to keep the event loop free, for
    the same reason the embedder does.
  - Web search is a native OpenAI tool (Method A, see HACKATHON.md). It is offered only on
    turns the guard validated, executed only through apu.tools.web_search with that turn's
    proof, and bounded to MAX_SEARCH_ROUNDS. Its sources are appended to the answer in the
    form the output channel needs (apu.modality.citations).
  - Akili's other TEU tools (calculator, course search, chapter loader) are not ported.
  - Messages are converted from LangChain message objects to the OpenAI dict format at the
    boundary. The graph state still carries LangChain messages because the add_messages
    reducer needs them.

Unchanged: retrieval and BMJ routing, the L1/L2 memory read, the prompt texts, the
tolerant extraction parser, the block detector, and the memory write-back running inline,
awaited before the turn returns (see apu.core.scheduler for why that is still open).
"""

import asyncio
import json
import os
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from apu import config
from apu.core.block_detector import detect_new_block_opportunity
from apu.core.extraction import parse_extraction
from apu.embeddings import local_embedder
from apu.guardrails import guard as topical_guard
from apu.guardrails.session import ValidatedTurn
from apu.logger import get_logger
from apu.mmu import cache_l1
from apu.mmu import dll as mmu
from apu.modality.citations import Source, render_answer
from apu.modality.mode import InputChannel, InteractionMode, OutputChannel
from apu.tools import web_search

logger = get_logger(__name__)

# Two searches are enough to refine a query once; the round after the last one is sent
# without tools, which forces a written answer instead of an unbounded loop.
MAX_SEARCH_ROUNDS = 2

SEARCH_INSTRUCTIONS = """
WEB SEARCH: when the course context and the student memory above are not enough to answer accurately, you may call the web_search tool. Use it only for the student's schoolwork. Do not list sources or URLs yourself: the sources you used are added to your answer automatically.
"""


def _nebius():
    """
    The Nebius client module, imported on first use rather than at import.

    nebius_client builds its OpenAI client at module scope and raises when
    NEBIUS_API_KEY is unset. Importing it at the top of this module would make the
    graph unimportable without a key, the failure Akili fixed by building its LLM
    clients lazily: a missing key should fail the first question, not the app start.
    """
    from apu.inference import nebius_client
    return nebius_client


# BaseMessage.type values mapped to the roles the chat completions API expects.
_ROLE_BY_MESSAGE_TYPE = {"system": "system", "human": "user", "ai": "assistant"}


def _to_str(content) -> str:
    """
    Normalize model output to plain text.

    The OpenAI client returns None for a message with no text (e.g. a completion
    that only produced reasoning or tool calls); str(None) would put the literal "None"
    into student memory. The list branch is kept from Akili for providers that return
    content blocks.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def _to_openai_messages(messages: List[BaseMessage]) -> List[dict]:
    converted = []
    for message in messages:
        role = _ROLE_BY_MESSAGE_TYPE.get(message.type)
        if role is None:
            # Only system/human/ai messages exist in the graph state; tool round trips
            # live in the per-turn conversation built by _answer, not in the state.
            raise ValueError(f"Cannot send a {message.type!r} message to the main model")
        converted.append({"role": role, "content": _to_str(message.content)})
    return converted


async def _call_extraction_model(prompt: str) -> str:
    # Deterministic and in JSON mode, as on every Akili branch: the output is parsed
    # as JSON, and temperature 0 keeps one exchange from producing different
    # memories on different runs. response_format is OpenAI's JSON mode; it has not
    # been verified against the live Token Factory endpoint for Nemotron Nano. If it
    # is rejected, the error is reported as a memory problem, not swallowed.
    raw_extraction = await asyncio.to_thread(
        _nebius().call_extraction_model,
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return _to_str(raw_extraction)


async def _run_tool_call(
    call, validated_turn: ValidatedTurn, sources: List[Source], tool_problems: List[str]
) -> str:
    """Execute one tool call and return the content of its tool message."""
    name = call.function.name
    if name != web_search.WEB_SEARCH_TOOL_NAME:
        return f"Unknown tool {name!r}: only {web_search.WEB_SEARCH_TOOL_NAME} is available."
    try:
        query = (json.loads(call.function.arguments or "{}").get("query") or "").strip()
    except (json.JSONDecodeError, AttributeError):
        query = ""
    if not query:
        return "web_search needs a non-empty 'query' argument."

    try:
        # The gate: search() refuses anything but this turn's ValidatedTurn. A
        # GuardViolation here would be a bug, so it is left to propagate.
        result = await web_search.get_web_search().search(query, validated_turn=validated_turn)
    except web_search.WebSearchUnavailable as error:
        logger.warning("Web search unavailable: %s", error)
        tool_problems.append(f"web search unavailable ({error})")
        return (
            "Web search is unavailable right now. Answer from the course context and your "
            "own knowledge, and say so if you are unsure."
        )

    known_urls = {source.url for source in sources}
    for source in result.sources:
        if source.url not in known_urls:
            sources.append(source)
            known_urls.add(source.url)
    return web_search.format_search_result_for_model(result)


async def _answer(
    conversation: List[dict], validated_turn: ValidatedTurn | None
) -> tuple[str, List[Source], List[str]]:
    """Main-model call, with web search tool rounds when the turn was validated."""
    sources: List[Source] = []
    tool_problems: List[str] = []

    for search_round in range(MAX_SEARCH_ROUNDS + 1):
        offer_search = validated_turn is not None and search_round < MAX_SEARCH_ROUNDS
        # temperature 0.7: the value every Akili provider branch used for the tutor.
        call_kwargs: dict = {"temperature": 0.7}
        if offer_search:
            call_kwargs["tools"] = [web_search.WEB_SEARCH_TOOL]

        message = await asyncio.to_thread(
            _nebius().call_main_model_message, conversation, **call_kwargs
        )
        tool_calls = getattr(message, "tool_calls", None) or []
        if not offer_search or not tool_calls:
            return _to_str(message.content), sources, tool_problems

        # Replay the assistant's tool request, then one tool message per call, as the
        # chat completions API requires before the model can use the results.
        conversation.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [call.model_dump() for call in tool_calls],
        })
        for call in tool_calls:
            conversation.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": await _run_tool_call(call, validated_turn, sources, tool_problems),
            })

    return "", sources, tool_problems  # not reached: the last round never offers tools


def _interaction_mode(state) -> InteractionMode:
    mode = state.get("interaction_mode")
    if mode is None:
        return InteractionMode(InputChannel.TEXT, OutputChannel.TEXT)
    if isinstance(mode, InteractionMode):
        return mode
    return InteractionMode(**mode)


# --- State Graph Definition ---
class AgentState(TypedDict, total=False):
    # add_messages so each node's returned messages accumulate in the state
    # instead of replacing the list.
    messages: Annotated[List[BaseMessage], add_messages]
    agent_id: str
    class_level: str
    subject: str
    memory_only_mode: bool
    needs_new_block: str
    proposed_block_config: dict
    memory_problems: List[str]
    # Guard session opened by the caller (dashboard, API): carries the class policy and the
    # off-topic counter. Required: no turn is answered without the topical guard.
    session_id: str
    off_topic: bool
    # Input/output channels of the session; text -> text when absent.
    interaction_mode: InteractionMode | dict
    # {"spoken": str | None, "written": str | None}, rendered for the output channel.
    rendered_answer: dict
    sources: List[dict]
    tool_problems: List[str]


class GuardSessionRequired(ValueError):
    """A turn reached the planner without a guard session."""


def build_message_window(history, prompt, exchanges: int) -> List[BaseMessage]:
    """
    The transcript sent to the model, trimmed to the last `exchanges` turns.

    `history` is the UI's own log: dicts of {"role", "content"}, oldest first,
    NOT including the prompt being asked now.

    Why this is a setting and not a constant: the transcript is the one part of
    the prompt that grows without bound, and what a device can afford depends on
    the model behind it. `exchanges=0` is memory-only: the tutor still sees the
    L1/L2 blocks, which carry continuity of topic and profile, just not the
    literal transcript.
    """
    window: List[BaseMessage] = []
    if exchanges > 0:
        for entry in history[-(exchanges * 2):]:
            content = entry.get("content", "")
            if entry.get("role") == "user":
                window.append(HumanMessage(content=content))
            else:
                window.append(AIMessage(content=content))
    window.append(HumanMessage(content=prompt))
    return window

# --- Memory Write-Back ---
async def _update_student_memory(
    user_query: str,
    agent_response: str,
    dll: dict
) -> List[str]:
    """
    Extract new information and save it to local LanceDB.

    Returns the list of problems encountered, empty when everything landed. The
    caller surfaces them: a student whose memory silently stopped updating has
    no way to know, and this used to be a single log line.
    """
    agent_response_str = _to_str(agent_response)

    extraction_prompt = f"""You are a memory extraction system for Akili education agent.
Extract information from this exchange to update the student memory blocks.

STUDENT: "{user_query}"
AKILI: "{agent_response_str[:400]}"

Return ONLY a valid JSON with the following keys (empty string if nothing to update):
- student_profile: Any personal info (name, age, grade, school, goals...)
- learning_preferences: Learning style, difficulty level, preferences...
- current_session: What the student is currently studying (topic, subject, chapter). Always fill this based on the conversation.

Rules:
- current_session MUST always be updated with the current topic.
- Use plain text sentences, not keywords.
- If no personal info is shared, leave student_profile and learning_preferences empty.

Example: {{"student_profile": "", "learning_preferences": "", "current_session": "The student is asking about the manorial system in the Middle Ages (5th Grade History)."}}
"""
    problems: List[str] = []
    try:
        raw_extraction = await _call_extraction_model(extraction_prompt)
    except Exception as e:
        logger.error(f"Error during memory extraction: {e}")
        return [f"the memory extractor could not be reached ({e})"]

    updates, problems = parse_extraction(raw_extraction)

    for block_id, new_info in updates.items():
        # Per block, so one failing write cannot discard the others.
        try:
            await mmu.update_node_content(block_id, new_info, dll)
        except Exception as e:
            logger.error("Failed to store memory block '%s': %s", block_id, e)
            problems.append(f"{block_id}: could not be saved ({e})")

    if not updates:
        problems.append("nothing was extracted from this exchange")
    for p in problems:
        logger.warning("Memory extraction: %s", p)
    return problems

# --- Nodes ---

async def planner_node(state: AgentState):
    """
    Main node that:
    0. Runs the topical guard
    1. Vectorizes the query
    2. Searches course data (LanceDB)
    3. Searches student memory (DLL)
    4. Generates a pedagogical response, with web search on validated turns
    """
    user_query = next(
        (m.content for m in reversed(state["messages"])
         if isinstance(m, HumanMessage)),
        state["messages"][-1].content,
    )

    # 0. Topical guard, before anything touches memory or a model. An off-topic turn gets
    # the guard's reply and nothing else: no retrieval, no answer, no memory write-back.
    session_id = state.get("session_id")
    if not session_id:
        raise GuardSessionRequired(
            "planner_node needs state['session_id'] from a guard session "
            "(apu.guardrails.session.sessions.open_session)."
        )
    decision = await topical_guard.get_topical_guard().check(session_id, user_query)
    if not decision.allowed:
        return {
            "messages": [AIMessage(content=decision.reply or "")],
            "needs_new_block": "False",
            "proposed_block_config": {},
            "memory_problems": [],
            "off_topic": True,
        }

    # 1. Query Vectorization — local ONNX, no network round trip. Resolved through
    # the module at call time: the embedder is process-cached and swappable.
    query_vector = await local_embedder.get_embedder().aembed_query(user_query)

    # 2. DLL Routing & Context Compilation
    dll = await mmu.load_dll()

    # `dll` is this turn's single DLL handle: search_memory applies the BMJ
    # promotion to it in place, and the memory write-back below persists that
    # same object. Passing it is what stops the write-back reverting the routing.
    relevant_blocks = await mmu.search_memory(
        query_vector,
        state["class_level"],
        state["subject"],
        dll=dll
    )

    context_text = "\n".join([f"--- {b['chapter_id']} ---\n{b['content']}" for b in relevant_blocks])

    # 3. Memory Context (Hybrid L1/L2 access)
    memory_context = ""
    for node_id in ["student_profile", "learning_preferences", "current_session"]:
        # Priority 1: L1 Cache (Hot RAM)
        content = cache_l1.get(node_id)
        if content:
            mmu.record_access(node_id, dll)

        # Priority 2: DLL Metadata (L2)
        if not content:
            node = dll["nodes"].get(node_id, {})
            content = node.get("content")

            # Write-back to L1 if found in L2
            if content:
                cache_l1.set(node_id, content, block_type=node.get("type"))
                mmu.record_access(node_id, dll)

        # Fallback to keywords for routing context
        if not content:
            node = dll["nodes"].get(node_id, {})
            content = ", ".join(node.get("keywords", []))

        if content:
            label = dll["nodes"].get(node_id, {}).get("label", node_id)
            memory_context += f"- {label}: {content}\n"

    # 4. Dynamic Pedagogical Prompt. prompts.json is written next to the LanceDB
    # directory by the registry sync (apu.sync.sync_manager); without it the generic
    # persona below applies.
    prompts_path = os.path.join(os.path.dirname(config.LANCE_DB_PATH), "prompts.json")
    base_instructions = "You are Akili, an expert academic tutor."
    class_guidelines = ""

    if os.path.exists(prompts_path):
        try:
            with open(prompts_path, "r") as f:
                prompts_data = json.load(f)
                # 1. Load general tutor persona
                base_instructions = prompts_data.get("system_tutor", base_instructions)
                # 2. Load class-specific guidelines (e.g., 6eme)
                class_guidelines = prompts_data.get(state["class_level"], "")
        except Exception as e:
            logger.warning(f"Failed to load dynamic prompts: {e}")

    system_prompt = f"""{base_instructions}

{class_guidelines}

CURRENT MISSION: Help the student master {state['subject']} ({state['class_level']}).

COURSE CONTEXT (Search Results):
{context_text}

STUDENT MEMORY (L1/L2):
{memory_context}

Respond as a helpful tutor. Keep it concise but warm. Use the Socratic method when possible.
"""

    return await _generate(state, user_query, system_prompt, dll, decision.validated_turn)


async def _generate(state: AgentState, user_query: str, system_prompt: str,
                    dll: dict, validated_turn: ValidatedTurn | None) -> dict:
    """Answer (with search when validated), render sources, then write memory back once."""
    # Search instructions only when search is actually offered: telling the model about a
    # tool it cannot call invites it to pretend it searched.
    instructions = system_prompt + (SEARCH_INSTRUCTIONS if validated_turn is not None else "")
    # SystemMessage, not HumanMessage: the instructions are not a student turn.
    conversation = _to_openai_messages([SystemMessage(content=instructions)] + state["messages"])
    answer_text, sources, tool_problems = await _answer(conversation, validated_turn)

    rendered = render_answer(answer_text, sources, _interaction_mode(state))
    shown = rendered.written if rendered.written is not None else rendered.spoken
    response = AIMessage(content=shown or "")

    # Memory is extracted from the answer itself, not from the appended source list,
    # which says nothing about the student.
    memory_problems = await _update_student_memory(user_query, answer_text, dll)

    history = [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content} for m in state["messages"]]
    history.append({"role": "assistant", "content": answer_text})

    proposal = detect_new_block_opportunity(history, dll)
    needs_new = "True" if proposal else "False"

    return {
        "messages": [response],
        "needs_new_block": needs_new,
        "proposed_block_config": proposal or {},
        # Surfaced to the caller. A parse failure used to be one log line, so a
        # student whose memory stopped updating had no way to know.
        "memory_problems": memory_problems,
        "rendered_answer": {"spoken": rendered.spoken, "written": rendered.written},
        "sources": [{"title": source.title, "url": source.url} for source in sources],
        "tool_problems": tool_problems,
    }


# --- Graph Assembly ---

def create_agent_graph():
    """
    Planner ──> END.

    A single node: search tool rounds happen inside the planner's turn, bounded by
    MAX_SEARCH_ROUNDS, so there is nothing for the graph to loop back from.
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("Planner", planner_node)
    workflow.set_entry_point("Planner")
    workflow.add_edge("Planner", END)
    return workflow.compile()
