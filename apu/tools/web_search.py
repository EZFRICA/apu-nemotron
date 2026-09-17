"""Web search through Tavily, only for turns the topical guard validated.

Gate: search() requires the ValidatedTurn that the topical rail issued for the session's
CURRENT turn. No proof, a proof from an earlier turn, or a proof for another session, and
the search is refused before any request leaves the process. This is what keeps web search
behind the guard whichever way the model ends up asking for it: native tool calls (method A)
or a parsed JSON action (method B). That choice waits on scripts/smoke_test_tool_calling.py
(see HACKATHON.md) and is not wired yet; either path must call search() with the turn's proof.

Excluded domains: config.GLOBAL_EXCLUDED_DOMAINS plus the class's own additions, taken from
the policy captured by the session, never from the caller. A class can add exclusions, never
remove a global one. Results are also re-checked locally, in case the API lets one through.

Every search returns its sources (title and URL); apu.modality.citations renders them for
the output channel.
"""

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from langchain_core.tools import ToolException
from langchain_tavily import TavilySearch as LangchainTavilySearch

from apu import config
from apu.guardrails.policy import ClassPolicy
from apu.guardrails.session import GuardSession, SessionRegistry, UnknownSession, ValidatedTurn
from apu.guardrails.session import sessions as default_sessions
from apu.modality.citations import Source


class GuardViolation(PermissionError):
    """A search was attempted for a turn the topical guard did not validate."""


class WebSearchUnavailable(RuntimeError):
    """Tavily could not be used (missing key, API error)."""


def _normalize_domain(domain: str) -> str:
    return domain.strip().lower().removeprefix("www.")


def excluded_domains_for(policy: ClassPolicy) -> list[str]:
    domains: list[str] = []
    for domain in (*config.GLOBAL_EXCLUDED_DOMAINS, *policy.tavily_excluded_domains):
        normalized = _normalize_domain(domain)
        if normalized and normalized not in domains:
            domains.append(normalized)
    return domains


def _is_excluded(url: str, excluded: list[str]) -> bool:
    host = _normalize_domain(urlparse(url).hostname or "")
    return any(host == domain or host.endswith("." + domain) for domain in excluded)


@dataclass(frozen=True)
class WebSearchResult:
    query: str
    sources: tuple[Source, ...]
    snippets: tuple[str, ...]


class TavilySearch:
    def __init__(
        self,
        *,
        sessions: SessionRegistry | None = None,
        tool_factory=None,
        max_results: int | None = None,
    ) -> None:
        self._sessions = sessions or default_sessions
        # tool_factory(max_results=..., exclude_domains=...) -> an object with ainvoke();
        # injectable so the gate and the result handling are testable without the API.
        self._tool_factory = tool_factory
        self._max_results = max_results or config.TAVILY_MAX_RESULTS

    def authorize(self, validated_turn: ValidatedTurn) -> GuardSession:
        if not isinstance(validated_turn, ValidatedTurn):
            raise GuardViolation("Web search requires the ValidatedTurn issued by the topical guard.")
        try:
            session = self._sessions.get(validated_turn.session_id)
        except UnknownSession as error:
            raise GuardViolation(f"Unknown guard session {validated_turn.session_id!r}.") from error
        if session.current_validated_turn_id != validated_turn.turn_id:
            raise GuardViolation("This turn is not the session's current validated turn.")
        return session

    async def search(self, query: str, *, validated_turn: ValidatedTurn) -> WebSearchResult:
        session = self.authorize(validated_turn)
        excluded = excluded_domains_for(session.policy)
        tool = self._build_tool(excluded)

        try:
            raw = await tool.ainvoke({"query": query})
        except ToolException:
            # langchain_tavily raises ToolException when a search simply found nothing.
            raw = {"results": []}

        if isinstance(raw, dict) and raw.get("error"):
            # langchain_tavily turns API failures into {"error": ...} instead of raising.
            raise WebSearchUnavailable(f"Tavily search failed: {raw['error']}")

        results = raw.get("results", []) if isinstance(raw, dict) else []
        sources: list[Source] = []
        snippets: list[str] = []
        for result in results:
            url = result.get("url")
            if not url or _is_excluded(url, excluded):
                continue
            sources.append(Source(title=result.get("title") or url, url=url))
            snippets.append(result.get("content") or "")
        return WebSearchResult(query=query, sources=tuple(sources), snippets=tuple(snippets))

    def _build_tool(self, excluded: list[str]):
        if self._tool_factory is not None:
            return self._tool_factory(max_results=self._max_results, exclude_domains=excluded)
        if not os.environ.get("TAVILY_API_KEY"):
            raise WebSearchUnavailable("TAVILY_API_KEY is not set (.env).")
        # Built per search: the exclusions depend on the session's class policy.
        return LangchainTavilySearch(max_results=self._max_results, exclude_domains=excluded)
