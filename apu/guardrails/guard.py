"""The topical guard: runs the shared NeMo Guardrails input rail on one student turn.

Every tutoring turn goes through check() before anything else happens. The result is:
  - off-topic: a reply to show the student, and nothing else (no answer, no memory write,
    no search);
  - on-topic: a ValidatedTurn, the only thing that lets web search run for that turn;
  - uncertain: the turn may be answered, but it is not validated;
  - the classifier failed: GuardUnavailable is raised, so no answer is produced unguarded.

Hot reload of class policies through NeMo Guardrails' multi-config API is NOT implemented
(TODO in HACKATHON.md): a policy is read when a session opens and applies to new sessions.
"""

import threading
import uuid
from dataclasses import dataclass

from nemoguardrails import LLMRails, RailsConfig

from apu import config
from apu.core.scheduler import DeferredWriteScheduler
from apu.guardrails.actions import build_actions
from apu.guardrails.session import SessionRegistry, TurnOutcome, ValidatedTurn
from apu.guardrails.session import sessions as default_sessions

# Only the input rail: dialog and output generation belong to the tutor, not to NeMo.
_INPUT_RAIL_ONLY = {
    "rails": {
        "input": True, "dialog": False, "output": False, "retrieval": False,
        "tool_input": False, "tool_output": False,
    },
}


class GuardUnavailable(RuntimeError):
    """The topical rail could not reach a verdict; the turn must not be answered."""


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    outcome: TurnOutcome
    reply: str | None = None
    validated_turn: ValidatedTurn | None = None


class TopicalGuard:
    def __init__(
        self,
        *,
        llm=None,
        sessions: SessionRegistry | None = None,
        scheduler: DeferredWriteScheduler | None = None,
        config_dir: str | None = None,
    ) -> None:
        self._sessions = sessions or default_sessions
        rails_config = RailsConfig.from_path(config_dir or config.GUARDRAILS_CONFIG_DIR)
        # llm=None: NeMo builds the main model from config.yml (Nemotron on Token Factory).
        self._rails = LLMRails(rails_config, llm=llm)
        for name, action in build_actions(self._sessions, scheduler).items():
            self._rails.register_action(action, name)

    async def check(self, session_id: str, user_text: str) -> GuardDecision:
        session = self._sessions.get(session_id)
        turn_id = str(uuid.uuid4())
        # A new turn starts unvalidated, whatever happened to the previous one.
        session.invalidate_current_turn()

        response = await self._rails.generate_async(
            messages=[
                {"role": "context", "content": {"session_id": session_id, "turn_id": turn_id}},
                {"role": "user", "content": user_text},
            ],
            options=_INPUT_RAIL_ONLY,
        )

        outcome = session.pop_turn_outcome(turn_id)
        if outcome is None or outcome is TurnOutcome.CLASSIFIER_ERROR:
            raise GuardUnavailable(
                f"The topical guard could not classify this turn: {session.last_guard_error or 'no verdict recorded'}"
            )
        if outcome is TurnOutcome.OFF_TOPIC:
            return GuardDecision(allowed=False, outcome=outcome, reply=_last_assistant_text(response))
        if outcome is TurnOutcome.ON_TOPIC:
            return GuardDecision(
                allowed=True, outcome=outcome, validated_turn=session.validated_turn(turn_id)
            )
        return GuardDecision(allowed=True, outcome=outcome)


def _last_assistant_text(response) -> str:
    messages = response.response if isinstance(response.response, list) else []
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message.get("content") or ""
    return ""


_guard: TopicalGuard | None = None
_guard_lock = threading.Lock()


def get_topical_guard() -> TopicalGuard:
    global _guard
    if _guard is None:
        with _guard_lock:
            if _guard is None:
                _guard = TopicalGuard()
    return _guard


def set_topical_guard(guard: TopicalGuard | None) -> None:
    global _guard
    with _guard_lock:
        _guard = guard
