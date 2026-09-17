"""Guard sessions: per-connection state for the topical rail.

A session captures its ClassPolicy once, when it opens, and never re-reads it: a changed
threshold applies to the next session only. The off-topic counter lives here, in memory,
and therefore starts at zero on every reconnection; only crossing the threshold is
persisted (as an EscalationEvent), never individual off-topic attempts.

ValidatedTurn is the proof that the topical rail accepted one specific turn. Only this
module can create one (see _MINT_KEY), and web search refuses to run without the proof
for the session's current turn.
"""

import threading
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from apu.guardrails.policy import ClassPolicy, get_class_policy_registry

_MINT_KEY = object()


class TurnOutcome(StrEnum):
    ON_TOPIC = "on_topic"
    OFF_TOPIC = "off_topic"
    # The classifier answered, but not with a usable verdict. The turn is answered, but
    # not validated: no web search, no off-topic count.
    UNCERTAIN = "uncertain"
    CLASSIFIER_ERROR = "classifier_error"


@dataclass(frozen=True)
class ValidatedTurn:
    session_id: str
    turn_id: str
    user_text: str
    _mint_key: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._mint_key is not _MINT_KEY:
            raise TypeError("A ValidatedTurn can only be issued by the topical rail.")


class UnknownSession(KeyError):
    pass


@dataclass
class GuardSession:
    session_id: str
    student_id: str
    policy: ClassPolicy
    off_topic_count: int = 0
    current_validated_turn_id: str | None = None
    last_guard_error: str | None = None
    _turn_outcomes: dict[str, TurnOutcome] = field(default_factory=dict, repr=False)
    _validated_turns: dict[str, ValidatedTurn] = field(default_factory=dict, repr=False)

    @property
    def class_id(self) -> str:
        return self.policy.class_id

    def record_turn_outcome(self, turn_id: str, outcome: TurnOutcome, error: str | None = None) -> None:
        self._turn_outcomes[turn_id] = outcome
        if error is not None:
            self.last_guard_error = error

    def pop_turn_outcome(self, turn_id: str) -> TurnOutcome | None:
        return self._turn_outcomes.pop(turn_id, None)

    def validate_turn(self, turn_id: str, user_text: str) -> ValidatedTurn:
        turn = ValidatedTurn(self.session_id, turn_id, user_text, _mint_key=_MINT_KEY)
        # Only the latest validated turn counts: a proof from an earlier turn cannot be
        # replayed to search on behalf of a later, unvalidated one.
        self._validated_turns = {turn_id: turn}
        self.current_validated_turn_id = turn_id
        return turn

    def validated_turn(self, turn_id: str) -> ValidatedTurn | None:
        return self._validated_turns.get(turn_id)

    def invalidate_current_turn(self) -> None:
        self._validated_turns = {}
        self.current_validated_turn_id = None


class SessionRegistry:
    def __init__(self, policy_registry_provider=get_class_policy_registry) -> None:
        self._policy_registry_provider = policy_registry_provider
        self._sessions: dict[str, GuardSession] = {}
        self._lock = threading.Lock()

    def open_session(self, student_id: str, class_id: str, session_id: str | None = None) -> GuardSession:
        policy = self._policy_registry_provider().get(class_id)
        session = GuardSession(
            session_id=session_id or str(uuid.uuid4()), student_id=student_id, policy=policy
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> GuardSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise UnknownSession(f"No open guard session {session_id!r}.")
        return session

    def close(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


sessions = SessionRegistry()
