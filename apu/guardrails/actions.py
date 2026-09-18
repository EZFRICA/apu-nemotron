"""Custom NeMo Guardrails actions for the topical rail.

They receive the session and turn ids through the rails context, and resolve everything
else (policy, counter) from the in-memory session registry, so the threshold used is the one
captured when the session opened, never a value supplied with the request.

Every action records the turn's outcome on the session. The guard reads that record rather
than NeMo's response text, because NeMo turns an exception inside an action into a generic
"internal error" reply that would otherwise be indistinguishable from a blocked turn.
"""

import uuid
from datetime import UTC, datetime

from apu.core.scheduler import DeferredWriteScheduler
from apu.escalation.jobs import submit_escalation_event
from apu.escalation.models import EscalationEvent
from apu.guardrails.classifier import build_classifier_prompt, parse_verdict
from apu.guardrails.session import SessionRegistry, TurnOutcome
from apu.logger import get_logger

logger = get_logger(__name__)

GENTLE_REPLY = (
    "I'm here to help you with your schoolwork. Try asking me about a lesson, an exercise "
    "or your revision instead, for example \"Explain fractions to me\" or \"Help me prepare "
    "for my history test\"."
)
FIRM_REPLY = (
    "I can only help with your schoolwork, and this request isn't part of it. Come back "
    "with a question about your lessons, your exercises or your revision."
)


def build_actions(sessions: SessionRegistry, scheduler: DeferredWriteScheduler | None = None) -> dict:
    async def check_school_topic(llm=None, context: dict | None = None) -> str:
        context = context or {}
        session = sessions.get(context["session_id"])
        turn_id = context["turn_id"]
        try:
            response = await llm.generate_async(
                build_classifier_prompt(context.get("user_message") or ""), temperature=0.0
            )
        except Exception as error:
            # Recorded, not raised: NeMo would swallow the exception into a generic reply.
            logger.error("Topical classifier call failed: %s", error)
            session.record_turn_outcome(turn_id, TurnOutcome.CLASSIFIER_ERROR, error=str(error))
            return "error"

        outcome = parse_verdict(response.content)
        if outcome is TurnOutcome.UNCERTAIN:
            logger.warning("Topical classifier gave no usable verdict: %r", response.content)
            session.record_turn_outcome(turn_id, TurnOutcome.UNCERTAIN)
        return outcome.value

    async def handle_off_topic_attempt(context: dict | None = None) -> str:
        context = context or {}
        session = sessions.get(context["session_id"])
        turn_id = context["turn_id"]

        session.invalidate_current_turn()
        session.off_topic_count += 1
        attempt = session.off_topic_count
        threshold = session.policy.escalation_threshold
        session.record_turn_outcome(turn_id, TurnOutcome.OFF_TOPIC)

        if attempt < threshold:
            return GENTLE_REPLY

        # Persist the crossing only, once per session: attempts below the threshold are
        # never stored, and attempts after it keep the firmer tone without adding events.
        if attempt == threshold:
            event = EscalationEvent(
                event_id=str(uuid.uuid4()),
                student_id=session.student_id,
                class_id=session.class_id,
                session_id=session.session_id,
                attempt_number_in_session=attempt,
                off_topic_request_text=context.get("user_message") or "",
                triggered_at=datetime.now(UTC),
            )
            submit_escalation_event(event, scheduler)
        return FIRM_REPLY

    async def mark_turn_validated(context: dict | None = None) -> bool:
        context = context or {}
        session = sessions.get(context["session_id"])
        turn_id = context["turn_id"]
        session.validate_turn(turn_id, context.get("user_message") or "")
        session.record_turn_outcome(turn_id, TurnOutcome.ON_TOPIC)
        return True

    return {
        "check_school_topic": check_school_topic,
        "handle_off_topic_attempt": handle_off_topic_attempt,
        "mark_turn_validated": mark_turn_validated,
    }
