"""Heuristic detector proposing a new dynamic DLL block. Ported unchanged from Akili."""


from apu.core.block_proposal import BlockProposal
from apu.logger import get_logger

logger = get_logger(__name__)

# Trigger thresholds for proposing a new block
MIN_TURNS_FOR_DETECTION = 4      # Minimum number of conversation turns
TOPIC_REPETITION_THRESHOLD = 2   # Number of times a topic should be mentioned

def detect_new_block_opportunity(history: list[dict], dll: dict) -> dict | None:
    """
    Analyzes recent history to detect if a new knowledge block
    should be created (e.g., note-taking on a new chapter).

    Returns a configuration dict if an opportunity is detected, else None.
    """
    # 1. Check minimum conversation threshold
    if len(history) < MIN_TURNS_FOR_DETECTION:
        return None

    # 2. Check that dynamic block limit isn't reached
    dynamic_count = dll.get("dynamic_block_count", 0)
    dynamic_max = dll.get("dynamic_block_max", 5)
    if dynamic_count >= dynamic_max:
        logger.debug(f"Dynamic block limit reached ({dynamic_count}/{dynamic_max}).")
        return None

    # 3. Heuristic analysis of recent student messages
    recent_user_messages = [
        m["content"] for m in history[-6:]
        if m.get("role") == "user" and isinstance(m.get("content"), str)
    ]

    if not recent_user_messages:
        return None

    # Look for patterns signaling a new topic to memorize
    learning_triggers = [
        "i don't understand", "explain", "what is", "how",
        "why", "i'm struggling", "difficult", "new chapter",
        "lesson", "exercise", "je ne comprends pas", "explique"
    ]

    trigger_count = sum(
        1 for msg in recent_user_messages
        for trigger in learning_triggers
        if trigger.lower() in msg.lower()
    )

    if trigger_count >= TOPIC_REPETITION_THRESHOLD:
        proposed_id = f"dynamic_block_{dynamic_count + 1}"
        logger.info(f"Block opportunity detected (triggers={trigger_count}): {proposed_id}")

        # The topic being learned, taken from the student's own words. This used
        # to be absent entirely, so the executor wrote a block with no content.
        topic = recent_user_messages[-1].strip()

        return BlockProposal(
            proposed_id=proposed_id,
            label="Topic currently being learned",
            # 'type', not 'block_type': the executor reads 'type', and the two
            # keys silently disagreeing is what produced typeless blocks.
            # 'temp' because a topic under study is recent context, and unlike
            # 'course' it is a type the rest of the system actually knows.
            type="temp",
            initial_content=topic[:500],
            keywords=[],
            reason=f"Student showed {trigger_count} active learning signals.",
        )

    return None
