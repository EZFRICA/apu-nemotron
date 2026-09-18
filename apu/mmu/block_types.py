"""Block types the tutoring memory must never contain.

An escalation event records that a student crossed their class's off-topic threshold. It is
about the student's behaviour, not their learning, and letting it into the DLL or into the
L3 tables the BMJ search reads would feed it back into tutoring prompts. The DLL and the L3
driver both refuse these types, and the search drops any row carrying one.
"""

ESCALATION_EVENT_BLOCK_TYPE = "escalation_event"

TUTORING_EXCLUDED_BLOCK_TYPES: frozenset[str] = frozenset({ESCALATION_EVENT_BLOCK_TYPE})


class ForbiddenBlockType(ValueError):
    """A block type that is kept out of the tutoring memory was offered to it."""


def refuse_non_tutoring_block_type(block_type: str | None) -> None:
    if block_type in TUTORING_EXCLUDED_BLOCK_TYPES:
        raise ForbiddenBlockType(
            f"Block type {block_type!r} is kept out of the tutoring memory; it is stored "
            "by apu.mmu.escalation_store and read only through the teacher API."
        )
