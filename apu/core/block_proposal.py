"""
The shape of a block proposal — one definition, used at both ends.

Ported unchanged from Akili (app_local/core/block_proposal.py).

The detector emitted `block_type` while the executor read `type`, and neither
`initial_content` nor `keywords` was ever produced. Nothing raised: a block with
`type=None` and `content=None` was created and reported as success. Having one
declared shape is what stops the two sides drifting again.
"""

from typing import TypedDict


class BlockProposal(TypedDict):
    """A proposed dynamic memory block, as passed from detector to executor."""
    proposed_id: str
    label: str
    type: str            # a DLL block type: temp | cours | fondamental
    initial_content: str
    keywords: list[str]
    reason: str


REQUIRED_FIELDS = ("proposed_id", "label", "type", "initial_content")


def validate(proposal: dict | None) -> str | None:
    """
    Return None if the proposal can be executed, else why it cannot.

    Executing a malformed proposal used to produce a permanently empty,
    typeless, unsearchable block that the UI announced as created. Refusing is
    the only useful behaviour.
    """
    if not proposal:
        return "proposal is empty"
    if not isinstance(proposal, dict):
        return f"proposal must be a dict, got {type(proposal).__name__}"

    missing = [f for f in REQUIRED_FIELDS
               if not str(proposal.get(f) or "").strip()]
    if missing:
        return f"missing or empty required field(s): {', '.join(missing)}"
    return None
