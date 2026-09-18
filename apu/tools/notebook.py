"""save_to_notebook: the tutor saves to the student's notebook when the student asks.

This is how a student who does not use the Save button keeps something:
"save that", "keep the key points", "note down the rule for adding fractions". Offered on
validated turns only, like web search, and executed with that turn's proof, so the student
the entry is filed under comes from the guard session, never from the model.

Write-only: there is no tool to read the notebook, and nothing adds it to the prompt.
"""

from apu.guardrails.session import SessionRegistry, UnknownSession, ValidatedTurn
from apu.guardrails.session import sessions as default_sessions
from apu.notebook import service
from apu.notebook.store import EntryKind, EntryOrigin, NotebookEntry, NotebookStore

SAVE_TO_NOTEBOOK_TOOL_NAME = "save_to_notebook"

SAVE_TO_NOTEBOOK_TOOL = {
    "type": "function",
    "function": {
        "name": SAVE_TO_NOTEBOOK_TOOL_NAME,
        "description": (
            "Save something to the student's notebook, ONLY when the student explicitly asks to "
            "save, keep or note something. The notebook is where the student keeps what matters "
            "to them and makes their revision sheets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "what": {
                    "type": "string",
                    "enum": [kind.value for kind in EntryKind],
                    "description": (
                        "full: your previous answer as written. key_points: the essential points "
                        "of your previous answer. excerpt: only the passage the student wants."
                    ),
                },
                "excerpt": {
                    "type": "string",
                    "description": "With what=excerpt: the exact passage to keep, as the student asked for it.",
                },
            },
            "required": ["what"],
        },
    },
}

NOTEBOOK_INSTRUCTIONS = """
NOTEBOOK: the student has a notebook where they keep what matters to them. When the student asks to save, keep, note, write down or remember something (for example "save that", "keep the key points", "just keep the rule for adding fractions", "note this in my notebook"), call save_to_notebook instead of answering again. Use what=full for your previous answer as it is, what=key_points for its essential points, and what=excerpt with the exact passage when the student wants only one part (copy that part from your previous answer). Never save without being asked. After saving, confirm it in one short sentence. You cannot read the notebook.
"""


class NotebookGateError(PermissionError):
    """A save was attempted without the current validated turn of the session."""


async def save_from_chat(
    arguments: dict, *, validated_turn: ValidatedTurn | None, previous_answer: str,
    class_level: str, subject: str, sessions: SessionRegistry | None = None,
    store: NotebookStore | None = None,
) -> NotebookEntry:
    """Validate the tool arguments and save. Raises ValueError for arguments the model can fix."""
    registry = sessions or default_sessions
    if validated_turn is None:
        raise NotebookGateError("save_to_notebook needs the turn the topical rail validated.")
    try:
        session = registry.get(validated_turn.session_id)
    except UnknownSession as error:
        raise NotebookGateError(str(error)) from error
    if session.current_validated_turn_id != validated_turn.turn_id:
        raise NotebookGateError("save_to_notebook needs the session's current validated turn.")

    try:
        kind = EntryKind(arguments.get("what"))
    except ValueError:
        raise ValueError("'what' must be one of: full, key_points, excerpt.") from None
    if kind is not EntryKind.EXCERPT and not previous_answer.strip():
        raise ValueError("There is no previous answer to save yet.")

    return await service.save_entry(
        student_id=session.student_id, class_level=class_level, subject=subject, kind=kind,
        answer=previous_answer, excerpt=arguments.get("excerpt"), origin=EntryOrigin.CHAT,
        store=store,
    )
