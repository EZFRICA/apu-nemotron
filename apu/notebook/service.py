"""Saving to the notebook, and turning it into a revision sheet.

Two Nemotron calls, both made only on the student's request:
  - key points: Nemotron Nano condenses one answer at save time (short, cheap, fast);
  - revision summary: Nemotron Super rewrites a set of entries into one sheet, where the
    quality of what the student will revise from matters more than latency.
Both calls are asked for plain text, so an entry stays readable wherever it is shown.
"""

import asyncio
from collections.abc import Sequence

from apu import config
from apu.modality.plain_text import plain_text
from apu.notebook.store import KIND_LABELS, EntryKind, EntryOrigin, NotebookEntry, NotebookStore, new_entry

PLAIN_TEXT_RULES = (
    "Write plain text only: no Markdown, no LaTeX, no emoji, no tables. Write formulas inline "
    "with ordinary characters (for example 1/4 + 1/6 = 5/12). Write in the language of the text."
)

KEY_POINTS_PROMPT = """Condense this tutor answer into the key points a student needs to keep for revision: definitions, rules, methods and at most one short example. Use only what the answer says: add nothing of your own. Leave out greetings, encouragement and questions to the student.

{rules}
One point per line, each line starting with "- ". At most 6 points.

ANSWER:
{answer}"""

SUMMARY_PROMPT = """Write one revision sheet from these notes a student saved during their lessons. The student will revise from it, so keep it short and well ordered: group related notes under a short title line, keep every definition, rule and method, drop repetitions.

{rules}

NOTES:
{notes}"""


class NotebookUnavailable(RuntimeError):
    """Nemotron could not produce the key points or the summary."""


def _nebius():
    # Imported on first use, as in apu.runtime.agent: a missing key fails the first save,
    # not the import of the interface.
    from apu.inference import nebius_client
    return nebius_client


async def _ask(call_name: str, prompt: str, temperature: float) -> str:
    """One plain-text call, retried once when Nemotron returns no text (seen live)."""
    call = getattr(_nebius(), call_name)
    for _ in range(2):
        reply = await asyncio.to_thread(call, [{"role": "user", "content": prompt}], temperature=temperature)
        if reply and reply.strip():
            # Nemotron ends lines with Markdown hard breaks ("  "), noise in a notebook.
            return "\n".join(line.rstrip() for line in reply.strip().splitlines())
    raise NotebookUnavailable("Nemotron returned no text twice.")


async def condense_key_points(answer: str) -> str:
    prompt = KEY_POINTS_PROMPT.format(rules=PLAIN_TEXT_RULES, answer=answer)
    return await _ask("call_extraction_model", prompt, temperature=0.0)


async def save_entry(
    *, student_id: str, class_level: str, subject: str, kind: EntryKind | str, answer: str,
    excerpt: str | None = None, origin: EntryOrigin | str, store: NotebookStore | None = None,
) -> NotebookEntry:
    """Keep the full answer, its key points, or the excerpt the student chose."""
    kind = EntryKind(kind)
    if kind is EntryKind.FULL:
        text = answer
    elif kind is EntryKind.KEY_POINTS:
        if not answer.strip():
            raise ValueError("There is no answer to condense.")
        text = await condense_key_points(answer)
    else:
        text = excerpt or ""
        if not text.strip():
            raise ValueError("An excerpt needs the passage to keep.")
    entry = new_entry(student_id, class_level, subject, kind, text, answer, origin)
    return (store or NotebookStore()).add(entry)


def entries_as_text(entries: Sequence[NotebookEntry]) -> str:
    """The selected entries as written, one numbered heading per entry."""
    blocks = [
        f"{number}. {entry.subject} ({entry.class_level}), {KIND_LABELS[entry.kind].lower()}\n{plain_text(entry.text)}"
        for number, entry in enumerate(entries, start=1)
    ]
    return "\n\n".join(blocks)


async def summarize_entries(entries: Sequence[NotebookEntry]) -> str:
    """
    One revision sheet from the chosen entries.

    Bounded on purpose: entries hold up to NOTEBOOK_MAX_ENTRY_CHARS each, so a student who
    selects their whole notebook would otherwise send a prompt of any size, at any cost, and
    get it refused by the API rather than by us.
    """
    if not entries:
        raise ValueError("There is nothing in the notebook to summarize.")
    if len(entries) > config.NOTEBOOK_MAX_SHEET_ENTRIES:
        raise ValueError(
            f"Choose at most {config.NOTEBOOK_MAX_SHEET_ENTRIES} entries for one sheet "
            f"({len(entries)} selected)."
        )
    notes = "\n\n".join(f"[{entry.course}] {entry.text}" for entry in entries)
    if len(notes) > config.NOTEBOOK_MAX_SHEET_CHARS:
        raise ValueError(
            f"The chosen entries are too long for one sheet ({len(notes)} characters, "
            f"limit {config.NOTEBOOK_MAX_SHEET_CHARS}). Choose fewer of them."
        )
    prompt = SUMMARY_PROMPT.format(rules=PLAIN_TEXT_RULES, notes=notes)
    return await _ask("call_main_model", prompt, temperature=0.3)
