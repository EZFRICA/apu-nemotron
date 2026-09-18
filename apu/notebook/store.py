"""Notebook entries and their SQLite store.

SQLite rather than a DLL block or an L3 table: entries are listed and printed, never searched
by vector, must never be paged out by the LRU, and must never reach the tutoring context.
"""

import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from apu import config


class EntryKind(StrEnum):
    FULL = "full"              # the tutor's answer, as written
    KEY_POINTS = "key_points"  # the answer condensed by Nemotron
    EXCERPT = "excerpt"        # a passage the student chose


class EntryOrigin(StrEnum):
    BUTTON = "button"          # the Save control under an answer
    CHAT = "chat"              # the student asked the tutor, which called save_to_notebook


KIND_LABELS = {
    EntryKind.FULL: "Full answer",
    EntryKind.KEY_POINTS: "Key points",
    EntryKind.EXCERPT: "Excerpt",
}

# What each choice keeps, in the student's words, shown next to the choice when saving.
KIND_DESCRIPTIONS = {
    EntryKind.FULL: "The whole answer, word for word.",
    EntryKind.KEY_POINTS: "A short list of the essentials (rules, methods, an example), written by Nemotron from this answer.",
    EntryKind.EXCERPT: "Only the part you choose: delete what you don't need in the box below.",
}


@dataclass(frozen=True)
class NotebookEntry:
    entry_id: str
    student_id: str
    class_level: str
    subject: str
    kind: EntryKind
    text: str                  # what is kept, and what a revision sheet is made from
    source_answer: str         # the answer it came from, kept for reference
    origin: EntryOrigin
    created_at: datetime

    @property
    def course(self) -> str:
        return f"{self.class_level}/{self.subject}"


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS notebook_entries (
    entry_id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    class_level TEXT NOT NULL,
    subject TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ({", ".join(f"'{k}'" for k in EntryKind)})),
    text TEXT NOT NULL,
    source_answer TEXT NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ({", ".join(f"'{o}'" for o in EntryOrigin)})),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS notebook_entries_by_student
    ON notebook_entries (student_id, class_level, subject, created_at);
"""


def new_entry(
    student_id: str, class_level: str, subject: str, kind: EntryKind | str, text: str,
    source_answer: str, origin: EntryOrigin | str, created_at: datetime | None = None,
) -> NotebookEntry:
    text = text.strip()
    if not text:
        raise ValueError("A notebook entry needs some text.")
    if len(text) > config.NOTEBOOK_MAX_ENTRY_CHARS:
        raise ValueError(f"A notebook entry holds at most {config.NOTEBOOK_MAX_ENTRY_CHARS} characters.")
    return NotebookEntry(
        entry_id=str(uuid.uuid4()), student_id=student_id, class_level=class_level,
        subject=subject, kind=EntryKind(kind), text=text, source_answer=source_answer,
        origin=EntryOrigin(origin), created_at=created_at or datetime.now(UTC),
    )


# Same reasoning as the escalation store: the notebook is written from the interface while
# a revision sheet may be reading it, so a writer must wait rather than fail.
_BUSY_TIMEOUT_SECONDS = 5.0
_schema_applied: set[str] = set()


class NotebookFull(ValueError):
    """The student has reached the number of entries one notebook may hold."""


class NotebookStore:
    def __init__(self, path: str | None = None) -> None:
        # Read at construction, not import, so a redirected data directory applies.
        self.path = path or config.NOTEBOOK_DB_PATH
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._connect().close()

    def _connect(self) -> sqlite3.Connection:
        # Checked before connecting, which creates the file: a database wiped under a
        # running process (the demo reset does exactly that) needs its schema again.
        existed = os.path.exists(self.path)
        connection = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_SECONDS)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute(f"PRAGMA busy_timeout = {int(_BUSY_TIMEOUT_SECONDS * 1000)}")
        if self.path not in _schema_applied or not existed:
            connection.executescript(_SCHEMA)
            _schema_applied.add(self.path)
        return connection

    def add(self, entry: NotebookEntry) -> NotebookEntry:
        # Ceiling per student: each save costs a model call and a row, and both are driven
        # by the student. Nothing is dropped silently; the save is refused and said so.
        if self.count(entry.student_id) >= config.NOTEBOOK_MAX_ENTRIES_PER_STUDENT:
            raise NotebookFull(
                f"Your notebook is full ({config.NOTEBOOK_MAX_ENTRIES_PER_STUDENT} entries). "
                "Delete something you no longer need before saving more."
            )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO notebook_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entry.entry_id, entry.student_id, entry.class_level, entry.subject, entry.kind.value,
                 entry.text, entry.source_answer, entry.origin.value, entry.created_at.isoformat()),
            )
        return entry

    def entries(self, student_id: str, class_level: str | None = None,
                subject: str | None = None) -> list[NotebookEntry]:
        """One student's entries, oldest first: the order a revision sheet is read in."""
        query = "SELECT * FROM notebook_entries WHERE student_id = ?"
        params: list[str] = [student_id]
        if class_level is not None:
            query += " AND class_level = ?"
            params.append(class_level)
        if subject is not None:
            query += " AND subject = ?"
            params.append(subject)
        with self._connect() as connection:
            rows = connection.execute(query + " ORDER BY created_at, entry_id", params).fetchall()
        return [_entry_from_row(row) for row in rows]

    def count(self, student_id: str) -> int:
        with self._connect() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM notebook_entries WHERE student_id = ?", (student_id,)
            ).fetchone()[0]

    def delete(self, student_id: str, entry_id: str) -> bool:
        """Remove one of the student's own entries. False when it is not theirs or not there."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM notebook_entries WHERE entry_id = ? AND student_id = ?",
                (entry_id, student_id),
            )
        return cursor.rowcount == 1


def _entry_from_row(row: sqlite3.Row) -> NotebookEntry:
    return NotebookEntry(
        entry_id=row["entry_id"], student_id=row["student_id"], class_level=row["class_level"],
        subject=row["subject"], kind=EntryKind(row["kind"]), text=row["text"],
        source_answer=row["source_answer"], origin=EntryOrigin(row["origin"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )
