"""Storage for escalation events: a block type kept apart from the tutoring memory.

Isolation is structural, not a filter someone could forget:
  - escalation events are not DLL nodes and not rows of the L3 tables the BMJ search reads
    (edu_registry, user_memory); they live in their own SQLite file;
  - apu.mmu.dll and apu.storage.lance_driver refuse the block type outright
    (apu.mmu.block_types), and the L3 search drops any row carrying it;
  - the only read path is the teacher/admin API (apu.api), behind authorize_view.

SQLite rather than LanceDB: nothing here is searched by vector at read time, and "resolved"
is a join between two append-only tables, which is what a relational store is for.
Writes arrive through the deferred-write scheduler (apu.escalation.jobs), never from the
student's request path.
"""

import json
import os
import sqlite3
from datetime import datetime

from apu import config
from apu.escalation.models import (
    EscalationCluster,
    EscalationClusterSnapshot,
    EscalationEvent,
    EscalationResolution,
)
from apu.mmu.block_types import ESCALATION_EVENT_BLOCK_TYPE

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS escalation_events (
    event_id TEXT PRIMARY KEY,
    block_type TEXT NOT NULL DEFAULT '{ESCALATION_EVENT_BLOCK_TYPE}'
        CHECK (block_type = '{ESCALATION_EVENT_BLOCK_TYPE}'),
    student_id TEXT NOT NULL,
    class_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    attempt_number_in_session INTEGER NOT NULL,
    off_topic_request_text TEXT NOT NULL,
    triggered_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS escalation_events_by_class
    ON escalation_events (class_id, triggered_at);

CREATE TABLE IF NOT EXISTS escalation_resolutions (
    event_id TEXT PRIMARY KEY REFERENCES escalation_events (event_id),
    resolved_by TEXT NOT NULL,
    resolved_at TEXT NOT NULL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS escalation_cluster_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id TEXT NOT NULL,
    computed_at TEXT NOT NULL,
    events_covered INTEGER NOT NULL,
    clusters_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS escalation_snapshots_by_class
    ON escalation_cluster_snapshots (class_id, snapshot_id);
"""


# Seconds a connection waits for a lock held by another writer before giving up.
_BUSY_TIMEOUT_SECONDS = 5.0

# Databases whose schema this process has already applied. The statements are idempotent,
# but replaying them on every single call cost a write transaction per read.
_schema_applied: set[str] = set()


class EventNotFound(LookupError):
    pass


class AlreadyResolved(ValueError):
    pass


def _event_from_row(row: sqlite3.Row) -> EscalationEvent:
    return EscalationEvent(
        event_id=row["event_id"],
        student_id=row["student_id"],
        class_id=row["class_id"],
        session_id=row["session_id"],
        attempt_number_in_session=row["attempt_number_in_session"],
        off_topic_request_text=row["off_topic_request_text"],
        triggered_at=datetime.fromisoformat(row["triggered_at"]),
    )


class EscalationStore:
    def __init__(self, db_path: str | None = None) -> None:
        # None: resolve config.ESCALATION_DB_PATH at each connection, so a redirected
        # path (tests, another data dir) is honoured by every store instance.
        self._db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        path = self._db_path or config.ESCALATION_DB_PATH
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # timeout + WAL: the deferred-write thread and a teacher's page read the same file at
        # the same time. In the default journal mode a writer locks out readers, and without
        # a timeout a concurrent write fails immediately with "database is locked" instead of
        # waiting the moment it takes.
        # Checked before connecting, which creates the file: a database wiped under a
        # running process (the demo reset does exactly that) needs its schema again.
        existed = os.path.exists(path)
        connection = sqlite3.connect(path, timeout=_BUSY_TIMEOUT_SECONDS)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute(f"PRAGMA busy_timeout = {int(_BUSY_TIMEOUT_SECONDS * 1000)}")
        if path not in _schema_applied or not existed:
            connection.executescript(_SCHEMA)
            _schema_applied.add(path)
        return connection

    # ── events ───────────────────────────────────────────────────────────────
    def append_event(self, event: EscalationEvent) -> bool:
        """Insert once. Returns False if the event was already stored (a retried write)."""
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO escalation_events (event_id, student_id, class_id, "
                "session_id, attempt_number_in_session, off_topic_request_text, triggered_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (event.event_id, event.student_id, event.class_id, event.session_id,
                 event.attempt_number_in_session, event.off_topic_request_text,
                 event.triggered_at.isoformat()),
            )
            return cursor.rowcount == 1

    def get_event(self, event_id: str) -> EscalationEvent | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM escalation_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return _event_from_row(row) if row else None

    def events_for_class(self, class_id: str) -> list[EscalationEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM escalation_events WHERE class_id = ? "
                "ORDER BY triggered_at, event_id",
                (class_id,),
            ).fetchall()
        return [_event_from_row(row) for row in rows]

    def delete_events_before(self, cutoff: datetime) -> int:
        """
        Erase events (and their resolutions) triggered before `cutoff`. Returns how many.

        These records are children's own messages, kept only so a teacher can act on a
        pattern. Nothing expires on its own, so this is what an operator runs to hold a
        retention period, and what answers a request to erase a pupil's history.
        """
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM escalation_resolutions WHERE event_id IN "
                "(SELECT event_id FROM escalation_events WHERE triggered_at < ?)",
                (cutoff.isoformat(),),
            )
            cursor = connection.execute(
                "DELETE FROM escalation_events WHERE triggered_at < ?", (cutoff.isoformat(),)
            )
            return cursor.rowcount

    def delete_student_events(self, student_id: str) -> int:
        """Erase one student's events and resolutions. Returns how many events were removed."""
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM escalation_resolutions WHERE event_id IN "
                "(SELECT event_id FROM escalation_events WHERE student_id = ?)",
                (student_id,),
            )
            cursor = connection.execute(
                "DELETE FROM escalation_events WHERE student_id = ?", (student_id,)
            )
            return cursor.rowcount

    def count_events(self, class_id: str) -> int:
        with self._connect() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM escalation_events WHERE class_id = ?", (class_id,)
            ).fetchone()[0]

    def list_events_with_resolutions(
        self, class_id: str, limit: int | None = None, offset: int = 0
    ) -> list[tuple[EscalationEvent, EscalationResolution | None]]:
        """Oldest first. `limit` bounds the page; None returns the whole class."""
        query = (
            "SELECT e.*, r.resolved_by, r.resolved_at, r.note "
            "FROM escalation_events e "
            "LEFT JOIN escalation_resolutions r ON r.event_id = e.event_id "
            "WHERE e.class_id = ? ORDER BY e.triggered_at, e.event_id"
        )
        params: list = [class_id]
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params += [limit, offset]
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        results = []
        for row in rows:
            resolution = (
                EscalationResolution(
                    event_id=row["event_id"],
                    resolved_by=row["resolved_by"],
                    resolved_at=datetime.fromisoformat(row["resolved_at"]),
                    note=row["note"],
                )
                if row["resolved_by"] is not None
                else None
            )
            results.append((_event_from_row(row), resolution))
        return results

    # ── resolutions ──────────────────────────────────────────────────────────
    def add_resolution(self, resolution: EscalationResolution) -> None:
        with self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM escalation_events WHERE event_id = ?", (resolution.event_id,)
            ).fetchone() is None:
                raise EventNotFound(resolution.event_id)
            try:
                connection.execute(
                    "INSERT INTO escalation_resolutions (event_id, resolved_by, resolved_at, note) "
                    "VALUES (?, ?, ?, ?)",
                    (resolution.event_id, resolution.resolved_by,
                     resolution.resolved_at.isoformat(), resolution.note),
                )
            except sqlite3.IntegrityError as error:
                # One resolution per event: a resolution is a record of who closed it and
                # when, and a second one would silently rewrite that history.
                raise AlreadyResolved(resolution.event_id) from error

    # ── cluster snapshots ────────────────────────────────────────────────────
    def save_snapshot(self, snapshot: EscalationClusterSnapshot, events_covered: int) -> None:
        clusters = [
            {"cluster_id": c.cluster_id, "event_ids": list(c.event_ids),
             "representative_text": c.representative_text}
            for c in snapshot.clusters
        ]
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO escalation_cluster_snapshots "
                "(class_id, computed_at, events_covered, clusters_json) VALUES (?, ?, ?, ?)",
                (snapshot.class_id, snapshot.computed_at.isoformat(), events_covered,
                 json.dumps(clusters, ensure_ascii=False)),
            )

    def latest_snapshot(self, class_id: str) -> EscalationClusterSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM escalation_cluster_snapshots WHERE class_id = ? "
                "ORDER BY snapshot_id DESC LIMIT 1",
                (class_id,),
            ).fetchone()
        if row is None:
            return None
        return EscalationClusterSnapshot(
            class_id=row["class_id"],
            computed_at=datetime.fromisoformat(row["computed_at"]),
            clusters=[
                EscalationCluster(
                    cluster_id=c["cluster_id"],
                    event_ids=list(c["event_ids"]),
                    representative_text=c["representative_text"],
                )
                for c in json.loads(row["clusters_json"])
            ],
        )

    def events_since_last_snapshot(self, class_id: str) -> int:
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM escalation_events WHERE class_id = ?", (class_id,)
            ).fetchone()[0]
            covered = connection.execute(
                "SELECT events_covered FROM escalation_cluster_snapshots WHERE class_id = ? "
                "ORDER BY snapshot_id DESC LIMIT 1",
                (class_id,),
            ).fetchone()
        return total - (covered[0] if covered else 0)
