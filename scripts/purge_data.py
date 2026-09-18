#!/usr/bin/env python3
"""
Erase stored student data: escalation events past a retention period, or everything one
student left behind.

    uv run python scripts/purge_data.py --older-than-days 180
    uv run python scripts/purge_data.py --student eleve-aya

Nothing expires on its own. These records are children's own messages, so a school running
this owes itself a retention period and a way to answer an erasure request; this script is
that way, and it is meant to be run on a schedule.

Escalations and notebooks only: the tutoring memory (L1/L2/L3) is reset from the interface
("Reset the student's memory") or by deleting the data directory.
"""

import argparse
import os
import sys
from datetime import UTC, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apu.mmu.escalation_store import EscalationStore  # noqa: E402
from apu.notebook.store import NotebookStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--older-than-days", type=int, help="erase escalations older than this")
    group.add_argument("--student", help="erase this student's escalations and notebook")
    parser.add_argument("--yes", action="store_true", help="do it without asking")
    args = parser.parse_args()

    if args.older_than_days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=args.older_than_days)
        what = f"escalations triggered before {cutoff.date().isoformat()}"
    else:
        what = f"everything stored for {args.student}"

    if not args.yes and input(f"Erase {what}? This cannot be undone. [y/N] ").strip().lower() != "y":
        print("Nothing was erased.")
        return 1

    escalations = EscalationStore()
    if args.older_than_days is not None:
        removed = escalations.delete_events_before(cutoff)
        print(f"Erased {removed} escalation event(s).")
    else:
        removed = escalations.delete_student_events(args.student)
        notebook = NotebookStore()
        entries = notebook.entries(args.student)
        for entry in entries:
            notebook.delete(args.student, entry.entry_id)
        print(f"Erased {removed} escalation event(s) and {len(entries)} notebook entry/entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
