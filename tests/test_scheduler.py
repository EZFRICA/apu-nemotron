"""
The ported LocalScheduler, as it actually behaves.

Target: apu/core/scheduler.py

Akili had no tests for it. These pin the ported behaviour, including the gaps
against the scaffold's description (no priority ordering, no retry), so that
implementing any of them shows up as a deliberate test change.
"""

import asyncio

from apu.core.scheduler import LocalScheduler


async def _run_until_drained(scheduler: LocalScheduler):
    worker = asyncio.create_task(scheduler.start())
    await asyncio.wait_for(scheduler.queue.join(), timeout=5)
    worker.cancel()


async def test_tasks_are_processed_fifo_regardless_of_priority():
    """
    The queue is a plain asyncio.Queue: `priority` is stored, not honoured. The
    scaffold describes a priority queue; this pins the gap.
    """
    scheduler = LocalScheduler()
    handled = []

    async def record(task_type, payload):
        handled.append(payload["id"])

    scheduler._handle_task = record
    await scheduler.push("GC_OPTIMIZE", {"id": "low"}, priority=3)
    await scheduler.push("GC_OPTIMIZE", {"id": "high"}, priority=0)
    await _run_until_drained(scheduler)

    assert handled == ["low", "high"]


async def test_a_failing_task_is_not_retried_and_does_not_stop_the_loop():
    scheduler = LocalScheduler()
    attempts = {"boom": 0, "ok": 0}

    async def flaky(task_type, payload):
        attempts[payload["id"]] += 1
        if payload["id"] == "boom":
            raise RuntimeError("fails")

    scheduler._handle_task = flaky
    await scheduler.push("X", {"id": "boom"})
    await scheduler.push("X", {"id": "ok"})
    await _run_until_drained(scheduler)

    assert attempts == {"boom": 1, "ok": 1}


async def test_gc_optimize_and_unknown_tasks_are_log_only():
    """GC_OPTIMIZE does not reorder anything; move-to-front lives in search_memory."""
    scheduler = LocalScheduler()
    await scheduler.push("GC_OPTIMIZE", {"block_id": "student_profile"})
    await scheduler.push("SOMETHING_ELSE", {})
    await _run_until_drained(scheduler)
    assert scheduler.queue.empty()
