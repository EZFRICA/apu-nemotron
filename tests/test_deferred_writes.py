"""
The deferred-write scheduler: off the request path, retried with exponential backoff.

Target: apu/core/scheduler.py (DeferredWriteScheduler)
"""

import threading

import pytest

from apu.core.scheduler import DeferredWriteScheduler


def _scheduler(**kwargs):
    delays: list[float] = []
    scheduler = DeferredWriteScheduler(sleep=delays.append, **kwargs)
    return scheduler, delays


def test_submit_returns_before_the_work_runs():
    scheduler, _ = _scheduler()
    release = threading.Event()
    done = []
    scheduler.register("slow", lambda payload: (release.wait(5), done.append(payload["n"])))

    scheduler.submit("slow", {"n": 1})
    assert done == [], "the caller did not wait for the write"

    release.set()
    assert scheduler.drain(5)
    assert done == [1]


def test_a_failing_write_is_retried_with_exponential_backoff():
    scheduler, delays = _scheduler(base_delay_seconds=0.5, max_delay_seconds=8.0)
    attempts = []

    def flaky(payload):
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("disk busy")

    scheduler.register("flaky", flaky)
    scheduler.submit("flaky", {})
    assert scheduler.drain(5)

    assert len(attempts) == 3
    assert delays == [0.5, 1.0]
    assert scheduler.dead_letters == []


def test_attempts_are_capped_and_the_failure_is_kept_visible():
    scheduler, delays = _scheduler(max_attempts=5, base_delay_seconds=1.0, max_delay_seconds=3.0)
    scheduler.register("broken", lambda payload: (_ for _ in ()).throw(RuntimeError("nope")))

    scheduler.submit("broken", {"id": "e1"})
    assert scheduler.drain(5)

    assert delays == [1.0, 2.0, 3.0, 3.0], "doubling, capped at max_delay"
    [dead] = scheduler.dead_letters
    assert dead.task_type == "broken" and dead.attempts == 5 and dead.payload == {"id": "e1"}


def test_one_failed_task_does_not_block_the_next():
    scheduler, _ = _scheduler(max_attempts=1)
    done = []
    scheduler.register("broken", lambda payload: 1 / 0)
    scheduler.register("fine", lambda payload: done.append(True))

    scheduler.submit("broken", {})
    scheduler.submit("fine", {})
    assert scheduler.drain(5)
    assert done == [True]


def test_an_unregistered_task_fails_at_submit_not_in_the_background():
    scheduler, _ = _scheduler()
    with pytest.raises(KeyError):
        scheduler.submit("never_registered", {})
