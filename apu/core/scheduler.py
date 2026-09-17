"""Background scheduling for the Control Unit.

Two schedulers live here:

  - DeferredWriteScheduler: the deferred-write mechanism the scaffold planned for L4. Work
    that must happen but must not delay the student's answer is queued and executed on a
    background thread, with retries and exponential backoff, capped attempts, and a
    dead-letter list so a write that finally failed is visible rather than lost. Escalation
    events and their clustering go through it (apu.escalation.jobs). Akili had no L4 tier;
    this is the first writer to use the mechanism.

  - LocalScheduler: Akili's scheduler, ported as-is. Its `priority` is stored but not
    honoured (a FIFO asyncio.Queue), GC_OPTIMIZE only logs, and nothing starts it. Kept for
    parity with the source; new work should not build on it.

Why a thread and not asyncio tasks for deferred writes: Streamlit runs each interaction in
its own asyncio.run() and tears the loop down when it returns, which would cancel a
fire-and-forget task before it ran (measured in Akili's COLD_PATH.md). A daemon thread
outlives the request that queued the work, and works the same under FastAPI.

The memory write-back (extraction call) still runs inline in apu.runtime.agent, as in Akili.
"""

import asyncio
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Dict

from apu.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DeadLetter:
    task_type: str
    payload: dict
    attempts: int
    error: str
    failed_at: datetime


class DeferredWriteScheduler:
    def __init__(
        self,
        *,
        max_attempts: int = 5,
        base_delay_seconds: float = 0.5,
        max_delay_seconds: float = 8.0,
        sleep: Callable[[float], None] = time.sleep,
        name: str = "apu-deferred-writes",
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self._sleep = sleep
        self._name = name
        self._queue: "queue.Queue[tuple[str, dict]]" = queue.Queue()
        self._handlers: dict[str, Callable[[dict], None]] = {}
        self._dead_letters: list[DeadLetter] = []
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def register(self, task_type: str, handler: Callable[[dict], None]) -> None:
        self._handlers[task_type] = handler

    def has_handler(self, task_type: str) -> bool:
        return task_type in self._handlers

    def submit(self, task_type: str, payload: dict) -> None:
        # Checked here, on the caller's side: an unregistered task type would otherwise be
        # discovered only in the background, after the caller believed the write was queued.
        if task_type not in self._handlers:
            raise KeyError(f"No handler registered for deferred task {task_type!r}")
        self.start()
        self._queue.put((task_type, payload))

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            task_type, payload = self._queue.get()
            try:
                self._execute_with_retries(task_type, payload)
            finally:
                self._queue.task_done()

    def backoff_delay(self, attempt: int) -> float:
        """Delay after the given failed attempt (1-based): base, 2x base, 4x base..., capped."""
        return min(self.base_delay_seconds * (2 ** (attempt - 1)), self.max_delay_seconds)

    def _execute_with_retries(self, task_type: str, payload: dict) -> None:
        handler = self._handlers[task_type]
        for attempt in range(1, self.max_attempts + 1):
            try:
                handler(payload)
                return
            except Exception as error:
                if attempt == self.max_attempts:
                    logger.error(
                        "Deferred %s failed after %d attempts, moved to dead letters: %s",
                        task_type, attempt, error,
                    )
                    self._dead_letters.append(DeadLetter(
                        task_type=task_type, payload=payload, attempts=attempt,
                        error=str(error), failed_at=datetime.now(UTC),
                    ))
                    return
                delay = self.backoff_delay(attempt)
                logger.warning(
                    "Deferred %s failed (attempt %d/%d), retrying in %.1fs: %s",
                    task_type, attempt, self.max_attempts, delay, error,
                )
                self._sleep(delay)

    def drain(self, timeout: float = 30.0) -> bool:
        """Block until every queued task has finished. For tests and shutdown, never on a request path."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    @property
    def dead_letters(self) -> list[DeadLetter]:
        return list(self._dead_letters)

    @property
    def pending(self) -> int:
        return self._queue.unfinished_tasks


deferred_writes = DeferredWriteScheduler()


class LocalScheduler:
    """
    Manages background asynchronous tasks for the local app (GC, Optimization).
    Ported as-is from Akili; see the module docstring for its limits.
    """
    def __init__(self):
        self.queue = asyncio.Queue()
        self._is_running = False

    async def push(self, task_type: str, payload: Dict[str, Any], priority: int = 1):
        """Adds a task to the queue."""
        await self.queue.put((priority, task_type, payload))
        logger.debug(f"Scheduler: Task added {task_type}")

    async def start(self):
        """Starts the scheduler loop."""
        if self._is_running: return
        self._is_running = True
        logger.info("Local scheduler started.")

        while self._is_running:
            priority, task_type, payload = await self.queue.get()
            try:
                await self._handle_task(task_type, payload)
            except Exception as e:
                logger.error(f"Scheduler error on {task_type}: {e}")
            finally:
                self.queue.task_done()

    async def _handle_task(self, task_type: str, payload: Dict[str, Any]):
        """Executes the task logic."""
        if task_type == "GC_OPTIMIZE":
            # Local LanceDB cleanup/optimization could be implemented here if needed
            logger.debug(f"GC: Optimization for block {payload.get('block_id')}")
        else:
            logger.warning(f"Unknown task: {task_type}")

# Global Instance
scheduler = LocalScheduler()
