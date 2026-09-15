"""Background task scheduler for the Control Unit.

Ported as-is from Akili (app_local/core/scheduler.py). The scaffold listed three
responsibilities for this module. The Akili source does not implement them, so they
are recorded here as open rather than invented during the port:

  - deferred writes to L4 (long-term archive). Akili has no L4 tier: L1 is RAM, L2
    the JSON DLL, L3 LanceDB. The deferred-write design exists only in the
    travel-agent APU (EZFRICA/Agent-Processor-Unit, apu/core/scheduler.py,
    SYNC_LETTA) and was not carried into Akili.
  - auto-retry with backoff. Also travel-agent APU only, and there it is a constant
    2 s sleep over 3 attempts with a priority bump, not an exponential backoff.
  - background GC / move-to-front during idle cycles. GC_OPTIMIZE below only logs.
    In Akili the move-to-front runs synchronously inside apu.mmu.dll.search_memory
    (BMJ), on the request path.

Two more properties of the ported code worth knowing before relying on it:

  - `priority` is stored but not honoured: the queue is a FIFO asyncio.Queue, not a
    PriorityQueue.
  - nothing starts or pushes to this scheduler, same as in Akili (its dashboard
    imported it and never called it).

The extraction call does not go through here either. In Akili the memory write-back
runs inline at the end of the turn (apu.runtime.agent._generate), awaited before the
answer is returned. Moving it off the critical path was prototyped and measured in
Akili's COLD_PATH.md (-34.6% perceived turn latency, needs a daemon thread and a
drain in the test fixtures) but deliberately not shipped. That decision is still open,
so the port keeps the inline behaviour.
"""

import asyncio
from typing import Any, Dict

from apu.logger import get_logger

logger = get_logger(__name__)

class LocalScheduler:
    """
    Manages background asynchronous tasks for the local app (GC, Optimization).
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
