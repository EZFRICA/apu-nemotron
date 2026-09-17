"""Deferred jobs for escalations: persist an event, then recluster its class when due.

Both run on the deferred-write scheduler, never on the student's request path:
  1. ESCALATION_EVENT_WRITE stores the event (idempotent, so a retried write is harmless),
     then, if the class has accumulated ESCALATION_CLUSTER_TRIGGER_COUNT events since its
     last snapshot, queues a recomputation for that class.
  2. ESCALATION_CLUSTER_RECOMPUTE clusters that class's events and stores a new snapshot.
The read side (apu.api) only ever reads the latest stored snapshot.
"""

import threading

from apu import config
from apu.core.scheduler import DeferredWriteScheduler, deferred_writes
from apu.escalation.clustering import EmbedTexts, cluster_class_events
from apu.escalation.models import EscalationEvent
from apu.logger import get_logger
from apu.mmu.escalation_store import EscalationStore

logger = get_logger(__name__)

ESCALATION_EVENT_WRITE = "escalation_event_write"
ESCALATION_CLUSTER_RECOMPUTE = "escalation_cluster_recompute"

_registration_lock = threading.Lock()


def recompute_class_snapshot(
    class_id: str, store: EscalationStore, embed_texts: EmbedTexts | None = None
) -> None:
    events = store.events_for_class(class_id)
    snapshot = cluster_class_events(class_id, events, embed_texts=embed_texts)
    store.save_snapshot(snapshot, events_covered=len(events))
    logger.info(
        "Escalation clusters recomputed for %s: %d events, %d clusters",
        class_id, len(events), len(snapshot.clusters),
    )


def register_escalation_jobs(
    scheduler: DeferredWriteScheduler,
    store_factory=EscalationStore,
    embed_texts: EmbedTexts | None = None,
) -> None:
    def write_event(payload: dict) -> None:
        event: EscalationEvent = payload["event"]
        store = store_factory()
        store.append_event(event)
        if store.events_since_last_snapshot(event.class_id) >= config.ESCALATION_CLUSTER_TRIGGER_COUNT:
            scheduler.submit(ESCALATION_CLUSTER_RECOMPUTE, {"class_id": event.class_id})

    def recompute(payload: dict) -> None:
        recompute_class_snapshot(payload["class_id"], store_factory(), embed_texts)

    scheduler.register(ESCALATION_EVENT_WRITE, write_event)
    scheduler.register(ESCALATION_CLUSTER_RECOMPUTE, recompute)


def ensure_escalation_jobs_registered(scheduler: DeferredWriteScheduler | None = None) -> None:
    target = scheduler or deferred_writes
    with _registration_lock:
        if not target.has_handler(ESCALATION_EVENT_WRITE):
            register_escalation_jobs(target)


def submit_escalation_event(
    event: EscalationEvent, scheduler: DeferredWriteScheduler | None = None
) -> None:
    target = scheduler or deferred_writes
    ensure_escalation_jobs_registered(target)
    target.submit(ESCALATION_EVENT_WRITE, {"event": event})
