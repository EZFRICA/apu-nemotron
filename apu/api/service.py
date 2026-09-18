"""Teacher and admin operations, shared by the HTTP API and the demo interface.

Both entry points go through these functions, so the interface cannot show or change
anything the API would refuse: every operation starts with authorize_view (or, for the
establishment listing, a per-class authorize_view), with the requester's scope taken from
the assignment registry.
"""

from datetime import UTC, datetime

from apu.auth import assignments
from apu.auth.authorization import authorize_view
from apu.core.scheduler import DeferredWriteScheduler, deferred_writes
from apu.escalation.jobs import ESCALATION_CLUSTER_RECOMPUTE, ensure_escalation_jobs_registered
from apu.escalation.models import EscalationClusterSnapshot, EscalationEvent, EscalationResolution
from apu.mmu.escalation_store import EscalationStore, EventNotFound


def _require_role(requester_id: str) -> None:
    if assignments.lookup_assignment(requester_id) is None:
        raise PermissionError(f"{requester_id} has no registered role.")


def list_escalations(
    requester_id: str, class_id: str, store: EscalationStore,
    limit: int | None = None, offset: int = 0,
) -> list[tuple[EscalationEvent, EscalationResolution | None]]:
    authorize_view(requester_id, class_id)
    return store.list_events_with_resolutions(class_id, limit=limit, offset=offset)


def count_escalations(requester_id: str, class_id: str, store: EscalationStore) -> int:
    authorize_view(requester_id, class_id)
    return store.count_events(class_id)


def latest_clusters(
    requester_id: str, class_id: str, store: EscalationStore
) -> EscalationClusterSnapshot | None:
    authorize_view(requester_id, class_id)
    # Read only: clusters are computed by the deferred job, never on this path.
    return store.latest_snapshot(class_id)


def resolve_escalation(
    requester_id: str, event_id: str, note: str | None, store: EscalationStore
) -> EscalationResolution:
    """Raises PermissionError, EventNotFound or AlreadyResolved."""
    # A requester with no role learns nothing, not even whether the event exists.
    _require_role(requester_id)
    event = store.get_event(event_id)
    if event is None:
        raise EventNotFound(event_id)
    authorize_view(requester_id, event.class_id)
    resolution = EscalationResolution(
        event_id=event_id,
        resolved_by=requester_id,
        resolved_at=datetime.now(UTC),
        note=note or None,
    )
    store.add_resolution(resolution)
    return resolution


def visible_classes(requester_id: str, establishment_id: str) -> list[str]:
    assignment = assignments.lookup_assignment(requester_id)
    if assignment is None:
        raise PermissionError(f"{requester_id} has no registered role.")
    if assignment.establishment_id != establishment_id:
        raise PermissionError("Establishment outside the requester's scope.")
    # Derived from the assignment registry, not from class policies; each class listed
    # only if authorize_view accepts it for this requester (admin: all, teacher: own).
    visible = []
    for class_id in assignments.get_assignment_registry().classes_in_establishment(establishment_id):
        try:
            authorize_view(requester_id, class_id)
        except PermissionError:
            continue
        visible.append(class_id)
    return visible


def request_cluster_recompute(
    requester_id: str, class_id: str, scheduler: DeferredWriteScheduler | None = None
) -> None:
    """Queue a recomputation on the deferred job (demo control); never computed inline."""
    authorize_view(requester_id, class_id)
    target = scheduler or deferred_writes
    ensure_escalation_jobs_registered(target)
    target.submit(ESCALATION_CLUSTER_RECOMPUTE, {"class_id": class_id})
