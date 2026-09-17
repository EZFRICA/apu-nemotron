"""Teacher and establishment-admin API (FastAPI).

    uv run uvicorn apu.api.app:app --reload

Every route authorizes through authorize_view before reading or writing anything, and the
requester's role and scope always come from the assignment registry, never from the request.

AUTHENTICATION IS A STUB (apu.auth.identity): the requester id is read from a plain
X-Requester-Id header. Not secure; see that module before deploying anything.
"""

from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from apu.auth import assignments
from apu.auth.authorization import authorize_view
from apu.auth.identity import requester_id_from_header
from apu.escalation.jobs import ensure_escalation_jobs_registered
from apu.escalation.models import EscalationEvent, EscalationResolution
from apu.mmu.escalation_store import AlreadyResolved, EscalationStore, EventNotFound


class ResolveRequest(BaseModel):
    # Anything else in the body (resolved_by, role, ...) is ignored: who resolved comes from
    # the authenticated requester, never from what the caller declares.
    model_config = ConfigDict(extra="ignore")
    note: str | None = None


def _resolution_to_dict(resolution: EscalationResolution | None) -> dict | None:
    if resolution is None:
        return None
    return {
        "event_id": resolution.event_id,
        "resolved_by": resolution.resolved_by,
        "resolved_at": resolution.resolved_at.isoformat(),
        "note": resolution.note,
    }


def _event_to_dict(event: EscalationEvent, resolution: EscalationResolution | None) -> dict:
    return {
        "event_id": event.event_id,
        "student_id": event.student_id,
        "class_id": event.class_id,
        "establishment_id": event.establishment_id,
        "session_id": event.session_id,
        "attempt_number_in_session": event.attempt_number_in_session,
        "off_topic_request_text": event.off_topic_request_text,
        "triggered_at": event.triggered_at.isoformat(),
        "status": "resolved" if resolution else "open",
        "resolution": _resolution_to_dict(resolution),
    }


def create_app(*, store_factory=EscalationStore) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Loaded at startup, like ClassPolicy. get_* keeps an already-installed registry.
        assignments.get_assignment_registry()
        ensure_escalation_jobs_registered()
        yield

    app = FastAPI(title="APU — teacher and admin API", lifespan=lifespan)

    @app.exception_handler(PermissionError)
    async def permission_denied(request: Request, error: PermissionError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content={"detail": str(error)})

    @app.get("/escalations")
    def list_escalations(class_id: str, requester_id: str = Depends(requester_id_from_header)) -> dict:
        authorize_view(requester_id, class_id)
        events = store_factory().list_events_with_resolutions(class_id)
        return {
            "class_id": class_id,
            "events": [_event_to_dict(event, resolution) for event, resolution in events],
        }

    @app.get("/escalations/clusters")
    def escalation_clusters(class_id: str, requester_id: str = Depends(requester_id_from_header)) -> dict:
        authorize_view(requester_id, class_id)
        # Read only: clusters are computed by the deferred job, never on this path.
        snapshot = store_factory().latest_snapshot(class_id)
        if snapshot is None:
            return {"class_id": class_id, "computed_at": None, "clusters": []}
        return {
            "class_id": snapshot.class_id,
            "computed_at": snapshot.computed_at.isoformat(),
            "clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "size": cluster.size,
                    "event_ids": list(cluster.event_ids),
                    "representative_text": cluster.representative_text,
                }
                for cluster in snapshot.clusters
            ],
        }

    @app.post("/escalations/{event_id}/resolve", status_code=status.HTTP_201_CREATED)
    def resolve_escalation(
        event_id: str,
        body: ResolveRequest | None = None,
        requester_id: str = Depends(requester_id_from_header),
    ) -> dict:
        # A requester with no role learns nothing, not even whether the event exists.
        if assignments.lookup_assignment(requester_id) is None:
            raise PermissionError(f"{requester_id} n'a aucun rôle enregistré.")
        store = store_factory()
        event = store.get_event(event_id)
        if event is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown escalation event.")
        authorize_view(requester_id, event.class_id)

        resolution = EscalationResolution(
            event_id=event_id,
            resolved_by=requester_id,
            resolved_at=datetime.now(UTC),
            note=body.note if body else None,
        )
        try:
            store.add_resolution(resolution)
        except AlreadyResolved:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Event already resolved.")
        except EventNotFound:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown escalation event.")
        return _resolution_to_dict(resolution)

    @app.get("/establishments/{establishment_id}/classes")
    def establishment_classes(
        establishment_id: str, requester_id: str = Depends(requester_id_from_header)
    ) -> dict:
        assignment = assignments.lookup_assignment(requester_id)
        if assignment is None:
            raise PermissionError(f"{requester_id} n'a aucun rôle enregistré.")
        if assignment.establishment_id != establishment_id:
            raise PermissionError("Établissement hors du périmètre du demandeur.")
        # Derived from the assignment registry, not from class policies; each class listed
        # only if authorize_view accepts it for this requester (admin: all, teacher: own).
        visible = []
        for class_id in assignments.get_assignment_registry().classes_in_establishment(establishment_id):
            try:
                authorize_view(requester_id, class_id)
            except PermissionError:
                continue
            visible.append(class_id)
        return {"establishment_id": establishment_id, "classes": visible}

    return app


app = create_app()
