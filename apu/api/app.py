"""Teacher and establishment-admin API (FastAPI).

    uv run uvicorn apu.api.app:app --reload

Every route authorizes through authorize_view before reading or writing anything, and the
requester's role and scope always come from the assignment registry, never from the request.
The operations themselves live in apu.api.service, shared with the demo interface.

AUTHENTICATION IS A STUB (apu.auth.identity): the requester id is read from a plain
X-Requester-Id header. Not secure; see that module before deploying anything.
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from apu.api import service
from apu.auth import assignments
from apu.auth.identity import requester_id_from_header
from apu.escalation.jobs import ensure_escalation_jobs_registered
from apu.escalation.models import EscalationEvent, EscalationResolution
from apu.mmu.escalation_store import AlreadyResolved, EscalationStore, EventNotFound

# The API is unversioned while it is pre-1.0 and consumed only by this repository's own
# interface. Give it a /v1 prefix before anyone else integrates against it.
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 500


class ResolveRequest(BaseModel):
    # Anything else in the body (resolved_by, role, ...) is ignored: who resolved comes from
    # the authenticated requester, never from what the caller declares.
    model_config = ConfigDict(extra="ignore")
    note: str | None = None


def resolution_to_dict(resolution: EscalationResolution | None) -> dict | None:
    if resolution is None:
        return None
    return {
        "event_id": resolution.event_id,
        "resolved_by": resolution.resolved_by,
        "resolved_at": resolution.resolved_at.isoformat(),
        "note": resolution.note,
    }


def event_to_dict(event: EscalationEvent, resolution: EscalationResolution | None) -> dict:
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
        "resolution": resolution_to_dict(resolution),
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
    def list_escalations(
        class_id: str,
        limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
        offset: int = Query(default=0, ge=0),
        requester_id: str = Depends(requester_id_from_header),
    ) -> dict:
        # Paged, always: a class accumulates events for a whole year, and a route that
        # returns "everything" is a route that one day returns everything.
        store = store_factory()
        events = service.list_escalations(requester_id, class_id, store, limit=limit, offset=offset)
        return {
            "class_id": class_id,
            "total": service.count_escalations(requester_id, class_id, store),
            "limit": limit,
            "offset": offset,
            "events": [event_to_dict(event, resolution) for event, resolution in events],
        }

    @app.get("/escalations/clusters")
    def escalation_clusters(class_id: str, requester_id: str = Depends(requester_id_from_header)) -> dict:
        snapshot = service.latest_clusters(requester_id, class_id, store_factory())
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
        try:
            resolution = service.resolve_escalation(
                requester_id, event_id, body.note if body else None, store_factory()
            )
        except EventNotFound as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Unknown escalation event."
            ) from error
        except AlreadyResolved as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="Event already resolved."
            ) from error
        return resolution_to_dict(resolution)

    @app.get("/establishments/{establishment_id}/classes")
    def establishment_classes(
        establishment_id: str, requester_id: str = Depends(requester_id_from_header)
    ) -> dict:
        return {
            "establishment_id": establishment_id,
            "classes": service.visible_classes(requester_id, establishment_id),
        }

    return app


app = create_app()
