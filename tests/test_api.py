"""
Authorization and the teacher/admin routes.

Target: apu/auth/, apu/api/app.py
"""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from apu.api.app import create_app
from apu.auth import assignments
from apu.auth.assignments import AssignmentRegistry, TeacherAssignment
from apu.auth.authorization import authorize_view
from apu.escalation.models import EscalationCluster, EscalationClusterSnapshot, EscalationEvent
from apu.mmu.escalation_store import EscalationStore

T0 = datetime(2026, 9, 17, 8, 0, tzinfo=UTC)

REGISTRY = AssignmentRegistry([
    TeacherAssignment("prof-kouassi", "teacher", "lycee-cocody", "lycee-cocody:3eA"),
    TeacherAssignment("prof-traore", "teacher", "lycee-cocody", "lycee-cocody:4eB"),
    TeacherAssignment("admin-cocody", "establishment_admin", "lycee-cocody"),
    TeacherAssignment("prof-bamba", "teacher", "college-yopougon", "college-yopougon:6eC"),
])


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(assignments, "_registry", REGISTRY)


def event(event_id, class_id="lycee-cocody:3eA"):
    return EscalationEvent(event_id, "eleve-1", class_id, "s1", 3, "Who won the match?", T0)


@pytest.fixture
def api(akili_paths):
    store = EscalationStore()
    store.append_event(event("e-3eA"))
    store.append_event(event("e-4eB", class_id="lycee-cocody:4eB"))
    return TestClient(create_app(store_factory=lambda: store)), store


def as_user(requester_id, **headers):
    return {"X-Requester-Id": requester_id, **headers}


# ── authorize_view ───────────────────────────────────────────────────────────

def test_a_teacher_is_scoped_to_their_class():
    scope = authorize_view("prof-kouassi", "lycee-cocody:3eA")
    assert (scope.role, scope.establishment_id, scope.class_id) == ("teacher", "lycee-cocody", "lycee-cocody:3eA")
    with pytest.raises(PermissionError, match="teacher's scope"):
        authorize_view("prof-kouassi", "lycee-cocody:4eB")


def test_an_admin_is_scoped_to_their_establishment():
    assert authorize_view("admin-cocody", "lycee-cocody:4eB").class_id is None
    with pytest.raises(PermissionError, match="admin's establishment"):
        authorize_view("admin-cocody", "college-yopougon:6eC")


def test_an_unknown_requester_has_no_scope():
    with pytest.raises(PermissionError, match="no registered role"):
        authorize_view("intrus", "lycee-cocody:3eA")


def test_an_establishment_prefix_is_not_confused_with_a_longer_name():
    """'lycee-cocody' must not grant 'lycee-cocody-2:...'."""
    with pytest.raises(PermissionError):
        authorize_view("admin-cocody", "lycee-cocody-2:1A")


@pytest.mark.parametrize("entry,message", [
    (dict(requester_id="p", role="teacher", establishment_id="e"), "no class_id"),
    (dict(requester_id="p", role="teacher", establishment_id="e", class_id="other:1A"), "not in establishment"),
    (dict(requester_id="p", role="establishment_admin", establishment_id="e", class_id="e:1A"), "must not be tied"),
    (dict(requester_id="p", role="superuser", establishment_id="e"), "Unknown role"),
])
def test_the_registry_refuses_inconsistent_assignments(entry, message):
    with pytest.raises(ValueError, match=message):
        TeacherAssignment(**entry)


def test_the_shipped_demo_registries_load():
    from apu import config
    from apu.guardrails.policy import ClassPolicyRegistry

    assert AssignmentRegistry.load(config.TEACHER_ASSIGNMENTS_PATH).lookup("admin-cocody").role == "establishment_admin"
    assert ClassPolicyRegistry.load(config.CLASS_POLICIES_PATH).get("lycee-cocody:3eA").escalation_threshold == 3


# ── authentication stub ──────────────────────────────────────────────────────

def test_no_requester_header_is_401(api):
    client, _ = api
    assert client.get("/escalations", params={"class_id": "lycee-cocody:3eA"}).status_code == 401


# ── GET /escalations ─────────────────────────────────────────────────────────

def test_a_teacher_lists_their_class(api):
    client, _ = api
    response = client.get("/escalations", params={"class_id": "lycee-cocody:3eA"}, headers=as_user("prof-kouassi"))
    assert response.status_code == 200
    [only] = response.json()["events"]
    assert only["event_id"] == "e-3eA" and only["status"] == "open" and only["resolution"] is None
    assert only["establishment_id"] == "lycee-cocody"


def test_a_declared_role_is_ignored(api):
    client, _ = api
    response = client.get(
        "/escalations",
        params={"class_id": "lycee-cocody:4eB", "role": "establishment_admin"},
        headers=as_user("prof-kouassi", **{"X-Role": "establishment_admin"}),
    )
    assert response.status_code == 403


@pytest.mark.parametrize("requester,class_id,status", [
    ("prof-kouassi", "lycee-cocody:4eB", 403),
    ("prof-bamba", "lycee-cocody:3eA", 403),
    ("intrus", "lycee-cocody:3eA", 403),
    ("admin-cocody", "lycee-cocody:4eB", 200),
    ("admin-cocody", "college-yopougon:6eC", 403),
])
def test_listing_is_authorized_from_the_registry(api, requester, class_id, status):
    client, _ = api
    assert client.get("/escalations", params={"class_id": class_id}, headers=as_user(requester)).status_code == status


def test_listing_is_paged(api):
    client, store = api
    for index in range(5):
        store.append_event(event(f"e-page-{index}"))

    first = client.get("/escalations", params={"class_id": "lycee-cocody:3eA", "limit": 2},
                       headers=as_user("prof-kouassi")).json()
    assert first["total"] == 6 and first["limit"] == 2 and len(first["events"]) == 2

    second = client.get("/escalations", params={"class_id": "lycee-cocody:3eA", "limit": 2, "offset": 2},
                        headers=as_user("prof-kouassi")).json()
    assert [e["event_id"] for e in second["events"]] != [e["event_id"] for e in first["events"]]

    over_the_cap = client.get("/escalations", params={"class_id": "lycee-cocody:3eA", "limit": 10_000},
                              headers=as_user("prof-kouassi"))
    assert over_the_cap.status_code == 422, "the page size has a ceiling"


# ── POST /escalations/{event_id}/resolve ─────────────────────────────────────

def test_resolving_records_the_authenticated_requester(api):
    client, store = api
    response = client.post(
        "/escalations/e-3eA/resolve",
        json={"note": "talked it through with the student", "resolved_by": "admin-cocody"},
        headers=as_user("prof-kouassi"),
    )
    assert response.status_code == 201
    assert response.json()["resolved_by"] == "prof-kouassi", "never taken from the body"

    [(stored, resolution)] = store.list_events_with_resolutions("lycee-cocody:3eA")
    assert stored == event("e-3eA"), "the event itself is untouched"
    assert resolution.note == "talked it through with the student"

    listed = client.get("/escalations", params={"class_id": "lycee-cocody:3eA"}, headers=as_user("prof-kouassi"))
    assert listed.json()["events"][0]["status"] == "resolved"


def test_resolving_twice_is_a_conflict(api):
    client, _ = api
    assert client.post("/escalations/e-3eA/resolve", headers=as_user("prof-kouassi")).status_code == 201
    assert client.post("/escalations/e-3eA/resolve", headers=as_user("admin-cocody")).status_code == 409


def test_a_teacher_cannot_resolve_another_class(api):
    client, store = api
    assert client.post("/escalations/e-4eB/resolve", headers=as_user("prof-kouassi")).status_code == 403
    assert store.list_events_with_resolutions("lycee-cocody:4eB")[0][1] is None


def test_an_unknown_event_is_404_but_only_for_someone_with_a_role(api):
    client, _ = api
    assert client.post("/escalations/ghost/resolve", headers=as_user("prof-kouassi")).status_code == 404
    assert client.post("/escalations/ghost/resolve", headers=as_user("intrus")).status_code == 403


# ── GET /escalations/clusters ────────────────────────────────────────────────

def test_clusters_are_read_from_the_last_snapshot_and_never_computed(api):
    client, store = api
    empty = client.get("/escalations/clusters", params={"class_id": "lycee-cocody:3eA"}, headers=as_user("prof-kouassi"))
    assert empty.status_code == 200 and empty.json() == {"class_id": "lycee-cocody:3eA", "computed_at": None, "clusters": []}
    assert store.latest_snapshot("lycee-cocody:3eA") is None, "reading did not compute anything"

    store.save_snapshot(EscalationClusterSnapshot(
        "lycee-cocody:3eA", T0, [EscalationCluster(0, ["e-3eA", "e-x"], "Who won the match?")]
    ), events_covered=2)
    body = client.get("/escalations/clusters", params={"class_id": "lycee-cocody:3eA"}, headers=as_user("prof-kouassi")).json()
    assert body["computed_at"] == T0.isoformat()
    assert body["clusters"] == [{"cluster_id": 0, "size": 2, "event_ids": ["e-3eA", "e-x"],
                                 "representative_text": "Who won the match?"}]


def test_clusters_of_another_class_are_forbidden(api):
    client, _ = api
    assert client.get("/escalations/clusters", params={"class_id": "lycee-cocody:4eB"},
                      headers=as_user("prof-kouassi")).status_code == 403


# ── GET /establishments/{establishment_id}/classes ───────────────────────────

@pytest.mark.parametrize("requester,establishment,status,classes", [
    ("admin-cocody", "lycee-cocody", 200, ["lycee-cocody:3eA", "lycee-cocody:4eB"]),
    ("prof-kouassi", "lycee-cocody", 200, ["lycee-cocody:3eA"]),
    ("prof-bamba", "lycee-cocody", 403, None),
    ("admin-cocody", "college-yopougon", 403, None),
    ("intrus", "lycee-cocody", 403, None),
])
def test_establishment_classes_come_from_the_assignment_registry(api, requester, establishment, status, classes):
    client, _ = api
    response = client.get(f"/establishments/{establishment}/classes", headers=as_user(requester))
    assert response.status_code == status
    if classes is not None:
        assert response.json() == {"establishment_id": establishment, "classes": classes}
