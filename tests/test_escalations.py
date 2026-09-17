"""
Escalation events: immutable records, resolution by join, isolation from tutoring memory,
per-class clustering computed by a deferred job.

Target: apu/escalation/, apu/mmu/escalation_store.py, apu/mmu/block_types.py
"""

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from apu import config
from apu.core.scheduler import DeferredWriteScheduler
from apu.escalation.clustering import cluster_class_events
from apu.escalation.jobs import ESCALATION_EVENT_WRITE, register_escalation_jobs
from apu.escalation.models import EscalationEvent, EscalationResolution
from apu.mmu import dll as mmu
from apu.mmu.block_types import ESCALATION_EVENT_BLOCK_TYPE, ForbiddenBlockType
from apu.mmu.escalation_store import AlreadyResolved, EscalationStore, EventNotFound
from apu.storage import lance_driver
from tests.conftest import V_A

T0 = datetime(2026, 9, 17, 8, 0, tzinfo=UTC)


def event(event_id, text="Qui a gagné le match ?", class_id="lycee-cocody:3eA", minutes=0, student="eleve-1"):
    return EscalationEvent(
        event_id=event_id, student_id=student, class_id=class_id, session_id=f"s-{student}",
        attempt_number_in_session=3, off_topic_request_text=text,
        triggered_at=T0 + timedelta(minutes=minutes),
    )


# Deterministic "embeddings": texts about football, texts about video games, one outlier.
def keyword_embed(texts):
    vectors = []
    for text in texts:
        lowered = text.lower()
        if "match" in lowered or "foot" in lowered:
            vectors.append([1.0, 0.02 * len(vectors), 0.0])
        elif "jeu" in lowered or "console" in lowered:
            vectors.append([0.02 * len(vectors), 1.0, 0.0])
        else:
            vectors.append([0.0, 0.0, 1.0])
    return vectors


@pytest.fixture
def store(akili_paths):
    return EscalationStore()


# ── the records ──────────────────────────────────────────────────────────────

def test_an_event_is_immutable_and_knows_its_establishment():
    e = event("e1")
    assert e.establishment_id == "lycee-cocody"
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.off_topic_request_text = "edited"


def test_a_malformed_class_id_is_refused():
    with pytest.raises(ValueError, match="etablissement_id:classe_code"):
        event("e1", class_id="3eA")


# ── the store ────────────────────────────────────────────────────────────────

def test_appending_is_idempotent_so_a_retried_write_is_harmless(store):
    assert store.append_event(event("e1")) is True
    assert store.append_event(event("e1")) is False
    assert [e.event_id for e in store.events_for_class("lycee-cocody:3eA")] == ["e1"]


def test_resolved_is_a_join_at_read_time(store):
    store.append_event(event("e1", minutes=0))
    store.append_event(event("e2", minutes=1))
    store.add_resolution(EscalationResolution("e1", "prof-kouassi", T0, note="vu en classe"))

    [(first, resolution), (second, none)] = store.list_events_with_resolutions("lycee-cocody:3eA")
    assert first == event("e1", minutes=0), "the event itself is unchanged"
    assert resolution.resolved_by == "prof-kouassi" and resolution.note == "vu en classe"
    assert second.event_id == "e2" and none is None


def test_an_event_is_resolved_once(store):
    store.append_event(event("e1"))
    store.add_resolution(EscalationResolution("e1", "prof-kouassi", T0))
    with pytest.raises(AlreadyResolved):
        store.add_resolution(EscalationResolution("e1", "admin-cocody", T0))


def test_resolving_an_unknown_event_is_refused(store):
    with pytest.raises(EventNotFound):
        store.add_resolution(EscalationResolution("ghost", "prof-kouassi", T0))


def test_reads_are_scoped_to_one_class(store):
    store.append_event(event("e1", class_id="lycee-cocody:3eA"))
    store.append_event(event("e2", class_id="lycee-cocody:4eB"))
    assert [e.event_id for e in store.events_for_class("lycee-cocody:3eA")] == ["e1"]


# ── isolation from the tutoring memory ───────────────────────────────────────

async def test_the_dll_refuses_an_escalation_block(akili_paths):
    dll = await mmu.init_dll()
    with pytest.raises(ForbiddenBlockType):
        await mmu.create_dynamic_block(
            "esc", "Escalation", ESCALATION_EVENT_BLOCK_TYPE, "texte", [], "test", dll, vector=list(V_A)
        )
    assert "esc" not in (await mmu.load_dll())["nodes"]
    assert "user_memory" not in lance_driver.list_table_names()
    with pytest.raises(ForbiddenBlockType):
        mmu.insert_node_by_type(ESCALATION_EVENT_BLOCK_TYPE, {"id": "esc"}, dll)


async def test_a_full_dll_does_not_page_out_a_block_to_make_room_for_a_forbidden_one(akili_paths):
    dll = await mmu.init_dll()
    dll["dynamic_block_max"] = 1
    dll = await mmu.create_dynamic_block("real", "Real", "temp", "c", [], "test", dll, vector=list(V_A))
    with pytest.raises(ForbiddenBlockType):
        await mmu.create_dynamic_block(
            "esc", "Escalation", ESCALATION_EVENT_BLOCK_TYPE, "texte", [], "test", dll, vector=list(V_A)
        )
    assert "real" in dll["nodes"]


async def test_the_l3_driver_refuses_an_escalation_block(akili_paths):
    with pytest.raises(ForbiddenBlockType):
        await lance_driver.upsert_local_block(
            block_id="esc", content="x", block_type=ESCALATION_EVENT_BLOCK_TYPE,
            class_level="6eme", subject="math", vector=list(V_A),
        )


async def test_the_pedagogical_search_never_returns_an_escalation_row(akili_paths):
    """Even a row that got into L3 some other way is dropped from search results."""
    lance_driver.get_db().create_table("user_memory", data=[
        {"id": "esc", "content": "Qui a gagné le match ?", "block_type": ESCALATION_EVENT_BLOCK_TYPE,
         "class_level": "6eme", "subject": "math", "vector": list(V_A), "updated_at": "2026-01-01"},
        {"id": "student_profile", "content": "Marc", "block_type": "fondamental",
         "class_level": "6eme", "subject": "math", "vector": list(V_A), "updated_at": "2026-01-01"},
    ])
    results = await lance_driver.search_block_index(list(V_A))
    assert [r["block_id"] for r in results] == ["student_profile"]


def test_escalations_live_outside_the_l3_store(store, akili_paths):
    store.append_event(event("e1"))
    assert not config.ESCALATION_DB_PATH.startswith(config.LANCE_DB_PATH)
    assert "escalation_events" not in lance_driver.list_table_names()


# ── clustering ───────────────────────────────────────────────────────────────

def test_similar_requests_cluster_and_an_isolated_one_stays_out():
    events = [
        event("foot-1", "Qui a gagné le match hier ?", minutes=0),
        event("jeu-1", "Quel jeu sur console acheter ?", minutes=1),
        event("foot-2", "Le score du match de foot ?", minutes=2),
        event("jeu-2", "Un bon jeu vidéo sur console ?", minutes=3),
        event("isole", "Quelle heure est-il à Tokyo ?", minutes=4),
    ]
    snapshot = cluster_class_events("lycee-cocody:3eA", events, embed_texts=keyword_embed, now=T0)

    assert [c.event_ids for c in snapshot.clusters] == [["foot-1", "foot-2"], ["jeu-1", "jeu-2"]]
    assert [c.representative_text for c in snapshot.clusters] == [
        "Qui a gagné le match hier ?", "Quel jeu sur console acheter ?",
    ], "the text of each cluster's earliest event"
    assert all("isole" not in c.event_ids for c in snapshot.clusters), "noise is not forced into a cluster"


def test_clustering_ignores_the_order_events_arrive_in():
    events = [
        event("foot-1", "match", minutes=0), event("foot-2", "foot", minutes=2),
        event("jeu-1", "jeu", minutes=1), event("jeu-2", "console", minutes=3),
    ]
    forward = cluster_class_events("lycee-cocody:3eA", events, embed_texts=keyword_embed, now=T0)
    backward = cluster_class_events("lycee-cocody:3eA", list(reversed(events)), embed_texts=keyword_embed, now=T0)
    assert forward == backward


def test_clustering_never_mixes_classes():
    events = [
        event("a", "match", class_id="lycee-cocody:3eA"),
        event("b", "match de foot", class_id="lycee-cocody:4eB"),
    ]
    snapshot = cluster_class_events("lycee-cocody:3eA", events, embed_texts=keyword_embed, now=T0)
    assert snapshot.class_id == "lycee-cocody:3eA"
    assert snapshot.clusters == [], "one event in this class cannot form a cluster"


def test_known_limit_a_single_group_of_similar_requests_forms_no_cluster():
    """
    Pins measured HDBSCAN behaviour with the specified parameters (cosine,
    min_cluster_size=2, allow_single_cluster left at False): when a class's events form
    only ONE group, HDBSCAN will not select it and labels everything noise. Setting
    allow_single_cluster=True fixes this case but merges unrelated requests into one
    cluster, which is worse. Open decision, see HACKATHON.md; change this test with it.
    """
    events = [event("a", "match", minutes=0), event("b", "foot", minutes=1), event("c", "match", minutes=2)]
    assert cluster_class_events("lycee-cocody:3eA", events, embed_texts=keyword_embed, now=T0).clusters == []


def test_fewer_than_two_events_yield_no_clusters():
    assert cluster_class_events("lycee-cocody:3eA", [], now=T0).clusters == []
    assert cluster_class_events("lycee-cocody:3eA", [event("a")], now=T0).clusters == []


# ── the deferred job ─────────────────────────────────────────────────────────

def test_clusters_are_recomputed_after_n_new_events_in_the_class(store, monkeypatch):
    monkeypatch.setattr(config, "ESCALATION_CLUSTER_TRIGGER_COUNT", 5)
    scheduler = DeferredWriteScheduler(sleep=lambda s: None)
    register_escalation_jobs(scheduler, store_factory=lambda: store, embed_texts=keyword_embed)

    texts = ["match", "jeu", "foot", "console", "Quelle heure est-il ?"]
    for n, text in enumerate(texts[:4]):
        scheduler.submit(ESCALATION_EVENT_WRITE, {"event": event(f"e{n}", text, minutes=n)})
    assert scheduler.drain(10)
    assert store.latest_snapshot("lycee-cocody:3eA") is None, "below N: nothing computed"

    scheduler.submit(ESCALATION_EVENT_WRITE, {"event": event("e4", texts[4], minutes=4)})
    assert scheduler.drain(10)
    snapshot = store.latest_snapshot("lycee-cocody:3eA")
    assert snapshot is not None
    assert [c.event_ids for c in snapshot.clusters] == [["e0", "e2"], ["e1", "e3"]]
    assert store.events_since_last_snapshot("lycee-cocody:3eA") == 0

    scheduler.submit(ESCALATION_EVENT_WRITE, {"event": event("e3", "match", class_id="lycee-cocody:4eB")})
    assert scheduler.drain(10)
    assert store.latest_snapshot("lycee-cocody:4eB") is None, "another class's counter is separate"
