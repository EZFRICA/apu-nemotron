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
from apu.escalation.jobs import (
    ESCALATION_CLUSTER_RECOMPUTE,
    ESCALATION_EVENT_WRITE,
    register_escalation_jobs,
)
from apu.escalation.models import EscalationEvent, EscalationResolution
from apu.mmu import dll as mmu
from apu.mmu.block_types import ESCALATION_EVENT_BLOCK_TYPE, ForbiddenBlockType
from apu.mmu.escalation_store import AlreadyResolved, EscalationStore, EventNotFound
from apu.storage import lance_driver
from tests.conftest import V_A

T0 = datetime(2026, 9, 17, 8, 0, tzinfo=UTC)


def event(event_id, text="Who won the match?", class_id="lycee-cocody:3eA", minutes=0, student="eleve-1"):
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
        if "match" in lowered or "football" in lowered:
            vectors.append([1.0, 0.02 * len(vectors), 0.0])
        elif "game" in lowered or "console" in lowered:
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
    with pytest.raises(ValueError, match="establishment_id:class_code"):
        event("e1", class_id="3eA")


# ── the store ────────────────────────────────────────────────────────────────

def test_appending_is_idempotent_so_a_retried_write_is_harmless(store):
    assert store.append_event(event("e1")) is True
    assert store.append_event(event("e1")) is False
    assert [e.event_id for e in store.events_for_class("lycee-cocody:3eA")] == ["e1"]


def test_resolved_is_a_join_at_read_time(store):
    store.append_event(event("e1", minutes=0))
    store.append_event(event("e2", minutes=1))
    store.add_resolution(EscalationResolution("e1", "prof-kouassi", T0, note="discussed in class"))

    [(first, resolution), (second, none)] = store.list_events_with_resolutions("lycee-cocody:3eA")
    assert first == event("e1", minutes=0), "the event itself is unchanged"
    assert resolution.resolved_by == "prof-kouassi" and resolution.note == "discussed in class"
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
            "esc", "Escalation", ESCALATION_EVENT_BLOCK_TYPE, "text", [], "test", dll, vector=list(V_A)
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
            "esc", "Escalation", ESCALATION_EVENT_BLOCK_TYPE, "text", [], "test", dll, vector=list(V_A)
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
        {"id": "esc", "content": "Who won the match?", "block_type": ESCALATION_EVENT_BLOCK_TYPE,
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


def test_data_can_be_erased_by_age_and_by_student(store):
    """Nothing expires on its own; these are what a retention period and an erasure use."""
    store.append_event(event("old", minutes=0, student="eleve-aya"))
    store.append_event(event("recent", minutes=120, student="eleve-koffi"))
    store.add_resolution(EscalationResolution("old", "prof-kouassi", T0))

    assert store.delete_events_before(T0 + timedelta(minutes=60)) == 1
    assert [e.event_id for e in store.events_for_class("lycee-cocody:3eA")] == ["recent"]

    assert store.delete_student_events("eleve-koffi") == 1
    assert store.events_for_class("lycee-cocody:3eA") == []
    assert store.delete_student_events("nobody") == 0


# ── clustering ───────────────────────────────────────────────────────────────

def test_similar_requests_cluster_and_an_isolated_one_stays_out():
    events = [
        event("football-1", "Who won the match yesterday?", minutes=0),
        event("game-1", "Which console game should I buy?", minutes=1),
        event("football-2", "What was the football match score?", minutes=2),
        event("game-2", "A good video game for my console?", minutes=3),
        event("isolated", "What time is it in Tokyo?", minutes=4),
    ]
    snapshot = cluster_class_events("lycee-cocody:3eA", events, embed_texts=keyword_embed, now=T0)

    assert [c.event_ids for c in snapshot.clusters] == [["football-1", "football-2"], ["game-1", "game-2"]]
    assert [c.representative_text for c in snapshot.clusters] == [
        "Who won the match yesterday?", "Which console game should I buy?",
    ], "the text of each cluster's earliest event"
    assert all("isolated" not in c.event_ids for c in snapshot.clusters), "noise is not forced into a cluster"


def test_clustering_ignores_the_order_events_arrive_in():
    events = [
        event("football-1", "match", minutes=0), event("football-2", "football", minutes=2),
        event("game-1", "game", minutes=1), event("game-2", "console", minutes=3),
    ]
    forward = cluster_class_events("lycee-cocody:3eA", events, embed_texts=keyword_embed, now=T0)
    backward = cluster_class_events("lycee-cocody:3eA", list(reversed(events)), embed_texts=keyword_embed, now=T0)
    assert forward == backward


def test_clustering_never_mixes_classes():
    events = [
        event("a", "match", class_id="lycee-cocody:3eA"),
        event("b", "football match", class_id="lycee-cocody:4eB"),
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
    events = [event("a", "match", minutes=0), event("b", "football", minutes=1), event("c", "match", minutes=2)]
    assert cluster_class_events("lycee-cocody:3eA", events, embed_texts=keyword_embed, now=T0).clusters == []


def test_fewer_than_two_events_yield_no_clusters():
    assert cluster_class_events("lycee-cocody:3eA", [], now=T0).clusters == []
    assert cluster_class_events("lycee-cocody:3eA", [event("a")], now=T0).clusters == []


# ── the deferred job ─────────────────────────────────────────────────────────

def test_clusters_are_recomputed_after_n_new_events_in_the_class(store, monkeypatch):
    monkeypatch.setattr(config, "ESCALATION_CLUSTER_TRIGGER_COUNT", 5)
    scheduler = DeferredWriteScheduler(sleep=lambda s: None)
    register_escalation_jobs(scheduler, store_factory=lambda: store, embed_texts=keyword_embed)

    texts = ["match", "game", "football", "console", "What time is it?"]
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


def test_only_one_recompute_is_queued_while_one_is_pending(store, monkeypatch):
    """
    Past the threshold the count stays high until a snapshot is stored, so without this
    every further event would queue another recomputation of the same class.
    """
    monkeypatch.setattr(config, "ESCALATION_CLUSTER_TRIGGER_COUNT", 2)
    recomputed: list[str] = []

    def counting_embed(texts):
        recomputed.append("run")
        return keyword_embed(texts)

    scheduler = DeferredWriteScheduler(sleep=lambda s: None)
    register_escalation_jobs(scheduler, store_factory=lambda: store, embed_texts=counting_embed)

    for n in range(5):
        scheduler.submit(ESCALATION_EVENT_WRITE, {"event": event(f"e{n}", "match", minutes=n)})
    assert scheduler.drain(10)

    assert len(recomputed) == 1, "the pending recomputation covers the later events too"
    assert store.events_since_last_snapshot("lycee-cocody:3eA") == 0

    # Once it has run, the next batch queues a fresh one.
    scheduler.submit(ESCALATION_EVENT_WRITE, {"event": event("e5", "match", minutes=5)})
    scheduler.submit(ESCALATION_EVENT_WRITE, {"event": event("e6", "foot", minutes=6)})
    assert scheduler.drain(10)
    assert len(recomputed) == 2


def test_a_failed_recompute_does_not_block_the_next_one(store, monkeypatch):
    monkeypatch.setattr(config, "ESCALATION_CLUSTER_TRIGGER_COUNT", 2)
    attempts: list[str] = []

    def failing_embed(texts):
        attempts.append("run")
        raise RuntimeError("embedder down")

    scheduler = DeferredWriteScheduler(max_attempts=1, sleep=lambda s: None)
    register_escalation_jobs(scheduler, store_factory=lambda: store, embed_texts=failing_embed)

    for n in range(2):
        scheduler.submit(ESCALATION_EVENT_WRITE, {"event": event(f"e{n}", "match", minutes=n)})
    assert scheduler.drain(10)
    assert len(attempts) == 1 and store.latest_snapshot("lycee-cocody:3eA") is None
    assert [letter.task_type for letter in scheduler.dead_letters] == [ESCALATION_CLUSTER_RECOMPUTE]

    scheduler.submit(ESCALATION_EVENT_WRITE, {"event": event("e2", "match", minutes=2)})
    assert scheduler.drain(10)
    assert len(attempts) == 2, "the class is not left without clusters for good"
