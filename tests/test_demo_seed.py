"""
Demo preparation: reset scope, local course registry import, example escalations.

Target: apu/demo/seed.py, apu/api/service.py (request_cluster_recompute)
"""

import asyncio
import os
import pathlib

import pytest

from apu import config
from apu.api import service
from apu.core.scheduler import DeferredWriteScheduler
from apu.demo import seed
from apu.escalation.jobs import register_escalation_jobs
from apu.guardrails.session import sessions
from apu.mmu import dll as mmu
from apu.mmu.escalation_store import EscalationStore
from tests.conftest import V_A, V_A_OPPOSITE, V_A_SCALED, V_B


def test_reset_only_touches_the_applications_own_state(akili_paths, tmp_path):
    data = pathlib.Path(config.DATA_DIR)
    stranger = data / "not-ours.txt"
    stranger.write_text("keep me")
    (data / "escalations.sqlite3").write_text("x")
    pathlib.Path(config.METADATA_LINKS_PATH).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(config.METADATA_LINKS_PATH).write_text("{}")
    sessions.open_session("eleve-aya", "lycee-cocody:3eA", session_id="to-close")

    seed.reset_demo_data(progress=lambda message: None)

    assert stranger.read_text() == "keep me"
    assert not (data / "escalations.sqlite3").exists()
    assert not pathlib.Path(config.METADATA_LINKS_PATH).exists()
    with pytest.raises(KeyError):
        sessions.get("to-close")


def test_the_stores_still_work_after_a_reset_under_a_running_process(akili_paths):
    """The schema is applied once per database; a wiped file must get it again."""
    from apu.escalation.models import EscalationEvent
    from apu.notebook.store import NotebookStore, new_entry

    now = __import__("datetime").datetime.now(__import__("datetime").UTC)
    EscalationStore().append_event(
        EscalationEvent("e1", "eleve-aya", "lycee-cocody:3eA", "s1", 3, "text", now))
    NotebookStore().add(new_entry("eleve-aya", "6eme", "math", "full", "kept", "kept", "button"))

    seed.reset_demo_data(progress=lambda message: None)

    assert EscalationStore().events_for_class("lycee-cocody:3eA") == []
    assert NotebookStore().entries("eleve-aya") == []
    EscalationStore().append_event(
        EscalationEvent("e2", "eleve-aya", "lycee-cocody:3eA", "s1", 3, "again", now))
    assert [e.event_id for e in EscalationStore().events_for_class("lycee-cocody:3eA")] == ["e2"]


def test_every_reset_path_is_inside_the_redirected_data_dir(akili_paths):
    for path in seed._state_paths():
        assert os.path.realpath(path).startswith(os.path.realpath(config.DATA_DIR)), path


def test_the_local_registry_build_imports_courses_without_gcs(akili_paths, monkeypatch, tmp_path, no_network):
    from cloud_registry.pipeline import batch_pipeline

    class StubEmbedder:
        async def aembed_query(self, text):
            return list(V_A_SCALED)

    registry = tmp_path / "registry"
    registry.mkdir()
    monkeypatch.setattr(batch_pipeline, "REGISTRY_DIR", registry)
    monkeypatch.setattr(batch_pipeline, "_embedder", StubEmbedder())

    courses, prompts_loaded = seed.load_local_course_registry(progress=lambda message: None)

    assert courses["6eme/math"] == 3 and courses["5eme/history"] == 3
    assert prompts_loaded
    assert pathlib.Path(config.LANCE_DB_PATH).parent.joinpath("prompts.json").exists()
    assert "6eme/math" in seed.loaded_courses()


def test_example_escalations_are_stored_and_clustered_per_class(akili_paths):
    # Two themes per class, as seeded: football vs Free Fire in 3eA, Didi B vs TikTok in 4eB.
    def embed(texts):
        return [list(V_A) if "PSG" in t else list(V_B) if "Free Fire" in t else list(V_A_OPPOSITE) for t in texts]

    counts, clusters = seed.seed_escalations(embed_texts=embed, progress=lambda message: None)

    assert counts == {"lycee-cocody:3eA": 6, "lycee-cocody:4eB": 4}
    assert clusters["lycee-cocody:3eA"] == 2
    events = EscalationStore().events_for_class("lycee-cocody:3eA")
    assert {e.attempt_number_in_session for e in events} == {3}, "the class threshold"


def test_prepare_demo_leaves_a_fresh_student_memory(akili_paths, stub_embeddings, monkeypatch):
    monkeypatch.setattr(seed, "seed_escalations", lambda **kwargs: ({}, {}))
    seed.prepare_demo(load_courses=False, progress=lambda message: None)
    assert asyncio.run(mmu.load_dll())["dynamic_block_count"] == 0


def test_the_demo_checks_report_missing_keys(akili_paths, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    checks = {check.label: check for check in seed.demo_checks()}
    assert checks["Tavily key"].ok is False
    assert checks["Courses loaded"].ok is False


def test_a_recompute_request_is_authorized_and_deferred(akili_paths):
    scheduler = DeferredWriteScheduler(sleep=lambda s: None)
    store = EscalationStore()
    register_escalation_jobs(scheduler, store_factory=lambda: store, embed_texts=lambda t: [[1.0]] * len(t))

    with pytest.raises(PermissionError):
        service.request_cluster_recompute("prof-kouassi", "lycee-cocody:4eB", scheduler)
    service.request_cluster_recompute("prof-kouassi", "lycee-cocody:3eA", scheduler)
    assert scheduler.drain(10)
    assert store.latest_snapshot("lycee-cocody:3eA") is not None
