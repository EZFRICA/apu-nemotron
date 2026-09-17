"""
The Streamlit dashboard, run headless with Streamlit's AppTest.

Target: apu/ui/dashboard.py

AppTest executes the script in-process, so the storage redirection, the stub
embedder, the fake Nebius client and the fake registry all apply to it.
"""

import pathlib

from streamlit.testing.v1 import AppTest

from apu.mmu import dll as mmu
from apu.storage import lance_driver
from tests.conftest import V_A
from tests.registry_fakes import install_fake_registry, make_registry_unreachable, manifest

DASHBOARD = str(pathlib.Path(__file__).resolve().parent.parent / "apu" / "ui" / "dashboard.py")


def _app():
    return AppTest.from_file(DASHBOARD, default_timeout=60)


def _button(at, label_part):
    return next(b for b in at.sidebar.button if label_part in b.label)


# ── registry unreachable ─────────────────────────────────────────────────────

def test_the_dashboard_boots_with_an_unreachable_registry_and_no_course(
    akili_paths, stub_embeddings, monkeypatch
):
    make_registry_unreachable(monkeypatch)
    at = _app().run()

    assert not at.exception, at.exception
    assert any("APU CONTROL CENTER" in t.value for t in at.title)
    assert any("Registry unreachable" in w.value for w in at.sidebar.warning)
    assert not at.sidebar.selectbox


def test_offline_the_courses_already_on_the_device_are_offered(
    akili_paths, stub_embeddings, monkeypatch
):
    make_registry_unreachable(monkeypatch)
    lance_driver.get_db().create_table("edu_registry", data=[
        {"id": f"ch_{cls}_{subj}", "chapter": "c", "content": "x",
         "block_type": "manual_chapter", "class_level": cls, "subject": subj,
         "vector": list(V_A), "updated_at": "2026-01-01T00:00:00"}
        for cls, subj in [("6eme", "math"), ("6eme", "history"), ("5eme", "math")]
    ])

    at = _app().run()

    assert not at.exception, at.exception
    assert any("Using local cache only" in w.value for w in at.sidebar.warning)
    grade, subject = at.sidebar.selectbox
    assert grade.options == ["5eme", "6eme"]
    assert grade.value == "6eme"
    assert subject.options == ["history", "math"]


# ── registry reachable ───────────────────────────────────────────────────────

def test_the_remote_catalog_is_offered_with_a_download_button(
    akili_paths, stub_embeddings, monkeypatch
):
    install_fake_registry(
        monkeypatch, akili_paths,
        manifest(catalog={"6eme": ["history", "math"], "5eme": ["math"]}),
    )
    at = _app().run()

    assert not at.exception, at.exception
    grade, subject = at.sidebar.selectbox
    assert grade.options == ["5eme", "6eme"]
    assert subject.options == ["history", "math"]
    assert _button(at, "Download & Activate")


def test_download_and_activate_imports_the_course_and_switches_context(
    akili_paths, stub_embeddings, monkeypatch
):
    import asyncio

    install_fake_registry(monkeypatch, akili_paths, manifest(catalog={"6eme": ["math"]}))
    at = _app().run()
    _button(at, "Download & Activate").click().run()

    assert not at.exception, at.exception
    assert "edu_registry" in lance_driver.list_table_names()
    dll = asyncio.run(mmu.load_dll())
    assert dll["course_selection"] == {"class": "6eme", "subject": "math"}
    assert any("Active context" in s.value for s in at.sidebar.success)


# ── the chat ─────────────────────────────────────────────────────────────────

def test_a_question_is_answered_by_nemotron_and_shown_in_the_chat(
    akili_paths, stub_embeddings, fake_nebius, monkeypatch
):
    make_registry_unreachable(monkeypatch)
    fake_nebius.main_replies = ["Bonjour ! Que sais-tu déjà des fractions ?"]
    fake_nebius.extraction_replies = [
        '{"current_session": "The student is starting fractions."}'
    ]

    at = _app().run()
    at.chat_input[0].set_value("Explique-moi les fractions").run()

    assert not at.exception, at.exception
    shown = [m.markdown[0].value for m in at.chat_message]
    assert shown == ["Explique-moi les fractions", "Bonjour ! Que sais-tu déjà des fractions ?"]
    assert len(fake_nebius.main_calls) == 1
    assert len(fake_nebius.extraction_calls) == 1


def test_an_inference_failure_is_shown_instead_of_a_traceback(
    akili_paths, stub_embeddings, fake_nebius, monkeypatch
):
    make_registry_unreachable(monkeypatch)
    fake_nebius.main_replies = [RuntimeError("NEBIUS_API_KEY is not set.")]

    at = _app().run()
    at.chat_input[0].set_value("Bonjour").run()

    assert not at.exception, at.exception
    assert any("NEBIUS_API_KEY" in e.value for e in at.error)
