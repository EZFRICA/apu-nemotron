"""
The demo interface, run headless with Streamlit's AppTest.

Target: apu/ui/app.py, apu/ui/views/*.py, apu/ui/common.py

AppTest executes each view in-process, so the storage redirection, the stub embedder, the
scripted topical guard, the fake Nebius client and the fake registry all apply to it.
"""

import pathlib
from datetime import UTC, datetime, timedelta

import pytest
from streamlit.testing.v1 import AppTest

from apu.escalation.models import EscalationEvent
from apu.mmu.escalation_store import EscalationStore
from apu.ui.common import DemoIdentity
from tests.conftest import OFF_TOPIC_MARKER, tool_call_reply
from tests.registry_fakes import install_fake_registry, manifest

UI = pathlib.Path(__file__).resolve().parent.parent / "apu" / "ui"
PROF = DemoIdentity("teacher", "prof-kouassi", "prof-kouassi", "lycee-cocody", "lycee-cocody:3eA")
ADMIN = DemoIdentity("establishment_admin", "admin-cocody", "admin-cocody", "lycee-cocody")


def view(name, identity=None):
    app = AppTest.from_file(str(UI / "views" / f"{name}.py"), default_timeout=60)
    if identity is not None:
        app.session_state["identity"] = identity
    return app


def texts(app):
    return " ".join(m.value for m in app.markdown) + " " + " ".join(c.value for c in app.caption)


@pytest.fixture
def ui(akili_paths, stub_embeddings, no_network):
    return akili_paths


# ── navigation ───────────────────────────────────────────────────────────────

def test_the_app_boots_on_the_student_page(ui):
    app = AppTest.from_file(str(UI / "app.py"), default_timeout=60).run()
    assert not app.exception, app.exception
    assert any("Sign in as" in s.label for s in app.sidebar.selectbox)
    assert any("Akili, your tutor" in t.value for t in app.title)


# ── student view ─────────────────────────────────────────────────────────────

def test_the_student_view_shows_the_guard_state(ui):
    app = view("student").run()
    assert not app.exception, app.exception
    assert app.metric[0].value == "0 / 3", "lycee-cocody:3eA threshold from the class policy"


def test_a_question_is_answered_with_its_search_and_sources(ui, fake_nebius, monkeypatch):
    from apu.tools import web_search

    class Tool:
        def __init__(self, **kwargs):
            pass

        async def ainvoke(self, payload):
            return {"results": [{"title": "Fraction — Wikipedia",
                                 "url": "https://fr.wikipedia.org/wiki/Fraction", "content": "..."}]}

    monkeypatch.setattr(web_search, "_web_search", web_search.TavilySearch(tool_factory=Tool))
    fake_nebius.main_replies = [tool_call_reply("web_search", {"query": "fractions"}), "A fraction is a share of a whole."]
    fake_nebius.extraction_replies = ["{}"]

    app = view("student").run()
    app.chat_input[0].set_value("Explain fractions to me").run()

    assert not app.exception, app.exception
    shown = texts(app)
    assert "A fraction is a share of a whole." in shown
    assert "https://fr.wikipedia.org/wiki/Fraction" in shown
    assert "“fractions”" in shown


def test_off_topic_questions_move_the_counter_and_show_the_guard(ui, fake_nebius):
    app = view("student").run()
    app.chat_input[0].set_value(f"{OFF_TOPIC_MARKER} Who won the match?").run()

    assert not app.exception, app.exception
    assert app.metric[0].value == "1 / 3"
    assert "Guard: off-topic" in texts(app)
    assert fake_nebius.calls == []


def test_switching_student_leaves_no_memory_from_the_previous_one(ui):
    """
    The tutoring memory is per device, not per student (README, "Known limits"), so the
    interface wipes it when the person changes. Otherwise the next student would open on the
    previous one's profile and current session.
    """
    import asyncio

    from apu.mmu import dll as mmu
    from apu.ui.common import clear_device_memory

    async def remember():
        dll = await mmu.init_dll()
        await mmu.update_node_content("student_profile", "Aya, wants to be a doctor.", dll)

    asyncio.run(remember())
    assert "Aya" in asyncio.run(mmu.load_dll())["nodes"]["student_profile"]["content"]

    clear_device_memory()

    profile = asyncio.run(mmu.load_dll())["nodes"]["student_profile"]
    assert "Aya" not in (profile.get("content") or ""), "the next student starts clean"


def test_a_teacher_identity_is_sent_away_from_the_student_page(ui):
    app = view("student", PROF).run()
    assert any("student page" in i.value for i in app.info)


def test_the_cloud_registry_is_only_read_on_demand(ui, monkeypatch):
    requested = install_fake_registry(monkeypatch, ui, manifest(catalog={"6eme": ["math"]}))
    app = view("student").run()
    assert requested == [], "no registry call on page load"

    next(b for b in app.button if b.label == "Browse the cloud registry").click().run()
    assert not app.exception, app.exception
    assert requested == ["manifest.json"]
    assert any(s.label == "Registry courses" for s in app.selectbox)


# ── teacher / admin view ─────────────────────────────────────────────────────

def _seed_event(event_id, class_id="lycee-cocody:3eA", text="Who won the match?"):
    EscalationStore().append_event(EscalationEvent(
        event_id, "eleve-aya", class_id, "s1", 3, text, datetime.now(UTC) - timedelta(minutes=5)))


def test_a_teacher_sees_and_resolves_their_class_escalations(ui):
    _seed_event("e1")
    _seed_event("e2", class_id="lycee-cocody:4eB", text="other class")
    app = view("teacher", PROF).run()

    assert not app.exception, app.exception
    assert [s.options for s in app.selectbox if s.label == "Class"] == [["lycee-cocody:3eA"]]
    assert "Who won the match?" in texts(app) and "other class" not in texts(app)

    app.button(key="FormSubmitter:resolve-e1-Mark as resolved").click().run()
    assert not app.exception, app.exception
    [(_, resolution)] = EscalationStore().list_events_with_resolutions("lycee-cocody:3eA")
    assert resolution.resolved_by == "prof-kouassi"


def test_a_students_text_cannot_inject_markup_into_the_teachers_page(ui):
    """
    An off-topic message is typed by a student and shown to their teacher inside a div
    with unsafe_allow_html. Streamlit strips scripts, but a raw <a href> or <style> would
    survive: a phishing link in the teacher's dashboard. It must arrive escaped.
    """
    payload = '<a href="https://evil.example/login">Reset your password</a><style>body{display:none}</style>'
    _seed_event("e-payload", text=payload)
    app = view("teacher", PROF).run()

    assert not app.exception, app.exception
    quoted = [m.value for m in app.markdown if "apu-quote" in m.value]
    assert quoted, "the request is shown to the teacher"
    assert all("<a href" not in value and "<style>" not in value for value in quoted)
    assert any("&lt;a href=" in value and "evil.example" in value for value in quoted), (
        "escaped, and still readable as the words the student typed"
    )


def test_an_admin_sees_every_class_of_the_establishment(ui):
    app = view("teacher", ADMIN).run()
    assert [s.options for s in app.selectbox if s.label == "Class"] == [["lycee-cocody:3eA", "lycee-cocody:4eB"]]


def test_access_outside_the_scope_is_refused_on_screen(ui):
    app = view("teacher", PROF).run()
    app.text_input(key="access_target").set_value("college-yopougon:6eC")
    next(b for b in app.button if b.label == "Open this class").click().run()
    assert any("403" in e.value and "teacher's scope" in e.value for e in app.error)


def test_a_student_identity_is_sent_away_from_the_teacher_page(ui):
    app = view("teacher").run()
    assert any("teacher / admin page" in i.value for i in app.info)


# ── demo page ────────────────────────────────────────────────────────────────

def test_the_demo_page_lists_checks_and_seeds_example_escalations(ui):
    app = view("demo").run()
    assert not app.exception, app.exception
    shown = texts(app)
    assert "Nebius Token Factory key" in shown and "Courses loaded" in shown

    next(b for b in app.button if b.label == "Example escalations only").click().run()
    assert not app.exception, app.exception
    assert len(EscalationStore().events_for_class("lycee-cocody:3eA")) == 6


def test_preparing_the_demo_requires_confirmation(ui):
    app = view("demo").run()
    prepare = next(b for b in app.button if b.label == "Prepare the demo")
    assert prepare.disabled


# ── rendering helpers ────────────────────────────────────────────────────────

def test_links_written_by_the_model_are_not_clickable_for_the_student():
    """
    A page the tutor read during a search can ask it to put a link in its answer. The
    address stays readable, but as code: nothing the model wrote is clickable.
    """
    from apu.ui.common import defang_links

    assert defang_links("See [my site](https://evil.example/login) now") == (
        "See my site (`https://evil.example/login`) now"
    )
    assert defang_links("Go to https://evil.example/x") == "Go to `https://evil.example/x`"
    assert defang_links("A fraction is a share.") == "A fraction is a share."
    assert "](http" not in defang_links("[a](https://a.fr) and [b](http://b.fr)")


def test_a_link_in_an_answer_reaches_the_page_defanged(ui, fake_nebius):
    fake_nebius.main_replies = ["Read more at [this page](https://evil.example/login)."]
    fake_nebius.extraction_replies = ["{}"]
    app = view("student").run()
    app.chat_input[0].set_value("Explain fractions").run()

    assert not app.exception, app.exception
    shown = texts(app)
    assert "](https://evil.example/login)" not in shown
    assert "`https://evil.example/login`" in shown


def test_nemotron_latex_delimiters_become_streamlit_math():
    from apu.ui.common import math_for_streamlit

    assert math_for_streamlit(r"Let us add \( \frac{1}{4} + \frac{1}{6} \).") == r"Let us add $\frac{1}{4} + \frac{1}{6}$."
    assert math_for_streamlit(r"\[ \frac{3}{12} + \frac{2}{12} \]") == "\n$$\n\\frac{3}{12} + \\frac{2}{12}\n$$\n"


def test_notebook_text_is_stripped_of_markdown_and_math_delimiters():
    from apu.ui.common import plain_text

    assert plain_text("**Step 1**: compute \\(1/4\\) and `$x$`") == "Step 1: compute 1/4 and x"
