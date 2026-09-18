"""Shared pieces of the demo interface: identity selector (stub), guard session, styling.

IDENTITY IS A STUB. The sidebar lets anyone act as any demo student, teacher or admin; no
password, no token. What IS real: a teacher's or admin's rights come from the assignment
registry through the same service functions as the API (apu.api.service), and a student's
class policy comes from the class policy registry.
"""

import asyncio
import html
import os
import re
import sys
from dataclasses import dataclass

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from apu import config  # noqa: E402
from apu.auth import assignments  # noqa: E402
from apu.demo.seed import load_demo_students  # noqa: E402
from apu.guardrails.policy import get_class_policy_registry  # noqa: E402
from apu.guardrails.session import UnknownSession  # noqa: E402
from apu.guardrails.session import sessions as guard_sessions  # noqa: E402
from apu.modality.plain_text import plain_text  # noqa: E402,F401  (re-exported for the views)

ROLE_LABELS = {
    "student": "Student",
    "teacher": "Teacher",
    "establishment_admin": "School admin",
}


@dataclass(frozen=True)
class DemoIdentity:
    role: str                 # student | teacher | establishment_admin
    person_id: str
    display_name: str
    establishment_id: str
    class_id: str | None = None

    @property
    def label(self) -> str:
        scope = self.class_id or self.establishment_id
        return f"{ROLE_LABELS[self.role]} — {self.display_name} ({scope})"


def run(coroutine):
    return asyncio.run(coroutine)


def available_identities() -> list[DemoIdentity]:
    identities = [
        DemoIdentity("student", s["student_id"], s["display_name"],
                     s["class_id"].split(":", 1)[0], s["class_id"])
        for s in load_demo_students()
    ]
    identities += [
        DemoIdentity(a.role, a.requester_id, a.requester_id, a.establishment_id, a.class_id)
        for a in assignments.get_assignment_registry().all()
    ]
    return identities


def default_identity() -> DemoIdentity:
    identities = available_identities()
    return next((i for i in identities if i.person_id == config.DEMO_STUDENT_ID), identities[0])


def current_identity() -> DemoIdentity:
    if "identity" not in st.session_state:
        st.session_state.identity = default_identity()
    return st.session_state.identity


def reset_conversation() -> None:
    st.session_state.chat = []
    st.session_state.last_turn = None
    st.session_state.pop("guard_session_id", None)
    st.session_state.pop("notebook_sheet", None)


def identity_sidebar() -> DemoIdentity:
    identities = available_identities()
    current = current_identity()
    labels = [identity.label for identity in identities]
    index = labels.index(current.label) if current.label in labels else 0
    choice = st.selectbox("Sign in as", labels, index=index, key="identity_choice")
    chosen = identities[labels.index(choice)]
    if chosen != current:
        st.session_state.identity = chosen
        # A new person is a new connection: new guard session, counter back to zero.
        reset_conversation()
        # And a clean tutoring memory. The DLL, the L1 cache and the archived memory rows are
        # per DEVICE, not per student (see "Known limits" in the README), so without this the
        # next student would open the app on the previous one's profile and current session.
        if "student" in (chosen.role, current.role):
            clear_device_memory()
        st.rerun()
    st.caption("⚠️ Simulated sign-in for the demo (insecure stub). "
               "Permissions still come from the registries.")
    return chosen


def clear_device_memory() -> None:
    """Wipe the tutoring memory this device holds (L1 + L2), leaving L3 archives alone."""
    from apu.mmu import cache_l1
    from apu.mmu import dll as mmu

    cache_l1.flush_all()
    run(mmu.force_reinit_dll())


def ensure_guard_session(identity: DemoIdentity) -> str:
    """The student's guard session, reopened if it vanished (demo reset, new identity)."""
    session_id = st.session_state.get("guard_session_id")
    if session_id:
        try:
            guard_sessions.get(session_id)
            return session_id
        except UnknownSession:
            pass
    session = guard_sessions.open_session(student_id=identity.person_id, class_id=identity.class_id)
    st.session_state.guard_session_id = session.session_id
    return session.session_id


def class_threshold(class_id: str) -> int:
    return get_class_policy_registry().get(class_id).escalation_threshold


def inject_css() -> None:
    st.markdown(
        """
        <style>
          .apu-badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem;
            font-weight:600; margin-right:6px; }
          .apu-badge.ok { background:rgba(74,222,128,.15); color:#86efac; border:1px solid rgba(74,222,128,.4); }
          .apu-badge.warn { background:rgba(251,146,60,.15); color:#fdba74; border:1px solid rgba(251,146,60,.4); }
          .apu-badge.info { background:rgba(56,189,248,.15); color:#7dd3fc; border:1px solid rgba(56,189,248,.4); }
          .apu-quote { border-left: 3px solid #38bdf8; padding: 4px 12px; color: #cbd5e1; }
        </style>
        """,
        unsafe_allow_html=True,
    )


_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
_BARE_URL = re.compile(r"(?<![(`\w])(https?://[^\s`)\]]+)")


def defang_links(text: str) -> str:
    """
    Show URLs as code, never as clickable links, in anything a model wrote.

    A web page the tutor read can tell it to put a link in its answer, and Markdown would
    render that link for the student to click. The words stay, the address stays visible,
    but nothing in the chat bubble is clickable. The verified Tavily sources of the turn are
    listed as real links in the "Last turn" tab, where they come from the search result and
    not from the model's text.
    """
    text = _MARKDOWN_LINK.sub(lambda match: f"{match.group(1)} (`{match.group(2)}`)", text)
    return _BARE_URL.sub(lambda match: f"`{match.group(1)}`", text)


def badge(text: str, kind: str = "info") -> str:
    return f'<span class="apu-badge {kind}">{text}</span>'


def quote(text: str) -> str:
    """
    A student's own words, shown to a teacher.

    Escaped, always: this text is typed by a student and rendered inside a div with
    unsafe_allow_html. Streamlit's sanitiser strips scripts, but a raw `<a href>` or
    `<style>` survives it, so an off-topic message could put a phishing link or a layout
    takeover in the teacher's dashboard. Escaping is what stops that; the div only styles it.
    """
    return f'<div class="apu-quote">{html.escape(text)}</div>'


_BLOCK_MATH = re.compile(r"\\\[(.+?)\\\]", re.S)
_INLINE_MATH = re.compile(r"\\\((.+?)\\\)", re.S)


def math_for_streamlit(text: str) -> str:
    """
    Convert LaTeX delimiters to the ones Streamlit's Markdown renders.

    Nemotron writes math as \\( ... \\) and \\[ ... \\]; Streamlit only renders $ ... $
    and $$ ... $$, so without this a formula shows up as raw backslash commands.
    """
    text = _BLOCK_MATH.sub(lambda match: f"\n$$\n{match.group(1).strip()}\n$$\n", text)
    return _INLINE_MATH.sub(lambda match: f"${match.group(1).strip()}$", text)
