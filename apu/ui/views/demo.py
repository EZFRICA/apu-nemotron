"""Demo setup: environment checks, live connectivity tests, data reset and seeding."""

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from apu.demo import seed  # noqa: E402
from apu.ui import common  # noqa: E402

st.title("🛠️ Demo setup")

st.subheader("Environment")
for check in seed.demo_checks():
    st.markdown(f"{'✅' if check.ok else '❌'} **{check.label}** — {check.detail}")

st.subheader("Live checks")
live = st.columns(2)
if live[0].button("Test Nemotron (Token Factory)", width="stretch"):
    started = time.monotonic()
    try:
        reply = seed.live_check_nemotron()
        live[0].success(f"Replied “{reply}” in {time.monotonic() - started:.1f} s")
    except Exception as error:
        live[0].error(f"Failed: {error}")
if live[1].button("Test Tavily", width="stretch"):
    started = time.monotonic()
    try:
        results = seed.live_check_tavily()
        live[1].success(f"{results} result(s) in {time.monotonic() - started:.1f} s")
    except Exception as error:
        live[1].error(f"Failed: {error}")

st.subheader("Demo data")
st.markdown(
    "- **Prepare the demo**: wipes the student memory, notebooks, courses and escalations stored locally, "
    "builds and imports the courses locally (no GCS), then creates example escalations with their clusters "
    "and a sample notebook for Aya.\n"
    "- **Example escalations only**: adds the example escalations without wiping anything."
)
confirmed = st.checkbox("I confirm wiping the local demo data", key="confirm_reset")
actions = st.columns(2)
if actions[0].button("Prepare the demo", type="primary", disabled=not confirmed, width="stretch"):
    with st.status("Preparing the demo…", expanded=True) as status:
        report = seed.prepare_demo(progress=status.write)
        status.update(label="Demo ready", state="complete")
    common.reset_conversation()
    st.success(f"Courses: {len(report.courses)} · escalations: {report.escalations_by_class} · "
               f"clusters: {report.clusters_by_class} · notebook entries: {report.notebook_entries}")
if actions[1].button("Example escalations only", width="stretch"):
    with st.status("Creating escalations…", expanded=True) as status:
        counts, clusters = seed.seed_escalations(progress=status.write)
        status.update(label="Escalations created", state="complete")

st.subheader("Suggested run")
st.markdown(
    "1. **Student Aya** (3eA): “How do I add two fractions with different denominators?” → answer grounded in the course.\n"
    "2. “Check online for the official BEPC 2026 exam dates in Côte d'Ivoire” → Tavily search, sources at the end.\n"
    "3. Three off-topic questions (“Who won the PSG vs Marseille match last night?”) → kind reply, then firmer, escalation.\n"
    "4. “Save the key points of your answer” → saved to the notebook; **📓 Notebook** → revision sheet "
    "from chosen entries or a Nemotron summary.\n"
    "5. **prof-kouassi** → Aya's escalation, clusters, “Mark as resolved”.\n"
    "6. Access control → a class from another school is refused; **admin-cocody** sees 3eA and 4eB.\n\n"
    "Details: `DEMO.md`."
)
