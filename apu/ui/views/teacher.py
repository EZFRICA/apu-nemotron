"""Teacher and admin view: escalations, clusters, resolution, access control.

Every read and write goes through apu.api.service, the same functions as the HTTP API, so
this page cannot show or change anything the registry does not allow for the chosen person.
"""

import os
import sys
from datetime import UTC, datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from apu import config  # noqa: E402
from apu.api import service  # noqa: E402
from apu.core.scheduler import deferred_writes  # noqa: E402
from apu.mmu.escalation_store import AlreadyResolved, EscalationStore  # noqa: E402
from apu.ui import common  # noqa: E402


def ago(moment: datetime) -> str:
    minutes = int((datetime.now(UTC) - moment).total_seconds() // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min ago"
    return f"{minutes // 60} h {minutes % 60:02d} ago"


identity = common.current_identity()
if identity.role == "student":
    st.info("This is the teacher / admin page. Pick a **Teacher** or **School admin** profile "
            "in the sidebar.")
    st.stop()

st.title("🧑‍🏫 Escalation follow-up")
st.caption(f"{common.ROLE_LABELS[identity.role]}: {identity.person_id} · school {identity.establishment_id}")
st.warning("Simulated authentication for the demo (insecure stub). The visible classes and the "
           "allowed actions come from the assignment registry, exactly as in the API.", icon="⚠️")

store = EscalationStore()
classes = service.visible_classes(identity.person_id, identity.establishment_id)
if not classes:
    st.info("No class is visible for this profile.")
    st.stop()

top = st.columns([3, 1])
class_id = top[0].selectbox("Class", classes, key="teacher_class")
if top[1].button("🔄 Refresh", width="stretch"):
    st.rerun()

events = service.list_escalations(identity.person_id, class_id, store)
snapshot = service.latest_clusters(identity.person_id, class_id, store)
open_events = [e for e, resolution in events if resolution is None]

metrics = st.columns(4)
metrics[0].metric("Escalations", len(events))
metrics[1].metric("Open", len(open_events))
metrics[2].metric("Resolved", len(events) - len(open_events))
metrics[3].metric("Clusters", len(snapshot.clusters) if snapshot else 0)

tab_events, tab_clusters, tab_access = st.tabs(["🚨 Escalations", "🧩 Clusters", "🔐 Access control"])

with tab_events:
    if not events:
        st.caption("No escalations for this class.")
    for event, resolution in sorted(events, key=lambda pair: pair[0].triggered_at, reverse=True):
        with st.container(border=True):
            header = st.columns([3, 2, 2])
            header[0].markdown(f"**{event.student_id}** · attempt #{event.attempt_number_in_session}")
            header[1].caption(ago(event.triggered_at))
            header[2].markdown(common.badge("resolved", "ok") if resolution else common.badge("open", "warn"),
                               unsafe_allow_html=True)
            st.markdown(common.quote(event.off_topic_request_text), unsafe_allow_html=True)
            if resolution:
                note = f" — “{resolution.note}”" if resolution.note else ""
                st.caption(f"Resolved by {resolution.resolved_by} {ago(resolution.resolved_at)}{note}")
            else:
                with st.form(f"resolve-{event.event_id}", border=False):
                    note = st.text_input("Note (optional)", key=f"note-{event.event_id}",
                                         placeholder="e.g. talked it through with the student")
                    if st.form_submit_button("Mark as resolved"):
                        refreshed = True
                        try:
                            service.resolve_escalation(identity.person_id, event.event_id, note, store)
                            st.toast("Escalation marked as resolved.", icon="✅")
                        except AlreadyResolved:
                            st.toast("Already resolved in the meantime.", icon="ℹ️")
                        except PermissionError as error:
                            # Shown without a rerun: a rerun would wipe the refusal off the
                            # screen before the teacher could read it. Toasts survive one.
                            st.error(f"Refused: {error}")
                            refreshed = False
                        if refreshed:
                            st.rerun()

with tab_clusters:
    pending = store.events_since_last_snapshot(class_id)
    if snapshot:
        st.caption(f"Computed {ago(snapshot.computed_at)} · {pending} new event(s) since · "
                   f"recomputed automatically every {config.ESCALATION_CLUSTER_TRIGGER_COUNT} events "
                   "(background job, never at read time).")
    else:
        st.caption("Not computed yet for this class.")
    if st.button("Recompute now (background job)"):
        service.request_cluster_recompute(identity.person_id, class_id)
        with st.spinner("Recomputing in the background…"):
            deferred_writes.drain(timeout=120)
        st.rerun()

    if snapshot and snapshot.clusters:
        events_by_id = {event.event_id: event for event, _ in events}
        for cluster in snapshot.clusters:
            with st.container(border=True):
                st.markdown(f"**{cluster.size} similar requests**")
                st.markdown(common.quote(cluster.representative_text), unsafe_allow_html=True)
                with st.expander("Show the requests"):
                    for event_id in cluster.event_ids:
                        event = events_by_id.get(event_id)
                        if event:
                            st.markdown(f"- {event.off_topic_request_text} · *{event.student_id}*")
    elif snapshot:
        st.info("No groups: this class's requests are too different from each other, or too few, "
                "to form clusters.")

with tab_access:
    st.markdown("Try to open a class outside your scope: the refusal comes from the assignment "
                "registry, not from the interface.")
    target = st.text_input("Class ID", value="college-yopougon:6eC", key="access_target")
    if st.button("Open this class"):
        try:
            count = len(service.list_escalations(identity.person_id, target, store))
            st.success(f"Allowed: {count} escalation(s) in {target}.")
        except PermissionError as error:
            st.error(f"403 — {error}")
