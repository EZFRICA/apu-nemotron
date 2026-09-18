"""Student view: the tutor, with the topical guard, web search sources and the student's
notebook with its revision sheets."""

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from apu import config  # noqa: E402
from apu.demo.seed import loaded_courses  # noqa: E402
from apu.guardrails.session import sessions as guard_sessions  # noqa: E402
from apu.mmu import cache_l1  # noqa: E402
from apu.mmu import dll as mmu  # noqa: E402
from apu.notebook import service as notebook  # noqa: E402
from apu.notebook.store import (  # noqa: E402
    KIND_DESCRIPTIONS,
    KIND_LABELS,
    EntryKind,
    EntryOrigin,
    NotebookStore,
)
from apu.runtime.agent import build_message_window, create_agent_graph  # noqa: E402
from apu.storage import lance_driver  # noqa: E402
from apu.sync import sync_manager  # noqa: E402
from apu.ui import common  # noqa: E402

OUTCOME_LABELS = {
    "on_topic": ("✅ school work", "ok"),
    "off_topic": ("🛡️ off-topic", "warn"),
    "uncertain": ("❔ uncertain", "info"),
}

identity = common.current_identity()
if identity.role != "student":
    st.info("This is the student page. Pick a **Student** profile in the sidebar.")
    st.stop()

st.session_state.setdefault("chat", [])
st.session_state.setdefault("last_turn", None)
session_id = common.ensure_guard_session(identity)
session = guard_sessions.get(session_id)
threshold = session.policy.escalation_threshold

if st.session_state.pop("toast", None):
    st.toast("Threshold reached: an escalation event was recorded for the teacher.", icon="🛡️")
if saved_label := st.session_state.pop("notebook_toast", None):
    st.toast(f"Saved to your notebook: {saved_label.lower()}.", icon="📓")

# ── header ───────────────────────────────────────────────────────────────────
st.title("💬 Akili, your tutor")
st.caption(f"{identity.display_name} · class {identity.class_id} · "
           f"answers by {config.MAIN_MODEL}, memory by {config.EXTRACTION_MODEL}")

history_turns = st.slider("Exchanges sent to the model", 0, 10, 3, key="history_turns")

# ── guard status ─────────────────────────────────────────────────────────────
count = session.off_topic_count
status_columns = st.columns(4)
status_columns[0].metric("Off-topic attempts", f"{count} / {threshold}")
status_columns[1].metric("Class threshold", threshold)
status_columns[2].metric("Escalation", "Recorded" if count >= threshold else "No")
last_outcome = (st.session_state.last_turn or {}).get("guard_outcome")
outcome_text, outcome_kind = OUTCOME_LABELS.get(last_outcome, ("—", "info"))
status_columns[3].markdown(f"**Last guard verdict**<br>{common.badge(outcome_text, outcome_kind)}",
                           unsafe_allow_html=True)
st.progress(min(count / threshold, 1.0))

# ── course selection ─────────────────────────────────────────────────────────
dll_state = common.run(mmu.load_dll())
course = dll_state.get("course_selection", {"class": config.EDU_DEFAULT_CLASS, "subject": config.EDU_DEFAULT_SUBJECT})
courses = loaded_courses()
with st.expander(f"📚 Active course: {course['class']} / {course['subject']}", expanded=False):
    local_column, registry_column = st.columns(2)
    with local_column:
        st.markdown("**On this device**")
        if not courses:
            st.warning("No course loaded. Download one from the cloud registry, or use the "
                       "**Demo setup** page.")
        else:
            current = f"{course['class']}/{course['subject']}"
            choice = st.selectbox("Available courses", courses,
                                  index=courses.index(current) if current in courses else 0)
            if choice != current and st.button("Activate this course", type="primary"):
                class_level, subject = choice.split("/", 1)
                cache_l1.flush_all()
                common.run(mmu.switch_course(class_level, subject))
                common.reset_conversation()
                st.rerun()
    with registry_column:
        st.markdown("**Cloud registry (GCS)**")
        # Fetched on demand only: an unreachable registry costs a Google auth attempt, which
        # must not slow down every interaction during a live demo.
        if st.button("Browse the cloud registry"):
            with st.spinner("Reading the manifest…"):
                st.session_state.remote_catalog = common.run(sync_manager.get_remote_catalog())
            if not st.session_state.remote_catalog:
                st.error("Registry unreachable (Google credentials, network or REGISTRY_MANIFEST_URL).")
        remote_catalog = st.session_state.get("remote_catalog") or {}
        remote_courses = sorted(f"{c}/{s}" for c, subjects in remote_catalog.items() for s in subjects)
        if remote_courses:
            remote_choice = st.selectbox("Registry courses", remote_courses, key="remote_choice")
            class_level, subject = remote_choice.split("/", 1)
            if sync_manager.is_course_available_locally(class_level, subject):
                st.caption("💾 Already downloaded.")
            elif st.button("⬇️ Download and activate", type="primary"):
                with st.spinner(f"Downloading {remote_choice}…"):
                    ok, message = common.run(sync_manager.download_course(class_level, subject))
                if ok:
                    cache_l1.flush_all()
                    common.run(mmu.switch_course(class_level, subject))
                    common.reset_conversation()
                    st.rerun()
                st.error(message)
        if st.button("🔄 Check for updates (prompts)"):
            ok, message = common.run(sync_manager.sync_with_registry())
            (st.success if ok else st.error)(message)
    # Akili's "Reset Memory": wipes L1 and L2 for this student; rows already archived in L3 stay.
    if st.button("🗑️ Reset the student's memory (L1 + L2)"):
        cache_l1.flush_all()
        common.run(mmu.force_reinit_dll())
        common.reset_conversation()
        st.rerun()


# ── rendering ────────────────────────────────────────────────────────────────
def render_assistant(message: dict, index: int) -> None:
    if message.get("off_topic"):
        st.markdown(common.badge("🛡️ Guard: off-topic", "warn"), unsafe_allow_html=True)
    st.markdown(common.defang_links(common.math_for_streamlit(message["content"])))
    details = []
    if message.get("searches"):
        details.append("🔎 " + " · ".join(f"“{query}”" for query in message["searches"]))
    if message.get("duration") is not None:
        details.append(f"⏱ {message['duration']:.1f} s")
    if message.get("saved"):
        details.append("📓 saved: " + ", ".join(KIND_LABELS[EntryKind(kind)].lower() for kind in message["saved"]))
    if details:
        st.caption("   ".join(details))
    if message.get("answer"):
        save_control(message, index)


def save_control(message: dict, index: int) -> None:
    """Keep this answer in the notebook: in full, as key points, or an excerpt the student picks."""
    with st.popover("💾 Save to notebook"):
        kind_label = st.radio("What do you want to keep in your notebook?", [KIND_LABELS[kind] for kind in EntryKind],
                              captions=[KIND_DESCRIPTIONS[kind] for kind in EntryKind], key=f"save-kind-{index}")
        kind = next(kind for kind in EntryKind if KIND_LABELS[kind] == kind_label)
        excerpt = None
        if kind is EntryKind.EXCERPT:
            excerpt = st.text_area("The part to keep (edit it down)", value=common.plain_text(message["answer"]),
                                   key=f"save-excerpt-{index}")
        if st.button("Save", key=f"save-{index}", type="primary"):
            try:
                with st.spinner("Saving…" if kind is not EntryKind.KEY_POINTS else "Nemotron is condensing the key points…"):
                    entry = common.run(notebook.save_entry(
                        student_id=identity.person_id, class_level=course["class"], subject=course["subject"],
                        kind=kind, answer=message["answer"], excerpt=excerpt, origin=EntryOrigin.BUTTON,
                    ))
            except Exception as error:  # shown next to the control rather than as a traceback
                st.error(f"Not saved: {error}")
                return
            message.setdefault("saved", []).append(entry.kind.value)
            st.session_state.notebook_toast = KIND_LABELS[entry.kind]
            st.rerun()


tab_chat, tab_notebook, tab_turn, tab_memory = st.tabs(
    ["💬 Conversation", "📓 Notebook", "🔍 Last turn", "🧠 APU memory"])

with tab_chat:
    chat_box = st.container(height=440)
    with chat_box:
        if not st.session_state.chat:
            st.caption("Ask a question about your lessons. Try an off-topic question too, "
                       "to see the guard at work.")
        for index, message in enumerate(st.session_state.chat):
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
                    render_assistant(message, index)


with tab_notebook:
    notebook_store = NotebookStore()
    scope = st.segmented_control("Show", ["This course", "All courses"], default="This course",
                                 key="notebook_scope") or "This course"
    if scope == "This course":
        entries = notebook_store.entries(identity.person_id, course["class"], course["subject"])
    else:
        entries = notebook_store.entries(identity.person_id)
    st.caption("What you chose to keep. The tutor never reads your notebook; revision sheets are made from it.")
    if not entries:
        st.info("Nothing saved yet. Use **💾 Save to notebook** under an answer, or ask the tutor: "
                "“save the key points of your answer”.")
    for entry in entries:
        with st.container(border=True):
            top = st.columns([5, 1])
            top[0].markdown(
                common.badge(KIND_LABELS[entry.kind], "info")
                + f" {entry.course} · {entry.created_at.astimezone().strftime('%d %b %H:%M')}"
                + (" · asked in chat" if entry.origin is EntryOrigin.CHAT else ""),
                unsafe_allow_html=True)
            if top[1].button("🗑️", key=f"delete-{entry.entry_id}", help="Remove from the notebook"):
                notebook_store.delete(identity.person_id, entry.entry_id)
                st.rerun()
            st.markdown(common.defang_links(common.math_for_streamlit(entry.text)))

    if entries:
        st.markdown("**📄 Revision sheet**")
        from_summary_label = "A summary written by Nemotron"
        sheet_source = st.radio("Made from", ["The selected entries, as written", from_summary_label],
                                key="sheet_source", horizontal=True)
        labels = {entry.entry_id: f"{KIND_LABELS[entry.kind]} · {entry.course} · {entry.text[:60]}"
                  for entry in entries}
        # Entries can be deleted or filtered out between runs; a stale selection would raise.
        if "sheet_entries" in st.session_state:
            st.session_state.sheet_entries = [i for i in st.session_state.sheet_entries if i in labels]
        else:
            st.session_state.sheet_entries = list(labels)
        selected = st.multiselect("Entries", list(labels), format_func=labels.get, key="sheet_entries")
        if st.button("Generate the revision sheet", type="primary", disabled=not selected):
            chosen = [entry for entry in entries if entry.entry_id in selected]
            try:
                if sheet_source == from_summary_label:
                    with st.spinner("Nemotron is writing the revision summary…"):
                        sheet_text = common.run(notebook.summarize_entries(chosen))
                else:
                    sheet_text = notebook.entries_as_text(chosen)
                st.session_state.notebook_sheet = sheet_text
            except Exception as error:  # Nemotron unreachable
                st.session_state.pop("notebook_sheet", None)
                st.error(f"Revision sheet unavailable: {error}")
        sheet = st.session_state.get("notebook_sheet")
        if sheet is not None:
            st.text(sheet)
            st.download_button("⬇ Download the sheet (text)", sheet, file_name="akili-notebook.txt",
                               mime="text/plain", key="sheet_download")

with tab_turn:
    turn = st.session_state.last_turn
    if not turn:
        st.caption("No turn yet.")
    else:
        columns = st.columns(3)
        columns[0].metric("Turn duration", f"{turn['duration']:.1f} s")
        columns[1].metric("Web searches", len(turn.get("searches") or []))
        columns[2].metric("Sources", len(turn.get("sources") or []))
        st.markdown("**Guard verdict** " + common.badge(*OUTCOME_LABELS.get(
            turn.get("guard_outcome"), ("—", "info"))), unsafe_allow_html=True)
        for query in turn.get("searches") or []:
            st.markdown(f"- Tavily query: `{query}`")
        for source in turn.get("sources") or []:
            st.markdown(f"- [{source['title']}]({source['url']})")
        for kind in ("memory_problems", "tool_problems", "answer_problems"):
            for problem in turn.get(kind) or []:
                st.warning(f"{kind.replace('_', ' ')}: {problem}")

with tab_memory:
    cached = cache_l1.get_all_cached()
    st.markdown("**L1 · RAM cache**")
    if cached:
        metrics = cache_l1.get_metrics()
        st.dataframe([{"block": block_id, "hit rate": f"{metrics.get(block_id, {}).get('hit_rate', 0):.0%}",
                       "content": str(content)[:120]} for block_id, content in cached.items()],
                     hide_index=True, width="stretch")
    else:
        st.caption("Empty: blocks arrive here when the agent reads them.")
    st.markdown("**L2 · DLL (HEAD → TAIL)**")
    st.dataframe([{"block": node["label"], "type": node.get("type"),
                   "content": (node.get("content") or ", ".join(node.get("keywords", [])))[:120]}
                  for node in mmu.get_all_nodes(dll_state)], hide_index=True, width="stretch")
    st.markdown("**L3 · LanceDB**")
    try:
        db = lance_driver.get_db()
        st.dataframe([{"table": name, "rows": db.open_table(name).count_rows()}
                      for name in lance_driver.list_table_names(db)], hide_index=True, width="stretch")
    except Exception as error:
        st.error(f"LanceDB: {error}")

# ── a turn ───────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask your question…"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat]
    previous_answer = next((m["answer"] for m in reversed(st.session_state.chat) if m.get("answer")), "")
    state = {
        "messages": build_message_window(history, prompt, history_turns),
        "session_id": session_id,
        "agent_id": dll_state.get("agent_id"),
        "class_level": course["class"],
        "subject": course["subject"],
        "memory_only_mode": history_turns == 0,
        "needs_new_block": "False",
        "proposed_block_config": {},
        "previous_answer": previous_answer,
    }
    count_before = session.off_topic_count
    started = time.monotonic()
    with st.spinner("Akili is thinking (guard, search if needed, answer, memory)…"):
        try:
            result = common.run(create_agent_graph().ainvoke(state))
            error = None
        except Exception as exc:  # shown in the chat rather than as a traceback
            result, error = None, exc
    duration = time.monotonic() - started

    st.session_state.chat.append({"role": "user", "content": prompt})
    if error is not None:
        st.session_state.chat.append({
            "role": "assistant", "duration": duration,
            "content": f"⚠️ This turn could not be completed: {error}",
        })
        st.session_state.last_turn = None
    else:
        st.session_state.chat.append({
            "role": "assistant",
            "content": result["messages"][-1].content,
            "off_topic": bool(result.get("off_topic")),
            "answer": None if result.get("off_topic") else result.get("answer_text"),
            "saved": [save["kind"] for save in result.get("notebook_saves") or []],
            "searches": result.get("searches") or [],
            "duration": duration,
        })
        st.session_state.last_turn = {**{k: result.get(k) for k in (
            "guard_outcome", "searches", "sources", "memory_problems", "tool_problems", "answer_problems")},
            "duration": duration}
        if result.get("notebook_saves"):
            st.session_state.notebook_toast = KIND_LABELS[EntryKind(result["notebook_saves"][-1]["kind"])]
        if count_before < threshold <= session.off_topic_count:
            st.session_state.toast = True
    st.rerun()
