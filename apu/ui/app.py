"""Akili on Nemotron: demo interface (Streamlit, multipage).

    uv run streamlit run apu/ui/app.py

Pages: the student tutor, the teacher/admin escalation view, and demo setup. The identity
selector in the sidebar is a stub (see apu.ui.common).
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Akili · APU on Nemotron", page_icon="🎓", layout="wide")

from apu.ui import common  # noqa: E402

common.inject_css()

with st.sidebar:
    st.markdown("## 🎓 Akili")
    st.caption("Agent Processor Unit on NVIDIA Nemotron · Nebius Token Factory")
    common.identity_sidebar()

navigation = st.navigation([
    st.Page("views/student.py", title="Student", icon="💬", default=True),
    st.Page("views/teacher.py", title="Teacher / Admin", icon="🧑‍🏫"),
    st.Page("views/demo.py", title="Demo setup", icon="🛠️"),
])
navigation.run()
