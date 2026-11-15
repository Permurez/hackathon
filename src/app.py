"""
app.py - Streamlit entry point: one module (plugin/fnirs_vascular), four views for an fNIRS researcher.

  ~/.venvs/fnirs/bin/streamlit run src/app.py      (from the repository root, so .streamlit/config.toml applies)

  Nagranie                  one .snirf file -> what is in the signal, conclusions, recording quality, uncertainty
  Przewidywanie aktywności  everything one session says + does the person train > 3 h/week, with the track record
  Badanie                   many recordings -> one table, group comparison, CSV export
  Pomiar powtarzany         the same person recorded several times -> stability, how many recordings to average
"""
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style  # noqa: E402

st.set_page_config(page_title="Pulse Under the Cap", page_icon="🫀", layout="wide")
style.apply()
st.navigation([
    st.Page(HERE / "page_recording.py", title="Recording", icon=":material/monitor_heart:", default=True),
    st.Page(HERE / "page_predict.py", title="Activity estimate", icon=":material/directions_run:"),
    st.Page(HERE / "page_study.py", title="Whole study", icon=":material/groups:"),
    st.Page(HERE / "page_repeat.py", title="Repeated recordings", icon=":material/repeat:"),
]).run()
