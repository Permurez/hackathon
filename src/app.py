"""
app.py - Streamlit entry point: one module (plugin/fnirs_vascular), three views for an fNIRS researcher.

  ~/.venvs/fnirs/bin/streamlit run src/app.py

  Nagranie            one .snirf file -> what is in the signal, conclusions, recording quality, uncertainty
  Badanie             many recordings -> one table, group comparison, CSV export
  Pomiar powtarzany   the same person recorded several times -> stability, how many recordings to average
"""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Puls pod czepkiem", layout="wide")
HERE = Path(__file__).resolve().parent
st.navigation([
    st.Page(HERE / "page_recording.py", title="Nagranie", icon=":material/monitor_heart:", default=True),
    st.Page(HERE / "page_predict.py", title="Przewidywanie aktywności", icon=":material/directions_run:"),
    st.Page(HERE / "page_study.py", title="Badanie: wiele nagrań", icon=":material/groups:"),
    st.Page(HERE / "page_repeat.py", title="Pomiar powtarzany", icon=":material/repeat:"),
]).run()
