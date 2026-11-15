"""
page_recording.py - "Nagranie" view of src/app.py: one 10-minute resting fNIRS recording -> two axes of vascular
health, conclusions with their uncertainty, recording quality (from plugin/fnirs_vascular).

Pick a recording from the dataset or upload a .snirf file; everything is computed live by vascular.py.
Group-level markers, not a diagnosis.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugin"))
import fnirs_vascular as fv  # noqa: E402
import vascular as V  # noqa: E402
import style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ICC_BAND, ICC_TI = 0.36, 0.95  # one 10-min recording: band across 4 rests a day (DOT, cautious value); TI split-half


@st.cache_data(show_spinner=False)
def run(path: str) -> dict:
    return V.analyse(path)


@st.cache_data(show_spinner=False)
def run_module(path: str) -> dict:
    return fv.analyze(path)


def pl(v, f=".2f") -> str:
    """Number with a Polish decimal comma."""
    return format(v, f)


def pct_range(pct: float, icc: float) -> tuple[float, float]:
    """80% range of the percentile from the measurement error of one recording (SEM = SD * sqrt(1 - ICC))."""
    z = stats.norm.ppf(min(max(pct / 100, 0.02), 0.98))
    h = 1.2816 * np.sqrt(1 - icc)
    return 100 * stats.norm.cdf(z - h), 100 * stats.norm.cdf(z + h)


@st.cache_data(show_spinner=False)
def load_cohort():
    return V.cohort()


def show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


coh = load_cohort()
files = sorted(V.BASE.glob("*.snirf"), key=lambda p: V.subject_of(p) or 0)


def label(p: Path) -> str:
    s = V.subject_of(p)
    r = coh[coh.subject == s]
    if len(r) and np.isfinite(r.age.iloc[0]):
        return f"Person {s} · {int(r.age.iloc[0])} y · questionnaire: {V.ACT.get(r.activity.iloc[0], '?')}"
    return p.name


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Recording")
mode = st.sidebar.radio("Source", ["From the dataset", "Upload a .snirf file"])
if mode == "From the dataset":
    if not files:
        st.error("No data. Run ./download_data.sh or upload a .snirf file.")
        st.stop()
    start = next((i for i, p in enumerate(files) if V.subject_of(p) == 17), 0)
    path = str(st.sidebar.selectbox("Person", files, index=start, format_func=label))
else:
    up = st.sidebar.file_uploader("A .snirf file: 10 min of rest, Cortivision C23", type=["snirf"])
    if up is None:
        st.info("Upload a .snirf recording in the left panel.")
        st.stop()
    tmp = Path(tempfile.gettempdir()) / up.name
    tmp.write_bytes(up.getvalue())
    path = str(tmp)
st.sidebar.caption("Reference: healthy people aged 18–42 without medication (n = 52), Cortivision supine dataset. "
                   "The person analysed is left out of the reference group.")

# ---------------------------------------------------------------- header + analysis
st.title("Pulse Under the Cap")
style.hero("10 minutes of lying still in an fNIRS cap tells us about your blood vessels: how actively they "
           "regulate (more in people who train) and how fast the pulse wave comes back (faster in older people). "
           "No extra equipment. These are hints compared with a group, not a diagnosis.",
           ["Pick a person from the dataset or upload your own .snirf file", "The app computes the result in a fraction of a second",
            "Read the tiles: each number shows how far you can trust it"])
style.legend()

with st.status("Analysing the recording…", expanded=False) as status:
    status.write("1. Raw light → optical density → oxygenated haemoglobin (HbO)")
    mod = run_module(path)
    try:
        res = run(path)
    except Exception as e:  # noqa: BLE001 - other montage: show what the module can say
        status.update(label=f"Different sensor layout than the reference group: {e}", state="error")
        st.subheader("Module output (no comparison with the group)")
        st.json(mod)
        st.stop()
    status.write("2. Spectrum of the forehead signal → strength of the slow vessel wave (0.0095–0.021 Hz)")
    status.write(f"3. {res['n_beats']} heartbeats averaged → wave from the heart and its echo → TI index")
    status.write("4. Comparison with a healthy group")
    status.update(label=f"Done: {res['file']} ({res['duration_min']:.1f} min)", state="complete")
sc = V.score(res, coh)

c1, c2, c4 = st.columns(3)
c1.metric("Vessel regulation", f"above {sc['pct_endo']:.0f}% of people",
          help="Strength of the slow vessel wave (linked to the vessel lining) vs healthy people aged 18–42.")
c2.metric("Pulse echo (TI)", f"above {sc['pct_ti']:.0f}% of people" if np.isfinite(sc["pct_ti"]) else "–",
          help="Higher = the pulse wave comes back faster; it rises with age. For one person it is a hint.")
c4.metric("Resting heart rate", f"{res['hr']:.0f} /min")
for col, kind in ((c1, "grupa"), (c2, "grupa"), (c4, "pomiar")):
    col.caption(style.badge(kind), unsafe_allow_html=True)

row = coh[coh.subject == res["subject"]]
if len(row) and np.isfinite(row.activity.iloc[0]):
    st.caption(f"For comparison, this person's questionnaire: {V.ACT.get(row.activity.iloc[0], '?')}, "
               f"age {int(row.age.iloc[0])} (the model does not see this).")

tab1, tab2 = st.tabs(["Signal and pulse wave", "Two axes vs the group"])
with tab1:
    a, b = st.columns(2)
    with a:
        show(V.fig_signal(res))
    with b:
        show(V.fig_spectrum(res))
    a, b = st.columns([1.4, 1])
    with a:
        show(V.fig_pulse(res))
    with b:
        st.subheader("What this recording shows")
        for line in V.interpret(res, sc):
            if not line.startswith("Model"):
                st.markdown(f"- {line}")
        st.markdown("- Estimated weekly training hours are on the *Activity estimate* page.")
        lo, hi = pct_range(sc["pct_endo"], ICC_BAND)
        st.markdown(f"- **Certainty:** in a repeat recording the vessel wave would usually land between the **{lo:.0f}th and "
                    f"{hi:.0f}th percentile** (one recording, repeatability {pl(ICC_BAND)}).")
        if np.isfinite(sc["pct_ti"]):
            lo, hi = pct_range(sc["pct_ti"], ICC_TI)
            st.markdown(f"- The TI index is steadier: **{lo:.0f}th–{hi:.0f}th percentile** "
                        f"(repeatability {pl(ICC_TI)}).")
        st.markdown("- To judge one person, average several recordings (*One-recording reliability* page); "
                    "to compare groups, one per person is enough (*Whole study* page).")
        with st.expander("Recording quality"):
            r = mod["recording"]
            st.markdown(f"{pl(r['duration_min'], '.1f')} min, {pl(r['sampling_hz'], '.1f')} Hz, {r['pairs']} sensor pairs, "
                        f"missing {pl(r['missing_samples_pct'], '.1f')}% of samples (filled in), "
                        f"dead pairs: {', '.join(r['dead_pairs']) or 'none'}; "
                        f"{mod['beats_used']} heartbeats, TI from {mod['arterial_timing']['channels_used']} channels.")
            for wmsg in mod["warnings"] or ["No warnings."]:
                st.markdown(f"- {wmsg}")
with tab2:
    a, b = st.columns([1.3, 1])
    with a:
        show(V.fig_two_axes(coh, res))
    with b:
        st.subheader("Two independent signals")
        st.markdown(
            "- **Vertical, regulation:** the slow vessel wave, about once a minute. In people training over 3 h "
            "a week it is on average **1.49× stronger**; in athletes (laser Doppler of the skin) 1.5× "
            "(Kvernmo et al. 2003).\n"
            "- **Horizontal, structure:** the pulse echo index. In stiffer arteries the echo comes back sooner. "
            "It rises with age even at 18–42 (r = +0.36), repeatability 0.95.\n"
            "- **The axes are independent** (correlation about 0): training and age leave two different traces.")
st.warning(V.DISCLAIMER + " It shows a group trend, not the exact training or artery stiffness of one person.")
