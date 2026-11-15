"""page_repeat.py - the same person recorded several times: how stable is the slow vasomotion band, how many
recordings must be averaged, and the smallest change that is more than noise (the numbers a training study needs).
Demo data: DOT dataset, 17 people x 4 ten-minute rests over one day (sitting, other montage -> raw values)."""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))
sys.path.insert(0, str(ROOT / "src"))
import fnirs_vascular as fv  # noqa: E402
import vascular as V  # noqa: E402
import style  # noqa: E402

DOT = ROOT / "dane/infopack_and_data/dot_montage_sitting/data/RS"
ICC_ALL_LONG = 0.36  # src/endo_followup.py: every long channel, no heartbeat criterion
ICC_MODULE = 0.56    # this module on the same DOT rests (long channels with a clear heartbeat)
SW_MODULE = 10.3     # pp: within-person SD of one recording, same DOT rests (sqrt of the within-person mean square)


@st.cache_data(show_spinner=False, persist="disk")
def run_many(paths: tuple[str, ...]) -> list[dict]:
    return fv.analyze_many(list(paths))


def pl(v, f=".2f") -> str:
    """Number with a Polish decimal comma."""
    return format(v, f)


def icc1(w: np.ndarray):
    """ICC(1,1) and within-person SD for a people x sessions array."""
    n, k = w.shape
    gm = w.mean()
    msb = k * ((w.mean(1) - gm) ** 2).sum() / (n - 1)
    msw = ((w - w.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw)), float(np.sqrt(msw))


def spearman_brown(icc, k):
    return k * icc / (1 + (k - 1) * icc)


def k_needed(icc, target):
    return int(np.ceil(target * (1 - icc) / (icc * (1 - target))))


st.title("How reliable is one recording?")
style.hero("Upload one recording: you get its vessel-wave value and the range a repeat recording of the same "
           "person would most likely fall in, so you know how much to trust a single result.",
           ["Upload one .snirf recording in the left panel", "Read its value and the likely range of a repeat",
            "See what it takes to track one person over time"])

src = st.sidebar.radio("Data", ["Upload one recording", "Example data (DOT dataset)"])

if src.startswith("Example"):
    paths = sorted(str(p) for p in DOT.glob("*_RS*.snirf"))
    if not paths:
        st.error("No DOT data. Run ./download_data.sh.")
        st.stop()
    with st.spinner(f"Processing {len(paths)} recordings…"):
        res = run_many(tuple(paths))
    rows = []
    for r in res:
        m = re.match(r"(\d+)_[A-Z]+_RS(\d)", r["file"])
        if m and "error" not in r:
            rows.append(dict(osoba=int(m[1]), sesja=int(m[2]), rytm=100 * r["vascular_regulation"]["endothelial_band_share"]))
    d = pd.DataFrame(rows)
    w = d.pivot(index="osoba", columns="sesja", values="rytm").dropna()
    icc, sw = icc1(w.to_numpy())
    k75, k80 = k_needed(icc, 0.75), k_needed(icc, 0.80)
    mdc1 = 1.96 * np.sqrt(2) * sw

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Repeatability of 1 recording (ICC)", pl(icc),
              help=f"ICC(1,1), {len(w)} people × {w.shape[1]} rests. The analysis on all long channels "
                   f"(src/endo_followup.py) gives {pl(ICC_ALL_LONG)}.")
    c2.metric("Mean of 4 recordings", pl(spearman_brown(icc, 4)))
    c3.metric("Recordings for ICC 0.75", f"{k75}")
    c4.metric("Smallest detectable change", f"{pl(mdc1, '.1f')} pp",
              help="MDC95 for comparing two single recordings, in percentage points of the band share.")

    who = st.sidebar.selectbox("Person", list(w.index), format_func=lambda s: f"Person {s}")
    a, b = st.columns([1.4, 1])
    with a:
        order = w.mean(axis=1).sort_values().index
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        for i, s in enumerate(order):
            hi = s == who
            ax.plot([i] * w.shape[1], w.loc[s], "o", ms=7 if hi else 5, color=V.ORANGE if hi else V.GREY,
                    alpha=1 if hi else 0.8, zorder=3 if hi else 2)
            ax.plot([i - 0.3, i + 0.3], [w.loc[s].mean()] * 2, color=V.INK if hi else V.MUTED, lw=2 if hi else 1)
        ax.set_xticks(range(len(order)), [str(s) for s in order], fontsize=8)
        ax.set_xlabel("person (sorted by mean)")
        ax.set_ylabel("slow vessel wave [% of signal]")
        ax.set_title("4 rests of the same person in one day (dots) and their mean (line)", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with b:
        ks = np.arange(1, 9)
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        ax.plot(ks, spearman_brown(icc, ks), "o-", color=V.BLUE, label=f"this module (ICC {pl(icc)})")
        ax.plot(ks, spearman_brown(ICC_ALL_LONG, ks), "o--", color=V.GREY, label=f"all channels ({pl(ICC_ALL_LONG)})")
        ax.axhline(0.75, color=V.MUTED, lw=0.8, ls=":")
        ax.text(8, 0.73, "good repeatability (0.75)", ha="right", va="top", fontsize=8, color=V.MUTED)
        ax.set_xlabel("number of averaged recordings")
        ax.set_ylabel("repeatability")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title("How many recordings to average", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    x = w.loc[who]
    st.markdown(f"**Person {who}:** " + "; ".join(pl(v, ".1f") for v in x) +
                f" % → mean {pl(x.mean(), '.1f')} %, spread ± {pl(x.std(), '.1f')} pp within one day.")
    st.markdown("#### Conclusion")
    st.markdown(
        f"- One 10-minute recording has repeatability **{pl(icc)}**: not enough to judge one person "
        f"from a single measurement.\n"
        f"- The mean of **{k75} recordings** gives repeatability 0.75, and of {k80} recordings 0.80.\n"
        f"- A difference between two single recordings smaller than **{pl(mdc1, '.1f')} pp** is noise. With means "
        f"of {k75} recordings before and after training the threshold drops to **{pl(mdc1 / np.sqrt(k75), '.1f')} pp**. For comparison: "
        "people who train and who do not differ by about 12 pp in the median.\n"
        "- These are the numbers needed to plan a training study: how many recordings per person and what change to look for.")
    st.caption("DOT data: sitting, a different sensor layout than the reference group, so raw values only (no percentiles). "
               "Band computed on long channels with a clear heartbeat; 18 Hz sampling is too slow for TI.")
else:
    up = st.sidebar.file_uploader("One .snirf recording (about 10 min of rest)", type=["snirf"])
    if up is None:
        st.info("Upload one resting .snirf recording in the left panel.")
        st.stop()
    tmp = Path(tempfile.mkdtemp(prefix="fnirs_one_")) / up.name
    tmp.write_bytes(up.getvalue())
    res = [r for r in run_many((str(tmp),)) if "error" not in r]
    if not res:
        st.error("This file could not be read as a resting fNIRS recording.")
        st.stop()
    r = res[0]
    v = 100 * r["vascular_regulation"]["endothelial_band_share"]
    # this view uses only the vessel wave: layout and pulse-timing warnings do not apply here
    notes = [w for w in r["warnings"] if not w.startswith(("Montage differs", "Sampling"))]
    lo, hi = max(v - SW_MODULE, 0.0), v + SW_MODULE
    k75 = k_needed(ICC_MODULE, 0.75)
    c1, c2, c3 = st.columns(3)
    c1.metric("Vessel wave", f"{pl(v, '.1f')} %")
    c2.metric("A repeat would likely give", f"{pl(lo, '.0f')}–{pl(hi, '.0f')} %")
    c3.metric("Repeatability of one recording", pl(ICC_MODULE))
    st.markdown(
        f"- **What this means:** the same person recorded again the same day usually lands within "
        f"±{pl(SW_MODULE, '.0f')} percentage points of this value.\n"
        "- **Good for:** comparing groups of people, where one recording per person is enough.\n"
        f"- **To track one person over time** (e.g. before and after training), average {k75} recordings: "
        "repeatability then reaches 0.75.")
    st.caption(f"{r['file']} · {pl(r['recording']['duration_min'], '.1f')} min · "
               + (" | ".join(notes) if notes else "recording OK"))

    # the same pictures as on the Recording page, for this one file
    try:
        full = V.analyse(str(tmp))
    except Exception:  # noqa: BLE001 - a file the plots cannot handle still gets the numbers above
        full = None
    if full is not None:
        st.subheader("This recording")
        a, b = st.columns(2)
        with a:
            st.pyplot(V.fig_signal(full), clear_figure=True)
        with b:
            st.pyplot(V.fig_spectrum(full), clear_figure=True)
        a, b = st.columns([1.4, 1])
        with a:
            st.pyplot(V.fig_pulse(full), clear_figure=True)
        with b:
            st.metric("Heart rate", f"{full['hr']:.0f} /min")
            st.metric("Beats averaged", f"{full['n_beats']}")
            st.caption("Top: the forehead signal and its spectrum, with the slow vessel wave in orange. "
                       "Left: hundreds of heartbeats averaged into one pulse, with the wave from the heart "
                       "and its echo.")
        plt.close("all")
