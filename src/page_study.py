"""page_study.py - many resting recordings at once: one table of markers, a group comparison with a plain-language
conclusion, and a CSV export. The use case: every fNIRS study records a rest; these markers come free with it and
show whether the compared groups also differ in their blood vessels (a confound for brain responses)."""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))
sys.path.insert(0, str(ROOT / "src"))
import fnirs_vascular as fv  # noqa: E402
import vascular as V  # noqa: E402
import style  # noqa: E402
from features import load_meta  # noqa: E402

MARKERS = {"vessel wave [% of signal]": "slow vessel wave", "TI [1/s]": "pulse echo index TI"}


def pl(v, f=".2f") -> str:
    """Number with a Polish decimal comma."""
    return format(v, f)


@st.cache_data(show_spinner=False, persist="disk")
def run_many(paths: tuple[str, ...]) -> list[dict]:
    return fv.analyze_many(list(paths))


def to_table(res: list[dict]) -> pd.DataFrame:
    rows = []
    for r in res:
        if "error" in r:
            rows.append({"file": r["file"], "warnings": "ERROR: " + r["error"]})
            continue
        v, t = r["vascular_regulation"], r["arterial_timing"]
        rows.append({"file": r["file"], "heart rate [bpm]": r["heart_rate_bpm"],
                     "vessel wave [% of signal]": round(100 * v["endothelial_band_share"], 1),
                     "vessel wave percentile": v["percentile_vs_reference"],
                     "TI [1/s]": t["timing_index_per_s"], "TI percentile": t["percentile_vs_reference"],
                     "length [min]": r["recording"]["duration_min"],
                     "warnings": " | ".join(r["warnings"])})
    return pd.DataFrame(rows)


def cliffs_delta(a, b):
    return float((np.sum(a[:, None] > b[None, :]) - np.sum(a[:, None] < b[None, :])) / (len(a) * len(b)))


def compare(df: pd.DataFrame, col: str, groups: list[str]):
    a = df.loc[df.group == groups[0], col].dropna().to_numpy(float)
    b = df.loc[df.group == groups[1], col].dropna().to_numpy(float)
    if len(a) < 3 or len(b) < 3:
        return None
    return dict(a=a, b=b, p=float(stats.mannwhitneyu(a, b).pvalue), med_a=float(np.median(a)),
                med_b=float(np.median(b)), delta=cliffs_delta(a, b))


def fig_groups(df, col, groups, c):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    rng = np.random.default_rng(0)
    for i, (g, x, color) in enumerate(((groups[0], c["a"], V.ORANGE), (groups[1], c["b"], V.GREY))):
        ax.boxplot(x, positions=[i], widths=0.5, showfliers=False, medianprops=dict(color=V.INK, lw=2))
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(x)), x, s=26, color=color, edgecolor="white", zorder=3)
    ax.set_xticks([0, 1], [f"{groups[0]}\n(n = {len(c['a'])})", f"{groups[1]}\n(n = {len(c['b'])})"])
    ax.set_ylabel(col)
    ax.set_title(f"{MARKERS[col]}\np = {pl(c['p'], '.4f')} (Mann–Whitney)", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- input
st.title("Whole study: many recordings at once")
style.hero("Have a whole study, e.g. dozens of people in two groups? Every fNIRS study records a rest anyway. "
           "From it the app computes two vessel measures for each participant and tells you whether the groups differ "
           "in their vessels too. Such a difference can look like a difference in brain activity, so it is worth taking into account.",
           ["Choose recordings and how to split them into groups", "The app processes all recordings (about 0.2 s each)",
            "Read the conclusion and download the CSV table"])

src = st.sidebar.radio("Recordings", ["Hackathon dataset (59 recordings, lying down)", "Upload your own .snirf files"])
groups_df, order = None, None
if src.startswith("Hackathon"):
    paths = sorted(str(p) for p in V.BASE.glob("*.snirf"))
    if not paths:
        st.error("No data. Run ./download_data.sh or upload files.")
        st.stop()
    split = st.sidebar.selectbox("Split into groups", ["Questionnaire: aerobic > 3 h/week vs less",
                                                      "Age: 30 or older vs under 30", "Sex"])
    meta = load_meta()
    meta["plik_id"] = meta.subject.astype(int)
    if split.startswith("Questionnaire"):
        order = ["aerobic > 3 h/week", "less"]
        meta["group"] = np.where(meta.activity == meta.activity.max(), *order)
    elif split.startswith("Age"):
        order = ["30 or older", "under 30"]
        meta["group"] = np.where(meta.age >= 30, *order)
    else:
        order = ["men", "women"]
        meta["group"] = np.where(meta.sex_male == 1, *order)
    groups_df = meta[["plik_id", "group"]]
else:
    ups = st.sidebar.file_uploader(".snirf files (rest, about 10 min)", type=["snirf"], accept_multiple_files=True)
    gcsv = st.sidebar.file_uploader("Optional CSV with groups: columns file, group", type=["csv"])
    if not ups:
        st.info("Upload .snirf files in the left panel. Optionally add a CSV with columns `file,group`.")
        st.stop()
    tmp = Path(tempfile.mkdtemp(prefix="fnirs_study_"))
    paths = []
    for u in ups:
        (tmp / u.name).write_bytes(u.getvalue())
        paths.append(str(tmp / u.name))
    if gcsv is not None:
        groups_df = pd.read_csv(gcsv)

with st.spinner(f"Processing {len(paths)} recordings…"):
    res = run_many(tuple(paths))
df = to_table(res)
ok = df["warnings"].fillna("").eq("").sum() if "warnings" in df else len(df)

c1, c2, c3 = st.columns(3)
c1.metric("Recordings", len(df))
c2.metric("No warnings", int(ok))
c3.metric("Compute time", "about 0.2 s / recording")

# ---------------------------------------------------------------- groups
if groups_df is not None:
    if "plik_id" in groups_df:
        df["group"] = df.file.str.extract(r"^(\d+)_")[0].astype(float).map(
            groups_df.set_index("plik_id").group)
    else:
        df["group"] = df.file.map(groups_df.set_index("file").group)
    names = order or sorted(map(str, df.group.dropna().unique()))
    if len(names) == 2:
        st.subheader("Do the groups differ in their vessels?")
        cols = st.columns(2)
        verdicts = []
        for col_ui, col in zip(cols, MARKERS):
            c = compare(df, col, names)
            if c is None:
                col_ui.info(f"{MARKERS[col]}: too few recordings in the groups.")
                continue
            with col_ui:
                st.pyplot(fig_groups(df, col, names, c), clear_figure=True)
            name = MARKERS[col][0].upper() + MARKERS[col][1:]
            if c["p"] < 0.05:
                verdicts.append(f"**{name}: the groups differ** (median {pl(c['med_a'])} vs "
                                f"{pl(c['med_b'])}, p = {pl(c['p'], '.4f')}, Cliff's delta {pl(c['delta'], '+.2f')}). When comparing "
                                "brain responses between these groups, add this measure as a control variable.")
            else:
                verdicts.append(f"{name}: no clear difference between the groups (p = {pl(c['p'])}).")
        st.markdown("#### Conclusion for this study")
        for v in verdicts:
            st.markdown(f"- {v}")
        st.caption("Comparing groups is the right use of these measures: reliable for groups, not for one person.")
    elif len(names):
        st.info(f"The comparison needs two groups; the data has {len(names)}: {', '.join(map(str, names))}.")

# ---------------------------------------------------------------- table + export
st.subheader("All recordings")
st.dataframe(df, hide_index=True, width="stretch")
buf = io.StringIO()
df.to_csv(buf, index=False)
st.download_button("Download table (CSV)", buf.getvalue(), "vessel_measures.csv", "text/csv")
st.caption("Percentiles only for the same sensor layout as the reference group (healthy, 18–42, n = 52). Recordings from this "
           "dataset belong to the reference group, so their percentile is computed against a group that includes them.")
with st.expander("The same result without the app (fnirs-vascular module)"):
    st.code("pip install ./plugin\nfnirs-vascular study_folder/ --out measures.csv", language="bash")
