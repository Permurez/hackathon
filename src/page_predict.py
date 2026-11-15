"""page_predict.py - "how much does this person exercise?" from one resting recording.

Model: logistic regression on the slow vasomotion band (the only input that helps; richer models are compared on the
page and do worse at n = 52). Training group = the healthy reference group of the module (18-42 y, no medication).
A person from the dataset is predicted by a model that has not seen them (leave-one-out); the track record and the
calibration ("people with a result like this one: X % really train > 3 h/week") come from the same leave-one-out run.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))
sys.path.insert(0, str(ROOT / "src"))
import fnirs_vascular as fv  # noqa: E402
import vascular as V  # noqa: E402
import style  # noqa: E402
from features import load_meta  # noqa: E402

SETS = {
    "fNIRS vessel wave": ["band"],
    "age, sex, BMI": ["age", "sex_male", "bmi"],
    "age, sex, BMI + resting heart rate": ["age", "sex_male", "bmi", "hr"],
    "vessel wave + heart rate": ["band", "hr"],
    "vessel wave + heart rate + TI": ["band", "hr", "TI"],
    "everything (vessel wave, heart rate, TI, age, sex, BMI)": ["band", "hr", "TI", "age", "sex_male", "bmi"],
}
LEVELS = ["niski", "średni", "wysoki"]
LEVELS_WITH = ["niskim", "średnim", "wysokim"]   # "z ... wynikiem"
# questionnaire answer -> typical aerobic hours per week: bin midpoints, the open "> 3 h" bin taken as 4 h
HOURS = {0.0: 0.0, 0.32: 0.5, 1.06: 0.5, 1.76: 2.0, 3.03: 4.0}
LV3 = ["under 1 h/week", "1–3 h/week", "over 3 h/week"]   # questionnaire 0-1.06 / 1.76 / 3.03


def pl(v, f=".2f") -> str:
    return format(v, f)


def model():
    return make_pipeline(StandardScaler(), LogisticRegression())


@st.cache_data(show_spinner=False, persist="disk")
def run_many(paths: tuple[str, ...]) -> list[dict]:
    return fv.analyze_many(list(paths))


@st.cache_data(show_spinner=False)
def training_table() -> pd.DataFrame:
    res = run_many(tuple(sorted(str(p) for p in V.BASE.glob("*.snirf"))))
    rows = [dict(file=r["file"], subject=int(r["file"].split("_")[0]), hr=r["heart_rate_bpm"],
                 band=r["vascular_regulation"]["endothelial_band_share"], TI=r["arterial_timing"]["timing_index_per_s"])
            for r in res if "error" not in r]
    d = pd.DataFrame(rows).merge(load_meta(), on="subject")
    d = d[(d.age < 60) & (d.medication_any == 0)].dropna(subset=["band", "hr", "TI", "activity", "age", "sex_male", "bmi"])
    d["active"] = (d.activity == d.activity.max()).astype(int)
    return d.sort_values("subject").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def evaluate(d: pd.DataFrame):
    y = d.active.to_numpy()
    rows, loo_band = [], None
    for name, cols in SETS.items():
        X = d[cols].to_numpy(float)
        loo = cross_val_predict(model(), X, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
        kf = [roc_auc_score(y, cross_val_predict(model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                                                 method="predict_proba")[:, 1]) for s in range(20)]
        rows.append({"input data": name, "AUC (leave-one-out)": round(roc_auc_score(y, loo), 2),
                     "AUC (5-fold, 20 powtórzeń)": round(float(np.mean(kf)), 2),
                     "trafienia": f"{int(((loo >= 0.5) == y).sum())}/{len(y)}"})
        if cols == ["band"]:
            loo_band = loo
    edges = np.quantile(loo_band, [1 / 3, 2 / 3])
    lvl = np.digitize(loo_band, edges)
    calib = [(int((lvl == k).sum()), float(y[lvl == k].mean())) for k in range(3)]
    return pd.DataFrame(rows), loo_band, edges, calib


def ord_probs(params, z):
    """P(level 0, 1, 2) of an ordinal logit with one predictor; statsmodels params = [beta, cut1, log(cut2 - cut1)]."""
    b, c1, c2 = params[0], params[1], params[1] + np.exp(params[2])
    lo, hi = expit(c1 - b * z), expit(c2 - b * z)
    return np.array([lo, hi - lo, 1 - hi])


@st.cache_data(show_spinner=False)
def ordinal(d: pd.DataFrame):
    """Three activity levels from the band: leave-one-out probabilities, full fit, calibration by model tertile."""
    y = np.where(d.activity >= 3.03, 2, np.where(d.activity >= 1.76, 1, 0))
    x = d.band.to_numpy(float)
    n = len(x)
    P = np.zeros((n, 3))
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        mu, sd = x[m].mean(), x[m].std()
        r = OrderedModel(y[m], ((x[m] - mu) / sd)[:, None], distr="logit").fit(method="bfgs", disp=False)
        P[i] = ord_probs(np.asarray(r.params), (x[i] - mu) / sd)
    mu, sd = x.mean(), x.std()
    full = OrderedModel(y, ((x - mu) / sd)[:, None], distr="logit").fit(method="bfgs", disp=False)
    ev = P @ np.arange(3)
    edges = np.quantile(ev, [1 / 3, 2 / 3])
    lvl = np.digitize(ev, edges)
    table = np.array([np.bincount(y[lvl == k], minlength=3) / max(int((lvl == k).sum()), 1) for k in range(3)])
    rho, p = stats.spearmanr(ev, y)
    hrs = np.round(d.activity, 2).map(HOURS).to_numpy(float)          # the questionnaire answer in hours
    rep = np.array([hrs[y == k].mean() for k in range(3)])             # typical hours of each of the 3 levels
    loo_mean = (hrs.sum() - hrs) / (n - 1)                             # baseline: the group mean, person left out
    summary = dict(rho=float(rho), p=float(p), acc=float((P.argmax(1) == y).mean()), base=float((y == 2).mean()),
                   counts=[int((y == k).sum()) for k in range(3)], tert=[int((lvl == k).sum()) for k in range(3)],
                   edges=[float(e) for e in edges], rep=[float(v) for v in rep],
                   mae=float(np.abs(P @ rep - hrs).mean()), mae_base=float(np.abs(loo_mean - hrs).mean()))
    return y, P, (float(mu), float(sd), np.asarray(full.params)), table, summary


def hours_txt(h: float) -> str:
    """Hours rounded to half an hour, Polish style: 'ok. 2,5 h', 'ok. 4 h'."""
    r = round(h * 2) / 2
    return "0 h" if r == 0 else f"about {pl(r, '.1f').replace('.0', '')} h"


@st.cache_data(show_spinner=False)
def module_result(path: str) -> dict:
    return fv.analyze(path)


@st.cache_data(show_spinner=False)
def breath_hold() -> pd.DataFrame:
    """Per person, from the breath-hold recording of the same session (src/bh_reflex.py, src/bh_co2.py)."""
    r = pd.read_csv(ROOT / "bh_reflex.csv").groupby("subject").agg(
        n_holds=("duration", "size"), longest_s=("duration", "max"), hr_change=("hr_change_pct", "mean"))
    return r.join(pd.read_csv(ROOT / "bh_co2.csv").groupby("subject").long_rise.mean().rename("hbo_rise"))


# ---------------------------------------------------------------- data + model
st.title("Activity estimate from one recording")
style.hero("One 10-minute recording, one profile: heart rate, how the vessels work, the reaction to holding the breath and "
           "an estimate of weekly training hours. Each number shows whether it is measured, compared with a "
           "group, or only a hint, and for the hours: how far off it may be.",
           ["Pick a person from the dataset or upload a .snirf file", "Read the profile at the top",
            "Below: roughly how many hours a week they train"])
with st.spinner("Preparing the model (the first run processes 59 recordings)…"):
    d = training_table()
    table, loo_band, edges, calib = evaluate(d)
y = d.active.to_numpy()
full = model().fit(d[["band"]].to_numpy(float), y)

src = st.sidebar.radio("Recording", ["Person from the dataset", "Upload a .snirf file"])
truth = None
if src.startswith("Person"):
    sid = st.sidebar.selectbox("Person", list(d.subject), format_func=lambda s: f"Person {s}",
                               index=int(np.flatnonzero(d.subject.to_numpy() == 17)[0]) if 17 in set(d.subject) else 0)
    i = int(np.flatnonzero(d.subject.to_numpy() == sid)[0])
    band, p = float(d.band[i]), float(loo_band[i])
    truth = d.iloc[i]
    st.sidebar.caption("Predicted by a model that has not seen this person.")
else:
    up = st.sidebar.file_uploader("A .snirf file: 10 min of rest, Cortivision C23", type=["snirf"])
    if up is None:
        st.info("Upload a .snirf recording in the left panel.")
        st.stop()
    tmp = Path(tempfile.mkdtemp(prefix="fnirs_pred_")) / up.name
    tmp.write_bytes(up.getvalue())
    r = fv.analyze(tmp)
    if not r["recording"]["reference_montage"]:
        st.error("Different sensor layout than the training group: an estimate would not be reliable.")
        st.json(r)
        st.stop()
    band = r["vascular_regulation"]["endothelial_band_share"]
    p = float(full.predict_proba([[band]])[0, 1])
    for wmsg in r["warnings"]:
        st.warning(wmsg)

y3, P3, (mu3, sd3, par3), table3, s3 = ordinal(d)
p3 = P3[i] if truth is not None else ord_probs(par3, (band - mu3) / sd3)
est = float(p3 @ np.array(s3["rep"]))
t3 = int(np.digitize([p3 @ np.arange(3)], s3["edges"])[0])
mod = module_result(str(V.BASE / d.file[i])) if truth is not None else r
bhs = breath_hold()
bh = bhs.loc[int(truth.subject)] if truth is not None and int(truth.subject) in bhs.index else None

# ---------------------------------------------------------------- profile: everything one session says
st.subheader("Everything this one session says")
style.legend()
vr, tm, q = mod["vascular_regulation"], mod["arterial_timing"], mod["recording"]
row1, row2 = st.columns(3), st.columns(3)
B = style.badge
row1[0].metric("Resting heart rate", f"{mod['heart_rate_bpm']:.0f} /min")
row1[0].caption(f"{B('pomiar')}in adults at rest usually 60–100/min", unsafe_allow_html=True)
row1[1].metric("Aerobic training", f"{hours_txt(est)} / week")
row1[1].caption(f"{B('wskazowka')}estimate from one recording; typical error ±{pl(s3['mae'], '.1f')} h "
                "(details below)", unsafe_allow_html=True)
row1[2].metric("Recording quality", "no warnings" if not mod["warnings"] else f"warnings: {len(mod['warnings'])}")
row1[2].caption(f"{B('pomiar')}{pl(q['duration_min'], '.1f')} min, {mod['beats_used']} heartbeats, missing "
                f"{pl(q['missing_samples_pct'], '.1f')}% of samples", unsafe_allow_html=True)
pe, pt = vr["percentile_vs_reference"], tm["percentile_vs_reference"]
row2[0].metric("Vessel regulation", "–" if pe is None else f"above {pe:.0f}% of people")
row2[0].caption(f"{B('grupa')}one recording is quite uncertain; a repeat may shift the result noticeably",
                unsafe_allow_html=True)
row2[1].metric("Pulse echo (TI)", "–" if pt is None else f"above {pt:.0f}% of people")
row2[1].caption(f"{B('grupa')}higher = the pulse wave comes back faster; rises with age; a stable measure",
                unsafe_allow_html=True)
if bh is not None and np.isfinite(bh.hr_change):
    rise = f"; HbO at the end {pl(bh.hbo_rise, '+.2f')} µM" if np.isfinite(bh.hbo_rise) else ""
    row2[2].metric("Breath holding", f"heart rate {bh.hr_change:+.0f}%")
    row2[2].caption(f"{B('pomiar')}same session, no norm; holds analysed: {int(bh.n_holds)}, "
                    f"longest {bh.longest_s:.0f} s{rise}", unsafe_allow_html=True)
else:
    row2[2].metric("Breath holding", "no recording")
    row2[2].caption("the breath-hold recording of this session was not loaded")
st.caption("Not shown: breathing rate. It could not be measured reliably from the resting supine signal "
           "(we checked), so we do not report it.")

# ---------------------------------------------------------------- hours per week
st.subheader("How many hours a week does this person train?")
h1, h2 = st.columns([1, 2])
h1.metric("Estimated aerobic training", f"{hours_txt(est)} / week")
h1.caption(f"{B('wskazowka')}typical error ±{pl(s3['mae'], '.1f')} h", unsafe_allow_html=True)
with h2:
    m1, m2, m3 = st.columns(3)
    for col, lab, v in zip((m1, m2, m3), LV3, p3):
        col.metric(lab, f"{100 * v:.0f}%")
    st.caption("Probability of each questionnaire range.")
if truth is not None:
    st.markdown(f"**This person's questionnaire:** {V.ACT.get(truth.activity, '?')} (age {int(truth.age)}), i.e. "
                f"{hours_txt(HOURS[round(float(truth.activity), 2)])} a week.")
rep = s3["rep"]
st.markdown(
    "- **How:** from one number in the recording (the slow vessel wave) the model gives the probability of each "
    f"questionnaire range; the estimate is their weighted average of typical values: {pl(rep[0], '.1f')} h, "
    f"{pl(rep[1], '.0f')} h and {pl(rep[2], '.0f')} h a week.\n"
    f"- **How accurate:** on people the model has not seen it is off by ±{pl(s3['mae'], '.1f')} h; "
    f"on average; giving everyone the group average is off by ±{pl(s3['mae_base'], '.1f')} h. The ranking of people holds "
    f"(ρ = {pl(s3['rho'])}).\n"
    "- **Limits:** the questionnaire cannot tell 4 from 10 hours, so “over 3 h” counts as 4 h, and for people who do not "
    f"train at all the estimate tends to be too high. A proof of concept on {len(d)} people; the user always makes "
    "just this one recording.")

# ---------------------------------------------------------------- honesty
st.subheader("Which data predict training best?")
show = table.drop(columns=["AUC (5-fold, 20 powtórzeń)", "trafienia"]).rename(
    columns={"AUC (leave-one-out)": "accuracy (AUC)"}).sort_values("accuracy (AUC)", ascending=False)
st.dataframe(show, hide_index=True, width="stretch")
st.caption("Accuracy (AUC): 0.5 is guessing, 1.0 is perfect. Measured on people the model "
           "has not seen.")
t = table.set_index("input data")
best = t["AUC (leave-one-out)"].idxmax()
dem = t.loc[["age, sex, BMI", "age, sex, BMI + resting heart rate"]]
st.markdown(
    f"- Best predictor: **{best}** (AUC {pl(t.loc[best, 'AUC (leave-one-out)'])}).\n"
    f"- Personal data alone (age, sex, BMI, heart rate): AUC {pl(dem['AUC (leave-one-out)'].min())}–"
    f"{pl(dem['AUC (leave-one-out)'].max())}.\n"
    "- **Conclusion:** the fNIRS recording carries information about training beyond personal data, but the advantage is moderate.")
st.caption("Honestly: the vessel wave was picked in a screen of this same data, so on new people accuracy will likely be "
           "lower; activity comes from a questionnaire; one recording has limited repeatability. It is a hint for "
           "people with a similar result, not a measurement of this person's activity.")
