"""activity.py - the training estimate for the app views: ordinal model on the slow vessel wave -> probability of
three questionnaire levels -> approximate aerobic hours per week. Same data and model as page_predict.py
(healthy reference group 18-42 y without medication; leave-one-out error reported)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.special import expit
from statsmodels.miscmodels.ordinal_model import OrderedModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))
sys.path.insert(0, str(ROOT / "src"))
import fnirs_vascular as fv  # noqa: E402
import vascular as V  # noqa: E402
from features import load_meta  # noqa: E402

# questionnaire answer -> typical aerobic hours per week: bin midpoints, the open "> 3 h" bin taken as 4 h
HOURS = {0.0: 0.0, 0.32: 0.5, 1.06: 0.5, 1.76: 2.0, 3.03: 4.0}
LV3 = ["under 1 h/week", "1–3 h/week", "over 3 h/week"]


@st.cache_data(show_spinner=False, persist="disk")
def _run_many(paths: tuple[str, ...]) -> list[dict]:
    return fv.analyze_many(list(paths))


@st.cache_data(show_spinner=False)
def training_table() -> pd.DataFrame:
    res = _run_many(tuple(sorted(str(p) for p in V.BASE.glob("*.snirf"))))
    rows = [dict(file=r["file"], subject=int(r["file"].split("_")[0]), hr=r["heart_rate_bpm"],
                 band=r["vascular_regulation"]["endothelial_band_share"], TI=r["arterial_timing"]["timing_index_per_s"])
            for r in res if "error" not in r]
    d = pd.DataFrame(rows).merge(load_meta(), on="subject")
    d = d[(d.age < 60) & (d.medication_any == 0)].dropna(subset=["band", "hr", "TI", "activity", "age", "sex_male", "bmi"])
    return d.sort_values("subject").reset_index(drop=True)


def ord_probs(params, z):
    """P(level 0, 1, 2) of an ordinal logit with one predictor; statsmodels params = [beta, cut1, log(cut2 - cut1)]."""
    b, c1, c2 = params[0], params[1], params[1] + np.exp(params[2])
    lo, hi = expit(c1 - b * z), expit(c2 - b * z)
    return np.array([lo, hi - lo, 1 - hi])


@st.cache_data(show_spinner=False)
def model(d: pd.DataFrame) -> dict:
    """Full fit, typical hours per level and the leave-one-out error of the hours estimate."""
    y = np.where(d.activity >= 3.03, 2, np.where(d.activity >= 1.76, 1, 0))
    x = d.band.to_numpy(float)
    hrs = np.round(d.activity, 2).map(HOURS).to_numpy(float)
    rep = np.array([hrs[y == k].mean() for k in range(3)])
    n = len(x)
    loo = np.zeros(n)
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        mu, sd = x[m].mean(), x[m].std()
        r = OrderedModel(y[m], ((x[m] - mu) / sd)[:, None], distr="logit").fit(method="bfgs", disp=False)
        loo[i] = ord_probs(np.asarray(r.params), (x[i] - mu) / sd) @ rep
    mu, sd = x.mean(), x.std()
    full = OrderedModel(y, ((x - mu) / sd)[:, None], distr="logit").fit(method="bfgs", disp=False)
    return dict(mu=float(mu), sd=float(sd), params=np.asarray(full.params), rep=rep,
                mae=float(np.abs(loo - hrs).mean()), rho=float(stats.spearmanr(loo, hrs)[0]), n=n)


def estimate_hours(band: float):
    """band = share of the slow vessel wave (0-1) -> (hours per week, probabilities of the 3 levels, typical error)."""
    m = model(training_table())
    p3 = ord_probs(m["params"], (band - m["mu"]) / m["sd"])
    return float(p3 @ m["rep"]), p3, m["mae"]


def hours_txt(h: float) -> str:
    r = round(h * 2) / 2
    return "0 h" if r == 0 else f"about {format(r, '.1f').replace('.0', '')} h"
