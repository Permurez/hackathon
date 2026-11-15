"""
bh_co2.py - does HbO rise with the CO2 build-up of a breath hold (cerebrovascular reactivity), and in which layer?

Per hold (marker 2 -> next marker 1, >= MIN_HOLD_S), slow HbO (low-pass 0.1 Hz):
  rise = mean of the last 5 s of the hold - mean of the 10 s before the onset   [uM]
  slope = rise / duration                                                       [uM/s]  (breath-holding-index-like)
Layers: skin = 10 mm channels, long = 30 mm channels, brain = long - beta*skin (OLS over the whole file).
SHAM: identical windows in the same person's resting baseline. Tests fixed in advance:
  1. hold vs sham rise (Wilcoxon, per person)   2. brain slope vs activity   3. longest hold vs activity
"""

from __future__ import annotations

import glob
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import haemo  # noqa: E402
from bh_reflex import BH_DIR, BS_DIR, LONG, META, MIN_HOLD_S, ROOT, SHORT  # noqa: E402

GRID = np.arange(-20, 45.01, 0.5)


def lowpass(x, fs, hi=0.1):
    sos = signal.butter(3, hi / (fs / 2), output="sos")
    return signal.sosfiltfilt(sos, x)


def holds_of(markers):
    out = []
    for k, (t2, code) in enumerate(markers):
        end = next((t for t, c in markers[k + 1:] if c == "1"), None) if code == "2" else None
        if end is not None and end - t2 >= MIN_HOLD_S:
            out.append((t2, end - t2))
    return out


def analyse(arg):
    path, windows = arg if isinstance(arg, tuple) else (arg, None)
    path = Path(path)
    sid = int(re.search(r"(?:_V2_|^)(\d+)_[A-Z]{2}_", path.name).group(1))
    h = haemo.convert(path)
    fs, t = h["fs"], h["time"]
    dead = {c.split()[0] for c, n in zip(h["od_channels"], h["nonpositive"]) if n}
    idx = {n: i for i, n in enumerate(h["pairs"])}
    L = [idx[n] for n in LONG if n not in dead]
    S = [idx[n] for n in SHORT if n not in dead]
    if not L or not S:
        return []
    skin = lowpass(h["hbo"][S].mean(0), fs)
    long_ = lowpass(h["hbo"][L].mean(0), fs)
    beta = float(np.dot(long_ - long_.mean(), skin - skin.mean()) / np.dot(skin - skin.mean(), skin - skin.mean()))
    layers = {"skin": skin, "long": long_, "brain": long_ - beta * skin}
    if windows is None:
        windows = holds_of(h["markers"])
    rows = []
    for t2, dur in windows:
        end = t2 + dur
        if t2 - 20 < 0 or end > t[-1]:
            continue
        r = dict(subject=sid, onset=t2, duration=dur, beta=beta)
        for name, x in layers.items():
            pre = x[(t >= t2 - 10) & (t < t2)].mean()
            late = x[(t >= end - 5) & (t < end)].mean()
            r[f"{name}_rise"] = late - pre
            r[f"{name}_slope"] = (late - pre) / dur
            r[f"_curve_{name}"] = np.interp(t2 + GRID, t, x - pre, left=np.nan, right=np.nan)
        rows.append(r)
    return rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import spearmanr, wilcoxon

    with Pool(6) as p:
        holds = pd.DataFrame([r for rows in p.map(analyse, sorted(glob.glob(str(BH_DIR / "*.snirf")))) for r in rows])
        bs = {int(re.match(r"(\d+)_", Path(f).name).group(1)): f for f in glob.glob(str(BS_DIR / "*.snirf"))}
        jobs = [(bs[s], list(zip(g.onset, g.duration))) for s, g in holds.groupby("subject") if s in bs]
        sham = pd.DataFrame([r for rows in p.map(analyse, jobs) for r in rows])
    num = [c for c in holds if not c.startswith("_")]
    per, per_s = holds[num].groupby("subject").mean(), sham[num].groupby("subject").mean()
    both = per.index.intersection(per_s.index)
    print(f"{len(holds)} holds >= {MIN_HOLD_S:.0f} s, {holds.subject.nunique()} people, median {holds.duration.median():.0f} s")
    print("\n1. HbO rise, last 5 s of the hold vs 10 s before [uM]  (per person; SHAM = same windows at rest)")
    for lay in ("skin", "long", "brain"):
        c = f"{lay}_rise"
        x, y = per[c], per_s[c]
        print(f"  {lay:<6} hold median {x.median():+.3f} (up in {100 * (x > 0).mean():.0f} %)   "
              f"sham median {y.median():+.3f} (up in {100 * (y > 0).mean():.0f} %)   "
              f"hold vs sham p = {wilcoxon(per.loc[both, c], per_s.loc[both, c]).pvalue:.2g}")
    m = pd.read_csv(META, decimal=",").rename(columns={"Nr osoby badanej": "subject", "Wiek": "age",
                                                       "Ocena aktywności fizycznej": "activity"})
    per = per.join(m.set_index("subject")[["age", "activity"]])
    per["max_hold"] = holds.groupby("subject").duration.max()
    for k, (c, lab) in enumerate([("brain_slope", "brain HbO slope (CO2 reactivity)"),
                                  ("max_hold", "longest marked hold [s]")], start=2):
        d = per[[c, "activity", "age"]].dropna()
        r, pv = spearmanr(d[c], d["activity"])
        ra, pa = spearmanr(d[c], d["age"])
        print(f"{k}. {lab:<34} vs activity rho {r:+.2f} (p {pv:.3f})   [vs age rho {ra:+.2f} (p {pa:.3f})]  n={len(d)}")
    holds[num].to_csv(ROOT / "results/bh_co2.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    H = holds[holds.duration >= 25]
    Sm = sham[sham.duration >= 25]
    for a, (lay, col) in zip(ax, [("skin", "#d97706"), ("long", "#7c3aed"), ("brain", "#2563eb")]):
        for D, c2, lab, ls in [(H, col, "bezdech", "-"), (Sm, "#7b8794", "kontrola: spoczynek", "--")]:
            C = np.vstack(D[f"_curve_{lay}"])
            P = np.array([np.nanmean(C[D.subject.values == s], 0) for s in np.unique(D.subject)])
            mu, se = np.nanmean(P, 0), np.nanstd(P, 0) / np.sqrt(np.sum(np.isfinite(P), 0))
            a.plot(GRID, mu, color=c2, lw=2, ls=ls, label=f"{lab} (n={len(P)})")
            a.fill_between(GRID, mu - 1.96 * se, mu + 1.96 * se, color=c2, alpha=0.15)
        a.axvline(0, color="k", lw=0.8); a.axvline(25, color="k", lw=0.5, ls=":")
        a.set_title({"skin": "skóra 10 mm", "long": "30 mm", "brain": "mózg = 30 mm − β·skóra"}[lay])
        a.set_xlabel("czas od początku bezdechu [s]")
        a.legend(fontsize=7)
    ax[0].set_ylabel("ΔHbO [µM] (średnia ± 95% CI)")
    fig.tight_layout()
    fig.savefig(ROOT / "results/bh_co2.png", dpi=120)
    print(f"figure -> {ROOT / 'results/bh_co2.png'}")


if __name__ == "__main__":
    main()
