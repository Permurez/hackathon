"""
spiroximetry.py - venous oxygen saturation (SvO2) from the respiratory oscillation of HbO / HbR
(near-infrared spiroximetry, Franceschini et al. 2002, J Appl Physiol 92:372), DOT sitting dataset.

Idea: breathing moves mostly VENOUS blood in and out of the head (intrathoracic pressure pump). If the
respiratory oscillation is a pure venous volume oscillation, then at the breathing frequency
     dHbO = SvO2 * dHbT,  dHbR = (1 - SvO2) * dHbT   ->   SvO2 = 1 / (1 + dHbR/dHbO)   (in-phase part)

Per file and channel: complex amplitude of HbO and HbR at the breathing frequency, per breath
  SPB (paced 6/min): lock-in at 0.10 Hz on each 10-s cycle (15 cycles, markers 21/23/25/27)
  RS  (free):        lock-in at the person's own breathing frequency, on consecutive 1-breath windows
Channel quality (chosen WITHOUT looking at SvO2): HbO-HbR phase within +-30 deg (volume oscillation)
and cross-breath coherence; per depth tier the TOP_K best-coherence channels are averaged.

Tests (fixed before running):
  1. depth: SvO2 skin 10 mm > 15-19 mm > 25-42 mm (the brain extracts more O2 than the scalp)
  2. reliability: ICC(1) of SvO2 over the 4 SPB sessions of a day
  3. agreement: SvO2 from free breathing (RS) vs paced breathing (SPB) in the same session
Every test is run twice: DPF from Scholkmann & Wolf (wavelength-dependent) and a flat DPF = 6.

  python src/spiroximetry.py         # -> spiroximetry.csv + spiroximetry.png + summary
"""

from __future__ import annotations

import glob
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import haemo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOT = ROOT / "dane/infopack_and_data/dot_montage_sitting"
TIERS = {"10mm": (9, 11), "17mm": (14, 20), "35mm": (24, 45)}
TOP_K = 3
MAX_PHASE = 30.0
DPF_MODES = {"dpf_age": dict(age=23), "dpf_flat6": dict(dpf=6.0)}


def windows(h, task):
    """(start, freq) of single-breath windows."""
    t, mk = h["time"], h["markers"]
    if task == "SPB":
        return [(o, 0.1) for o, c in mk if c in ("21", "23", "25", "27") and o + 10 <= t[-1]]
    # RS: own breathing frequency from the long-channel HbT spectrum, first 600 s
    from scipy import signal
    m = h["distance_mm"] > 24
    x = (h["hbo"][m] + h["hbr"][m]).mean(0)
    fs = h["fs"]
    f, p = signal.welch(x[: int(600 * fs)] - x.mean(), fs, nperseg=int(120 * fs))
    band = (f > 0.12) & (f < 0.5)
    fr = f[band][np.argmax(p[band])]
    T = 1 / fr
    return [(s, fr) for s in np.arange(0, min(600, t[-1]) - T, T)]


def channel_amplitudes(h, wins):
    """Complex amplitudes per breath: arrays [n_breaths, n_channels] for HbO and HbR."""
    t = h["time"]
    Ao, Ar = [], []
    for s, fr in wins:
        m = (t >= s) & (t < s + 1 / fr)
        ref = np.exp(-2j * np.pi * fr * t[m])
        o = h["hbo"][:, m] - h["hbo"][:, m].mean(1, keepdims=True)
        r = h["hbr"][:, m] - h["hbr"][:, m].mean(1, keepdims=True)
        Ao.append((o * ref).mean(1))
        Ar.append((r * ref).mean(1))
    return np.array(Ao), np.array(Ar)


def analyse(path):
    path = Path(path)
    sid, task, ses = re.match(r"(\d+)_[A-Z]+_(RS|SPB)(\d)", path.name).groups()
    rows = []
    for mode, kw in DPF_MODES.items():
        h = haemo.convert(path, **kw)
        dead = {c.split()[0] for c, n in zip(h["od_channels"], h["nonpositive"]) if n}
        alive = np.array([p not in dead for p in h["pairs"]])
        wins = windows(h, task)
        Ao, Ar = channel_amplitudes(h, wins)
        H = (Ar * np.conj(Ao)).mean(0) / (np.abs(Ao) ** 2).mean(0)        # HbR/HbO transfer at breathing freq
        coh = np.abs((Ar * np.conj(Ao)).mean(0)) / np.sqrt((np.abs(Ar) ** 2).mean(0) * (np.abs(Ao) ** 2).mean(0))
        phase = np.degrees(np.angle(H))
        S = 1 / (1 + np.real(H))
        r = dict(subject=int(sid), task=task, session=int(ses), dpf=mode, breath_hz=wins[0][1], n_breaths=len(wins))
        for tier, (lo, hi) in TIERS.items():
            ok = alive & (h["distance_mm"] >= lo) & (h["distance_mm"] <= hi) & (np.abs(phase) < MAX_PHASE)
            ok &= (S > 0) & (S < 1.5)  # drop numerically broken channels (|HbO| ~ 0), NOT a plausibility cut
            idx = np.flatnonzero(ok)
            best = idx[np.argsort(-coh[idx])][:TOP_K]
            r[f"svo2_{tier}"] = float(np.mean(S[best])) if len(best) else np.nan
            r[f"n_ok_{tier}"] = int(ok.sum())
            r[f"coh_{tier}"] = float(np.mean(coh[best])) if len(best) else np.nan
        rows.append(r)
    return rows


def icc1(df, col):
    """One-way random-effects ICC(1) over sessions (rows: subject x session)."""
    d = df.dropna(subset=[col]).groupby("subject").filter(lambda g: len(g) >= 2)
    g = d.groupby("subject")[col]
    k = g.size().mean()
    msb = (g.mean().sub(d[col].mean()) ** 2 * g.size()).sum() / (g.ngroups - 1)
    msw = ((d[col] - g.transform("mean")) ** 2).sum() / (len(d) - g.ngroups)
    return (msb - msw) / (msb + (k - 1) * msw), g.ngroups


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, wilcoxon

    files = sorted(glob.glob(str(DOT / "data/SPB/*.snirf")) + glob.glob(str(DOT / "data/RS/*.snirf")))
    with Pool(6) as p:
        df = pd.DataFrame([r for rows in p.map(analyse, files) for r in rows])
    df.to_csv(ROOT / "spiroximetry.csv", index=False)
    T = list(TIERS)
    for mode in DPF_MODES:
        d = df[df.dpf == mode]
        spb, rs = d[d.task == "SPB"], d[d.task == "RS"]
        print(f"\n===== {mode} =====")
        print("usable channels per file (median):", {t: int(spb[f'n_ok_{t}'].median()) for t in T},
              " coherence of chosen:", {t: round(spb[f'coh_{t}'].median(), 2) for t in T})
        per = spb.groupby("subject")[[f"svo2_{t}" for t in T]].mean()
        print("1. DEPTH (paced breathing, person mean of 4 sessions)")
        for t in T:
            x = per[f"svo2_{t}"].dropna()
            print(f"   SvO2 {t:<5} median {100 * x.median():5.1f}%  [range {100 * x.min():.0f}-{100 * x.max():.0f}]  n={len(x)}")
        for a, b in [("10mm", "17mm"), ("17mm", "35mm"), ("10mm", "35mm")]:
            x = per[[f"svo2_{a}", f"svo2_{b}"]].dropna()
            diff = x.iloc[:, 0] - x.iloc[:, 1]
            print(f"   {a} > {b}: in {100 * (diff > 0).mean():.0f}% of people, median diff {100 * diff.median():+.1f} pp, "
                  f"Wilcoxon p = {wilcoxon(diff).pvalue:.2g}")
        print("2. RELIABILITY over 4 paced sessions, ICC(1):",
              {t: f"{icc1(spb, f'svo2_{t}')[0]:.2f} (n={icc1(spb, f'svo2_{t}')[1]})" for t in T})
        m = spb.merge(rs, on=["subject", "session"], suffixes=("_spb", "_rs"))
        print(f"3. FREE vs PACED breathing (same session; RS breathing {60 * rs.breath_hz.median():.1f}/min):")
        for t in T:
            x = m[[f"svo2_{t}_spb", f"svo2_{t}_rs"]].dropna()
            pp = x.groupby(m.loc[x.index, "subject"]).mean()
            rho, pv = spearmanr(pp.iloc[:, 0], pp.iloc[:, 1])
            print(f"   {t:<5} median RS {100 * x.iloc[:, 1].median():.1f}% vs SPB {100 * x.iloc[:, 0].median():.1f}%; "
                  f"between-person rho {rho:+.2f} (p {pv:.3f}, n {len(pp)})")

    d = df[(df.dpf == "dpf_age") & (df.task == "SPB")]
    per = d.groupby("subject")[[f"svo2_{t}" for t in T]].mean() * 100
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), facecolor="#fcfcfb")
    for _, r in per.iterrows():
        ax[0].plot(range(3), r.values, color="#c3c2b7", lw=0.9, marker="o", ms=3)
    ax[0].plot(range(3), per.median().values, color="#2563eb", lw=2.5, marker="o", ms=8, label="mediana")
    ax[0].set_xticks(range(3), ["skóra 10 mm", "15–19 mm", "25–42 mm\n(sięga kory)"])
    ax[0].set_ylabel("saturacja krwi żylnej SvO₂ [%]"); ax[0].legend(fontsize=8)
    ax[0].set_title("Saturacja żylna vs głębokość (każda linia = 1 osoba)", fontsize=10)
    for s, g in d.groupby("subject"):
        ax[1].plot(g.session, 100 * g["svo2_35mm"], marker="o", ms=4, lw=1, alpha=0.7)
    ax[1].set_xticks([1, 2, 3, 4], ["9:30", "12:00", "15:00", "17:30"])
    ax[1].set_ylabel("SvO₂ 25–42 mm [%]"); ax[1].set_title("Ta sama osoba w 4 sesjach (każda linia = 1 osoba)", fontsize=10)
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "spiroximetry.png", dpi=130, facecolor=fig.get_facecolor())
    print(f"\n-> spiroximetry.csv, spiroximetry.png")


if __name__ == "__main__":
    main()
