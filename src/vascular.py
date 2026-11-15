"""
vascular.py - core of the demo: one 10-min resting fNIRS recording -> two markers of vascular health.

  regulation = power of the slow vasomotion band 0.0095-0.021 Hz (attributed to the endothelium), forehead HbO
  structure  = reflected-wave timing index TI = 1 / (t_reflected - t_systolic), Afkhami et al. 2021

Both are compared with the healthy reference group of the supine dataset (18-42 y, no medication, n = 52).
Group-level markers, not a diagnosis.

  python src/vascular.py --figs        # cohort "two axes" figure + an example card for the slides
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal, stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import haemo  # noqa: E402
import timing_index as tim  # noqa: E402
from features import FOREHEAD, load_meta  # noqa: E402
from ti_followup import ti_segment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "dane/infopack_and_data/fnirs_high_freq_supine/data/baseline"
LO, HI = 0.0095, 0.021
ACT = {0.0: "no activity", 0.32: "light activity regularly", 1.06: "aerobic < 1 h/week",
       1.76: "aerobic 1–3 h/week", 3.03: "aerobic > 3 h/week"}
INK, MUTED, BLUE, ORANGE, GREY = "#14213d", "#5b6472", "#2a78d6", "#d97706", "#b4bcc6"
DISCLAIMER = "A group-level hint compared with healthy people aged 18–42, not a medical diagnosis."
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def subject_of(path) -> int | None:
    m = re.match(r"(\d+)_", Path(path).name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
def analyse(path) -> dict:
    """One recording -> both markers, plus the signals the figures need."""
    path = Path(path)
    h = haemo.convert(path, dpf=6.0)
    fs = h["fs"]
    dead = {c.split()[0] for c, n in zip(h["od_channels"], h["nonpositive"]) if n}
    idx = {n: i for i, n in enumerate(h["pairs"])}
    fore = [idx[n] for n in FOREHEAD if n not in dead]
    if not fore:
        raise ValueError("no usable forehead channel in this recording")
    x = h["hbo"][fore].mean(0)

    # regulation: share of the slow vasomotion band in the resting HbO spectrum
    down = int(fs // 5)
    fsd = fs / down
    xd = signal.detrend(signal.decimate(x - x.mean(), down, ftype="fir"))
    f, p = signal.welch(xd, fsd, nperseg=int(min(len(xd), 300 * fsd)))
    band, tot = (f >= LO) & (f <= HI), (f >= LO) & (f <= 2.0)
    endo = float(np.trapezoid(p[band], f[band]) / np.trapezoid(p[tot], f[tot]))

    # structure: averaged pulse (for display) and the timing index (same code as the cohort analysis)
    ref = tim.bp(x, fs, 0.5, 5)
    fw, pw = signal.welch(ref, fs, nperseg=int(20 * fs))
    m = (fw > 0.7) & (fw < 2.5)
    period = 1 / fw[m][np.argmax(pw[m])]
    tr, _ = signal.find_peaks(-ref, distance=int(0.6 * period * fs), prominence=0.3 * ref.std())
    L = int(np.median(np.diff(tr)))
    tr = [a for a, b in zip(tr[:-1], tr[1:]) if abs((b - a) - L) < 0.2 * L and a + L < len(ref)]
    beat = np.mean([ref[a:a + L] for a in tr], axis=0)
    beat = beat - np.linspace(beat[0], beat[-1], L)
    t_sys, t_refl = tim.channel_times(beat, fs)
    TI = ti_segment(h, 0, h["time"][-1])
    return dict(file=path.name, subject=subject_of(path), t_min=np.arange(len(xd)) / fsd / 60, hbo=xd, f=f, p=p,
                endo=endo, beat=(beat - beat.min()) / np.ptp(beat), beat_ms=1000 * np.arange(L) / fs,
                t_sys=t_sys, t_refl=t_refl, TI=TI, hr=60 / period, n_beats=len(tr),
                duration_min=float(h["time"][-1] / 60))


def cohort() -> pd.DataFrame:
    """Both markers for every person of the supine dataset (precomputed by bands.py and ti_followup.py)."""
    b = pd.read_csv(ROOT / "bands_supine.csv")[["subject", "l30_hbo_endothelial_rel"]] \
        .rename(columns={"l30_hbo_endothelial_rel": "endo"})
    t = pd.read_csv(ROOT / "ti_followup.csv")[["subject", "TI_full"]].rename(columns={"TI_full": "TI"})
    m = load_meta()[["subject", "age", "sex_male", "activity", "medication_any"]]
    d = b.merge(t, on="subject", how="outer").merge(m, on="subject", how="left")
    d["healthy_ref"] = (d.age < 60) & (d.medication_any == 0)
    return d


def score(res: dict, coh: pd.DataFrame) -> dict:
    """Percentiles in the healthy reference group (the person left out) + the activity model."""
    ref = coh[coh.healthy_ref & (coh.subject != res["subject"])].dropna(subset=["endo", "TI", "activity"])
    y = (ref.activity == ref.activity.max()).astype(int)
    mu, sd = ref.endo.mean(), ref.endo.std()
    model = LogisticRegression().fit(((ref.endo - mu) / sd).values[:, None], y)
    return dict(n_ref=len(ref),
                pct_endo=100 * float(np.mean(ref.endo < res["endo"])),
                pct_ti=100 * float(np.mean(ref.TI < res["TI"])) if np.isfinite(res["TI"]) else np.nan,
                prob_active=float(model.predict_proba([[(res["endo"] - mu) / sd]])[0, 1]))


def interpret(res: dict, sc: dict) -> list[str]:
    e, s = sc["pct_endo"], sc["pct_ti"]
    out = [f"Vessel regulation (slow wave 0.0095–0.021 Hz): stronger than in {e:.0f}% of the healthy group."]
    if np.isfinite(s):
        out.append(f"Pulse echo (TI): comes back {'faster' if s >= 50 else 'slower'} than in "
                   f"{s if s >= 50 else 100 - s:.0f}% of the group. Higher TI goes with older age "
                   "(stiffer arteries); for one person it is a hint, not a stiffness measurement.")
    out.append(f"Model (logistic regression, AUC 0.67): {100 * sc['prob_active']:.0f}% chance "
               "that the person does aerobic exercise over 3 h a week.")
    return out


# ---------------------------------------------------------------------------
def _ax(ax, size=(6.4, 3.8)):
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        return fig, ax
    return ax.figure, ax


def fig_signal(res, ax=None):
    fig, ax = _ax(ax)
    ax.plot(res["t_min"], res["hbo"], lw=0.6, color=BLUE)
    ax.set_xlabel("time [min]")
    ax.set_ylabel("HbO [µM]")
    ax.set_title("Forehead signal at rest")
    return fig


def fig_spectrum(res, ax=None):
    fig, ax = _ax(ax)
    f, p = res["f"], res["p"]
    band = (f >= LO) & (f <= HI)
    ax.semilogx(f[1:], p[1:], color=INK)
    ax.fill_between(f[band], p[band], color=ORANGE, alpha=0.65, label="slow vessel wave (endothelium)")
    for x0, lab, y0 in ((0.1, "Mayer waves", 0.97), (0.25, "breathing", 0.88), (1.1, "heart", 0.97)):
        ax.axvline(x0, color=GREY, lw=0.8, ls=":")
        ax.text(x0, y0, lab, fontsize=8, color=MUTED, ha="center", va="top", transform=ax.get_xaxis_transform())
    ax.set_xlim(0.005, 2)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("power")
    ax.set_title(f"Spectrum: vessel wave = {100 * res['endo']:.0f}% of signal")
    ax.legend(fontsize=8, loc="center right")
    return fig


def fig_pulse(res, ax=None):
    fig, ax = _ax(ax)
    ax.plot(res["beat_ms"], res["beat"], color=INK, lw=2)
    ts, tf = res["t_sys"], res["t_refl"]
    if np.isfinite(ts) and np.isfinite(tf):
        for t, lab, c in ((ts, "wave from the heart", BLUE), (tf, "echo wave", ORANGE)):
            i = int(round(t * 1000 / (res["beat_ms"][1] - res["beat_ms"][0])))
            ax.plot(1000 * t, res["beat"][min(i, len(res["beat"]) - 1)], "o", ms=10, color=c, label=lab)
        ax.annotate("", xy=(1000 * tf, 0.06), xytext=(1000 * ts, 0.06), arrowprops=dict(arrowstyle="<->", color=MUTED))
        ax.text(500 * (ts + tf), 0.1, f"{1000 * (tf - ts):.0f} ms on this averaged pulse", ha="center", color=MUTED)
    ax.set_xlabel("time from beat start [ms]")
    ax.set_yticks([])
    ti = f"TI = {res['TI']:.2f} 1/s (channel mean)" if np.isfinite(res["TI"]) else "TI: n/a"
    ax.set_title(f"Averaged pulse wave ({res['n_beats']} beats) · {ti}")
    ax.legend(fontsize=8, loc="upper right")
    return fig


def fig_two_axes(coh, res=None, ax=None):
    fig, ax = _ax(ax, size=(7.6, 5.6))
    ref = coh[coh.healthy_ref].dropna(subset=["endo", "TI", "activity"])
    if res is not None:
        ref = ref[ref.subject != res["subject"]]
    top = ref.activity == ref.activity.max()
    ax.scatter(ref.TI[~top], 100 * ref.endo[~top], s=46, color=GREY, edgecolor="white", label="aerobic ≤ 3 h/week")
    ax.scatter(ref.TI[top], 100 * ref.endo[top], s=46, color=ORANGE, edgecolor="white", label="aerobic > 3 h/week")
    mx, my = ref.TI.median(), 100 * ref.endo.median()
    ax.axvline(mx, color=MUTED, lw=0.8, ls="--")
    ax.axhline(my, color=MUTED, lw=0.8, ls="--")
    if res is not None and np.isfinite(res["TI"]):
        ax.scatter([res["TI"]], [100 * res["endo"]], s=380, marker="*", color=INK, zorder=5, label="this person")
    ax.margins(x=0.08, y=0.12)
    kw = dict(fontsize=8.5, color=MUTED, transform=ax.transAxes)
    ax.text(0.02, 0.98, "more elastic · strong regulation", va="top", **kw)
    ax.text(0.98, 0.98, "stiffer · strong regulation", va="top", ha="right", **kw)
    ax.text(0.02, 0.02, "more elastic · weak regulation", va="bottom", **kw)
    ax.text(0.98, 0.02, "stiffer · weak regulation", va="bottom", ha="right", **kw)
    r, _ = stats.spearmanr(ref.TI, ref.endo)
    ax.set_xlabel("pulse echo index TI [1/s]   →  stiffer arteries")
    ax.set_ylabel("slow vessel wave [% of signal]   →  stronger regulation")
    ax.set_title(f"Two independent axes of vessel health\nhealthy people 18–42 without medication, n = {len(ref)}, axis correlation ρ = {r:+.2f}",
                 fontsize=11)
    ax.legend(fontsize=8.5, loc="center right")
    return fig


def fig_card(res, coh, sc):
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25])
    fig_signal(res, fig.add_subplot(gs[0, 0]))
    fig_spectrum(res, fig.add_subplot(gs[0, 1]))
    fig_pulse(res, fig.add_subplot(gs[1, 0:2]))
    fig_two_axes(coh, res, fig.add_subplot(gs[:, 2]))
    who = f"Osoba {res['subject']}" if res["subject"] else res["file"]
    pti = f"{sc['pct_ti']:.0f}." if np.isfinite(sc["pct_ti"]) else "–"
    fig.suptitle(f"{who}: regulacja naczyń {sc['pct_endo']:.0f}. percentyl · sztywność (TI) {pti} percentyl · "
                 f"model: {100 * sc['prob_active']:.0f}% szans na aeroby > 3 h/tydz", fontsize=13, color=INK)
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return fig


# ---------------------------------------------------------------------------
if __name__ == "__main__" and "--figs" in sys.argv:
    coh = cohort()
    ref = coh[coh.healthy_ref].dropna(subset=["endo", "TI", "activity"])
    r, p = stats.spearmanr(ref.TI, ref.endo)
    print(f"healthy reference n={len(ref)}: TI vs endothelial band rho {r:+.2f} (p {p:.2f}) -> independent axes")
    r, p = stats.spearmanr(ref.endo, ref.activity)
    print(f"endothelial band vs activity, no medication: rho {r:+.2f} (p {p:.4f})")
    fig = fig_two_axes(coh)
    fig.tight_layout()
    fig.savefig(ROOT / "fig_two_axes.png", dpi=150)
    path = next(BASE.glob("17_*.snirf"))
    res = analyse(path)
    sc = score(res, coh)
    fig_card(res, coh, sc).savefig(ROOT / "demo_card_17.png", dpi=130)
    print("\n".join(interpret(res, sc)))
    print("-> fig_two_axes.png, demo_card_17.png")
