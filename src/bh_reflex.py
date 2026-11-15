"""
bh_reflex.py - is a diving-type response visible during the breath holds of the supine dataset?

Two textbook signs of the (dry) diving response, checked hold by hold against the free breathing just before:
  1. BRADYCARDIA          heart rate during the hold < before
  2. SKIN VASOCONSTRICTION  pulse amplitude in the 10 mm (skin-only) channels during the hold < before
     (the same logic as a finger PPG: smaller pulse = narrower skin vessels)
For comparison the same pulse amplitude is reported for the 30 mm channels.

Only holds with a marked end (marker 2 -> next marker 1) of at least MIN_HOLD_S are used.
Per person: mean over their holds; test across people: Wilcoxon signed-rank (one per sign, fixed in advance).

  python src/bh_reflex.py            # -> table in the terminal + bh_reflex.png + bh_reflex.csv
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

ROOT = Path(__file__).resolve().parents[1]
BH_DIR = ROOT / "dane/infopack_and_data/fnirs_high_freq_supine/data/BH"
BS_DIR = ROOT / "dane/infopack_and_data/fnirs_high_freq_supine/data/baseline"
META = ROOT / "dane/infopack_and_data/fnirs_high_freq_supine/subjects_description.csv"
LONG = ["S6_D4", "S5_D3", "S6_D5", "S5_D4"]  # forehead 30 mm, no hair
SHORT = ["S7_D4", "S8_D5"]                   # 10 mm, skin only
MIN_HOLD_S = 15.0
BASE = (-20.0, -2.0)     # free breathing before the hold [s from onset]
GRID = np.arange(-20, 40.01, 0.5)  # time axis of the averaged curves


def beats(x, fs):
    """Beat times [s] (steepest upstroke) and beat amplitudes (trough -> next peak) of a cardiac signal."""
    xc = haemo_band(x, fs, 0.5, 5.0)
    f, p = signal.welch(xc, fs, nperseg=int(20 * fs))
    m = (f > 0.7) & (f < 2.5)
    f0 = f[m][np.argmax(p[m])]
    tr, _ = signal.find_peaks(-xc, distance=int(0.55 * fs / f0), prominence=0.2 * np.std(xc))
    t, amp = [], []
    d = np.gradient(xc)
    for a, b in zip(tr[:-1], tr[1:]):
        i = a + int(np.argmax(d[a:a + max(3, (b - a) // 2)]))
        t.append(i / fs)
        amp.append(xc[a:b].max() - xc[a])
    t, amp = np.array(t), np.array(amp)
    ibi = np.diff(t)
    ok = np.ones(len(ibi), bool)
    med = signal.medfilt(ibi, 9) if len(ibi) > 9 else np.full(len(ibi), np.median(ibi))
    ok &= np.abs(ibi - med) < 0.25 * med  # drop missed / false beats against the LOCAL median
    return t[1:][ok], 60.0 / ibi[ok], amp[1:][ok]


def haemo_band(x, fs, lo, hi):
    sos = signal.butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    return signal.sosfiltfilt(sos, x)


def mean_in(t, y, t0, t1):
    m = (t >= t0) & (t < t1)
    return float(np.mean(y[m])) if m.sum() >= 3 else np.nan


def curve(t, y, onset, ref):
    """y relative to the pre-hold reference, on GRID (s from onset)."""
    return np.interp(onset + GRID, t, y, left=np.nan, right=np.nan) / ref


def analyse(arg):
    path, windows = arg if isinstance(arg, tuple) else (arg, None)
    path = Path(path)
    sid = int(re.search(r"(?:_V2_|^)(\d+)_[A-Z]{2}_", path.name).group(1))
    h = haemo.convert(path)
    fs = h["fs"]
    dead = {c.split()[0] for c, n in zip(h["od_channels"], h["nonpositive"]) if n}
    idx = {n: i for i, n in enumerate(h["pairs"])}
    longs = [idx[n] for n in LONG if n not in dead]
    shorts = [idx[n] for n in SHORT if n not in dead]
    if not longs or not shorts:
        return []
    xl = h["hbo"][longs].mean(0)
    xs = h["hbo"][shorts].mean(0)
    tl, hr, amp_l = beats(xl, fs)      # HR from the 30 mm average (strongest pulse)
    ts, _, amp_s = beats(xs, fs)
    mk = h["markers"]
    if windows is None:  # real holds: marker 2 -> next marker 1
        windows = []
        for k, (t2, code) in enumerate(mk):
            end = next((t for t, c in mk[k + 1:] if c == "1"), None) if code == "2" else None
            if end is not None and end - t2 >= MIN_HOLD_S:
                windows.append((t2, end - t2))
    rows = []
    T = h["time"][-1]
    for t2, dur in windows:
        if t2 + BASE[0] < 0 or t2 + dur > T:
            continue
        end = t2 + dur
        w = dict(subject=sid, onset=t2, duration=dur)
        for name, (t, y) in {"hr": (tl, hr), "amp_skin": (ts, amp_s), "amp_long": (tl, amp_l)}.items():
            pre = mean_in(t, y, t2 + BASE[0], t2 + BASE[1])
            late = mean_in(t, y, t2 + dur / 2, end)          # second half of the hold
            nadir = np.nanmin(np.interp(np.arange(t2 + 5, end, 0.5), t, y)) if name == "hr" else np.nan
            w[f"{name}_pre"], w[f"{name}_late"] = pre, late
            w[f"{name}_change_pct"] = 100 * (late / pre - 1)
            if name == "hr":
                w["hr_nadir_change_pct"] = 100 * (nadir / pre - 1)
            w[f"_curve_{name}"] = curve(t, y, t2, pre)
        rows.append(w)
    return rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import wilcoxon

    files = sorted(glob.glob(str(BH_DIR / "*.snirf")))
    with Pool(6) as p:
        holds = pd.DataFrame([r for rows in p.map(analyse, files) for r in rows])
        # SHAM control: the same windows (onset, duration) cut from the same person's resting baseline
        bs = {int(re.match(r"(\d+)_", Path(f).name).group(1)): f for f in glob.glob(str(BS_DIR / "*.snirf"))}
        jobs = [(bs[s], list(zip(g.onset, g.duration))) for s, g in holds.groupby("subject") if s in bs]
        sham = pd.DataFrame([r for rows in p.map(analyse, jobs) for r in rows])
    print(f"{len(files)} BH files, {len(holds)} holds >= {MIN_HOLD_S:.0f} s with a marked end, "
          f"{holds.subject.nunique()} people; hold duration median {holds.duration.median():.0f} s "
          f"(range {holds.duration.min():.0f}-{holds.duration.max():.0f})")
    num = [c for c in holds if not c.startswith("_")]
    per = holds[num].groupby("subject").mean()
    per_sham = sham[num].groupby("subject").mean()
    both = per.index.intersection(per_sham.index)
    print("\nper person (mean of their holds), second half of the hold vs 20 s of breathing before;")
    print("SHAM = identical windows in the same person's resting baseline (no breath hold):")
    for c, label in [("hr_change_pct", "heart rate"), ("hr_nadir_change_pct", "heart rate, lowest point"),
                     ("amp_skin_change_pct", "pulse amplitude SKIN 10 mm"),
                     ("amp_long_change_pct", "pulse amplitude 30 mm")]:
        x = per[c].dropna()
        p = wilcoxon(x).pvalue
        print(f"  {label:<28} median {x.median():+6.1f} %  [IQR {x.quantile(.25):+.1f} .. {x.quantile(.75):+.1f}]"
              f"   lower in {100 * (x < 0).mean():3.0f} % of people   Wilcoxon p = {p:.2g}   n = {len(x)}")
        y = per_sham[c].dropna()
        pd_ = wilcoxon(per.loc[both, c], per_sham.loc[both, c], nan_policy="omit").pvalue
        print(f"  {'   SHAM':<28} median {y.median():+6.1f} %  [IQR {y.quantile(.25):+.1f} .. {y.quantile(.75):+.1f}]"
              f"   lower in {100 * (y < 0).mean():3.0f} % of people   hold vs sham p = {pd_:.2g}   n = {len(y)}")
    holds[num].to_csv(ROOT / "results/bh_reflex.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    long_holds = holds[holds.duration >= 25]
    for a, (key, lab, col) in zip(ax, [("hr", "tętno", "#b91c1c"), ("amp_skin", "amplituda pulsu skóry 10 mm", "#d97706")]):
        C = 100 * (np.vstack(long_holds[f"_curve_{key}"]) - 1)
        subj = long_holds.subject.values
        P = np.array([np.nanmean(C[subj == s], 0) for s in np.unique(subj)])  # one curve per person
        m, se = np.nanmean(P, 0), np.nanstd(P, 0) / np.sqrt(np.sum(np.isfinite(P), 0))
        a.fill_between(GRID, m - 1.96 * se, m + 1.96 * se, color=col, alpha=0.2)
        a.plot(GRID, m, color=col, lw=2, label="bezdech")
        S = sham[sham.duration >= 25]
        Cs = 100 * (np.vstack(S[f"_curve_{key}"]) - 1)
        Ps = np.array([np.nanmean(Cs[S.subject.values == s_], 0) for s_ in np.unique(S.subject)])
        a.plot(GRID, np.nanmean(Ps, 0), color="#7b8794", lw=1.5, ls="--", label="kontrola: spoczynek, te same okna")
        a.legend(fontsize=8)
        a.axvline(0, color="k", lw=0.8); a.axhline(0, color="k", lw=0.5, ls=":")
        a.set_xlabel("czas od początku bezdechu [s]"); a.set_ylabel("zmiana vs oddychanie przed [%]")
        a.set_title(f"{lab} (bezdechy ≥ 25 s, n = {len(P)} osób, średnia ± 95% CI)")
    fig.tight_layout()
    fig.savefig(ROOT / "results/bh_reflex.png", dpi=120)
    print(f"\nfigure -> {ROOT / 'results/bh_reflex.png'}  (curves end where the shortest included holds end)")


if __name__ == "__main__":
    main()
