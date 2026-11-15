"""
timing_index.py - replication of the NIRS timing index TI_NIRS (Afkhami et al. 2021, Sci Rep 11:14812)
on the supine 141.79 Hz baseline.

TI = 1 / (t_refl - t_sys)   [1/s]; reflected wave returns sooner in stiffer arteries -> higher TI = stiffer.
Per channel (all long channels 20-35 mm, cardiac quality >= 4):
  HbO, band-pass 0.5-5 Hz, ensemble average of beats aligned to the beat foot (no ECG here),
  s2 = 2nd derivative; zero-crossings of s2 (ZC1, ZC2, ...);
  t_sys  = earlier of (first local max of the pulse, ZC2)
  t_refl = positive peak of the pulse between ZC3 and ZC4, else the minimum of s2 there
  channel kept if 40 ms < t_refl - t_sys < 400 ms; per person channels outside mean +- 1.5 SD dropped;
  person value = mean over >= 3 channels.
Hypotheses fixed in advance (directions from the paper): TI up with age, TI down with CRF (Jurca 2005).

  python src/timing_index.py      # -> timing_index.csv + summary
"""

from __future__ import annotations

import glob
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import haemo  # noqa: E402
from features import FOREHEAD, cardiac_quality, load_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "dane/infopack_and_data/fnirs_high_freq_supine/data/baseline"


def bp(x, fs, lo, hi):
    return signal.sosfiltfilt(signal.butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos"), x)


def channel_times(w, fs):
    """t_sys, t_refl [s] from one averaged beat (foot at index 0)."""
    s = (w - w.min()) / (np.ptp(w) + 1e-30)
    s2 = np.gradient(np.gradient(signal.savgol_filter(s, max(7, int(0.03 * fs) | 1), 3)))
    zc = np.flatnonzero(np.diff(np.sign(s2)) != 0)
    if len(zc) < 4:
        return np.nan, np.nan
    pk, _ = signal.find_peaks(s, prominence=0.002)
    t_sys = min(pk[0], zc[1]) if len(pk) else zc[1]
    a, b = zc[2], zc[3]
    pk2 = pk[(pk > a) & (pk <= b)]
    t_refl = pk2[0] if len(pk2) else a + int(np.argmin(s2[a:b + 1]))
    return t_sys / fs, t_refl / fs


def analyse(path):
    sid = int(re.match(r"(\d+)_", Path(path).name).group(1))
    h = haemo.convert(path, dpf=6.0)
    fs = h["fs"]
    dead = {c.split()[0] for c, n in zip(h["od_channels"], h["nonpositive"]) if n}
    idx = {n: i for i, n in enumerate(h["pairs"])}
    ref = bp(h["hbo"][[idx[n] for n in FOREHEAD if n not in dead]].mean(0), fs, 0.5, 5)
    f, p = signal.welch(ref, fs, nperseg=int(20 * fs))
    m = (f > 0.7) & (f < 2.5)
    period = 1 / f[m][np.argmax(p[m])]
    tr, _ = signal.find_peaks(-ref, distance=int(0.6 * period * fs), prominence=0.3 * ref.std())
    L = int(np.median(np.diff(tr)))
    tr = [a for a, b in zip(tr[:-1], tr[1:]) if abs((b - a) - L) < 0.2 * L and a + L < len(ref)]
    out = dict(subject=sid, hr_fnirs=60 / period, n_beats=len(tr))
    ti = []
    for name, i in idx.items():
        d = h["distance_mm"][i]
        if name in dead or not (20 <= d <= 35):
            continue
        x = bp(h["hbo"][i], fs, 0.5, 5)
        if cardiac_quality(h["hbo"][i], fs) < 4:
            continue
        w = np.mean([x[a:a + L] for a in tr], axis=0)
        w = w - np.linspace(w[0], w[-1], L)  # remove drift across the beat
        ts, tf = channel_times(w, fs)
        if np.isfinite(ts) and 0.040 < tf - ts < 0.400:
            ti.append(1 / (tf - ts))
            out[f"ti_{name}"] = ti[-1]
    ti = np.array(ti)
    if len(ti) >= 3:
        keep = np.abs(ti - ti.mean()) <= 1.5 * ti.std() if ti.std() > 0 else np.ones(len(ti), bool)
        if keep.sum() >= 3:
            out["TI"] = float(ti[keep].mean())
            out["n_ch"] = int(keep.sum())
    return out


def main():
    from scipy.stats import pearsonr, spearmanr

    with Pool(6) as p:
        df = pd.DataFrame(p.map(analyse, sorted(glob.glob(str(BASE / "*.snirf")))))
    df = df.merge(load_meta(), on="subject")
    # Jurca et al. 2005 non-exercise CRF (METs), formula from the organisers' supplementary material
    df["CRF"] = 2.77 * df.sex_male - 0.10 * df.age - 0.17 * df.bmi - 0.03 * df.hr_fnirs + 1.00 * df.activity + 18.07
    df.to_csv(ROOT / "results/timing_index.csv", index=False)
    d = df.dropna(subset=["TI"])
    print(f"TI obtained for {len(d)}/{len(df)} people (paper: 32/38); channels per person median {d.n_ch.median():.0f}; "
          f"TI median {d.TI.median():.2f} 1/s = {1000 / d.TI.median():.0f} ms between systolic and reflected wave")
    rng = np.random.default_rng(0)

    def test(col, sub=None, label=None):
        x = d if sub is None else d[sub]
        r, p = pearsonr(x.TI, x[col])
        rs, ps = spearmanr(x.TI, x[col])
        perm = np.mean([abs(spearmanr(rng.permutation(x.TI.values), x[col])[0]) >= abs(rs) for _ in range(2000)])
        print(f"  TI vs {label or col:<22} Pearson r {r:+.2f} (p {p:.3f})   Spearman {rs:+.2f} (p {ps:.3f}, perm p {perm:.3f})   n {len(x)}")

    print("PRE-SPECIFIED (paper: age r = +0.53, CRF r = -0.44):")
    test("age")
    test("CRF", label="CRF (Jurca)")
    print("secondary:")
    test("age", d.age < 60, "age (< 60 y)")
    test("CRF", d.age < 60, "CRF (< 60 y)")
    test("activity")
    test("hr_fnirs", label="heart rate")
    test("height")
    test("NEG_CONTROL_group", label="NEG control (Grupa)")


if __name__ == "__main__":
    main()
