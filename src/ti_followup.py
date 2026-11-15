"""ti_followup.py - reflected-wave timing index without anyone on medication: age relation (Pearson + Spearman)
and split-half reliability (first vs second half of the same 10-min recording)."""
import glob, re, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy import signal, stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
import haemo, timing_index as tim
from features import FOREHEAD, cardiac_quality, load_meta
BASE = Path(__file__).resolve().parents[1] / "dane/infopack_and_data/fnirs_high_freq_supine/data/baseline"

def ti_segment(h, t0, t1):
    fs = h["fs"]; a0, a1 = int(t0 * fs), int(t1 * fs)
    dead = {c.split()[0] for c, n in zip(h["od_channels"], h["nonpositive"]) if n}
    idx = {n: i for i, n in enumerate(h["pairs"])}
    ref = tim.bp(h["hbo"][[idx[n] for n in FOREHEAD if n not in dead]].mean(0)[a0:a1], fs, 0.5, 5)
    f, p = signal.welch(ref, fs, nperseg=int(20 * fs)); m = (f > 0.7) & (f < 2.5); period = 1 / f[m][np.argmax(p[m])]
    tr, _ = signal.find_peaks(-ref, distance=int(0.6 * period * fs), prominence=0.3 * ref.std())
    L = int(np.median(np.diff(tr)))
    tr = [a for a, b in zip(tr[:-1], tr[1:]) if abs((b - a) - L) < 0.2 * L and a + L < len(ref)]
    ti = []
    for name, i in idx.items():
        if name in dead or not (20 <= h["distance_mm"][i] <= 35): continue
        seg = h["hbo"][i][a0:a1]
        if cardiac_quality(seg, fs) < 4: continue
        x = tim.bp(seg, fs, 0.5, 5); w = np.mean([x[a:a + L] for a in tr], axis=0); w = w - np.linspace(w[0], w[-1], L)
        ts, tf = tim.channel_times(w, fs)
        if np.isfinite(ts) and 0.040 < tf - ts < 0.400: ti.append(1 / (tf - ts))
    ti = np.array(ti)
    if len(ti) < 3: return np.nan
    keep = np.abs(ti - ti.mean()) <= 1.5 * ti.std() if ti.std() > 0 else np.ones(len(ti), bool)
    return float(ti[keep].mean()) if keep.sum() >= 3 else np.nan

def run(path):
    sid = int(re.match(r"(\d+)_", Path(path).name).group(1)); h = haemo.convert(path, dpf=6.0); T = h["time"][-1]
    return dict(subject=sid, TI_full=ti_segment(h, 0, T), TI_half1=ti_segment(h, 0, T / 2), TI_half2=ti_segment(h, T / 2, T))

if __name__ == "__main__":
    with Pool(6) as p: R = pd.DataFrame(p.map(run, sorted(glob.glob(str(BASE / "*.snirf")))))
    d = R.merge(load_meta(), on="subject")
    for lab, g in (("everyone", d), ("NO medication", d[d.medication_any == 0])):
        x = g.dropna(subset=["TI_full", "age"])
        rp, pp = stats.pearsonr(x.TI_full, x.age); rs, ps = stats.spearmanr(x.TI_full, x.age)
        print(f"{lab:<14} n={len(x)} (age {x.age.min()}-{x.age.max()}): TI vs age Pearson {rp:+.2f} (p {pp:.4f}) | Spearman {rs:+.2f} (p {ps:.4f})")
    g = d[d.medication_any == 0].dropna(subset=["TI_half1", "TI_half2"])
    r_half = stats.spearmanr(g.TI_half1, g.TI_half2)[0]; sb = 2 * r_half / (1 + r_half)
    print(f"reliability (no medication, n={len(g)}): first vs second half rho {r_half:+.2f} -> whole-recording reliability (Spearman-Brown) {sb:.2f}")
    print("medicated people (excluded):"); print(d[d.medication_any == 1][["subject", "age", "antihypertensive", "TI_full"]].round(2).to_string(index=False))
    R.to_csv(Path(__file__).resolve().parents[1] / "results/ti_followup.csv", index=False)
