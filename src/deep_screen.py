"""deep_screen.py - exploratory screen in the most homogeneous group (18-24 y), with FDR, bootstrap stability and a
negative control. Adds niche scale-free variability features (1/f slope, DFA, sample entropy, skew/kurtosis,
beat-to-beat pulse amplitude variability, respiratory modulation of the pulse)."""
import glob, re, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy import signal, stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
import haemo
from features import FOREHEAD, load_meta
ROOT = Path(__file__).resolve().parents[1]
SUP = ROOT / "dane/infopack_and_data/fnirs_high_freq_supine/data/baseline"

def sampen(x, m=2, r=0.2):
    x = (x - x.mean()) / x.std(); N = len(x); r = r
    def count(mm):
        E = np.lib.stride_tricks.sliding_window_view(x, mm)[: N - m]
        d = np.max(np.abs(E[:, None, :] - E[None, :, :]), axis=2)
        return (np.sum(d <= r) - len(E)) / 2
    B, A = count(m), count(m + 1)
    return -np.log(A / B) if A > 0 and B > 0 else np.nan

def dfa(x, scales=(8, 12, 16, 24, 32, 48, 64, 96, 128)):
    y = np.cumsum(x - x.mean()); F = []
    for s in scales:
        n = len(y) // s; seg = y[: n * s].reshape(n, s); t = np.arange(s)
        res = [np.sqrt(np.mean((v - np.polyval(np.polyfit(t, v, 1), t)) ** 2)) for v in seg]; F.append(np.mean(res))
    return np.polyfit(np.log(scales), np.log(F), 1)[0]

def niche(path):
    sid = int(re.match(r"(\d+)_", Path(path).name).group(1))
    h = haemo.convert(path, dpf=6.0); fs = h["fs"]
    dead = {c.split()[0] for c, k in zip(h["od_channels"], h["nonpositive"]) if k}
    idx = {n: i for i, n in enumerate(h["pairs"])}
    x = h["hbo"][[idx[n] for n in FOREHEAD if n not in dead]].mean(0)
    out = {"subject": sid}
    d10 = signal.decimate(signal.decimate(x - x.mean(), 7, ftype="fir"), 2, ftype="fir"); f10 = fs / 14   # ~10 Hz
    f, p = signal.welch(signal.detrend(d10), f10, nperseg=int(120 * f10))
    m = (f >= 0.02) & (f <= 0.5); out["spec_slope_1f"] = np.polyfit(np.log(f[m]), np.log(p[m]), 1)[0]
    sos = signal.butter(3, 0.5 / (f10 / 2), output="sos"); slow = signal.sosfiltfilt(sos, d10)[:: 5]   # 2 Hz
    out["dfa_alpha"] = dfa(np.diff(slow)) if len(slow) > 300 else np.nan
    out["sampen_slow"] = sampen(slow[:1200])
    out["skew_slow"], out["kurt_slow"] = stats.skew(slow), stats.kurtosis(slow)
    xc = signal.sosfiltfilt(signal.butter(3, [0.5 / (fs / 2), 8 / (fs / 2)], btype="band", output="sos"), x)
    fq, pp = signal.welch(xc, fs, nperseg=int(20 * fs)); mc = (fq > 0.7) & (fq < 2.5); per = 1 / fq[mc][np.argmax(pp[mc])]
    tr, _ = signal.find_peaks(-xc, distance=int(0.6 * per * fs), prominence=0.3 * xc.std())
    amp = np.array([xc[a:b].max() - xc[a] for a, b in zip(tr[:-1], tr[1:])]); tb = tr[:-1] / fs
    ok = np.abs(amp - np.median(amp)) < 3 * stats.median_abs_deviation(amp)
    amp, tb = amp[ok], tb[ok]
    out["pulse_amp_cv"] = amp.std() / amp.mean()
    out["pulse_amp_succdiff"] = np.sqrt(np.mean(np.diff(amp) ** 2)) / amp.mean()
    g = np.arange(tb[0], tb[-1], 0.25); a4 = np.interp(g, tb, amp / amp.mean())
    fa, pa = signal.welch(a4 - a4.mean(), 4, nperseg=240)
    out["pulse_amp_resp_mod"] = np.trapezoid(pa[(fa > 0.15) & (fa < 0.5)], fa[(fa > 0.15) & (fa < 0.5)]) / np.trapezoid(pa[(fa > 0.01) & (fa < 1.5)], fa[(fa > 0.01) & (fa < 1.5)])
    out["pulse_amp_mayer_mod"] = np.trapezoid(pa[(fa > 0.08) & (fa < 0.15)], fa[(fa > 0.08) & (fa < 0.15)]) / np.trapezoid(pa[(fa > 0.01) & (fa < 1.5)], fa[(fa > 0.01) & (fa < 1.5)])
    return out

if __name__ == "__main__":
    with Pool(6) as p: N = pd.DataFrame(p.map(niche, sorted(glob.glob(str(SUP / "*.snirf")))))
    N.to_csv(ROOT / "results/niche_features.csv", index=False)
    f = pd.read_csv(ROOT / "results/features.csv")
    df = f.merge(pd.read_csv(ROOT / "results/bands_supine.csv"), on="subject", how="left") \
          .merge(pd.read_csv(ROOT / "results/timing_index.csv").drop(columns=[c for c in load_meta().columns if c != "subject"] + ["CRF"], errors="ignore"), on="subject", how="left") \
          .merge(pd.read_csv(ROOT / "results/endo_followup_supine.csv")[["subject", "lr_sync"]], on="subject", how="left").merge(N, on="subject", how="left")
    g = df[df.age.between(18, 24)].copy()
    targets = ["sex_male", "height", "weight", "bmi", "head_circ", "hair_darkness", "hair_curl", "hair_thickness",
               "fitzpatrick", "medication_any", "activity", "treadmill_kmh", "NEG_CONTROL_group"]
    feats = [c for c in df if c not in targets + ["subject", "age", "antihypertensive", "n_ch", "n_beats", "hr_fnirs"] and df[c].dtype != object]
    rng = np.random.default_rng(0); rows = []
    for t in targets:
        for c in feats:
            d = g[[c, t]].dropna()
            if len(d) < 30 or d[t].nunique() < 2 or d[c].nunique() < 5: continue
            r, pv = stats.spearmanr(d[c], d[t]); rows.append(dict(feature=c, target=t, rho=r, p=pv, n=len(d)))
    R = pd.DataFrame(rows).dropna(subset=["p"]); R["q"] = stats.false_discovery_control(R.p)
    cand = R[R.q < 0.10].copy()
    stab = []
    for _, row in cand.iterrows():
        d = g[[row.feature, row.target]].dropna().values; k = 0
        for _ in range(500):
            s = d[rng.integers(0, len(d), len(d))]
            if np.ptp(s[:, 1]) == 0: continue
            rr, pp = stats.spearmanr(s[:, 0], s[:, 1]); k += (pp < 0.05) and (np.sign(rr) == np.sign(row.rho))
        stab.append(k / 500)
    cand["bootstrap_stable"] = stab
    print(f"group 18-24: n = {len(g)}; {len(feats)} features x {len(targets)} targets = {len(R)} tests; p<.05 {(R.p<.05).sum()} (chance {0.05*len(R):.0f}); q<.10 {len(cand)}")
    print("NEG control (Grupa) min q:", round(R[R.target == "NEG_CONTROL_group"].q.min(), 3))
    print(cand.sort_values("q").round(3).to_string(index=False))
    print("\nnew niche features, best hit each:")
    for c in N.columns[1:]:
        x = R[R.feature == c].sort_values("p").head(1)
        if len(x): print(f"  {c:<22} best: {x.target.iloc[0]:<14} rho {x.rho.iloc[0]:+.2f} p {x.p.iloc[0]:.3f} q {x.q.iloc[0]:.2f}")
    R.to_csv(ROOT / "results/deep_screen_18_24.csv", index=False)
    # covariance structure: PCA of scale-free physiology within the group
    phys = ["hr_bpm", "rmssd_ms", "sd1_sd2", "pulse_p1_phase", "pulse_relax_area", "resp_rate_bpm", "pow_mayer_rel", "pow_resp_rel",
            "l30_hbo_endothelial_rel", "l30_hbo_neurogenic_rel", "l30_hbo_myogenic_rel", "TI", "spec_slope_1f", "dfa_alpha", "sampen_slow", "pulse_amp_cv", "pulse_amp_resp_mod"]
    X = g[phys].dropna(); Z = (X - X.mean()) / X.std(); C = np.corrcoef(Z.values.T)
    w, V = np.linalg.eigh(C); w, V = w[::-1], V[:, ::-1]
    print(f"\nPCA within 18-24 (n {len(X)}): variance % of first 5 PCs {np.round(100*w[:5]/w.sum()).astype(int)}")
    meta = g.loc[X.index, ["sex_male", "height", "bmi", "activity"]]
    for k in range(3):
        sc = Z.values @ V[:, k]; top = np.argsort(-np.abs(V[:, k]))[:4]
        print(f"  PC{k+1}: " + ", ".join(f"{phys[i]} {V[i,k]:+.2f}" for i in top) + "  | " + ", ".join(f"{m} {stats.spearmanr(sc, meta[m])[0]:+.2f}" for m in meta))
