"""multiverse.py - robustness of the two findings over every reasonable analysis variant (all reported, none selected)."""
import glob, re, sys, itertools
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy import signal
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import haemo, timing_index as tim
from features import load_meta
BASE = Path(__file__).resolve().parents[1] / "dane/infopack_and_data/fnirs_high_freq_supine/data/baseline"
WINS = (200, 300, 400)                                     # Welch window [s]
EDGES = {"0.0095-0.021": (0.0095, 0.021), "0.008-0.02": (0.008, 0.020), "0.01-0.025": (0.010, 0.025)}

def endo(path):
    sid = int(re.match(r"(\d+)_", Path(path).name).group(1))
    h = haemo.convert(path, dpf=6.0); fs = h["fs"]; down = int(fs // 5); fsd = fs / down
    dead = {c.split()[0] for c, n in zip(h["od_channels"], h["nonpositive"]) if n}
    out = {"subject": sid}
    for i, name in enumerate(h["pairs"]):
        if name in dead: continue
        for chrom in ("hbo", "hbr", "hbt"):
            x = h["hbo"][i] + h["hbr"][i] if chrom == "hbt" else h[chrom][i]
            x = signal.detrend(signal.decimate(x - x.mean(), down, ftype="fir"))
            for w in WINS:
                fq, p = signal.welch(x, fsd, nperseg=int(w * fsd))
                tot = np.trapezoid(p[(fq >= 0.008) & (fq <= 2)], fq[(fq >= 0.008) & (fq <= 2)])
                for e, (lo, hi) in EDGES.items():
                    m = (fq >= lo) & (fq <= hi)
                    out[(name, chrom, w, e)] = np.trapezoid(p[m], fq[m]) / tot
    return out

def ti_variant(band):
    lo, hi = band; orig = tim.bp
    tim.bp = lambda x, fs, a, b: orig(x, fs, lo, hi)
    with Pool(6) as p: d = pd.DataFrame(p.map(tim.analyse, sorted(glob.glob(str(BASE / "*.snirf")))))
    tim.bp = orig
    return d

def summarize(label, rhos, ps, sign):
    rhos, ps = np.array(rhos), np.array(ps)
    print(f"{label}: {len(rhos)} variants | same direction {100*np.mean(np.sign(rhos)==sign):.0f}% | "
          f"same direction AND p<0.05 {100*np.mean((np.sign(rhos)==sign)&(ps<.05)):.0f}% | rho median {np.median(rhos):+.2f} "
          f"[min {rhos.min():+.2f}, max {rhos.max():+.2f}]")

if __name__ == "__main__":
    meta = load_meta()
    with Pool(6) as p: E = p.map(endo, sorted(glob.glob(str(BASE / "*.snirf"))))
    keys = sorted({k for e in E for k in e if k != "subject"}, key=str)
    print("=== ENDOTHELIAL BAND vs ACTIVITY (expected +) ===")
    res = []
    for k in keys:
        d = pd.DataFrame([{"subject": e["subject"], "v": e.get(k, np.nan)} for e in E]).merge(meta, on="subject").dropna(subset=["v"])
        if len(d) < 40: continue
        r, pv = spearmanr(d.v, d.activity); res.append(dict(ch=k[0], chrom=k[1], win=k[2], edge=k[3], rho=r, p=pv))
    R = pd.DataFrame(res); R.to_csv("multiverse_endothelial.csv", index=False)
    summarize("ALL variants", R.rho, R.p, +1)
    for col in ("ch", "chrom", "win", "edge"):
        g = R.groupby(col).apply(lambda x: pd.Series({"n": len(x), "rho_median": x.rho.median(), "pct_p05": 100 * np.mean((x.rho > 0) & (x.p < .05))}), include_groups=False)
        print(f"-- by {col}:\n" + g.round(2).to_string())
    print("\n=== TI vs AGE (expected +), per channel and per band-pass ===")
    rr, pp = [], []
    for band in ((0.5, 5), (0.5, 8), (0.8, 5), (0.5, 3.5)):
        d = ti_variant(band).merge(meta, on="subject")
        for c in [c for c in d if c.startswith("ti_")] + ["TI"]:
            x = d[[c, "age"]].dropna()
            if len(x) < 30: continue
            r, pv = spearmanr(x[c], x.age); rr.append(r); pp.append(pv)
            print(f"  band {band} {c:<10} rho {r:+.2f} p {pv:.3f} n {len(x)}")
    summarize("TI ALL variants", rr, pp, +1)
